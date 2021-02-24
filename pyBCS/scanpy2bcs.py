import scanpy
import h5py
import numpy as np
import scipy
import os
import json
import pandas as pd
import uuid
import time
import shutil
import zipfile
from pandas.api.types import is_numeric_dtype
from abc import ABC, abstractmethod
import abc

class DataObject(ABC):
    def __init__(self, source):
        """Constructor of DataObject class

        Keyword arguments:
            source: Path to input file or folder

        Returns:
            None
        """
        self.source = source

    def get_n_cells(self):
        """Gets the number of cells

        Returns:
            The number of cells of the data
        """
        return len(self.get_barcodes())

    @abc.abstractclassmethod
    def get_barcodes(self):
        """Gets barcode names

        Returns:
            An array contains the barcode names
        """
        pass

    @abc.abstractclassmethod
    def get_raw_barcodes(self):
        """Gets the raw barcode names

        Returns:
            An array contains the raw barcode names
        """
        pass

    @abc.abstractclassmethod
    def get_features(self):
        """Gets the gene names

        Returns:
            An array contains the gene names
        """
        pass

    @abc.abstractclassmethod
    def get_raw_features(self):
        """Get the raw gene names

        Returns:
            An array contains the raw gene names
        """
        pass

    @abc.abstractclassmethod
    def get_raw_matrix(self):
        """Gets the raw matrix

        Returns:
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the matrix
        """
        pass

    @abc.abstractclassmethod
    def get_normalized_matrix(self):
        """Gets the normalized matrix

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the normalized matrix
        """
        pass

    def get_raw_data(self):
        """Gets raw data

        Returns:
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the matrix
            An array contains the raw barcode names
            An array contains the raw gene names
        """
        M = self.get_raw_matrix()
        barcodes = self.get_raw_barcodes()
        features = self.get_raw_features()
        return M, barcodes, features

    def get_normalized_data(self):
        """Gets the normalized data

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the normalized matrix
            An array contains the barcode names
            An array contains the gene names
        """
        M = self.get_normalized_matrix()
        barcodes = self.get_barcodes()
        features = self.get_features()
        return M, barcodes, features

    @abc.abstractclassmethod
    def get_metadata(self):
        """Gets metadata

        Returns:
            A pandas.DataFrame contains the metadata
        """
        pass

    @abc.abstractclassmethod
    def get_dimred(self):
        """Gets dimentional reduction data

        Returns:
            A dictionary whose each value is a numpy.ndarray contains the dimentional reduced data
        """
        pass

    def sync_data(self, norm, raw):
        """Synces normalized and raw data

        Keyword arguments:
            norm: A tuple contains normalized data, which is the output of DataObject.get_normalized_data
            raw: A tuple contains raw data, which is the output of DataObject.get_raw_data

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the synced normalized matrix
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the synced raw matrix
            An array contains the synced barcode names
            An array contains the synced gene names
            A boolean value indicates that if raw data is available
        """
        norm_M, norm_barcodes, norm_features = norm
        raw_M, raw_barcodes, raw_features = raw
        has_raw = True
        if raw_M is None:
            print("Raw data is not available, using normalized data as raw data", flush=True)
            raw_M = norm_M.tocsr()
            barcodes = norm_barcodes
            features = norm_features
            has_raw = False
        elif raw_M.shape == norm_M.shape:
            print("Normalized data and raw data are both available", flush=True)
            barcodes = norm_barcodes
            features = norm_features
        else:
            print("Shape of normalized data (%d, %d) does not equal shape of raw data (%d %d), using raw data as normalized data" % (norm_M.shape + raw_M.shape), flush=True)
            norm_M = raw_M.tocsc()
            barcodes = raw_barcodes
            features = raw_features
        return norm_M, raw_M, barcodes, features, has_raw

    def get_synced_data(self):
        """Gets synced version of normalized and raw data

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the synced normalized matrix
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the synced raw matrix
            An array contains the synced barcode names
            An array contains the synced gene names
            A boolean value indicates that if raw data is available
        """
        norm_M, norm_barcodes, norm_features = self.get_normalized_data()
        try:
            raw_M, raw_barcodes, raw_features = self.get_raw_data()
        except Exception as e:
            print("Cannot read raw data: %s" % str(e))
            raw_M = raw_barcodes = raw_features = None
        return self.sync_data((norm_M, norm_barcodes, norm_features),
                                    (raw_M, raw_barcodes, raw_features))

    def write_metadata(self, zobj, root_name, replace_missing="Unassigned"):
        """Writes metadata to zip file

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file
            replace_missing: A string indicates what missing values in metadata should be named

        Returns:
            None
        """
        print("Writing main/metadata/metalist.json")
        metadata = self.get_metadata()
        for metaname in metadata.columns:
            try:
                metadata[metaname] = pd.to_numeric(metadata[metaname],
                                                    downcast="float")
            except:
                print("Cannot convert %s to numeric, treating as categorical" % metaname)

        content = {}
        all_clusters = {}
        numeric_meta = metadata.select_dtypes(include=["number"]).columns
        category_meta = metadata.select_dtypes(include=["category"]).columns
        for metaname in metadata.columns:
            uid = generate_uuid()

            if metaname in numeric_meta:
                all_clusters[uid] = list(metadata[metaname])
                lengths = 0
                names = "NaN"
                _type = "numeric"
            elif metaname in category_meta:
                if replace_missing not in metadata[metaname].cat.categories:
                    metadata[metaname] = add_category_to_first(metadata[metaname],
                                                                new_category=replace_missing)
                metadata[metaname].fillna(replace_missing, inplace=True)

                value_to_index = {}
                for x, y in enumerate(metadata[metaname].cat.categories):
                    value_to_index[y] = x
                all_clusters[uid] = [value_to_index[x] for x in metadata[metaname]]
                index, counts = np.unique(all_clusters[uid], return_counts = True)
                lengths = np.array([0] * len(metadata[metaname].cat.categories))
                lengths[index] = counts
                lengths = [x.item() for x in lengths]
                _type = "category"
                names = list(metadata[metaname].cat.categories)
            else:
                print("\"%s\" is not numeric or categorical, ignoring" % metaname)
                continue


            content[uid] = {
                "id":uid,
                "name":metaname,
                "clusterLength":lengths,
                "clusterName":names,
                "type":_type,
                "history":[generate_history_object()]
            }

        graph_based_history = generate_history_object()
        graph_based_history["hash_id"] = "graph_based"
        n_cells = self.get_n_cells()
        content["graph_based"] = {
            "id":"graph_based",
            "name":"Graph-based clusters",
            "clusterLength":[0, n_cells],
            "clusterName":["Unassigned", "Cluster 1"],
            "type":"category",
            "history":[graph_based_history]
        }
        with zobj.open(root_name + "/main/metadata/metalist.json", "w") as z:
            z.write(json.dumps({"content":content, "version":1}).encode("utf8"))


        for uid in content:
            print("Writing main/metadata/%s.json" % uid, flush=True)
            if uid == "graph_based":
                clusters = [1] * n_cells
            else:
                clusters = all_clusters[uid]
            obj = {
                "id":content[uid]["id"],
                "name":content[uid]["name"],
                "clusters":clusters,
                "clusterName":content[uid]["clusterName"],
                "clusterLength":content[uid]["clusterLength"],
                "history":content[uid]["history"],
                "type":[content[uid]["type"]]
            }
            with zobj.open(root_name + ("/main/metadata/%s.json" % uid), "w") as z:
                z.write(json.dumps(obj).encode("utf8"))

    def write_dimred(self, zobj, root_name):
        """Writes dimred data to the zip file

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file

        Returns:
            None
        """
        print("Writing dimred")
        data = {}
        default_dimred = None
        dimred_data = self.get_dimred()
        if len(dimred_data.keys()) == 0:
            raise Exception("No dimred data found")
        for dimred in dimred_data:
            print("Writing %s" % dimred)
            matrix = dimred_data[dimred]
            if matrix.shape[1] > 3:
                print("%s has more than 3 dimensions, using only the first 3 of them" % dimred)
                matrix = matrix[:, 0:3]
            n_shapes = matrix.shape

            matrix = [list(map(float, x)) for x in matrix]
            dimred_history = generate_history_object()
            coords = {
                "coords":matrix,
                "name":dimred,
                "id":dimred_history["hash_id"],
                "size":list(n_shapes),
                "param":{"omics":"RNA", "dims":len(n_shapes)},
                "history":[dimred_history]
            }
            if default_dimred is None:
                default_dimred = coords["id"]
            data[coords["id"]] = {
                "name":coords["name"],
                "id":coords["id"],
                "size":coords["size"],
                "param":coords["param"],
                "history":coords["history"]
            }
            with zobj.open(root_name + "/main/dimred/" + coords["id"], "w") as z:
                z.write(json.dumps(coords).encode("utf8"))
        meta = {
            "data":data,
            "version":1,
            "bbrowser_version":"2.7.38",
            "default":default_dimred,
            "description":"Created by converting scanpy to bbrowser format"
        }
        print("Writing main/dimred/meta", flush=True)
        with zobj.open(root_name + "/main/dimred/meta", "w") as z:
            z.write(json.dumps(meta).encode("utf8"))

    def write_matrix(self, zobj, dest_hdf5):
        """Writes expression data to the zip file

        Keyword arguments:
            zobj: The opened-for-write zip file
            dest_hdf5: An opened-for-write hdf5 file where the expression should be written to

        Returns:
            An array that contains the barcode names that are actually written to the hdf5 file
            An array that contains the gene names that are actually written to the hdf5 file
            A boolean indicates that if raw data is available
        """
        #TODO: Reduce memory usage
        norm_M, raw_M, barcodes, features, has_raw = self.get_synced_data()

        print("Writing group \"bioturing\"")
        bioturing_group = dest_hdf5.create_group("bioturing")
        bioturing_group.create_dataset("barcodes",
                                        data=encode_strings(barcodes))
        bioturing_group.create_dataset("features",
                                        data=encode_strings(features))
        bioturing_group.create_dataset("data", data=raw_M.data)
        bioturing_group.create_dataset("indices", data=raw_M.indices)
        bioturing_group.create_dataset("indptr", data=raw_M.indptr)
        bioturing_group.create_dataset("feature_type", data=["RNA".encode("utf8")] * len(features))
        bioturing_group.create_dataset("shape", data=[len(features), len(barcodes)])

        if has_raw:
            print("Writing group \"countsT\"")
            raw_M_T = raw_M.tocsc()
            countsT_group = dest_hdf5.create_group("countsT")
            countsT_group.create_dataset("barcodes",
                                            data=encode_strings(features))
            countsT_group.create_dataset("features",
                                            data=encode_strings(barcodes))
            countsT_group.create_dataset("data", data=raw_M_T.data)
            countsT_group.create_dataset("indices", data=raw_M_T.indices)
            countsT_group.create_dataset("indptr", data=raw_M_T.indptr)
            countsT_group.create_dataset("shape", data=[len(barcodes), len(features)])
        else:
            print("Raw data is not available, ignoring \"countsT\"")

        print("Writing group \"normalizedT\"")
        normalizedT_group = dest_hdf5.create_group("normalizedT")
        normalizedT_group.create_dataset("barcodes",
                                        data=encode_strings(features))
        normalizedT_group.create_dataset("features",
                                        data=encode_strings(barcodes))
        normalizedT_group.create_dataset("data", data=norm_M.data)
        normalizedT_group.create_dataset("indices", data=norm_M.indices)
        normalizedT_group.create_dataset("indptr", data=norm_M.indptr)
        normalizedT_group.create_dataset("shape", data=[len(barcodes), len(features)])

        print("Writing group \"colsum\"")
        norm_M = norm_M.tocsr()
        n_cells = len(barcodes)
        sum_lognorm = np.array([0.0] * n_cells)
        if has_raw:
            sum_log = np.array([0.0] * n_cells)
            sum_raw = np.array([0.0] * n_cells)

        for i in range(n_cells):
            l, r = raw_M.indptr[i:i+2]
            sum_lognorm[i] = np.sum(norm_M.data[l:r])
            if has_raw:
                sum_raw[i] = np.sum(raw_M.data[l:r])
                sum_log[i] = np.sum(np.log2(raw_M.data[l:r] + 1))

        colsum_group = dest_hdf5.create_group("colsum")
        colsum_group.create_dataset("lognorm", data=sum_lognorm)
        if has_raw:
            colsum_group.create_dataset("log", data=sum_log)
            colsum_group.create_dataset("raw", data=sum_raw)
        return barcodes, features, has_raw

    def write_main_folder(self, zobj, root_name):
        """Writes data to "main" folder

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file

        Returns:
            A boolean indicates that if raw data is available
        """
        print("Writing main/matrix.hdf5", flush=True)
        tmp_matrix = "." + str(uuid.uuid4())
        with h5py.File(tmp_matrix, "w") as dest_hdf5:
            barcodes, features, has_raw = self.write_matrix(zobj, dest_hdf5)
        print("Writing to zip", flush=True)
        zobj.write(tmp_matrix, root_name + "/main/matrix.hdf5")
        os.remove(tmp_matrix)

        print("Writing main/barcodes.tsv", flush=True)
        with zobj.open(root_name + "/main/barcodes.tsv", "w") as z:
            z.write("\n".join(barcodes).encode("utf8"))

        print("Writing main/genes.tsv", flush=True)
        with zobj.open(root_name + "/main/genes.tsv", "w") as z:
            z.write("\n".join(features).encode("utf8"))

        print("Writing main/gene_gallery.json", flush=True)
        obj = {"gene":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"gene"},"version":1,"protein":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"protein"}}
        with zobj.open(root_name + "/main/gene_gallery.json", "w") as z:
            z.write(json.dumps(obj).encode("utf8"))
        return has_raw

    def write_runinfo(self, zobj, root_name, unit):
        """Writes run_info.json

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file
            unit: Unit of the study

        Returns:
            None
        """
        print("Writing run_info.json", flush=True)
        runinfo_history = generate_history_object()
        runinfo_history["hash_id"] = root_name
        date = time.time() * 1000
        run_info = {
            "species":"human",
            "hash_id":root_name,
            "version":16,
            "n_cell":self.get_n_cells(),
            "modified_date":date,
            "created_date":date,
            "addon":"SingleCell",
            "matrix_type":"single",
            "n_batch":1,
            "platform":"unknown",
            "omics":["RNA"],
            "title":["Created by bbrowser converter"],
            "history":[runinfo_history],
            "unit":unit
        }
        with zobj.open(root_name + "/run_info.json", "w") as z:
            z.write(json.dumps(run_info).encode("utf8"))

    def write_bcs(self, root_name, output_name, replace_missing="Unassigned"):
        """Writes data to bcs file

        Keyword arguments:
            root_name: Name of the root directory that contains the whole study in the zip file
            output_name: Relative path to output file
            replace_missing: A string indicates what missing values in metadata should be named

        Returns:
            Relative path to output file
        """
        zobj = zipfile.ZipFile(output_name, "w")
        self.write_metadata(zobj, root_name, replace_missing)
        self.write_dimred(zobj, root_name)
        has_raw = self.write_main_folder(zobj, root_name)
        unit = "umi" if has_raw else "lognorm"
        self.write_runinfo(zobj, root_name, unit)
        zobj.close()
        return output_name

class ScanpyData(DataObject):
    def __init__(self, source, raw_key="counts"):
        DataObject.__init__(self, source=source)
        self.object = scanpy.read_h5ad(source, "r")
        self.raw_key = raw_key

    def get_barcodes(self):
        return self.object.obs_names

    def get_features(self):
        return self.object.var_names

    def get_raw_barcodes(self):
        return self.get_barcodes()

    def get_raw_features(self):
        try:
            return self.object.raw.var.index
        except:
            return self.get_features()

    def get_raw_matrix(self):
        try:
            return self.object.raw.X[:][:].tocsr()
        except:
            return self.object.layers[self.raw_key].tocsr()

    def get_normalized_matrix(self):
        return self.object.X[:][:].tocsc()

    def get_metadata(self):
        return self.object.obs

    def get_dimred(self):
        res = {}
        for dimred in self.object.obsm:
            if isinstance(self.object.obsm[dimred], np.ndarray) == False:
                print("%s is not a numpy.ndarray, ignoring" % dimred)
                continue
            res[dimred] = self.object.obsm[dimred]
        return res

class SpringData(DataObject):
    def __init__(self, source):
        DataObject.__init__(self, source=source)
        self.barcodes = None
        self.features = None
        self.raw_barcodes = None
        self.raw_features = None

    def get_barcodes(self):
        if self.barcodes is None:
            try:
                self.barcodes = np.load(os.path.join(self.source,
                                                    "FullDataset_v1",
                                                    "cell_filter.npy"))
                self.barcodes = [str(x) for x in self.barcodes]
            except:
                with open(os.path.join(self.source, "FullDataset_v1",
                                        "cell_filter.txt"),
                            "r") as f:
                    lines = f.readlines()

                self.barcodes = [x[0:-1] for x in lines]
        return self.barcodes

    def get_raw_barcodes(self):
        if self.raw_barcodes is None:
            self.raw_barcodes = self.get_barcodes()
        return self.raw_barcodes

    def get_features(self):
        if self.features is None:
            with open(os.path.join(self.source, "genes.txt"), "r") as f:
                lines = f.readlines()
            self.features = [x[0:-1] for x in lines]
        return self.features

    def get_raw_features(self):
        if self.raw_features is None:
            self.raw_features = self.get_features()
        return self.raw_features

    def get_raw_matrix(self):
        with np.load(os.path.join(self.source, "counts_norm.npz"), "r") as f:
            sparse_format = f["format"].astype("str")
            if sparse_format == "csc":
                return scipy.sparse.csc_matrix((f["data"], f["indices"], f["indptr"]),
                                                shape=f["shape"])\
                                    .tocsr()
            elif sparse_format == "csr":
                return scipy.sparse.csr_matrix((f["data"], f["indices"], f["indptr"]),
                                                shape=f["shape"])
            else:
                raise Exception("Format %s is not supported" % sparse_format)

    def get_normalized_matrix(self):
        with np.load(os.path.join(self.source, "counts_norm.npz"), "r") as f:
            sparse_format = f["format"].astype("str")
            if sparse_format == "csc":
                return scipy.sparse.csc_matrix((f["data"], f["indices"], f["indptr"]),
                                                shape=f["shape"])
            elif sparse_format == "csr":
                return scipy.sparse.csr_matrix((f["data"], f["indices"], f["indptr"]),
                                                shape=f["shape"])\
                                    .tocsc()
            else:
                raise Exception("Format %s is not supported" % sparse_format)

    def get_metadata(self):
        with open(os.path.join(self.source, "FullDataset_v1",
                                "categorical_coloring_data.json"),
                    "r") as f:
            obj = json.load(f)
        for key in obj:
            obj[key] = obj[key]["label_list"]
        metadata = pd.DataFrame.from_dict(obj)
        for meta in metadata:
            if len(np.unique(metadata[meta])) < len(metadata[meta]):
                metadata[meta] = pd.Series(metadata[meta], dtype="category")
        return metadata

    def get_dimred(self):
        df = pd.read_csv(os.path.join(self.source, "FullDataset_v1",
                                        "coordinates.txt"),
                            names=["index", "x", "y"],
                            index_col="index")
        df.index = df.index.map(str)
        coordinates = df.loc[self.get_barcodes(), :].to_numpy()
        return {"coordinates" : coordinates}



def generate_uuid(remove_hyphen=True):
    """Generates a unique uuid string

    Keyword arguments:
        remove_hyphen: True if the hyphens should be removed from the uuid, False otherwise
    """
    res = str(uuid.uuid4())
    if remove_hyphen == True:
        res = res.replace("-", "")
    return res

def encode_strings(strings, encode_format="utf8"):
    """Converts an array/list of strings into utf8 representation
    """
    return [x.encode(encode_format) for x in strings]

def generate_history_object():
    """Generates a Bioturing-format history object
    """
    return {
        "created_by":"bbrowser_format_converter",
        "created_at":time.time() * 1000,
        "hash_id":generate_uuid(),
        "description":"Created by converting scanpy object to bbrowser format"
    }

def add_category_to_first(column, new_category):
    """Adds a new category to a pd.Categorical object

    Keyword arguments:
        column: The pd.Categorical object
        new_category: The new category to be added

    Returns:
        A new pd.Categorical object that is almost the same as the given one,
            except for a new category is added (if it is not already included in the original object).
            The new category is added to first in the categories list.
    """
    if column.dtype.name != "category":
        raise Exception("Object is not a pandas.Categorical")

    if new_category in column.cat.categories:
        raise Exception("%s is already in categories list" % new_category)

    column = column.copy()
    column = column.cat.add_categories(new_category)
    cat = column.cat.categories.tolist()
    cat = cat[0:-1]
    cat.insert(0, new_category)
    column = column.cat.reorder_categories(cat)
    return column

def format_data(source, output_name, input_format="h5ad", raw_key="counts", replace_missing="Unassigned"):
    study_id = generate_uuid(remove_hyphen=False)
    if input_format == "h5ad":
        data_object = ScanpyData(source, raw_key)
    elif input_format == "spring":
        data_object = SpringData(source)
    else:
        raise Exception("Invalid input format: %s" % input_format)
    return data_object.write_bcs(root_name=study_id, output_name=output_name,
                                    replace_missing=replace_missing)
