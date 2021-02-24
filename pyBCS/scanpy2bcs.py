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
    def __init__(self, source, root_name, output_name, replace_missing="Unassigned"):
        self.source = source
        self.root_name = root_name
        self.zobj = zipfile.ZipFile(output_name, "w")
        self.output_name = output_name
        self.replace_missing = replace_missing

    def get_n_cells(self):
        return len(self.get_barcodes())

    @abc.abstractclassmethod
    def get_barcodes(self):
        pass

    @abc.abstractclassmethod
    def get_raw_barcodes(self):
        pass

    @abc.abstractclassmethod
    def get_features(self):
        pass

    @abc.abstractclassmethod
    def get_raw_features(self):
        pass

    @abc.abstractclassmethod
    def get_raw_matrix(self):
        pass

    @abc.abstractclassmethod
    def get_normalized_matrix(self):
        pass

    @abc.abstractclassmethod
    def get_raw_data(self):
        pass

    @abc.abstractclassmethod
    def get_normalized_data(self):
        pass

    @abc.abstractclassmethod
    def get_metadata(self):
        pass

    @abc.abstractclassmethod
    def get_dimred(self):
        pass

    def sync_data(self, norm, raw):
        norm_M, norm_barcodes, norm_features = norm
        raw_M, raw_barcodes, raw_features = raw
        has_raw = True
        if raw_M is None:
            raw_M = norm_M.tocsr()
            barcodes = norm_barcodes
            features = norm_features
            has_raw = False
        elif raw_M.shape == norm_M.shape:
            barcodes = norm_barcodes
            features = norm_features
        else:
            norm_M = raw_M.tocsc()
            barcodes = raw_barcodes
            features = raw_features
        return norm_M, raw_M, barcodes, features, has_raw

    def get_synced_data(self):
        norm_M, norm_barcodes, norm_features = self.get_normalized_data()
        try:
            raw_M, raw_barcodes, raw_features = self.get_raw_data()
        except Exception as e:
            print("Cannot read raw data: %s" % str(e))
            raw_M = raw_barcodes = raw_features = None
        return self.sync_data((norm_M, norm_barcodes, norm_features),
                                    (raw_M, raw_barcodes, raw_features))

    def write_metadata(self):
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
                if self.replace_missing not in metadata[metaname].cat.categories:
                    metadata[metaname] = add_category_to_first(metadata[metaname],
                                                                new_category=self.replace_missing)
                metadata[metaname].fillna(self.replace_missing, inplace=True)

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
        with self.zobj.open(self.root_name + "/main/metadata/metalist.json", "w") as z:
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
            with self.zobj.open(self.root_name + ("/main/metadata/%s.json" % uid), "w") as z:
                z.write(json.dumps(obj).encode("utf8"))

    def write_dimred(self):
        print("Writing dimred")
        data = {}
        default_dimred = None
        dimred_data = self.get_dimred()
        if len(dimred_data.keys()) == 0:
            raise Exception("No dimred data found")
        for dimred in dimred_data:
            print("--->Writing %s" % dimred)
            matrix = dimred_data[dimred]
            if matrix.shape[1] > 3:
                print("--->%s has more than 3 dimensions, using only the first 3 of them" % dimred)
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
            with self.zobj.open(self.root_name + "/main/dimred/" + coords["id"], "w") as z:
                z.write(json.dumps(coords).encode("utf8"))
        meta = {
            "data":data,
            "version":1,
            "bbrowser_version":"2.7.38",
            "default":default_dimred,
            "description":"Created by converting scanpy to bbrowser format"
        }
        print("Writing main/dimred/meta", flush=True)
        with self.zobj.open(self.root_name + "/main/dimred/meta", "w") as z:
            z.write(json.dumps(meta).encode("utf8"))

    def write_matrix(self, dest_hdf5):
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

    def write_main_folder(self):
        print("Writing main/matrix.hdf5", flush=True)
        tmp_matrix = "." + str(uuid.uuid4())
        with h5py.File(tmp_matrix, "w") as dest_hdf5:
            barcodes, features, has_raw = self.write_matrix(dest_hdf5)
        print("Writing to zip", flush=True)
        self.zobj.write(tmp_matrix, self.root_name + "/main/matrix.hdf5")
        os.remove(tmp_matrix)

        print("Writing main/barcodes.tsv", flush=True)
        with self.zobj.open(self.root_name + "/main/barcodes.tsv", "w") as z:
            z.write("\n".join(barcodes).encode("utf8"))

        print("Writing main/genes.tsv", flush=True)
        with self.zobj.open(self.root_name + "/main/genes.tsv", "w") as z:
            z.write("\n".join(features).encode("utf8"))

        print("Writing main/gene_gallery.json", flush=True)
        obj = {"gene":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"gene"},"version":1,"protein":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"protein"}}
        with self.zobj.open(self.root_name + "/main/gene_gallery.json", "w") as z:
            z.write(json.dumps(obj).encode("utf8"))
        return has_raw

    def write_runinfo(self, unit):
        print("Writing run_info.json", flush=True)
        runinfo_history = generate_history_object()
        runinfo_history["hash_id"] = self.root_name
        date = time.time() * 1000
        run_info = {
            "species":"human",
            "hash_id":self.root_name,
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
        with self.zobj.open(self.root_name + "/run_info.json", "w") as z:
            z.write(json.dumps(run_info).encode("utf8"))

    def write_bcs(self):
        self.write_metadata()
        self.write_dimred()
        has_raw = self.write_main_folder()
        unit = "umi" if has_raw else "lognorm"
        self.write_runinfo(unit)
        return self.output_name

class ScanpyData(DataObject):
    def __init__(self, source, root_name, output_name, raw_key="counts"):
        DataObject.__init__(self, source=source, root_name=root_name,
                            output_name=output_name)
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

    def get_raw_data(self):
        M = self.get_raw_matrix()
        barcodes = self.get_raw_barcodes()
        features = self.get_raw_features()
        return M, barcodes, features

    def get_normalized_data(self):
        M = self.get_normalized_matrix()
        barcodes = self.get_barcodes()
        features = self.get_features()
        return M, barcodes, features

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


def generate_uuid(remove_hyphen=True):
    """Generates a unique uuid string

    Keyword arguments:
        remove_hyphen: True if the hyphens should be removed from the uuid, False otherwise
    """
    res = str(uuid.uuid4())
    if remove_hyphen == True:
        res = res.replace("-", "")
    return res

#def get_barcodes(scanpy_obj):
#    """Reads barcodes from a scanpy object
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the barcodes
#
#    Returns:
#        A numpy array that contains the barcodes
#    """
#    return scanpy_obj.obs_names
#
#def get_features(scanpy_obj):
#    """Reads feature names/ids from a scanpy object
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the features
#
#    Returns:
#        A numpy array that contains the feature names
#    """
#    return scanpy_obj.var_names
#
#def get_raw_features(scanpy_obj):
#    """Reads original/raw feature names (before filtering) from a scanpy object
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the features
#
#    Returns:
#        A numpy array that contains the feature names
#    """
#    return scanpy_obj.raw.var.index
#
#def get_raw_from_rawX(scanpy_obj):
#    """Reads raw data from slot 'raw.X' in a scanpy object
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the data
#
#    Returns:
#        A scipy.sparse.csr_matrix that represents a matrix of size (cells x genes)
#        A numpy array that contains the cell names of the matrix
#        A numpy array that contains the gene names of the matrix
#    """
#    M = scanpy_obj.raw.X[:][:].tocsr()
#    barcodes = get_barcodes(scanpy_obj)
#    features = get_raw_features(scanpy_obj)
#    return M, barcodes, features
#
#def get_raw_from_layers(scanpy_obj, raw_key):
#    """Reads raw data from a 'layers' in a scanpy object
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the data
#        raw_key: The key of the raw data in slot 'layers'
#
#    Returns:
#        A scipy.sparse.csr_matrix that represents a matrix of size (cells x genes)
#        A numpy array that contains the cell names of the matrix
#        A numpy array that contains the gene names of the matrix
#    """
#    M = scanpy_obj.layers[raw_key].tocsr()
#    barcodes = get_barcodes(scanpy_obj)
#    features = get_features(scanpy_obj)
#    return M, barcodes, features
#
#def get_raw_data(scanpy_obj, raw_key="auto"):
#    """Finds and reads raw data from a scanpy object
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the data
#        raw_key: A string that specifies where to look for raw data
#                    'raw.X': Reads raw data from scanpy_obj.raw.X
#                    'auto': Looks for raw data in scanpy_obj.raw.X,
#                            scanpy_obj.layers["counts"] and scanpy_obj.layers["raw"]
#                    other: Reads raw data in scanpy_obj.layers[raw_key]
#
#    Returns:
#        A scipy.sparse.csr_matrix that represents a matrix of size (cells x genes)
#        A numpy array that contains the cell names of the matrix
#        A numpy array that contains the gene names of the matrix
#    """
#    if raw_key == "auto":
#        try:
#            res = get_raw_from_rawX(scanpy_obj)
#        except Exception as e:
#            print("--->Error when reading \"raw.X\": ", e)
#            print("--->Trying possible keys")
#            candidate_keys = ["counts", "raw"]
#            for key in candidate_keys:
#                try:
#                    res = get_raw_from_layers(scanpy_obj, key)
#                except:
#                    continue
#                print("--->Found raw data in /layers/%s" % key)
#                return res
#            raise Exception("Raw data not found")
#    elif raw_key == "raw.X":
#        res = get_raw_from_rawX(scanpy_obj)
#    else:
#        res = get_raw_from_layers(scanpy_obj, raw_key)
#    return res
#
#def get_normalized_data(scanpy_obj, raw_data, raw_barcodes, raw_features):
#    """Reads normalized data from a scanpy object or if in need, obtains it by normalizing raw data
#
#    Keyword arguments:
#        scanpy_obj: The scanpy obj that stores the data
#        raw_data: A scipy.sparse matrix of size (cells x genes) that contains raw data
#        raw_barcodes: An array of barcodes of raw data
#        raw_features: An array of feature names of raw data
#
#    Returns:
#        A scipy.sparse.csc_matrix of shape (cells x genes) that stores log-normalized data
#        An array of barcodes of normalized data
#        An array of feature names of normalized data
#    """
#    M = scanpy_obj.X[:][:].tocsc()
#    norm_barcodes = scanpy_obj.obs.index
#    norm_features = scanpy_obj.var.index
#    if (raw_data is None) or (M.shape == raw_data.shape):
#        return M, norm_barcodes, norm_features
#    else:
#        return raw_data.tocsc(), raw_barcodes, raw_features

def encode_strings(strings, encode_format="utf8"):
    """Converts an array/list of strings into utf8 representation
    """
    return [x.encode(encode_format) for x in strings]

#def write_matrix(scanpy_obj, dest_hdf5, raw_key="auto"):
#    """Extracts data from a scanpy object and writes to /main/matrix.hdf5
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the data
#        dest_hdf5: An opened-for-write h5py.File
#        raw_key: Where to look for raw data in the scanpy object. See 'get_raw_data'
#                    for mor details
#
#    Returns:
#        An array that contains the cell barcodes
#        An array that contains the gene names
#        A boolean variable indicating if the scanpy object has raw data
#    """
#    try:
#        raw_M, barcodes, features = get_raw_data(scanpy_obj, raw_key)
#    except Exception as e:
#        print("--->Cannot read raw data: %s" % str(e))
#        raw_M = barcodes = features = None
#
#    norm_M, barcodes, features = get_normalized_data(scanpy_obj, raw_M,
#                                                        barcodes,
#                                                        features)
#
#    if raw_M is None:
#        has_raw = False
#        print("--->Using normalized data as raw data")
#        raw_M = norm_M.tocsr()
#    else:
#        has_raw = True
#
#    print("--->Writing group \"bioturing\"")
#    bioturing_group = dest_hdf5.create_group("bioturing")
#    bioturing_group.create_dataset("barcodes",
#                                    data=encode_strings(barcodes))
#    bioturing_group.create_dataset("features",
#                                    data=encode_strings(features))
#    bioturing_group.create_dataset("data", data=raw_M.data)
#    bioturing_group.create_dataset("indices", data=raw_M.indices)
#    bioturing_group.create_dataset("indptr", data=raw_M.indptr)
#    bioturing_group.create_dataset("feature_type", data=["RNA".encode("utf8")] * len(features))
#    bioturing_group.create_dataset("shape", data=[len(features), len(barcodes)])
#
#    if has_raw:
#        print("--->Writing group \"countsT\"")
#        raw_M_T = raw_M.tocsc()
#        countsT_group = dest_hdf5.create_group("countsT")
#        countsT_group.create_dataset("barcodes",
#                                        data=encode_strings(features))
#        countsT_group.create_dataset("features",
#                                        data=encode_strings(barcodes))
#        countsT_group.create_dataset("data", data=raw_M_T.data)
#        countsT_group.create_dataset("indices", data=raw_M_T.indices)
#        countsT_group.create_dataset("indptr", data=raw_M_T.indptr)
#        countsT_group.create_dataset("shape", data=[len(barcodes), len(features)])
#    else:
#        print("--->Raw data is not available, ignoring \"countsT\"")
#
#    print("--->Writing group \"normalizedT\"")
#    normalizedT_group = dest_hdf5.create_group("normalizedT")
#    normalizedT_group.create_dataset("barcodes",
#                                    data=encode_strings(features))
#    normalizedT_group.create_dataset("features",
#                                    data=encode_strings(barcodes))
#    normalizedT_group.create_dataset("data", data=norm_M.data)
#    normalizedT_group.create_dataset("indices", data=norm_M.indices)
#    normalizedT_group.create_dataset("indptr", data=norm_M.indptr)
#    normalizedT_group.create_dataset("shape", data=[len(barcodes), len(features)])
#
#    print("--->Writing group \"colsum\"")
#    norm_M = norm_M.tocsr()
#    n_cells = len(barcodes)
#    sum_lognorm = np.array([0.0] * n_cells)
#    if has_raw:
#        sum_log = np.array([0.0] * n_cells)
#        sum_raw = np.array([0.0] * n_cells)
#
#    for i in range(n_cells):
#        l, r = raw_M.indptr[i:i+2]
#        sum_lognorm[i] = np.sum(norm_M.data[l:r])
#        if has_raw:
#            sum_raw[i] = np.sum(raw_M.data[l:r])
#            sum_log[i] = np.sum(np.log2(raw_M.data[l:r] + 1))
#
#    colsum_group = dest_hdf5.create_group("colsum")
#    colsum_group.create_dataset("lognorm", data=sum_lognorm)
#    if has_raw:
#        colsum_group.create_dataset("log", data=sum_log)
#        colsum_group.create_dataset("raw", data=sum_raw)
#    return barcodes, features, has_raw

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

#def write_metadata(scanpy_obj, dest, zobj, replace_missing="Unassigned"):
#    """Reads metadata from a scanpy object and writes to /main/metadata
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the metadata
#        dest: The directory that stores the study after the extraction
#        zobj: A opened-for-write ZipFile object that stores the compressed study
#        replace_missing: Replace nan/missing values by this value
#
#    Returns:
#        None
#    """
#    print("Writing main/metadata/metalist.json")
#    metadata = scanpy_obj.obs.copy()
#    for metaname in metadata.columns:
#        try:
#            metadata[metaname] = pd.to_numeric(metadata[metaname], downcast="float")
#        except:
#            print("--->Cannot convert %s to numeric, treating as categorical" % metaname)
#
#    content = {}
#    all_clusters = {}
#    numeric_meta = metadata.select_dtypes(include=["number"]).columns
#    category_meta = metadata.select_dtypes(include=["category"]).columns
#    for metaname in metadata.columns:
#        uid = generate_uuid()
#
#        if metaname in numeric_meta:
#            all_clusters[uid] = list(metadata[metaname])
#            lengths = 0
#            names = "NaN"
#            _type = "numeric"
#        elif metaname in category_meta:
#            if replace_missing not in metadata[metaname].cat.categories:
#                metadata[metaname] = add_category_to_first(metadata[metaname],
#                                                            new_category=replace_missing)
#            metadata[metaname].fillna(replace_missing, inplace=True)
#
#            value_to_index = {}
#            for x, y in enumerate(metadata[metaname].cat.categories):
#                value_to_index[y] = x
#            all_clusters[uid] = [value_to_index[x] for x in metadata[metaname]]
#            index, counts = np.unique(all_clusters[uid], return_counts = True)
#            lengths = np.array([0] * len(metadata[metaname].cat.categories))
#            lengths[index] = counts
#            lengths = [x.item() for x in lengths]
#            _type = "category"
#            names = list(metadata[metaname].cat.categories)
#        else:
#            print("--->\"%s\" is not numeric or categorical, ignoring" % metaname)
#            continue
#
#
#        content[uid] = {
#            "id":uid,
#            "name":metaname if metaname != "seurat_clusters" else "Graph-based clusters",
#            "clusterLength":lengths,
#            "clusterName":names,
#            "type":_type,
#            "history":[generate_history_object()]
#        }
#
#    graph_based_history = generate_history_object()
#    graph_based_history["hash_id"] = "graph_based"
#    n_cells = scanpy_obj.n_obs
#    content["graph_based"] = {
#        "id":"graph_based",
#        "name":"Graph-based clusters",
#        "clusterLength":[0, n_cells],
#        "clusterName":["Unassigned", "Cluster 1"],
#        "type":"category",
#        "history":[graph_based_history]
#    }
#    with zobj.open(dest + "/main/metadata/metalist.json", "w") as z:
#        z.write(json.dumps({"content":content, "version":1}).encode("utf8"))
#
#
#    for uid in content:
#        print("Writing main/metadata/%s.json" % uid, flush=True)
#        if uid == "graph_based":
#            clusters = [1] * n_cells
#        else:
#            clusters = all_clusters[uid]
#        obj = {
#            "id":content[uid]["id"],
#            "name":content[uid]["name"],
#            "clusters":clusters,
#            "clusterName":content[uid]["clusterName"],
#            "clusterLength":content[uid]["clusterLength"],
#            "history":content[uid]["history"],
#            "type":[content[uid]["type"]]
#        }
#        with zobj.open(dest + ("/main/metadata/%s.json" % uid), "w") as z:
#            z.write(json.dumps(obj).encode("utf8"))

#def write_main_folder(scanpy_obj, dest, zobj, raw_key="auto"):
#    """Reads data from a scanpy object and write it to /main
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the data
#        dest: The directory that stores the study after the extraction
#        zobj: A opened-for-write ZipFile object that stores the compressed study
#        raw_key: Where to look for raw data. See 'get_raw_data' for more details
#
#    Returns:
#        None
#    """
#    print("Writing main/matrix.hdf5", flush=True)
#    tmp_matrix = "." + str(uuid.uuid4())
#    with h5py.File(tmp_matrix, "w") as dest_hdf5:
#        barcodes, features, has_raw = write_matrix(scanpy_obj, dest_hdf5,
#                                                    raw_key=raw_key)
#    print("--->Writing to zip", flush=True)
#    zobj.write(tmp_matrix, dest + "/main/matrix.hdf5")
#    os.remove(tmp_matrix)
#
#    print("Writing main/barcodes.tsv", flush=True)
#    with zobj.open(dest + "/main/barcodes.tsv", "w") as z:
#        z.write("\n".join(barcodes).encode("utf8"))
#
#    print("Writing main/genes.tsv", flush=True)
#    with zobj.open(dest + "/main/genes.tsv", "w") as z:
#        z.write("\n".join(features).encode("utf8"))
#
#    print("Writing main/gene_gallery.json", flush=True)
#    obj = {"gene":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"gene"},"version":1,"protein":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"protein"}}
#    with zobj.open(dest + "/main/gene_gallery.json", "w") as z:
#        z.write(json.dumps(obj).encode("utf8"))
#    return has_raw

#def write_dimred(scanpy_obj, dest, zobj):
#    """Reads dimred data from a scanpy object and writes it to /main/dimred
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the dimred
#        dest: The directory that stores the study after the extraction
#        zobj: A opened-for-write ZipFile object that stores the compressed study
#
#    Returns:
#        None
#    """
#    print("Writing dimred")
#    data = {}
#    default_dimred = None
#    has_dimred = False
#    for dimred in scanpy_obj.obsm:
#        if isinstance(scanpy_obj.obsm[dimred], np.ndarray) == False:
#            print("--->%s is not a numpy.ndarray, ignoring" % dimred)
#            continue
#        print("--->Writing %s" % dimred)
#        matrix = scanpy_obj.obsm[dimred]
#        if matrix.shape[1] > 3:
#            print("--->%s has more than 3 dimensions, using only the first 3 of them" % dimred)
#            matrix = matrix[:, 0:3]
#        n_shapes = matrix.shape
#
#        matrix = [list(map(float, x)) for x in matrix]
#        dimred_history = generate_history_object()
#        coords = {
#            "coords":matrix,
#            "name":dimred,
#            "id":dimred_history["hash_id"],
#            "size":list(n_shapes),
#            "param":{"omics":"RNA", "dims":len(n_shapes)},
#            "history":[dimred_history]
#        }
#        if default_dimred is None:
#            default_dimred = coords["id"]
#        data[coords["id"]] = {
#            "name":coords["name"],
#            "id":coords["id"],
#            "size":coords["size"],
#            "param":coords["param"],
#            "history":coords["history"]
#        }
#        with zobj.open(dest + "/main/dimred/" + coords["id"], "w") as z:
#            z.write(json.dumps(coords).encode("utf8"))
#        has_dimred = True
#    if has_dimred == False:
#        raise Exception("No embeddings in \"obsm\" found")
#    meta = {
#        "data":data,
#        "version":1,
#        "bbrowser_version":"2.7.38",
#        "default":default_dimred,
#        "description":"Created by converting scanpy to bbrowser format"
#    }
#    print("Writing main/dimred/meta", flush=True)
#    with zobj.open(dest + "/main/dimred/meta", "w") as z:
#        z.write(json.dumps(meta).encode("utf8"))


#def write_runinfo(scanpy_obj, dest, study_id, zobj, unit="umi"):
#    """Writes runinfo.json
#
#    Keyword arguments:
#        scanpy_obj: The scanpy object that stores the data
#        dest: The directory that stores the study after the extraction
#        study_id: A uuid-generated study id
#        zobj: A opened-for-write ZipFile object that stores the compressed study
#        unit: Unit of the dataset, must be in ["umi", "lognorm"]
#    """
#    print("Writing run_info.json", flush=True)
#    runinfo_history = generate_history_object()
#    runinfo_history["hash_id"] = study_id
#    date = time.time() * 1000
#    run_info = {
#        "species":"human",
#        "hash_id":study_id,
#        "version":16,
#        "n_cell":scanpy_obj.n_obs,
#        "modified_date":date,
#        "created_date":date,
#        "addon":"SingleCell",
#        "matrix_type":"single",
#        "n_batch":1,
#        "platform":"unknown",
#        "omics":["RNA"],
#        "title":["Created by bbrowser converter"],
#        "history":[runinfo_history],
#        "unit":unit
#    }
#    with zobj.open(dest + "/run_info.json", "w") as z:
#        z.write(json.dumps(run_info).encode("utf8"))

#def format_data(source, output_name, raw_key="auto"):
#    """Converts from a scanpy object to BioTuring Compressed Study format
#
#    Keyword arguments:
#        source: Path to the scanpy object
#        output_name: Path to output file
#        raw_key: A string that specifies where to look for raw data
#                    'raw.X': Reads raw data from scanpy_obj.raw.X
#                    'auto': Looks for raw data in scanpy_obj.raw.X,
#                            scanpy_obj.layers["counts"] and scanpy_obj.layers["raw"]
#                    other: Reads raw data in scanpy_obj.layers[raw_key]
#
#    Returns:
#        Path to output file
#    """
#    scanpy_obj = scanpy.read_h5ad(source, "r")
#    zobj = zipfile.ZipFile(output_name, "w")
#    study_id = generate_uuid(remove_hyphen=False)
#    dest = study_id
#    with h5py.File(source, "r") as s:
#        write_metadata(scanpy_obj, dest, zobj)
#        write_dimred(scanpy_obj, dest, zobj)
#        has_raw = write_main_folder(scanpy_obj, dest, zobj, raw_key=raw_key)
#        unit = "umi" if has_raw else "lognorm"
#        write_runinfo(scanpy_obj, dest, study_id, zobj, unit)
#
#    return output_name

def format_data(source, output_name, input_format="h5ad", raw_key="counts"):
    study_id = generate_uuid(remove_hyphen=False)
    if input_format == "h5ad":
        data_object = ScanpyData(source, study_id, output_name, raw_key)
    else:
        raise Exception("Invalid input format: %s" % input_format)
    return data_object.write_bcs()
