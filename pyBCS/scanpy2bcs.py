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

def copy_dataset(source, dest, name, chunk_size=16*1024):
    dataset = dest.create_dataset(name, shape=source[name].shape, dtype=source[name].dtype)
    i = 0
    while i + chunk_size <= source[name].shape[0]:
        dataset[i:i+chunk_size] = source[name][i:i+chunk_size]
        i += chunk_size

    if i < source[name].shape[0]:
        dataset[i:] = source[name][i:]


def generate_uuid(remove_hyphen=True):
    res = str(uuid.uuid4())
    if remove_hyphen == True:
        res = res.replace("-", "")
    return res

def write_matrix(source_hdf5, dest_hdf5):
    print("--->Writing group \"bioturing\"")
    bioturing_group = dest_hdf5.create_group("bioturing")
    barcodes = [x.encode("utf8") for x in source_hdf5["obs"]["index"][:]]
    bioturing_group.create_dataset("barcodes", data=barcodes)
    features = [x.encode("utf8") for x in source_hdf5["var"]["index"][:]]
    bioturing_group.create_dataset("features", data=features)
    M = scipy.sparse.csr_matrix((source_hdf5["layers"]["counts"]["data"],
                                 source_hdf5["layers"]["counts"]["indices"],
                                 source_hdf5["layers"]["counts"]["indptr"]),
                                shape=(len(features), len(barcodes)))
    M = M.tocsc()
    bioturing_group.create_dataset("data", data=M.data)
    bioturing_group.create_dataset("indices", data=M.indices)
    bioturing_group.create_dataset("indptr", data=M.indptr)
    bioturing_group.create_dataset("feature_type", data=["RNA".encode("utf8")] * len(features))
    bioturing_group.create_dataset("shape", data=[len(features), len(barcodes)])

    print("--->Writing group \"countsT\"")
    countsT_group = dest_hdf5.create_group("countsT")
    countsT_group.create_dataset("barcodes", data=features)
    countsT_group.create_dataset("features", data=barcodes)
    copy_dataset(source_hdf5["layers"]["counts"], countsT_group, "indptr")
    copy_dataset(source_hdf5["layers"]["counts"], countsT_group, "indices")
    copy_dataset(source_hdf5["layers"]["counts"], countsT_group, "data")
    countsT_group.create_dataset("shape", data=[len(barcodes), len(features)])

    print("--->Writing group \"normalizedT\"")
    normalizedT_group = dest_hdf5.create_group("normalizedT")
    normalizedT_group.create_dataset("barcodes", data=features)
    normalizedT_group.create_dataset("features", data=barcodes)
    copy_dataset(source_hdf5["X"], normalizedT_group, "indptr")
    copy_dataset(source_hdf5["X"], normalizedT_group, "indices")
    copy_dataset(source_hdf5["X"], normalizedT_group, "data")
    normalizedT_group.create_dataset("shape", data=[len(barcodes), len(features)])

def generate_history_object():
    return {
        "created_by":"bbrowser_format_converter",
        "created_at":time.time() * 1000,
        "hash_id":generate_uuid(),
        "description":"Created by converting scanpy object to bbrowser format"
    }

def is_all_numeric(a):
    try:
        a = [float(x) for x in a]
    except:
        return False
    return True

def convert_to_int(a):
    try:
        return [int(x) for x in a]
    except:
        return None

def convert_to_float(a):
    try:
        return [float(x) for x in a]
    except:
        return None

def write_metadata(source_hdf5, dest, zobj):
    print("Writing main/metadata/metalist.json")
    content = {}
    categories = source_hdf5["obs"]["__categories"]
    n_cells = len(source_hdf5["obs"]["index"])
    all_clusters = {}
    for metaname in source_hdf5["obs"]:
        if metaname in ["__categories", "index"]:
            continue
        if len(np.unique(categories[metaname])) == n_cells:
            print("--->Ignore metadata %s" % metaname)
            continue
        uid = generate_uuid()
        if metaname in categories:
            if is_all_numeric(categories[metaname][:]):
                values = convert_to_int(categories[metaname][:])
                if values is None:
                    values = convert_to_float(categories[metaname][:])
                values = np.array(values)
                all_clusters[uid] = values[source_hdf5["obs"][metaname][:]]
            else:
                all_clusters[uid] = source_hdf5["obs"][metaname][:] + 1
        else:
            all_clusters[uid] = source_hdf5["obs"][metaname][:]

        all_clusters[uid] = [x.item() for x in all_clusters[uid]]

        if metaname in categories and not is_all_numeric(categories[metaname][:]):
            names = ["Unassigned"] + list(categories[metaname][:])
            lengths = np.array([0] * len(names))
            ids, counts = np.unique(source_hdf5["obs"][metaname][:], return_counts=True)
            lengths[ids + 1] = counts
            lengths = [x.item() for x in lengths]
            _type = "category"
        else:
            lengths = 0
            names = "NaN"
            _type = "numeric"
        content[uid] = {
            "id":uid,
            "name":metaname if metaname != "seurat_clusters" else "Graph-based clusters",
            "clusterLength":lengths,
            "clusterName":names,
            "type":_type,
            "history":[generate_history_object()]
        }

    graph_based_history = generate_history_object()
    graph_based_history["hash_id"] = "graph_based"
    content["graph_based"] = {
        "id":"graph_based",
        "name":"Graph-based clusters",
        "clusterLength":[0, n_cells],
        "clusterName":["Unassigned", "Cluster 1"],
        "type":"category",
        "history":[graph_based_history]
    }
    with zobj.open(dest + "/main/metadata/metalist.json", "w") as z:
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
        with zobj.open(dest + ("/main/metadata/%s.json" % uid), "w") as z:
            z.write(json.dumps(obj).encode("utf8"))

def write_main_folder(source_hdf5, dest, zobj):
    print("Writing main/matrix.hdf5", flush=True)
    tmp_matrix = "." + str(uuid.uuid4())
    with h5py.File(tmp_matrix, "w") as dest_hdf5:
        write_matrix(source_hdf5, dest_hdf5)
    zobj.write(tmp_matrix, dest + "/main/matrix.hdf5")
    os.remove(tmp_matrix)

    print("Writing main/barcodes.tsv", flush=True)
    barcodes = "\n".join(source_hdf5["obs"]["index"][:]).encode("utf8")
    with zobj.open(dest + "/main/barcodes.tsv", "w") as z:
        z.write(barcodes)

    print("Writing main/genes.tsv", flush=True)
    features = "\n".join(source_hdf5["var"]["index"][:]).encode("utf8")
    with zobj.open(dest + "/main/genes.tsv", "w") as z:
        z.write(features)

    print("Writing main/gene_gallery.json", flush=True)
    obj = {"gene":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"gene"},"version":1,"protein":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"protein"}}
    with zobj.open(dest + "/main/gene_gallery.json", "w") as z:
        z.write(json.dumps(obj).encode("utf8"))

def write_dimred(source_hdf5, dest, zobj):
    print("Writing dimred")
    data = {}
    default_dimred = None
    for dimred in source_hdf5["obsm"]:
        try:
            x = source_hdf5["obsm"][dimred].shape
        except Exception as e:
            print("--->Ignoring %s" % dimred)
            continue
        print("--->Writing %s" % dimred)
        matrix = source_hdf5["obsm"][dimred][:][:]
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
        with zobj.open(dest + "/main/dimred/" + coords["id"], "w") as z:
            z.write(json.dumps(coords).encode("utf8"))
    meta = {
        "data":data,
        "version":1,
        "bbrowser_version":"2.7.38",
        "default":default_dimred,
        "description":"Created by converting scanpy to bbrowser format"
    }
    print("Writing main/dimred/meta", flush=True)
    with zobj.open(dest + "/main/dimred/meta", "w") as z:
        z.write(json.dumps(meta).encode("utf8"))


def write_runinfo(source_hdf5, dest, study_id, zobj):
    print("Writing run_info.json", flush=True)
    runinfo_history = generate_history_object()
    runinfo_history["hash_id"] = study_id
    date = time.time() * 1000
    run_info = {
        "species":"human",
        "hash_id":study_id,
        "version":16,
        "n_cell":len(source_hdf5["obs"]["index"]),
        "modified_date":date,
        "created_date":date,
        "addon":"SingleCell",
        "matrix_type":"single",
        "n_batch":1,
        "platform":"unknown",
        "omics":["RNA"],
        "title":["Created by bbrowser converter"],
        "history":[runinfo_history]
    }
    with zobj.open(dest + "/run_info.json", "w") as z:
        z.write(json.dumps(run_info).encode("utf8"))

def check_format(source):
    dataset_paths = ["layers/counts/indptr",
                    "layers/counts/indices",
                    "layers/counts/data",
                    "X/indptr",
                    "X/indices",
                    "X/data",
                    "obs/index",
                    "var/index",
                    "obs/__categories",
                    "obsm"]
    with h5py.File(source, "r") as f:
        for p in dataset_paths:
            try:
                x = f[p]
            except Exception as e:
                raise type(e)("Error when checking %s: %s" % (p, str(e)))

def format_data(source, output_name, overwrite=False):
    check_format(source)

    zobj = zipfile.ZipFile(output_name, "w")
    study_id = generate_uuid(remove_hyphen=False)
    dest = study_id
    with h5py.File(source, "r") as s:
        write_main_folder(s, dest, zobj)
        write_metadata(s, dest, zobj)
        write_dimred(s, dest, zobj)
        write_runinfo(s, dest, study_id, zobj)

    return output_name

