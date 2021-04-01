# pyBCS

This is a python library to create a BioTuring Compressed Study (`bcs`) file from an AnnData (scanpy) object.

`bcs` files can be imported directly into [BBrowser](https://bioturing.com/bbrowser), a software for single-cell data.

Visit our [github](https://github.com/bioturing/pyBCS) for more detail.

## Installation

pip install pyBCS-bioturing

## Example

### Scanpy

```python
from pyBCS import scanpy2bcs
scanpy2bcs.format_data("/mnt/example/data.h5ad", "/mnt/example/data.bcs",
                        input_format="h5ad", graph_based="louvain")
```


### SPRING

```python
from pyBCS import scanpy2bcs
scanpy2bcs.format_data("/mnt/example/spring_study", "/mnt/example/data.bcs",
                        input_format="spring",
                        graph_based="louvain")
```

### Loom

```python
from pyBCS import scanpy2bcs
scanpy2bcs.format_data("/mnt/example/data.loom", "/mnt/example/data.bcs",
                        input_format="loom",
                        barcode_name="CellID",
                        feature_name="Gene",
                        dimred_keys={"tsne":["tsne1", "tsne2"]})
```

### Abloom

```python
from pyBCS import scanpy2bcs
scanpy2bcs.format_data("/mnt/example/data.loom", "/mnt/example/data.bcs",
                        input_format="abloom",
                        barcode_name="observation_id",
                        feature_name="accession_id",
                        graph_based="cluster")
```
