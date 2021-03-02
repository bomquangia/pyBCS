# pyBCS

This is a python library to create a BioTuring Compressed Study (`bcs`) file from an AnnData (scanpy) object.

`bcs` files can be imported directly into [BBrowser](https://bioturing.com/bbrowser), a software for single-cell data.

Visit our [github](https://github.com/bioturing/pyBCS) for more detail.

## Example

### Scanpy

```python
from pyBCS import scanpy2bcs
scanpy2bcs.format_data('/mnt/example/data.h5ad', '/mnt/example/data.bcs')
```


### SPRING

```python
from pyBCS import scanpy2bcs
scanpy2bcs.format_data('/mnt/example/spring_study', '/mnt/example/data.bcs', input_format='spring')
```
