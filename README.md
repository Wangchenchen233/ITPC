## Spectral Clustering and Embedding with Inter-class Topology-preserving
This repository provides a python implementation of ITPC and its variants as described in the paper ["Spectral Clustering and Embedding with Inter-class Topology-preserving"](https://doi.org/10.1016/j.knosys.2023.111278).


## Datasets
All data sets can be obtained from the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets.php) or scikit-feature selection repository (https://jundongl.github.io/scikit-feature/datasets.html) or MLData repository (http://www.cad.zju.edu.cn/home/dengcai/Data/MLData.html).


## Environment Settings
- numpy
- matplotlib
- scikit-learn
- scipy
- pandas

## Running the code
You can run the following demo function in "./simulation/" directly
```sh
simi_four_classes.py
```
or run the following parameter search function for ITPC
```sh
para_search_ITPC.py
```
or run the following parameter search function for ITPPC or ITPPC_2
```sh
para_search_ITPPC.py
```

## Citation
```sh
@article{WANG2023ITPPC,
title = {Spectral clustering and embedding with inter-class topology-preserving},
author = {Chenchen Wang and Zhichen Gu and Jin-Mao Wei},
journal = {Knowledge-Based Systems},
pages = {111278},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.111278},
url = {https://www.sciencedirect.com/science/article/pii/S0950705123010262}
```
# Contact
If you have any questions, please feel free to contact me with wangc@mail.nankai.edu.cn
