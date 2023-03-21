# PrivGraph
Implementation of PrivGraph
## Requirements


```
numpy >= 1.20.1
pandas >= 1.2.4
networkx >= 2.5
scikit-learn >= 0.24.1
python-louvain >= 0.15
python >= 3.8
```

## Contents

The project contains 3 folders and 6 files.

1. data (folder): All datasets are in this folder.
2. comm (folder): This folder is used for community discovery.
3. result (folder): This folder is used to store the results and contains four examples of synthetic graphs.
4. main.py (file): The file is used to obtain the results of PrivGraph for End-to-End experiments.
5. main_vary_N.py (file): The file is used to obtain the results for different number of nodes.
6. main_vary_eps.py (file): The file is used to obtain the results for different privacy budget allocations.
7. main_vary_t.py (file): The file is used to obtain the results for different resolution parameters.
8. IM_spread.py (file): The file is used to obtain the results of influence maximization.
9. utils.py (file): The file includes some functions that are needed for other files.

## Run


```
###### Example 1: End to End ######
python main.py

###### Example 2: Impact of the number of nodes ######
python main_vary_N.py

###### Example 3: Impact of the privacy budget allocation ######
python main_vary_eps.py

###### Example 4: Impact of the resolution parameter ######
python main_vary_t.py

###### Example 5: Influence Maximization ######
python IM_spread.py
```

## Citation

```
 @inproceedings{YZDCCS23,
    author = {Quan Yuan and Zhikun Zhang and Linkang Du and Min Chen and Peng Cheng and Mingyang Sun},
    title = {{PrivGraph: Differentially Private Graph Data Publication by Exploiting Community Information}},
    booktitle = {{USENIX Security}},
    publisher = {},
    year = {2023},
}
```