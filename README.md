# SAENAS-NE
This repo is the official implementation of "Surrogate-assisted evolutionary neural architecture search with network embedding"

## Overview
![overview](images/overview.png)
## Dataset
CIFAR-10

## Experiments


### Step 1. Prepare the architecture dataset on DARTS search space and train the graph2vec model
```shell
bash scripts/data_json.sh
```

### Step 2. Search the best architecture on NASBench search space
```shell
bash scripts/search.sh
```

## Citation
If you find this project useful in your research, please consider cite:

```
@article{fan2022surrogate,
  title={Surrogate-assisted evolutionary neural architecture search with network embedding},
  author={Fan, Liang and Wang, Handing},
  journal={Complex \& Intelligent Systems},
  pages={1--19},
  year={2022},
  publisher={Springer}
}