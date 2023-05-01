# SAENAS-NE
This repo is the official implementation of "Surrogate-assisted evolutionary neural architecture search with network embedding"

## Overview
![overview](images/overview.png)
## Dataset
CIFAR-10

## Experiments

### Prepare the Nasbench space
Nasbench-101: https://github.com/google-research/nasbench (nasbench_full.tfrecord)

Nasbench-201: https://github.com/D-X-Y/NAS-Bench-201 (NAS-Bench-201-v1_0-e61699.pth )

Nasbench-301: https://github.com/automl/nasbench301 (v1.0)

Download the datasets for different Nasbench spaces and change the location of nas_bench_dir in nasbench.yaml


### Step 1. Prepare the architecture dataset on Nasbench space and train the graph2vec model
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