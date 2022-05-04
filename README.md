
## 🔥News🔥
Hi, if you like this work, you may be interested in our new work, too. 

Welcome to our latest work [Automatic Relation-aware Graph Network Proliferation](https://github.com/phython96/ARGNP). 

**This work has been accepted by CVPR2022 and selected as an ORAL presentation.**

In the latest work, we have achieved state-of-the-art results on SBM_CLUSTER (77.35% OA), ZINC_100k (0.136 MAE), CIFAR10 (73.90% OA), TSP (0.855 F1-score) datasets and so on. 

### What's new? 

1. **We devise a novel dual relation-aware graph search space that comprises both node and relation learning operations.**
So, the ARGNP can leverage the edge attributes in some datasets, such as ZINC. 
It significantly improves the graph representative capability. 
Interestingly, we also observe the performance improvement even if there is no available edge attributes in some datasets. 

2. **We design a network proliferation search paradigm (NPSP) to progressively determine the GNN architectures by iteratively performing network division and differentiation.**
The network proliferation search paradigm decomposes the training of global supernet into sequential local supernets optimization, which alleviates the interference among child graph neural architectures. It reduces the spatial-time complexity from quadratic to linear and enables the search to thoroughly free from the cell-sharing trick. 

3. **Our framework is suitable for solving node-level, edge-level, and graph-level tasks. The codes are easy to use.**

---

# Rethinking Graph Neural Architecture Search from Message-passing

<a href="https://arxiv.org/abs/2103.14282"><img src = "https://img.shields.io/badge/arxiv-2103.14282-critical"></img></a> <a href="https://opensource.org/licenses/MIT"><img src = "https://img.shields.io/badge/License-MIT-yellow.svg"></img></a> 



## Getting Started

### 0. Prerequisites

+ Linux
+ NVIDIA GPU + CUDA CuDNN 

### 1. Setup Python Environment

```sh
# clone Github repo
conda install git
git clone https://github.com/phython96/GNAS-MP.git
cd GNAS-MP

# Install python environment
conda env create -f environment_gpu.yml
conda activate gnasmp
```

### 2. Download datasets

The datasets are provided by project [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns), you can click [here](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/02_download_datasets.md) to download all the required datasets. 

### 3. Search Architectures

```sh
sh scripts/search_molecules_zinc.sh [gpu_id]
```

### 4. Train & Test

```
sh scripts/train_molecules_zinc.sh [gpu_id] '[path_to_genotypes]/example.yaml'
```

## Reference
```latex
@inproceedings{cai2021rethinking,
  title={Rethinking Graph Neural Architecture Search from Message-passing},
  author={Cai, Shaofei and Li, Liang and Deng, Jincan and Zhang, Beichen and Zha, Zheng-Jun and Su, Li and Huang, Qingming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6657--6666},
  year={2021}
}
```
