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
python scripts/search_molecules_zinc.sh [gpu_id]
```

### 4. Train & Test

```
python scripts/train_molecules_zinc.sh [gpu_id] '[path_to_genotypes]/example.yaml'
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