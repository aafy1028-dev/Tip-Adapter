# Tip-Adapter-Dual
Dual-Encoder Extension of Tip-Adapter for Few-shot Image Classification

This project extends the original **Tip-Adapter** framework by introducing a **dual-encoder architecture** that combines:

- SigLIP2 Vision Encoder
- DINOv3 Vision Encoder

The extracted features from both encoders are concatenated to improve representation ability for **few-shot image classification** and **open-set recognition**.

The implementation is based on the official Tip-Adapter repository.

---

# Paper

Original Paper:

Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification  
ECCV 2022

Paper link:

https://arxiv.org/pdf/2207.09519.pdf

---

# Method Overview

Tip-Adapter is a **training-free adaptation method** for CLIP to perform few-shot classification using a **key-value cache model**.

Compared with traditional fine-tuning approaches, Tip-Adapter:

- avoids full model training
- uses cached features for classification
- achieves competitive performance with significantly lower training cost

This project extends the original framework with **dual encoder feature extraction**.

### Dual Encoder Pipeline

```

Image
│
├── SigLIP2 Encoder
│
├── DINOv3 Encoder
│
Feature Concatenation
│
Tip-Adapter Cache Model
│
Few-shot Classification

```

The final representation is:

```

Feature = Concat( SigLIP2(image), DINOv3(image) )

```

---

# Project Structure

```

Tip-Adapter
│
├ configs
│   ├ food101_dual.yaml
│   ├ food101_siglip.yaml
│
├ models
│   ├ **init**.py
│   └ dual_encoder.py
│
├ main.py
├ utils.py
├ requirements.txt
└ README.md

````

Main components:

| File | Description |
|-----|-------------|
| main.py | Main training / evaluation script |
| dual_encoder.py | Dual encoder implementation |
| utils.py | Utility functions |
| configs/ | Experiment configuration files |

---

# Installation

Clone repository

```bash
git clone https://github.com/aafy1028-dev/Tip-Adapter.git
cd Tip-Adapter
````

Create conda environment

```bash
conda create -n tip_adapter python=3.10
conda activate tip_adapter
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch

```bash
conda install pytorch torchvision cudatoolkit
```

---

# Dataset

This project uses the **Food-101 dataset**.

Download dataset:

[https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

After downloading, place the dataset under:

```
Tip-Adapter/food-101/
```

Dataset structure:

```
food-101
│
├ images
├ meta
├ train.txt
└ test.txt
```

---

# Running Experiments

Run the dual encoder experiment:

```bash
python main.py --config configs/food101_dual.yaml
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/food101_dual.yaml
```

---

# Key Modifications

Compared with the original Tip-Adapter repository:

• Added **DualEncoder module**

• Added **SigLIP2 feature extraction**

• Added **DINOv3 feature extraction**

• Added new configuration files

• Improved feature representation using encoder fusion

---

# Results

Experimental results depend on:

* encoder architecture
* dataset shots
* hyperparameters

Users can modify configuration files in:

```
configs/
```

to perform different experiments.

---

# Acknowledgement

This project is built upon the original Tip-Adapter repository:

[https://github.com/gaopengcuhk/Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter)

We thank the authors for their excellent work.

This project also benefits from the following works:

* CLIP
* DINOv3
* SigLIP
* CLIP-Adapter

---

# Citation

If you use this project, please cite the original Tip-Adapter paper.

```
@article{zhang2021tip,
title={Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling},
author={Zhang, Renrui and Fang, Rongyao and Gao, Peng and Zhang, Wei and Li, Kunchang and Dai, Jifeng and Qiao, Yu and Li, Hongsheng},
journal={arXiv preprint arXiv:2111.03930},
year={2021}
}
```

---

# Author

GitHub: [https://github.com/aafy1028-dev](https://github.com/aafy1028-dev)

---

# License

This project follows the license of the original Tip-Adapter repository.

````

---

