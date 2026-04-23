[English](./README.md) | [简体中文](./README.zh-CN.md)

# GRAU-Approx-Toolkit

**GRAU-Approx-Toolkit** is a Python toolkit for building and evaluating **hardware-friendly approximations** of quantized activation pipelines based on:

- **Piecewise Linear Fitting (PWLF)**
- **Power-of-Two (PoT)** slope quantization
- **Additive Power-of-Two (APoT)** slope quantization

The toolkit is developed for the GRAU project (**Generic Reconfigurable Activation Unit**), which targets a reusable activation design for quantized neural network hardware accelerators.

---

## Abstract

With the continuous growth of neural network scales, low-precision quantization is widely used in edge accelerators. Classic multi-threshold activation hardware requires \(2^n\) thresholds for \(n\)-bit outputs, causing a rapid increase in hardware cost as precision increases. We propose a reconfigurable activation hardware, **GRAU**, based on piecewise linear fitting, where the segment slopes are approximated by powers of two. Our design requires only basic comparators and 1-bit right shifters, supporting mixed-precision quantization and nonlinear functions such as **SiLU**. Compared with multi-threshold activators, GRAU reduces LUT consumption by over 90%, achieving higher hardware efficiency, flexibility, and scalability. The best trade-off is usually achieved with 6-8 segments, while complex nonlinearities under aggressive low-cost settings may suffer larger accuracy degradation.

---

## Overview

This repository contains three parts:

- **`src/`**: source code of the GRAU approximation toolkit
- **`demo/`**: small and easy-to-follow examples for toolkit usage and toy networks
- **`experiment/`**: full experiment code used in the paper
  - **`experiment/report/`** stores training and evaluation results

The goal of this repository is twofold:

1. provide a reusable **Python toolkit** for PWLF / PoT / APoT approximation;
2. provide the **research code** used to reproduce the experiments in the GRAU paper.

---

## Features

Currently, the toolkit includes:

### `GRAU_PWLF`
A standalone greedy integer-aware PWLF engine.

It supports:
- greedy breakpoint selection
- integer-aware breakpoints
- minimum-gap constraints
- PoT slope quantization
- APoT slope quantization
- plotting and comparison utilities

### `GRAU_Linear`
A wrapper for approximating the folded **BN + QuantAct** stage after a quantized linear layer.

It supports:
- `"PWLF"`
- `"PoT-PWLF"`
- `"APoT-PWLF"`

### `GRAU_Conv2D`
A convolutional version of the GRAU approximation wrapper for quantized convolution layers.

### `GRAU_ResConv2D`
A residual convolution wrapper designed for ResNet-style structures.

### `extend_min_max_ranges`
A utility function for extending recorded layer-wise output ranges before fitting.

---

## Repository Structure

```text
GRAU-Approx-Toolkit/
├── src/                    # source code of the toolkit
├── demo/                   # small demos and toy examples
│   ├── toolkit-demo/       # basic examples for GRAU_PWLF and GRAU_Linear
│   └── network-demo/       # tiny MLP/CNN examples on MNIST
├── experiment/             # full experiment code used in the paper
│   ├── report/             # training logs / result reports
│   ├── ResNet18_*.ipynb
│   ├── SFC_*.ipynb
│   └── VGG16_*.ipynb
├── pyproject.toml
└── README.md
```

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <your-repo-url>
cd GRAU-Approx-Toolkit
pip install -e .
```

You may also need the common dependencies used in the notebooks, such as:

```bash
pip install torch torchvision numpy matplotlib jupyter brevitas
```

---

## Getting Started

To keep this README short and focused, the usage examples are provided as notebooks in the `demo/` folder.

### Toolkit demos
See:

- `demo/toolkit-demo/grau_pwlf.ipynb`
- `demo/toolkit-demo/grau_linear.ipynb`

These notebooks show how to:
- build PWLF approximations,
- inspect breakpoints and slopes,
- compare original / PWLF / PoT-PWLF / APoT-PWLF results,
- apply GRAU to quantized linear layers.

### Small network demos
See:

- `demo/network-demo/TinyMLP_*.ipynb`
- `demo/network-demo/TinyCNN_*.ipynb`

These notebooks provide lightweight examples on **MNIST** for quickly understanding the workflow before moving to larger experiments.

---

## Paper Experiments

The `experiment/` folder contains the notebooks used for the paper experiments.

It includes experiments on multiple model / dataset / activation / precision settings, including combinations such as:

- **SFC**
- **VGG16**
- **ResNet18**

with different activations and quantization settings, such as:

- ReLU
- Sigmoid
- SiLU
- 4-bit
- 8-bit
- mixed precision

This folder is intended for:
- reproducing the paper results,
- checking detailed training settings,
- and extending the experiments to new configurations.

---

## Reports and Results

The training outputs and result summaries are stored under:

```text
experiment/report/
```

This directory contains the recorded results from the experiment notebooks.

---

## What This Repository Is For

This repository is useful if you want to:

- study hardware-friendly activation approximation for QNNs,
- experiment with PWLF / PoT / APoT representations,
- replace folded activation stages in quantized models,
- reproduce the experiments from the GRAU paper,
- or prepare approximation parameters for future FPGA / ASIC implementations.

---

## Relation to the GRAU Paper

This toolkit accompanies the GRAU project:

**GRAU: Generic Reconfigurable Activation Unit Design for Neural Network Hardware Accelerators**

The paper studies how folded activation pipelines can be approximated in a hardware-friendly way while supporting:
- runtime reconfiguration,
- mixed-precision quantization,
- and non-monotonic activations.

---

## Notes

- The `demo/` folder is the best starting point for new users.
- The `experiment/` folder is more research-oriented and contains the full paper experiment code.
- Some parts of the broader project are still under active development, so this repository currently focuses on the completed and usable toolkit components plus reproducible experiments.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{liu2026graugenericreconfigurableactivation,
      title={GRAU: Generic Reconfigurable Activation Unit Design for Neural Network Hardware Accelerators}, 
      author={Yuhao Liu and Salim Ullah and Akash Kumar},
      year={2026},
      eprint={2602.22352},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2602.22352}, 
}
```
