[English](./README.md) | [简体中文](./README.zh-CN.md)

# GRAU-Approx-Toolkit

**GRAU-Approx-Toolkit** 是一个 Python 工具包，用于构建和评估**量化激活流水线的硬件友好近似**，其核心方法包括：

- **分段线性拟合（Piecewise Linear Fitting, PWLF）**
- **2 的幂（Power-of-Two, PoT）**斜率量化
- **加性 2 的幂（Additive Power-of-Two, APoT）**斜率量化

该工具包围绕 GRAU 项目（**Generic Reconfigurable Activation Unit**）开发，目标是为量化神经网络硬件加速器提供一种可复用的激活设计方案。

---

## 摘要

随着神经网络规模的持续增长，低精度量化被广泛应用于边缘加速器中。经典的多阈值激活硬件在输出为 \(n\) 比特时需要 \(2^n\) 个阈值，因而随着精度提升，硬件代价会快速增加。我们提出了一种可重构激活硬件 **GRAU**，其基于分段线性拟合，并将各段斜率近似为 2 的幂。该设计仅需要基础比较器和 1-bit 右移器，就可以支持混合精度量化以及 **SiLU** 等非线性函数。与多阈值激活器相比，GRAU 可将 LUT 消耗降低 90% 以上，同时获得更高的硬件效率、灵活性和可扩展性。通常情况下，6–8 段可以取得较好的折中；而在更激进的低成本设置下，复杂非线性函数可能会出现更明显的精度下降。

---

## 仓库概览

该仓库主要包含三个部分：

- **`src/`**：GRAU 近似工具包的源代码
- **`demo/`**：工具包使用示例以及小型网络示例
- **`experiment/`**：论文中使用的完整实验代码
  - **`experiment/report/`** 用于存放训练和评估结果

该仓库主要有两个目标：

1. 提供一个可复用的 **PWLF / PoT / APoT** 近似 Python 工具包；
2. 提供用于复现 GRAU 论文实验的**研究代码**。

---

## 功能

目前工具包包含以下模块：

### `GRAU_PWLF`
一个独立的、整数感知的贪心 PWLF 引擎。

它支持：
- 贪心式断点选择
- 整数感知断点
- 最小间隔约束
- PoT 斜率量化
- APoT 斜率量化
- 绘图与对比分析工具

### `GRAU_Linear`
用于近似量化线性层后 **BN + QuantAct** 折叠阶段的封装模块。

支持的推理模式：
- `"PWLF"`
- `"PoT-PWLF"`
- `"APoT-PWLF"`

### `GRAU_Conv2D`
面向量化卷积层的 GRAU 卷积版近似封装。

### `GRAU_ResConv2D`
面向 ResNet 风格结构的残差卷积封装。

### `extend_min_max_ranges`
用于在拟合前扩展各层输出范围记录的辅助函数。

---

## 仓库结构

```text
GRAU-Approx-Toolkit/
├── src/                    # 工具包源代码
├── demo/                   # 小型 demo 与示例网络
│   ├── toolkit-demo/       # GRAU_PWLF 与 GRAU_Linear 的基础示例
│   └── network-demo/       # 基于 MNIST 的 Tiny MLP/CNN 示例
├── experiment/             # 论文中的完整实验代码
│   ├── report/             # 训练日志 / 结果报告
│   ├── ResNet18_*.ipynb
│   ├── SFC_*.ipynb
│   └── VGG16_*.ipynb
├── pyproject.toml
└── README.md
```

---

## 安装

克隆仓库并以 editable 模式安装：

```bash
git clone <your-repo-url>
cd GRAU-Approx-Toolkit
pip install -e .
```

你可能还需要安装 notebook 中常用的一些依赖，例如：

```bash
pip install torch torchvision numpy matplotlib jupyter brevitas
```

---

## 快速开始

为了让 README 保持简洁，具体使用示例放在 `demo/` 目录中的 notebook 中。

### 工具包示例
请查看：

- `demo/toolkit-demo/grau_pwlf.ipynb`
- `demo/toolkit-demo/grau_linear.ipynb`

这些 notebook 展示了如何：
- 构建 PWLF 近似；
- 查看断点和斜率；
- 对比 original / PWLF / PoT-PWLF / APoT-PWLF 的结果；
- 将 GRAU 应用于量化线性层。

### 小型网络示例
请查看：

- `demo/network-demo/TinyMLP_*.ipynb`
- `demo/network-demo/TinyCNN_*.ipynb`

这些 notebook 提供了基于 **MNIST** 的轻量级示例，方便你在进入更大规模实验前先快速理解整体流程。

---

## 论文实验

`experiment/` 目录包含论文中使用的实验 notebook。

其中涵盖了多种 **模型 / 数据集 / 激活函数 / 量化精度** 组合，例如：

- **SFC**
- **VGG16**
- **ResNet18**

以及不同的激活函数和量化设置，例如：

- ReLU
- Sigmoid
- SiLU
- 4-bit
- 8-bit
- mixed precision

该目录主要用于：
- 复现论文结果；
- 查看详细训练设置；
- 在此基础上扩展到新的实验配置。

---

## 报告与结果

训练输出和结果汇总保存在：

```text
experiment/report/
```

该目录保存了实验 notebook 产生的记录结果。

---

## 这个仓库适合做什么

如果你希望：

- 研究 QNN 中硬件友好的激活近似方法，
- 实验 PWLF / PoT / APoT 表达形式，
- 替换量化模型中折叠后的激活阶段，
- 复现 GRAU 论文实验，
- 或者为后续 FPGA / ASIC 实现准备近似参数，

那么这个仓库会比较适合你。

---

## 与 GRAU 论文的关系

该工具包配套于 GRAU 项目：

**GRAU: Generic Reconfigurable Activation Unit Design for Neural Network Hardware Accelerators**

论文研究了如何以硬件友好的方式近似折叠后的激活流水线，同时支持：
- 运行时可重构，
- 混合精度量化，
- 非单调激活函数。

---

## 说明

- 对新用户来说，最推荐从 `demo/` 目录开始。
- `experiment/` 目录更偏向研究复现，包含完整的论文实验代码。
- 整个项目仍有部分内容处于持续开发中，因此当前仓库主要聚焦于已经完成、可直接使用的工具模块以及可复现的实验部分。

---

## 引用

如果你在研究中使用了这个仓库，请引用：

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
