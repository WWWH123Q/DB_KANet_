# DB-KANet: A Lightweight Network for Real-Time Precipitation Nowcasting

<p align="center">
  <img src="assets/architecture.png" width="850">
</p>

<p align="center">
  <b>A lightweight dual-branch Kolmogorov–Arnold network for efficient precipitation nowcasting</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-red">
  <img src="https://img.shields.io/badge/Task-Precipitation%20Nowcasting-green">
  <img src="https://img.shields.io/badge/Model-DB--KANet-orange">
</p>

---

## Overview

This repository provides the official PyTorch implementation of:

**DB-KANet: A Lightweight Network for Real-Time Precipitation Nowcasting**

DB-KANet is designed for real-time precipitation nowcasting under resource-constrained environments. It adopts a lightweight U-shaped encoder–decoder architecture and integrates Kolmogorov–Arnold Network principles into spatiotemporal precipitation prediction.

The proposed model introduces two key modules:

- **Dual-Branch Attention Module (DBAM)**: captures both global spatiotemporal evolution and fine-grained local rainfall structures.
- **Convolutional Kolmogorov–Arnold Network (CKAN)**: enhances nonlinear representation capacity for abrupt and complex precipitation dynamics.

With only **1.32M parameters** and **2.58 GFLOPs**, DB-KANet achieves a strong balance between forecasting accuracy and computational efficiency, making it suitable for operational nowcasting and edge-deployment scenarios.

---

## Highlights

- Lightweight U-shaped encoder–decoder architecture.
- Global–local feature modeling through DBAM.
- Nonlinear precipitation dynamics modeling through CKAN.
- Efficient inference with only **1.32M parameters** and **2.58 GFLOPs**.
- Evaluated on both regional-scale and urban-scale precipitation datasets.
- Designed for real-time precipitation monitoring and disaster early-warning applications.

---

## Network Architecture

<p align="center">
  <img src="assets/architecture.png" width="850">
</p>

The overall framework of DB-KANet follows an encoder–decoder design with skip connections. DBAM and CKAN blocks are embedded into the network to improve feature representation while maintaining low computational cost.

### Dual-Branch Attention Module

DBAM divides feature channels into two specialized branches:

- A **global branch** for long-range dependency modeling.
- A **local branch** for fine-grained spatial structure extraction.

The two branches are fused through lightweight convolution and residual connection, enabling efficient global–local spatiotemporal feature aggregation.

### Convolutional KAN Block

CKAN introduces Kolmogorov–Arnold Network principles into convolutional feature learning. It combines:

- A nonlinear basis-function path for adaptive nonlinear modeling.
- A residual convolutional path for preserving local inductive bias.
- Lightweight fusion for efficient feature transformation.

This design improves the model's ability to represent nonlinear and rapidly evolving precipitation patterns.

---

## Repository Structure

The recommended repository structure is:

```text
DB-KANet/
├── assets/
│   ├── architecture.png
│   ├── laps_visualization.png
│   ├── shanghai_visualization.png
│   └── results_curve.png
├── configs/
│   └── config_setting.py
├── dataset/
│   ├── Shanghai.py
│   └── metrics.py
├── models/
│   └── DB_KANet.py
├── dataprepare/
├── engine.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
└── LICENSE
