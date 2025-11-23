# DGCNN for Point Cloud Segmentation

This repository implements a complete **Dynamic Graph CNN (DGCNN)** architecture for point cloud segmentation, following the original paper:

> *Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", 2019*

DGCNN is a neural network designed to operate directly on unordered point clouds, without voxels or meshes. It dynamically builds graphs in feature space at each layer, capturing local geometric relationships in a flexible, permutation‑invariant way.

---

## 🚀 Key Idea of DGCNN

Instead of treating each point independently (like PointNet), DGCNN constructs a **graph structure** around each point during every layer of the network.

This graph is built using **k‑Nearest Neighbors (k‑NN)**, not in Euclidean space necessarily, but in **current feature space**. As features evolve, the neighborhood structure changes — hence the term *dynamic* graph.

For each point pair (i, j) in its neighborhood, the network computes an **edge feature**:

```
e_ij = concat( x_j - x_i , x_i )
```

This captures both:

* **local differences** between points (geometry)
* **absolute context** of each point

These edge features go through a series of **EdgeConv** layers to extract increasingly rich descriptors.

---

## 🧱 Architecture Overview

A typical DGCNN segmentation model follows this structure:

### 1. **Input**

* Shape: `(B, C, N)` where C=3 (xyz) or more for extra features

### 2. **Dynamic Graph + EdgeConv Layers**

We build 4 EdgeConv blocks:

1. **EdgeConv 1** → 64 channels
2. **EdgeConv 2** → 64 channels
3. **EdgeConv 3** → 128 channels
4. **EdgeConv 4** → 256 channels

Each block builds a k‑NN graph and applies:

```
Conv2d → BatchNorm → LeakyReLU → Max over neighbors
```

This yields multi‑scale per‑point features:

* `x1`: 64‑dim
* `x2`: 64‑dim
* `x3`: 128‑dim
* `x4`: 256‑dim

We concatenate them → **512‑dim local feature per point**.

---

## 🌐 Global Feature

On the concatenated 512‑dim per‑point features, we apply:

1. `Conv1d(512 → 1024)`
2. Global max pooling over all points

This results in a **1024‑dim global descriptor** of the whole object.
We then broadcast this descriptor back to each point.

Final point representation:

```
local_per_point (512) + global_context (1024) = 1536 channels per point
```

---

## 🎯 Segmentation Head

A small MLP applied point‑wise:

```
Conv1d(1536 → 512)
Conv1d(512 → 256)
Dropout
Conv1d(256 → num_classes)
```

Outputs **class logits per point**.

---

## 📌 Why DGCNN Works Well

* Captures **local geometric relationships**
* Learns features that are **invariant to permutation** of input points
* Recomputes neighborhoods dynamically → adapts to learned feature space
* Provides strong performance on segmentation + classification

DGCNN tends to outperform PointNet/PointNet++ in tasks with:

* Complex local geometry
* High curvature regions
* Fine‑grained part boundaries

---

## 🧪 Training & Loss

The model exposes a standard interface:

```
logits = model(points)
loss = cross_entropy(logits, labels)
```

This integrates directly with your existing training loop and dataset loaders.

---

## 📁 File Structure

```
segmentation_models/
│
├── dgcnn.py                # DGCNN implementation
├── pointnet2.py            # PointNet++ implementation
├── pointnet2_utils.py      # Utility functions (FPS, ball query, etc.)
└── ...
```

---

## 📖 References

* **DGCNN Paper**: [https://arxiv.org/abs/1801.07829](https://arxiv.org/abs/1801.07829)
* EdgeConv concept introduced in this work

---

If you'd like, I can also generate:

* Architecture diagrams
* Comparison table (PointNet vs PointNet++ vs DGCNN)
* Example usage code
* Visualization tools for graph edges / neighborhoods
