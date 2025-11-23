import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Compute k-NN for each point in the batch.

    Args:
        x: (B, C, N) features (often C=3 for xyz or higher for later layers)
    Returns:
        idx: (B, N, k) indices of k nearest neighbors in feature space
    """
    B, C, N = x.shape
    # Compute pairwise distance on features (C-dim)
    # dist[b, i, j] = ||x[b,:,i] - x[b,:,j]||^2
    xx = torch.sum(x ** 2, dim=1, keepdim=True)          # (B,1,N)
    yy = xx.transpose(2, 1)                              # (B,N,1)
    # dist = ||x_i||^2 - 2 x_i x_j + ||x_j||^2
    dist = xx - 2 * torch.matmul(x.transpose(2, 1), x) + yy  # (B,N,N)

    # Take k smallest distances (excluding self if you want; this version keeps it simple)
    _, idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)  # (B,N,k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Build "edge features" for DGCNN.

    Given per-point features x, build for each point i and each neighbor j:
        e_ij = concat( x_j - x_i, x_i )

    Args:
        x: (B, C, N) input features
        k: number of neighbors
        idx: optional precomputed knn idx (B, N, k)

    Returns:
        edge_features: (B, 2*C, N, k)
    """
    B, C, N = x.shape

    if idx is None:
        idx = knn(x, k=k)  # (B,N,k)

    # Create batch indices for advanced indexing
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N  # (B,1,1)

    # Flatten idx to a single dimension indexing into (B*N, N)
    idx = idx + idx_base  # (B,N,k)
    idx = idx.view(-1)    # (B*N*k,)

    # Flatten x to (B*N, C)
    x = x.transpose(2, 1).contiguous()     # (B,N,C)
    feature = x.view(B * N, C)             # (B*N,C)

    # Gather neighbor features
    neighbor = feature[idx, :]             # (B*N*k, C)
    neighbor = neighbor.view(B, N, k, C)   # (B,N,k,C)

    # Central point features, expanded along neighbor dim
    x_central = x.view(B, N, 1, C).repeat(1, 1, k, 1)  # (B,N,k,C)

    # Edge feature: [x_j - x_i, x_i]
    edge_feature = torch.cat((neighbor - x_central, x_central), dim=-1)  # (B,N,k,2C)

    # Return as (B, 2C, N, k) for Conv2d
    edge_feature = edge_feature.permute(0, 3, 1, 2).contiguous()
    return edge_feature
