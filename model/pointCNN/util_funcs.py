# External Modules
import torch
from torch import cuda, FloatTensor, LongTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from typing import Union

# Types to allow for both CPU and GPU models.
UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]

'''
rep_pts：代表性点（Representative points）。这些点通常是通过某种方式从原始点云中提取出来的一小部分点。它们可能是原始点云中的一部分子集，也可能是通过某种特定算法（比如聚类算法）得到的代表性点。

pts：点云（Point cloud）。这是原始的点集合，可能是从传感器（如激光雷达或摄像头）采集到的原始数据，或者是从其他来源获取的点云数据。它通常包含大量的点，可能包含噪音和冗余信息。
'''


def knn_indices_func_cpu(rep_pts: FloatTensor,  # (N, pts, dim)
                         pts: FloatTensor,  # (N, x, dim)
                         K: int, D: int
                         ) -> LongTensor:  # (N, pts, K)
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()
    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D * K + 1, algorithm="ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:, 1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis=0))
    return region_idx


def knn_indices_func_gpu(rep_pts: cuda.FloatTensor,  # (N, pts, dim)
                         pts: cuda.FloatTensor,  # (N, x, dim)
                         k: int, d: int
                         ) -> cuda.LongTensor:  # (N, pts, K)
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []

    for n, qry in enumerate(rep_pts):
        ref = pts[n]
        n, d = ref.size()
        m, d = qry.size()
        mref = ref.expand(m, n, d)
        mqry = qry.expand(n, m, d).transpose(0, 1)
        dist2 = torch.sum((mqry - mref) ** 2, 2).squeeze()
        _, inds = torch.topk(dist2, k * d + 1, dim=1, largest=False)
        region_idx.append(inds[:, 1::d])

    region_idx = torch.stack(region_idx, dim=0)
    return region_idx
