"""
Author: Austin J. Garrett

PyTorch implementation of the PointCNN paper, as specified in:
  https://arxiv.org/pdf/1801.07791.pdf
Original paper by: Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen
"""

# External Modules
import torch
import torch.nn as nn
from torch import FloatTensor
import numpy as np
from typing import Tuple, Callable, Optional

# Internal Modules
from util_funcs import UFloatTensor, ULongTensor
from util_layers import Conv, SepConv, Dense, EndChannels

class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.
    定义了一个对单个点及其邻居进行卷积操作的模块
    """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int,
                 P : int, C_mid : int, depth_multiplier : int) -> None:
        """
        :param C_in: Input dimension of the points' features.  输入点云特征维度
        :param C_out: Output dimension of the representative point features. 代表性点的输出特征维度
        :param dims: Spatial dimensionality of points. 点的空间维度
        :param K: Number of neighbors to convolve over. 邻居个数
        :param P: Number of representative points. 选取的特征点的个数
        :param C_mid: Dimensionality of lifted point features. 提升点特征的维度
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution. 用于内部深度可分离卷积的深度乘法器
        """
        super(XConv, self).__init__()

        if __debug__:
            # Only needed for assertions.
            self.C_in = C_in
            self.C_mid = C_mid
            self.dims = dims
            self.K = K

        self.P = P

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        self.dense1 = Dense(dims, C_mid).cuda()
        self.dense2 = Dense(C_mid, C_mid).cuda()

        # Layers to generate X 为了生成X矩阵的层
        self.x_trans = nn.Sequential(
            EndChannels(Conv(
                in_channels = dims,
                out_channels = K*K,
                kernel_size = (1, K),
                with_bn = False
            )).cuda(),
            Dense(K*K, K*K, with_bn = False).cuda(),
            Dense(K*K, K*K, with_bn = False, activation = None).cuda()
        )

        # 将中心点的特征与邻居点的特征进行拼接
        self.end_conv = EndChannels(SepConv(
            in_channels = C_mid + C_in,
            out_channels = C_out,
            kernel_size = (1, K),
            depth_multiplier = depth_multiplier
        )).cuda()

    def forward(self, x : Tuple[UFloatTensor,
                                UFloatTensor,
                                Optional[UFloatTensor]]  # (N, P, dims)# (N, P, K, dims)# (N, P, K, C_in)
               ) -> UFloatTensor:                        # (N, K, C_out)
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x
        # 输入张量 (rep_pt, pts, fts) 包含了代表性点、区域点云和区域特征
        '''
        rep_pt 是代表性点，形状为 (N, P, dims)
        pts 是区域点云，形状为 (N, P, K, dims)
        fts 是区域特征，形状为 (N, P, K, C_in)
        '''

        if fts is not None:
            assert(rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # Check P is equal.
            assert(pts.size()[2] == fts.size()[2] == self.K)            # Check K is equal.
            assert(fts.size()[3] == self.C_in)                          # Check C_in is equal.
        else:
            assert(rep_pt.size()[0] == pts.size()[0])                   # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1])                   # Check P is equal.
            assert(pts.size()[2] == self.K)                             # Check K is equal.
        assert(rep_pt.size()[2] == pts.size()[3] == self.dims)          # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim = 2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        p_center = p_center.cuda()
        pts_local = pts - p_center  # (N, P, K, dims)
        pts_local = pts_local.cuda()
        # pts_local = self.pts_layernorm(pts - p_center)

        # Individually lift each point into C_mid space.
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted  = self.dense2(fts_lifted0)  # (N, P, K, C_mid)

        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1).cuda()  # (N, P, K, C_mid + C_in)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        # Weight and permute fts_cat with the learned X.
        fts_X = torch.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim = 2)
        return fts_p

class PointCNN(nn.Module):
    """ Pointwise convolutional model.
     逐点卷积模式
     """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,
                                            UFloatTensor,
                                            int, int],
                                           ULongTensor]    # (N, P, dims)# (N, x, dims)# (N, P, K)
                ) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points.
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.

        C_in：输入点特征的维度。它表示每个点的特征向量的长度或维度。
        C_out：代表性点特征的输出维度。这是在进行卷积操作后，每个代表性点的特征向量的长度或维度。
        dims：点的空间维度。通常，对于 3D 点云数据，这个值是 3，表示点的坐标有三个分量。
        K：要进行卷积操作的邻居点的数量。对于每个代表性点，将使用其周围的 K 个邻居点进行卷积操作。
        D：邻居点的 "spread" 或者说 "范围"。在这个上下文中，它可能表示每个代表性点周围的邻居点的范围或距离的限制。这个参数可能用于某些选择邻居点的算法中。
        P：代表性点的数量。在卷积操作中，将从输入的点云中选取 P 个代表性点进行操作。
        r_indices_func：选择器函数，用于选择邻居点的函数。这个函数接受代表性点和点云作为输入，然后返回每个代表性点的邻居点的索引。在卷积操作中，邻居点将与每个代表性点一起使用来进行卷积操作。

          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
          pts_idx：指向pts的索引数组，使得pts[pts_idx]是rep_pt周围“区域”中的一组点。
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4

        if C_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2).cuda() if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    def select_region(self, pts : UFloatTensor,  # (N, x, dims)
                      pts_idx : ULongTensor      # (N, P, K)
                     ) -> UFloatTensor:          # (P, K, dims)
        """
        Selects neighborhood points based on output of r_indices_func.
        :param pts: Point cloud to select regional points from.
        :param pts_idx: Indices of points in region to be selected.
        :return: Local neighborhoods around each representative point.
        """
        pts_idx = pts_idx.cuda()
        pts = pts.cuda()
        # print("pts_idx device:", pts_idx.device)

        regions = torch.stack([
            pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))
        ], dim = 0)
        return regions

    def forward(self, x : Tuple[FloatTensor,
                                FloatTensor,
                                FloatTensor]  # (N, P, dims)# (N, x, dims)# (N, x, C_in)
               ) -> FloatTensor:              # (N, P, C_out)
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :param x: (rep_pts, pts, fts) where
          - rep_pts: Representative points.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the
          feature associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated to rep_pts.
        """

        rep_pts, pts, fts = x
        pts = pts.cuda()
        fts = self.dense(fts) if fts is not None else fts

        # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu()).cuda()
        # -------------------------------------------------------------------------- #

        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))

        return fts_p

class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points.
     具有随机子采样代表点的PointCNN
     """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                                            UFloatTensor,  # (N, x, dims)
                                            int, int],
                                           ULongTensor]    # (N, P, K)
                ) -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P

    def forward(self, x : Tuple[UFloatTensor,  # (N, x, dims)
                                UFloatTensor]  # (N, x, dims)
               ) -> Tuple[UFloatTensor,        # (N, P, dims)
                          UFloatTensor]:       # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
            rep_pts = pts[:,idx,:]
        else:
            # All input points are representative points.
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        return rep_pts, rep_pts_fts





if __name__ == '__main__':
    from util_funcs import knn_indices_func_gpu
    # 创建一个 PointCNN 实例
    point_cnn = RandPointCNN(0,64,3,16,16,200,knn_indices_func_gpu)
    # 假设 pts 和 fts 是点云数据
    # pts 是点的坐标信息，fts 是点的特征信息

    # 生成假设的点云数据
    pts = torch.randn(3, 2048, 3)  # 第一维是batchsize 假设有 2048 个点，每个点有 3 维坐标
    # fts = torch.randn(1, 2048, 3)  # 假设每个点有 3 维特征

    pts = pts.cuda()
    # fts = fts.cuda()


    # 将点云数据传递给 PointCNN 的前向方法
    rep_pts,rep_pts_fts =point_cnn((pts,None))

    # rep_pts 是新的代表性点集
    # rep_pts_fts 是从原始点云中投影而来的特征
    print(rep_pts_fts.shape)
