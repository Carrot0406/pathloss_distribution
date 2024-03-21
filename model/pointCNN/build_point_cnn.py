from pointCNN import RandPointCNN
from  util_funcs import knn_indices_func_cpu,knn_indices_func_gpu
import torch.nn as nn
from util_layers import Dense
import torch

# 先定义一个点坐标维度固定的随机采样点pointCNN
AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)

NUM_CLASS = 10

# 多层使用pointCNN
class PointCNN(nn.Module):

    def __init__(self):
        super(PointCNN, self).__init__()

        # 第一层不随机采样，计算全部点的特征 开始的点没有特征，从邻居点获得特征
        self.pcnn1 = AbbPointCNN(0, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            # 第二三层也是计算全部点的特征
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            # 第四五层开始更改输出维度
            # 最终输出的维度是N P C_out  N  e  b
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 30, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, NUM_CLASS, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        # print(x[0].shape) # 0是点
        # print(x[1].shape) # 1是特征
        x = self.pcnn2(x)[1]  # grab features 是 N 120 100
        x = x.view(x.size(0),3,1200)

        # logits = self.fcn(x)
        # logits_mean = torch.mean(logits, dim=1)
        return x

if __name__ == '__main__':
    pointcnn = PointCNN()
    input = torch.randn(1,2048,3)
    input = input.cuda()
    out = pointcnn((input,None))
    print(out.shape)
    # 最终返回的特征是N 3 1200