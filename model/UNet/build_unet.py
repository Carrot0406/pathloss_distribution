from unet_model import UNet
import torch
import torch.nn as nn


'''
创建UNet 输入输出channel为4 最终输出的结构shape为N 4 1920 1080 也改变成N 4 6400
'''
unet = UNet(n_channels=4, n_classes=4)
# print(unet)

# 引入Unet和自定义MLP 方便将点云与图片数据进行拼接 都处理成 N C 6400
class myUnet(nn.Module):
    def __init__(self):
        super(myUnet,self).__init__()

        self.unet = unet
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(4*41*25,4*1200)
    
    def forward(self,x):
        x = self.unet(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        x = x.view(x.size(0),4,-1)
        return x
    
if __name__ == '__main__':
    unet = myUnet().cuda(1)
    input = torch.randn(1,4,500,300)
    input = input.cuda(1)
    out = unet(input)
    print(out.shape)
    # 最终返回的特征是N 4 1200