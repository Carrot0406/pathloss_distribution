import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
from ..classifier import Classifier


class Fusionet(nn.Module):
    def __init__(self, img_encoder, cloud_encoder, opt):
        super(Fusionet, self).__init__()
        self.img_encoder = img_encoder
        self.cloud_encoder = cloud_encoder
        self._initialize_weights()
        self.relu = nn.LeakyReLU()
        self.method = opt.method
        self.classifier = Classifier.Classifier()
        # 通道数目 可以不用这一步
        # self.downsample = nn.Conv2d(35, 14, kernel_size=(1, 1))
        self.downsample = nn.Conv2d(4, 4, kernel_size=(1, 1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, img, cloud):
        # img = self.downsample(img)
        img_latent = self.img_encoder(img)
        cloud_latent = self.cloud_encoder(cloud)
        feature = []

        if self.method == 'concat':
            feature = torch.cat((img_latent, cloud_latent), dim=1)
            feature = feature.view(2, 7, 1200)
            out = self.classifier(feature)
        elif self.method == 'attention':
            feature = torch.cat((img_latent, cloud_latent), dim=1)
            feature = feature.view(2, 7, 1200)
            out = self.classifier(feature)
        return out
