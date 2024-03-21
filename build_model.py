import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
from model.UNet import unet_model
from model.fusion_model import FusionModel
from model.pointCNN import util_funcs
from model.pointCNN import build_point_cnn
from model.UNet import build_unet

UFloatTensor = util_funcs.UFloatTensor
ULongTensor = util_funcs.ULongTensor

'''
创建UNet 输入输出channel为4 最终输出的结构shape为N 4 1920 1080 也改变成N 4 1200
'''
unet = build_unet.myUnet()

'''
创建PointCNN
'''
# 最终返回的特征是N 3 1200
point_cnn = build_point_cnn.PointCNN()


def build(CUDA, opt):
    global model
    if opt.modality == 'img':
        model = unet
    elif opt.modality == 'point':
        model = point_cnn
    elif opt.modality == 'img_point':
        encoder_v = unet
        encoder_c = point_cnn
        model = FusionModel.Fusionet(encoder_v, encoder_c, opt)
    if CUDA:
        model.cuda()
    return model




def set_optimizer(model, opt):
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer