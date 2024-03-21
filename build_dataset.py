import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

def default_loader_rgb(image_path):
    return Image.open(image_path).convert('RGB')


def default_loader_dep(image_path):
    return Image.open(image_path).convert('L')


def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def get_pl_data(file_path):
    """
    把路径损耗的值转换成tensor
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [float(line.strip()) for line in lines]
        tensor = torch.tensor(data)
    # shape 100
    return tensor


def get_cloud_numpy(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        num_points = len(lines) // 3
        point_cloud = np.empty((num_points, 3), dtype=float)
        for i in range(num_points):
            x = float(lines[i * 3].strip())
            y = float(lines[i * 3 + 1].strip())
            z = float(lines[i * 3 + 2].strip())
            point_cloud[i] = [x, y, z]
    transposed_array = np.transpose(point_cloud)
    # shape 3*2048
    return transposed_array

'''
只用图片数据构建数据集 验证PL预测程度
'''
class DatasetImg(Dataset):
    def __init__(self, opt, mode, transform=None):
        self.mode = mode
        path_to_rgb = readlines(opt.rgb_path)
        path_to_depth = readlines(opt.depth_path)
        path_to_pl = readlines(opt.pl_path)
        self.path_to_rgb = path_to_rgb
        self.path_to_depth = path_to_depth
        self.path_to_pl = path_to_pl
        self.transform = transform
        self.loader_rgb = default_loader_rgb
        self.loader_dep = default_loader_dep

    def __len__(self):
        return len(self.path_to_rgb)

    def __getitem__(self, index):
        path_rgb = self.path_to_rgb[index]
        path_dep = self.path_to_depth[index]
        rgb = self.loader_rgb(path_rgb)
        depth = self.loader_dep(path_dep)
        if self.transform is not None:
            rgb = self.transform(rgb)
            depth = self.transform(depth)
        pl = get_pl_data(self.path_to_pl[index])
        return rgb, depth, pl

'''
只用点云数据构建数据集 验证PL预测程度
'''
class DatasetPoint(Dataset):
    def __init__(self, opt, mode):
        # get labels
        self.mode = mode
        path_to_cloud = readlines(opt.lidar_path)
        path_to_pl = readlines(opt.pl_path)
        self.path_to_cloud = path_to_cloud
        self.path_to_pl = path_to_pl


    def __getitem__(self, index):
        path_cloud = self.path_to_cloud[index]
        cloud_numpy = get_cloud_numpy(path_cloud)
        to_tensor = transforms.ToTensor()
        cloud = to_tensor(cloud_numpy)
        pl = get_pl_data(self.path_to_pl[index])
        return cloud,pl

    def __len__(self):
        return len(self.path_to_cloud)

'''
图片和点云数据构建数据集 验证PL预测程度
'''
class DatasetImg_Point(Dataset):
    def __init__(self, opt, mode, transform):
        self.mode = mode
        self.transform = transform
        self.loader_rgb = default_loader_rgb
        self.loader_dep = default_loader_dep
        self.path_to_pl_gt = opt.pl_path
        self.depth_fpath = opt.depth_path
        self.rgb_fpath = opt.rgb_path
        self.lidar_fpath = opt.lidar_path

        self.depth_filenames = readlines(self.depth_fpath)
        self.rgb_filenames = readlines(self.rgb_fpath)
        self.lidar_filenames = readlines(self.lidar_fpath)
        self.pathloss_filenames = readlines(self.path_to_pl_gt)

    def __getitem__(self, index):
        rgb = self.transform(self.loader_rgb(self.rgb_filenames[index]))
        depth = self.transform(self.loader_dep(self.depth_filenames[index]))
        cloud_numpy = get_cloud_numpy(self.lidar_filenames[index])
        to_tensor = transforms.ToTensor()
        cloud = to_tensor(cloud_numpy)
        pathloss = get_pl_data(self.pathloss_filenames[index])
        return rgb, depth, cloud, pathloss

    def __len__(self):
        return len(self.depth_filenames)

def build_dataset(opt, mode, transform):
    if opt.modality == 'img':  # to pretrain visual channel
        dataset = DatasetImg(opt, mode, transform)
    elif opt.modality == 'point':  # to pretrain semantic channel
        dataset = DatasetPoint(opt, mode)
    elif opt.modality == 'img_point':  # to pretrain image channel
        dataset = DatasetImg_Point(opt, mode, transform)
    else:
        assert 1 < 0, 'Please fill the correct train stage!'
    return dataset

if __name__ == '__main__':
    '''
    测试pl
    '''
    # file_path = r'E:\python\dataset\pl_distrubution\Bus3\time1_pl.txt'
    # pl_tensor = get_pl_data(file_path)
    # print(pl_tensor.shape)

    '''
    测试lidar
    '''
    # file_path = r'E:\python\dataset\lidar\Car7\down_2048\time1000_pointcloud.txt'
    # point = get_cloud_numpy(file_path)
    # print(point.dtype)
    # print(point.shape)

    '''
    测试构建数据集
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pl_path',type=str,default='E:\python\pathloss_distribution\split\pl_distribution_val.txt')
    parser.add_argument('--rgb_path',type=str,default='E:\python\pathloss_distribution\split\RGB_val.txt')
    parser.add_argument('--depth_path',type=str,default='E:\python\pathloss_distribution\split\depth_val.txt')
    parser.add_argument('--lidar_path',type=str,default='E:\python\pathloss_distribution\split\lidar_val.txt')
    parser.add_argument('--modality', type=str, default='point', help='choose modality for experiment: img, point, img_point')

    opt = parser.parse_args()

    transform_img_train = transforms.Compose([
        transforms.Resize([1080, 1920]),
        transforms.ToTensor(), ])

    dataset = build_dataset(opt, mode='train', transform=transform_img_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, sampler=None)
    for batch_idx, (point, pl_gt) in enumerate(train_loader):
        print(point.shape)
        break

