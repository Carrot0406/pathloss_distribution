import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # 导入ThreadPoolExecutor类

def read_point_cloud(file_path):
    """读取点云数据"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = []
    for i in range(0, len(lines), 3):
        x = float(lines[i].strip())
        y = float(lines[i+1].strip())
        z = float(lines[i+2].strip())
        points.append([x, y, z])
    return np.array(points)

def voxel_grid_downsample(point_cloud, num_points):
    """体素网格下采样"""
    min_values = np.min(point_cloud, axis=0)
    max_values = np.max(point_cloud, axis=0)
    grid_size = np.ceil((max_values - min_values) * 10).astype(int)  # 根据点云范围设置网格大小
    voxel_grid = np.zeros(grid_size)

    # 将点云映射到体素网格中
    indices = np.floor((point_cloud - min_values) * 10).astype(int)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    # 随机选择体素中的点
    non_zero_indices = np.nonzero(voxel_grid)
    selected_indices = np.random.choice(len(non_zero_indices[0]), num_points, replace=False)
    selected_points = np.array([non_zero_indices[0][selected_indices],
                                non_zero_indices[1][selected_indices],
                                non_zero_indices[2][selected_indices]]).T

    # 将选定的点转换回原始坐标空间
    selected_points = selected_points / 10 + min_values

    return selected_points

def save_point_cloud(point_cloud, file_path):
    """保存点云数据到txt文件"""
    with open(file_path, 'w') as file:
        for point in point_cloud:
            file.write(f"{point[0]}\n")
            file.write(f"{point[1]}\n")
            file.write(f"{point[2]}\n")

def process_file(file):
    """处理单个文件"""
    # 读取点云数据
    points = read_point_cloud(file)

    # 体素网格下采样
    downsampled_point_cloud = voxel_grid_downsample(points, num_points)

    # 保存降采样后的点云数据到文件
    output_path = os.path.join(device_out_path, os.path.basename(file))
    save_point_cloud(downsampled_point_cloud, output_path)
    print('Saved {}'.format(output_path))

base_root = r'E:\PythonProject\car_pl\project\dataset\28GHz\lidar'
num_points = 2048
# 其中包含各个设备的子文件夹
devices = ['Bus3', 'Car5', 'Car7', 'Car9', 'Car10', 'RSF5', 'RSF8']

with ThreadPoolExecutor() as executor:  # 使用ThreadPoolExecutor类创建线程池
    for device in devices:
        path = os.path.join(base_root, device, 'raw')
        device_out_path = os.path.join(base_root, device, 'down_2048')
        os.makedirs(device_out_path, exist_ok=True)
        files = [os.path.join(path, f) for f in os.listdir(path)]
        executor.map(process_file, files)

