# 将从WI获取的PL数据处理成分布 前提是先将PL数据分离出来
import os

import numpy as np
import matplotlib.pyplot as plt


# 参数pl文件路径、分布输出路径、是否可视化图像并存储以及图片存储路径
def get_distrubution(in_path, out_path, is_vision, vision_path):
    # 读取原始数据
    data = np.loadtxt(in_path)

    # 计算分布情况
    hist, bin_edges = np.histogram(data, bins=np.arange(0, 252.5, 2.5), density=True)

    # 将结果存储到新的 txt 文件中
    with open(out_path, 'w') as f:
        # f.write("BinCenter Probability\n")
        for i in range(len(hist)):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            f.write(f"{hist[i]}\n")

    if is_vision:
        # 绘制直方图
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers, hist, width=2.5, align='center', color='blue', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.title('Distribution of Values')
        plt.grid(True)
        plt.savefig(vision_path)  # 保存为图片
        # plt.show()


if __name__ == '__main__':
    base_root = r'E:\PythonProject\car_pl\project\dataset\28GHz\pathloss'
    out_root = r'E:\PythonProject\car_pl\project\dataset\28GHz\pl_distrubution'
    vision_root = r'E:\PythonProject\car_pl\project\dataset\28GHz\pl_dis_vision'
    # 其中包含各个设备的子文件夹
    devices = ['Car5', 'Car7', 'Car9', 'Car10', 'RSF5', 'RSF8']
    for device in devices:
        device_path = os.path.join(base_root, device)
        files = os.listdir(device_path)
        for file in files:
            in_path = os.path.join(device_path, file)
            out_dir = os.path.join(out_root, device)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, file)
            vis_dir = os.path.join(vision_root, device)
            os.makedirs(vis_dir, exist_ok=True)
            img_name = file.replace('txt','png')
            vis_path = os.path.join(vis_dir, img_name)
            get_distrubution(in_path, out_path, is_vision=False, vision_path=vis_path)
            print(out_path)
