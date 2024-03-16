import os

# 看一下最大值与最小值
def read_point_cloud(file_path):
    points = []  # 存储点云数据
    with open(file_path, 'r') as file:
        for line in file:
            # 读取每行数据并转换为浮点数
            x = float(line.strip())
            y = float(next(file).strip())
            z = float(next(file).strip())
            # 将点的xyz坐标添加到列表中
            points.append((x, y, z))
    return points


def compute_z_extremes(points):
    if not points:
        return None, None
    z_values = [point[2] for point in points]  # 提取z坐标
    return min(z_values), max(z_values)


# 其中包含各个设备的子文件夹
devices = ['Bus3', 'Car5', 'Car7', 'Car9', 'Car10', 'RSF5', 'RSF8']
out = open('max_min_z.txt', 'w', encoding='utf')

for device in devices:
    out.write(device + '\n')
    for i in range(1, 50):
        path = r'E:\PythonProject\car_pl\project\dataset\28GHz\lidar'
        device_path = os.path.join(path, device, 'raw')
        files = os.listdir(device_path)
        for file in files:
            if i > 10:
                break
            file_path = os.path.join(device_path, file)
            points = read_point_cloud(file_path)
            # 计算z坐标的最大值和最小值
            min_z, max_z = compute_z_extremes(points)
            out.write("{},{},{},{}\n".format('最小', min_z, '最大', max_z))
            print(min_z, max_z)
            i += 1
out.close()

