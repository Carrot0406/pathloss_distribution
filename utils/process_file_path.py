# 将训练集 验证集 测试集使用的文件路径进行划分
import os
import re
import numpy as np


def get_number_from_filename(filename):
    return int(filename.split("time")[1].split("_")[0])

# 间隔取样
def get_files_5(folder_path):
    '''
    拿到一个文件夹下面合适的路径
    :param folder_path:
    :return:
    '''
    # 匹配文件名中的数字，并保存模 5 为 1 的文件路径
    file_paths = []
    file_names = os.listdir(folder_path)
    sorted(file_names,key=get_number_from_filename)
    for file_name in file_names:
        numbers = re.findall(r'\d+', file_name)  # 提取文件名中的所有数字
        if numbers:
            number = int(numbers[0])  # 提取第一个数字
            if number % 5 == 1:
                file_paths.append(os.path.join(folder_path, file_name))
    # # 对文件路径进行排序
    # sorted(file_paths, key=get_number_from_filename)
    return file_paths

# 全部取样
def get_files(folder_path):
    '''
    拿到一个文件夹下面合适的路径
    :param folder_path:
    :return:
    '''
    file_paths = []
    file_names = os.listdir(folder_path)
    sorted(file_names,key=get_number_from_filename)
    for file_name in file_names:
        file_paths.append(os.path.join(folder_path, file_name))
    # # 对文件路径进行排序 注意需要开一个新的变量
    file_paths_new = sorted(file_paths, key=get_number_from_filename)
    return file_paths_new

base_root = r'/home/bailu/radio_science/dataset'
out_root = r'/home/bailu/radio_science/dataset'

model_list = ['pathloss','depth', 'RGB', 'lidar']

train_list = ['Car5', 'Car9', 'Car10', 'RSF5']
test_list = ['Car7', 'RSF8']
val_list = ['Bus3']

for model in model_list:
    train_txt = []
    test_txt = []
    val_txt = []
    for train in train_list:
        train_path = os.path.join(base_root, model, train)
        single_list = get_files(train_path)
        train_txt += single_list
    # 将文件路径保存到 txt 文件中
    output_file_path = model + '_train' + '.txt'
    with open(output_file_path, 'w') as f:
        for file_path in train_txt:
            f.write(file_path + '\n')

    for test in test_list:
        test_path = os.path.join(base_root, model, test)
        single_list = get_files(test_path)
        test_txt += single_list
    # 将文件路径保存到 txt 文件中
    output_file_path = model + '_test' + '.txt'
    with open(output_file_path, 'w') as f:
        for file_path in test_txt:
            f.write(file_path + '\n')

    for val in val_list:
        val_path = os.path.join(base_root, model, val)
        single_list = get_files(val_path)
        val_txt += single_list
    # 将文件路径保存到 txt 文件中
    output_file_path = model + '_val' + '.txt'
    with open(output_file_path, 'w') as f:
        for file_path in val_txt:
            f.write(file_path + '\n')
