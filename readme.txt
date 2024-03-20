预测路径损耗分布
使用三种感知数据:
    点云需要处理一下，去掉不用的点，进行降采样
    间隔取样处理，比如隔5个time采用一个
    depth：E:\PythonProject\car_pl\project\dataset\28GHz\depth
    rgb:E:\PythonProject\car_pl\project\dataset\28GHz\RGB
    lidar:E:\PythonProject\car_pl\project\dataset\28GHz\lidar 里面的down是处理过的点云，但是处理方式只进行了降采样，使用的是体素降采样
    pl:E:\PythonProject\car_pl\project\dataset\28GHz\pathloss 是已经从wi的raw data处理好的文件
分布：需要自己处理一下
    在pl的基础上进行，处理成0-250dB 2.5dB为间隔 100个bin



