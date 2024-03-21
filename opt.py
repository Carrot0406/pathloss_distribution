import argparse


def opt_algorithm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_path', type=str, default=r'E:\python\pathloss_distribution\split\RGB_train.txt',
                        help='indicator to dataset')
    parser.add_argument('--depth_path', type=str, default=r'E:\python\pathloss_distribution\split\depth_train.txt',
                        help='indicator to dataset')
    parser.add_argument('--lidar_path', type=str,default=r'E:\python\pathloss_distribution\split\lidar_train.txt',
                        help='indicator to dataset')
    parser.add_argument('--pl_path', type=str, default=r'E:\python\pathloss_distribution\split\pl_distribution_train_train.txt', help='indicator to dataset')

    parser.add_argument('--result_path', type=str, default=r'E:\python\pathloss_distribution\result', help='indicator to dataset')

    parser.add_argument('--method', type=str, default='concat', help='')
    # experiment controls
    parser.add_argument('--modality', type=str, default='img_point', help='choose modality for experiment: v, s, v+s')
    parser.add_argument('--mode', type=str, default='train',
                        help='select from train, val, test. Used in dataset creation')

    # turning parameters

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_finetune', type=float, default=1e-4, help='fine-tune learning rate')
    parser.add_argument('--lrd_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--lrd_rate_finetune', type=float, default=0.1, help='decay rate of fine-tune learning rate')
    parser.add_argument('--lr_decay', type=int, default=4, help='decay rate of learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')

    args = parser.parse_args()

    return args