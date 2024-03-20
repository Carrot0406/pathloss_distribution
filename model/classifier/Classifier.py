import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)

        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(384, 100)  # 更新这里的输入尺寸
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(100, 512)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(512, 1024)
        self.relu7 = nn.ReLU()
        self.fc4 = nn.Linear(1024, 2048)
        self.relu8 = nn.ReLU()
        self.fc5 = nn.Linear(2048, 4096)
        self.relu9 = nn.ReLU()
        self.fc6 = nn.Linear(4096, 5159)
        self.relu10 = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = x.view(x.size(0), -1)  # 改变张量形状
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)
        x = self.relu7(x)
        x = self.fc4(x)
        x = self.relu8(x)
        x = self.fc5(x)
        x = self.relu9(x)
        x = self.fc6(x)
        x = x.view(x.size(0), 5159)
        x = self.relu10(x)  # 改变张量形状
        return x