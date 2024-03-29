'''
Simple CNN for MNIST dataset, the figure in MNIST has 1 channel

'''
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from model.resnet_gn import resnet18

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, mean=0, std=1)
        #     elif isinstance(m,nn.Linear):
        #          nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __deepcopy__(self):
        # 创建一个新的 VGG 实例
        new_model = CNN1(copy.deepcopy(self.conv1), self.classifier[0].in_features, self.classifier[-1].out_features)
        # 复制每个层的参数
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                new_model.classifier[i].weight = copy.deepcopy(layer.weight)
                new_model.classifier[i].bias = copy.deepcopy(layer.bias)
        return new_model

    def copy(self):
        return copy.deepcopy(self)

        
'''
Simple CNN for FMNIST dataset, the figure in FMNIST has 3 channel

'''

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1)
            elif isinstance(m,nn.Linear):
                 nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def VGG11s():
    # if dataset_name == 'CIFAR10':
    #     in_channels = 3
    # else:
    #     in_channels = 1
    in_channels = 1
    return VGG(make_layers([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'], in_channels), size=128)

def VGG11s_3():
    # if dataset_name == 'CIFAR10':
    #     in_channels = 3
    # else:
    #     in_channels = 1
    in_channels = 3
    return VGG(make_layers([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'], in_channels), size=128)

def VGG11():
    # if dataset_name == 'CIFAR10':
    #     in_channels = 3
    # else:
    in_channels = 1
    return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], in_channels))

def make_layers(cfg, in_channels):
    layers = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    '''
    VGG model 
    '''

    def __init__(self, features, size=512, out=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def __deepcopy__(self, memo):
        # 创建一个新的 VGG 实例
        new_model = VGG(copy.deepcopy(self.features), self.classifier[0].in_features, self.classifier[-1].out_features)
        # 复制每个层的参数
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                new_model.classifier[i].weight = copy.deepcopy(layer.weight)
                new_model.classifier[i].bias = copy.deepcopy(layer.bias)
        return new_model

    def copy(self):
        return copy.deepcopy(self)

class ResNet18(nn.Module):

    def __init__(self, group_norm=2):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=False, num_classes=100, num_channels_per_group=group_norm)
    def forward(self, x):
        x = self.resnet(x)
        return x