from torchvision import datasets, transforms
import numpy as np


class CIFAR10:
    def __init__(self):
        # 获取数据集
        train_datasets = datasets.CIFAR10(root='../data/', train=True,
                                        transform=transforms.ToTensor(), download=True)
        test_datasets = datasets.CIFAR10(root='../data/', train=False,
                                       transform=transforms.ToTensor(), download=True)
        train_datasets.data = train_datasets.data.transpose((0, 3, 1, 2))
        test_datasets.data = test_datasets.data.transpose((0, 3, 1, 2))
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets

if __name__ == '__main__':
    CIFAR10()