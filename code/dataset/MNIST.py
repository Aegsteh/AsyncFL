from torchvision import datasets, transforms


class MNIST:
    def __init__(self):
        # get dataset
        self.train_datasets = datasets.MNIST(root='../data/', train=True,download=True)
        self.test_datasets = datasets.MNIST(root='../data/', train=False,download=True)

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets