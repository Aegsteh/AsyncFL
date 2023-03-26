from torchvision import transforms
from dataset.CIFAR10 import CIFAR10
from dataset.FashionMNIST import FashionMNIST
from dataset.MNIST import MNIST


def get_default_data_transforms(name, train=True, verbose=False):
    name = name.lower()
    transforms_train = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        # (0.24703223, 0.24348513, 0.26158784)
        'geo': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]),
        'kws': None,
        'speechcommands':None
    }
    transforms_eval = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),  #
        'geo': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]),
        'kws': None,
        'speechcommands':None
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return (transforms_train[name], transforms_eval[name])

def get_dataset(dataset_name):
    if dataset_name == 'MNIST':
        return MNIST()
    elif dataset_name == 'FMNIST':
        return FashionMNIST()
    elif dataset_name == 'CIFAR10':
        return CIFAR10()