from client.baseClient import baseClient
from dataset.CIFAR10 import CIFAR10
from dataset.MNIST import MNIST
import torch
import tools.utils
from tools import jsonTool


# load config
config = jsonTool.generate_config('config.json')            # read config.json file and generate config dict
client_config = config["client"]                    # get client's config

device = tools.utils.get_device(config["device"])               # get training device according to os platform, gpu or cpu

mnist = MNIST()               # load whole dataset
train_set = mnist.get_train_dataset()               # get global training set
train_data,train_label = train_set.data,train_set.targets       # get training data and training label
dataset0 = [train_data,train_label]                             # packege training data and training label

test_set = mnist.get_test_dataset()                             # get testing data and testing label

client0 = baseClient(cid=0,
                     dataset=dataset0,
                     client_config=client_config,
                     device=device)
client0.start()