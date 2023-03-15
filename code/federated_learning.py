from client.baseClient import baseClient
from dataset.CIFAR10 import CIFAR10
from dataset.MNIST import MNIST
import torch
import tools.utils
from tools import jsonTool
from tools.IID import split_data


if __name__ == "__main__":
    # load config
    config = jsonTool.generate_config('config.json')            # read config.json file and generate config dict
    client_config = config["client"]                            # get client's config
    data_distribution_config = config["data_distribution"]
    global_config = config["global"]

    device = tools.utils.get_device(config["device"])               # get training device according to os platform, gpu or cpu
    n_clients = global_config["n_clients"]

    
    # dataset
    mnist = MNIST()               # load whole dataset
    train_set = mnist.get_train_dataset()               # get global training set
    split = split_data(data_distribution_config,n_clients,train_set)

    # clients
    clients = []
    for i in range(n_clients):
        clients += [baseClient(cid=i,
                        dataset=split[i],
                        client_config=client_config,
                        device=device)]