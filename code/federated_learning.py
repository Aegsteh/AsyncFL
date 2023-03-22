from client.BaseClient import BaseClient
from dataset.CIFAR10 import CIFAR10
from dataset.MNIST import MNIST
import torch
import tools.utils
from tools import jsonTool
from tools.IID import split_data

from server.AsyncServer import AsyncServer


if __name__ == "__main__":
    # load config
    config = jsonTool.generate_config('config.json')            # read config.json file and generate config dict
    client_config = config["client"]                            # get client's config
    data_distribution_config = config["data_distribution"]
    global_config = config["global"]
    client_config["model"] = global_config["model"]             # add model config to client
    client_config["dataset"] = global_config["dataset"]

    device = tools.utils.get_device(config["device"])               # get training device according to os platform, gpu or cpu
    n_clients = global_config["n_clients"]

    compressor_config = config["compressor"]        # gradient compression config

    
    # dataset
    mnist = MNIST()               # load whole dataset
    train_set = mnist.get_train_dataset()               # get global training set
    split = split_data(data_distribution_config,n_clients,train_set)
    test_set = mnist.get_test_dataset()

    # clients
    clients = []
    for i in range(n_clients):
        clients += [BaseClient(cid=i,
                        dataset=split[i],
                        client_config=client_config,
                        compressor_config=compressor_config,
                        device=device)]
    
    # server
    server = AsyncServer(global_config=global_config,
                        dataset=test_set,
                        compressor_config=compressor_config,
                        clients=clients,
                        device=device)
    
    # set server for each client
    for client in clients:
        client.set_server(server)
    
    # start training
    server.start()