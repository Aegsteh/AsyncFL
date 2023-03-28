from client.BaseClient import BaseClient
from client.SyncClient import SyncClient

import dataset.utils as du
import torch
import tools.utils
from tools import jsonTool
from tools.IID import split_data,get_global_data

import tools.resultTools as rt
import tools.delayTools as dt

from server.AsyncServer import AsyncServer
from server.SyncServer import SyncServer


if __name__ == "__main__":
    # load config
    config = jsonTool.generate_config('config.json')            # read config.json file and generate config dict
    client_config = config["client"]                            # get client's config
    data_distribution_config = config["data_distribution"]
    global_config = config["global"]
    client_config["model"] = global_config["model"]             # add model config to client
    client_config["dataset"] = global_config["dataset"]
    client_config["loss function"] = global_config["loss function"]

    device = tools.utils.get_device(config["device"])               # get training device according to os platform, gpu or cpu
    n_clients = global_config["n_clients"]

    compressor_config = config["compressor"]        # gradient compression config

    # dataset
    dataset = du.get_dataset(global_config["dataset"])               # load whole dataset
    train_set = dataset.get_train_dataset()               # get global training set
    split = split_data(data_distribution_config, n_clients, train_set)
    test_set = dataset.get_test_dataset()
    test_set = get_global_data(test_set)

    # clients
    clients = []
    if global_config["mode"] == 'sync':
        # synchronous clients
        for i in range(n_clients):
            clients += [SyncClient(cid=i,
                                dataset=split[i],
                                client_config=client_config,
                                compression_config=compressor_config,
                                device=device)]
        # server
        server = SyncServer(global_config=global_config,
                            dataset=test_set,
                            compressor_config=compressor_config,
                            clients=clients,
                            device=device)
    elif global_config["mode"] == 'async':   
        # simulate delay
        delays = dt.generate_delays(global_config)
        for i in range(n_clients):
            clients += [BaseClient(cid=i,
                            dataset=split[i],
                            client_config=client_config,
                            compression_config=compressor_config,
                            delay=delays[i],
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