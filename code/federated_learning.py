from client.SyncClient import SyncClient
from client.AsyncClient import AsyncClient
from client.AFOClient import AFOClient
from client.FedBuffClient import FedBuffClient

import dataset.utils as du
import torch
import ctypes
import tools.utils
from tools import jsonTool
from tools.IID import split_data, get_global_data
import multiprocessing
from multiprocessing import Manager
import tools.resultTools as rt
import tools.delayTools as dt

from server.AsyncServer import AsyncServer
from server.SyncServer import SyncServer
from server.AFOServer import AFOServer
from server.FedBuffServer import FedBuffServer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == "__main__":
    mode = 'sync'
    # load config
    multiprocessing.set_start_method('spawn', force=True)
    # read config json file and generate config dict
    config_file = jsonTool.get_config_file(mode=mode)
    config = jsonTool.generate_config(config_file)
    # get client's config
    client_config = config["client"]
    data_distribution_config = config["data_distribution"]
    global_config = config["global"]
    # add model config to client
    client_config["model"] = global_config["model"]
    client_config["dataset"] = global_config["dataset"]
    client_config["loss function"] = global_config["loss function"]

    # get training device according to os platform, gpu or cpu
    device = tools.utils.get_device(config["device"])
    n_clients = global_config["n_clients"]

    # gradient compression config
    compressor_config = config["compressor"]

    # dataset
    # load whole dataset
    dataset = du.get_dataset(global_config["dataset"])
    train_set = dataset.get_train_dataset()               # get global training set
    split = split_data(data_distribution_config, n_clients, train_set)
    test_set = dataset.get_test_dataset()
    test_set = get_global_data(test_set)

    # clients
    for cr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    # for cr in [0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625]:
        for li in [10, 20, 30, 40, 50, 60]:
            client_config["local epoch"] = global_config["local epoch"] = li
            compressor_config["uplink"]["params"]["cr"] = cr
            clients = []
            if mode == 'sync':
                clients = []
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
                for client in clients:
                    client.set_server(server)
                server.start()
            elif mode == 'period':
                clients = []
                # print config
                jsonTool.print_config(config)
                MANAGER = Manager()         # multiprocessing manager
                # initialize STOP_EVENT, representing for if global training stops
                STOP_EVENT = MANAGER.Value(ctypes.c_bool, False)
                SELECTED_EVENT = MANAGER.list(
                    [False for i in range(global_config["n_clients"])])
                GLOBAL_QUEUE = MANAGER.Queue()
                GLOBAL_INFO = MANAGER.list([0])
                # simulate delay
                delays = dt.generate_delays(global_config)
                for i in range(n_clients):
                    clients += [AsyncClient(cid=i,
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
                # start training
                server.start(STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO)
            elif mode == 'afo':
                clients = []
                # print config
                jsonTool.print_config(config)
                MANAGER = Manager()  # multiprocessing manager
                # initialize STOP_EVENT, representing for if global training stops
                STOP_EVENT = MANAGER.Value(ctypes.c_bool, False)
                SELECTED_EVENT = MANAGER.list(
                    [False for i in range(global_config["n_clients"])])
                GLOBAL_QUEUE = MANAGER.Queue()
                GLOBAL_INFO = MANAGER.list([0])
                # simulate delay
                delays = dt.generate_delays(global_config)
                for i in range(n_clients):
                    clients += [AFOClient(cid=i,
                                            dataset=split[i],
                                            client_config=client_config,
                                            compression_config=compressor_config,
                                            delay=delays[i],
                                            device=device)]

                # server
                server = AFOServer(global_config=global_config,
                                     dataset=test_set,
                                     compressor_config=compressor_config,
                                     clients=clients,
                                     device=device)
                # start training
                server.start(STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO)
            elif mode == 'FedBuff':
                clients = []
                # print config
                jsonTool.print_config(config)
                MANAGER = Manager()  # multiprocessing manager
                # initialize STOP_EVENT, representing for if global training stops
                STOP_EVENT = MANAGER.Value(ctypes.c_bool, False)
                SELECTED_EVENT = MANAGER.list(
                    [False for i in range(global_config["n_clients"])])
                GLOBAL_QUEUE = MANAGER.Queue()
                GLOBAL_INFO = MANAGER.list([0])
                # simulate delay
                delays = dt.generate_delays(global_config)
                for i in range(n_clients):
                    clients += [FedBuffClient(cid=i,
                                            dataset=split[i],
                                            client_config=client_config,
                                            compression_config=compressor_config,
                                            delay=delays[i],
                                            device=device)]

                # server
                server = FedBuffServer(global_config=global_config,
                                     dataset=test_set,
                                     compressor_config=compressor_config,
                                     clients=clients,
                                     device=device)
                # start training
                server.start(STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO)