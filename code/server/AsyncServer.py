import torch
from model.CNN import CNN1
from model.CNN import CNN3

from compressor.topk import TopkCompressor
from compressor import NoneCompressor

class AsyncServer:
    def __init__(self,global_config,compressor_config,clients,device):
        # model
        self.model_name = global_config["model"]              
        self.model = self.init_model().to(device)       # mechine learning model

        # global iteration
        self.global_iter = 0        # indicate the version of global model

        # scheduler
        self.scheduler = ServerScheduler()      # create a scheduler object
        self.scheduler.register_clients(clients)        # register all clients

        # sender
        self.sender = ServerSender(compressor_config=compressor_config)
        self.scatter_init_model()       # send initial model to all clients

    def init_model(self):
        if self.model_name == 'CNN1':
            return CNN1()
        elif self.model_name == 'CNN3':
            return CNN3()
    
    def scatter_init_model(self):
        clients = self.scheduler.get_all_registered_clients()       # get all clients
        self.sender.send_to_multi_clients(clients)

class ServerSender:
    def __init__(self,compressor_config):
        self.compressor_config = compressor_config["downlink"]
        self.compressor = self.get_compressor(self.compressor_config)
    
    def get_compressor(self,compressor_config):
        compressor_method = compressor_config["method"]
        if compressor_method == 'topk':
            return TopkCompressor(compressor_config["params"]["cr"])
        elif compressor_method == 'None':
            return NoneCompressor()
    
    def send_to_multi_clients(self,clients):
        for cid,client in clients.items():
            self.send_to_one_clients(client)

    def send_to_one_clients(self,client):
        params = client.get_model_params()      # get client model parameter dict
        compress_model_params = self.compress_all(params)
        client.receive(compress_model_params)

    
    def compress_all(self,params):
        compressed_model_params = {}        # all compressed model params
        for name, param in params.items():
            compressed_param,attribute = self.compress_one(name,param)      # one compressed model params
            compressed_model_params[name] = (compressed_param,attribute)
        return compressed_model_params

    
    def compress_one(self,name,param):
        compressed_param,attribute = self.compressor.compress(param,name)
        return compressed_param,attribute


class ServerScheduler:
    def __init__(self):
        self.registered_clients = {}

    def register_clients(self,clients): # add multi-clients to server scheduler
        for client in clients:
            self.add_client(client)
     
    def add_client(self,client):        # add one client to server scheduler
        cid = client.cid
        if cid in self.registered_clients.keys():
                raise Exception("Client id conflict.")
        self.registered_clients[cid] = client

    def get_all_registered_clients(self):       # get all registered clients
        return self.registered_clients