import torch
from model.CNN import CNN1,CNN3,VGG11s,VGG11

import threading
import queue
import copy
import schedule

from compressor.topk import TopkCompressor
from compressor import NoneCompressor

from dataset.CustomerDataset import CustomerDataset
from dataset.utils import get_default_data_transforms

import server.ScheduleClass as sc

import tools.jsonTool
import tools.tensorTool as tl
import tools.resultTools as rt

config = tools.jsonTool.generate_config('config.json')
global_config = config["global"]

class SyncServer:
    def __init__(self,global_config,dataset,compressor_config,clients,device):
        # global_config
        self.global_config = global_config
        self.schedule_config = global_config["schedule"]

        # device
        self.device = device

        # model
        self.model_name = global_config["model"]              
        self.model = self.init_model().to(device)       # mechine learning model
        self.W = {name : value for name, value in self.model.named_parameters()}
        self.dW_compress = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        # receive queue
        self.parameter_queue = queue.Queue()            

        # dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset

        # global iteration
        self.current_epoch = 0       # indicate the version of global model
        self.total_epoch = global_config["epoch"]

        # loss function
        self.loss_fun_name = global_config["loss function"]        # loss function
        self.loss_func = self.init_loss_fun()

        self.compressor_config = compressor_config

        # results
        self.staleness_list = []
        self.loss_list = []
        self.accuracy_list = []

        # global manager
        self.global_manager = SyncGlobalManager(clients=clients,
                                                 dataset=dataset,
                                                 global_config=global_config)
        
    def start(self):        # start the whole training priod
        print("Start global training...\n")

        self.update()
        
        # Exit
        print("Global Updater Exit.\n")

    def update(self):
        for epoch in range(self.global_config["epoch"]):
            # select clients
            participating_clients = self.schedule(self.global_manager.clients_list,self.schedule_config)
            for client in participating_clients:
                client.run()
            
            client_gradients = []           # save multi local_W
            data_nums = []
            while not self.parameter_queue.empty():
                transmit_dict = self.parameter_queue.get()   # get information from client,(cid, client_gradient, data_num, timestamp)
                cid = transmit_dict["cid"]                                    # cid
                client_gradient = transmit_dict["client_gradient"]            # client gradient
                data_num = transmit_dict["data_num"]                          # number of data samples

                client_gradients.append(client_gradient)            
                data_nums.append(data_num)

            data_nums = torch.Tensor(data_nums)
            tl.weighted_average(target=self.dW, 
                                sources=client_gradients, 
                                weights=data_nums)             # global gradient
            tl.add(target=self.W, source=self.dW)

            self.eval_model()

            # save results
            global_loss, global_acc = self.get_accuracy_and_loss_list()
            staleness_list = self.get_staleness_list()
            rt.save_results(config["result"]["path"],
                        dir_name="{}_{}_{}".format(global_config["model"],global_config["dataset"],self.compressor_config["uplink"]["params"]["cr"]),
                        config=config,
                        global_loss=global_loss,
                        global_acc=global_acc,
                        staleness=staleness_list)
      
    def init_model(self):
        if self.model_name == 'CNN1':
            return CNN1()
        elif self.model_name == 'CNN3':
            return CNN3()
        elif self.model_name == 'VGG11s':
            return VGG11s()
        elif self.model_name == 'VGG11':
            return VGG11()
        
    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()
    
    def scatter_init_model(self):
        for cid,client in self.global_manager.get_clients_dict().items():
            client.synchronize_with_server(self)
            model_timestamp = copy.deepcopy(self.current_epoch)["t"]
            client.model_timestamp = model_timestamp
    
    def schedule(self,clients, schedule_config, **kwargs):
        participating_clients = sc.schedule(clients,schedule_config)
        return participating_clients
    
    def select_clients(self,participating_clients):
        for client in participating_clients:
            client.set_selected_event(True)

    
    def receive(self,transmit_dict):
        self.parameter_queue.put(transmit_dict)
      
    def eval_model(self):
        self.model.eval()
        data_loader = self.global_manager.test_loader
        test_correct = 0.0
        test_loss = 0.0
        test_num = 0
        for data in data_loader:
            features, labels = data
            features = features.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(features)  # predict
            _, id = torch.max(outputs.data, 1)
            test_loss += self.loss_func(outputs, labels).item()
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_num += len(features)
        accuracy = test_correct / test_num
        loss = test_loss / test_num

        self.accuracy_list.append(accuracy)
        self.loss_list.append(loss)
        print("Server: Global Epoch {}, Test Accuracy: {} , Test Loss: {}".format(self.current_epoch, accuracy, loss))
    
    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_staleness_list(self):
        return self.staleness_list

class SyncGlobalManager:       # Manage clients and global information
    def __init__(self,clients,dataset,global_config):
        # clients
        self.clients_num = len(clients)
        self.clients_list = clients
        self.clients_dict = {}
        self.register_clients(clients)

        # global infromation
        self.global_epoch = global_config["epoch"]      # global epoch/iteration
        self.global_acc = []            # test accuracy
        self.global_loss = []           # training loss

        # global test dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset      # the test dataset of server, a list with 2 elements, the first is all data, the second is all label
        self.x_test = dataset[0]
        self.y_test = dataset[1]
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval), 
                                                        batch_size=8,
                                                        shuffle=False)
        
    
    def find_client_by_cid(self,cid):       # find client by cid
        for client in self.clients:
            if client.cid == cid:
                return client
        return None
    
    def get_clients_dict(self):
        return self.clients_dict
    
    def register_clients(self,clients): # add multi-clients to server scheduler
        for client in clients:
            self.add_client(client)
     
    def add_client(self,client):        # add one client to server scheduler
        cid = client.cid
        if cid in self.clients_dict.keys():
                raise Exception("Client id conflict.")
        self.clients_dict[cid] = client
    
    def start_one_client(self,cid):
        clients_dict = self.get_clients_dict()      # get all clients
        for c in clients_dict.keys():
            if c == cid:
                clients_dict[c].start()
