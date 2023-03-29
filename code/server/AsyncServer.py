import torch
from model.CNN import CNN1,CNN3,VGG11s,VGG11

import threading
import queue
import copy
import time
import schedule
import multiprocessing 

from compressor.topk import TopkCompressor
from compressor import NoneCompressor

from dataset.CustomerDataset import CustomerDataset
from dataset.utils import get_default_data_transforms

import server.ScheduleClass as sc

import tools.jsonTool
import tools.tensorTool as tl
import tools.resultTools as rt

from client.BaseClient import run_client

config = tools.jsonTool.generate_config('config.json')
global_config = config["global"]

class AsyncServer:
    def __init__(self,global_config,dataset,compressor_config,clients,device):
        # mutiple processes valuable
        self.global_stop_event = False

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
        self.global_manager = AsyncGlobalManager(clients=clients,
                                                 dataset=dataset,
                                                 global_config=global_config,
                                                 stop_event=self.global_stop_event)
        
    def start(self):        # start the whole training priod
        print("Start global training...\n")

        # Start Training
        self.global_manager.start_clients()     # start all clients for global training

        # start updater and scheduler
        schedule.every(self.global_config["epoch_time"]).seconds.do(self.update)
        while self.current_epoch < self.total_epoch:
            schedule.run_pending()
            
        self.global_stop_event.set()
        
        # Exit
        print("Global Updater Exit.\n")
        self.global_manager.stop_clients()

    def update(self):
        if self.current_epoch >= self.total_epoch:        # if global training is going on
            return
        
        if not self.parameter_queue.empty():            # if server has received some gradients from clients
            client_gradients = []           # save multi local_W
            data_nums = []
            stalenesses = []
            while not self.parameter_queue.empty():
                transmit_dict = self.parameter_queue.get()   # get information from client,(cid, client_gradient, data_num, timestamp)
                cid = transmit_dict["cid"]                                    # cid
                client_gradient = transmit_dict["client_gradient"]            # client gradient
                data_num = transmit_dict["data_num"]                          # number of data samples
                timestamp = transmit_dict["timestamp"]                        # timestamp of client gradient
                staleness = self.current_epoch - timestamp                  # staleness

                client_gradients.append(client_gradient)            
                data_nums.append(data_num)
                stalenesses.append(staleness)
                self.staleness_list.append(staleness)
            tl.weighted_average(target=self.dW, 
                                sources=client_gradients, 
                                weights=torch.Tensor(data_nums))             # global gradient
            tl.add(target=self.W, source=self.dW)

            self.eval_model()
        
        self.current_epoch += 1
        print("Current Epoch: {}".format(self.current_epoch))

        # save result
        self.save_result()

        # schedule
        participating_clients = self.schedule(self.global_manager.clients_list,self.schedule_config)
        self.select_clients(participating_clients)
    
    def save_result(self):
        global_loss, global_acc = self.get_accuracy_and_loss_list()
        staleness_list = self.get_staleness_list()
        rt.save_results(config["result"]["path"],
                        dir_name="{}_{}_{}".format(
                            global_config["model"], global_config["dataset"], self.compressor_config["uplink"]["params"]["cr"]),
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
    
    def schedule(self,clients, schedule_config, **kwargs):
        participating_clients = sc.schedule(clients,schedule_config)
        return participating_clients
    
    def select_clients(self,participating_clients):
        for client in participating_clients:
            client.set_selected_event(True)

    
    def receive(self,transmit_dict):
        self.server_lock.acquire()      # lock
        self.parameter_queue.put(transmit_dict)
        self.server_lock.release()      # unlock
      
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

class AsyncGlobalManager:       # Manage clients and global information
    def __init__(self,clients,dataset,global_config,stop_event):
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
        self.x_test = dataset[0].numpy()
        self.y_test = dataset[1].numpy()
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval), 
                                                        batch_size=8,
                                                        shuffle=False)
        
        # multiple process valuable
        self.stop_event = stop_event        # False for initialization
        
    
    def find_client_by_cid(self,cid):       # find client by cid
        for client in self.clients:
            if client.cid == cid:
                return client
        return None
    
    def get_clients_dict(self):
        return self.clients_dict
    
    def get_clients_list(self):
        return self.clients_list
    
    def register_clients(self,clients): # add multi-clients to server scheduler
        for client in clients:
            self.add_client(client)
     
    def add_client(self,client):        # add one client to server scheduler
        cid = client.cid
        if cid in self.clients_dict.keys():
                raise Exception("Client id conflict.")
        self.clients_dict[cid] = client
    
    def start_clients(self):        # start all clients training 
        client_pool = multiprocessing.Pool(len(self.get_clients_list()))
        print("Start all client-threads\n")
        clients_dict = self.get_clients_dict()
        for cid,client_thread in clients_dict.items():
            client_thread.set_stop_event(self.stop_event)       # set stop event false, i.e. start training
            client_pool.apply_async(run_client, args=(
                client_thread,), error_callback=err_call_back)       # add process to process pool
        client_pool.close()
        client_pool.join()

    
    def stop_clients(self):
        clients_list = self.get_clients_dict()      # get all clients
        for cid,client_thread in clients_list.items():
            client_thread.set_stop_event(self.stop_event)               # start all clients

def err_call_back(err):
        print(f'Error: {str(err)}')