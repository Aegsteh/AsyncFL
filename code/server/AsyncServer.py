import torch
from model.CNN import CNN1,CNN3,VGG11s,VGG11

import threading
import queue
import copy
import time
import schedule

from compressor.topk import TopkCompressor
from compressor import NoneCompressor

from dataset.CustomerDataset import CustomerDataset
from dataset.utils import get_default_data_transforms

import server.ScheduleClass as sc
from server.ScheduleClass import find_ScheduleClass

import tools.jsonTool
import tools.tensorTool as tl
import tools.resultTools as rt

config = tools.jsonTool.generate_config('config.json')
global_config = config["global"]

class AsyncServer:
    def __init__(self,global_config,dataset,compressor_config,clients,device):
        # mutiple processes valuable
        self.global_stop_event = threading.Event()
        self.global_stop_event.clear()
        self.server_lock = threading.Lock()

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

        # initilize 
        # self.scatter_init_model()           # ditribute initial global model to all clients
        # print("Transmit initialized global model to all clients\n")

        # Start Training
        self.global_manager.start_clients()     # start all clients fro global training

        # start updater and scheduler
        schedule.every(self.global_config["epoch_time"]).seconds.do(self.update)
        while self.current_epoch < self.total_epoch:
            schedule.run_pending()

        # Process join
        # self.global_manager.join_clients()      # join all clients to train
        # self.updater.join()                 # updater join
        # self.scheduler.join()               # scheduler join
        self.global_stop_event.set()
        
        # Exit
        print("Global Updater Exit.\n")
        self.global_manager.stop_clients()

    def update(self):
        if self.current_epoch < self.total_epoch:        # if global training is going on
            self.server_lock.acquire()          # lock server to update global model

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
                data_nums = torch.Tensor(data_nums)
                tl.weighted_average(target=self.dW, 
                                    sources=client_gradients, 
                                    weights=data_nums)             # global gradient
                tl.add(target=self.W, source=self.dW)

                self.eval_model()
            
            self.current_epoch += 1
            print("Current Epoch: {}".format(self.current_epoch))
            self.server_lock.release()

            # save results
            global_loss, global_acc = self.get_accuracy_and_loss_list()
            staleness_list = self.get_staleness_list()
            rt.save_results(config["result"]["path"],
                        dir_name="{}_{}_{}".format(global_config["model"],global_config["dataset"],self.compressor_config["uplink"]["params"]["cr"]),
                        config=config,
                        global_loss=global_loss,
                        global_acc=global_acc,
                        staleness=staleness_list)

            # schedule
            participating_clients = self.schedule(self.global_manager.clients_list,self.schedule_config)
            self.select_clients(participating_clients)
      
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
        self.x_test = dataset[0]
        self.y_test = dataset[1]
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval), 
                                                        batch_size=8,
                                                        shuffle=False)
        
        # multiple process valuable
        self.stop_event = stop_event
        
    
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

    def get_all_registered_clients(self):       # get all registered clients
        return self.registered_clients
    
    def start_clients(self):        # start all clients training 
        print("Start all client-threads\n")
        clients_dict = self.get_clients_dict()
        for cid,client_thread in clients_dict.items():
            client_thread.set_stop_event(self.stop_event)       # set stop event false, i.e. start training
            client_thread.start()               # start client
    
    def start_one_client(self,cid):
        clients_dict = self.get_clients_dict()      # get all clients
        for c in clients_dict.keys():
            if c == cid:
                clients_dict[c].start()

    def join_clients(self):
        clients_list = self.get_clients_dict()      # get all clients
        for cid,client_thread in clients_list.items():
            client_thread.join()               # start all clients
    
    def stop_clients(self):
        clients_list = self.get_clients_dict()      # get all clients
        for cid,client_thread in clients_list.items():
            client_thread.stop_event.set()               # start all clients

# class ServerUpdater(threading.Thread):
#     def __init__(self,updater_config,server,parameter_collection,current_epoch,total_epoch,server_lock,global_stop_event,global_W):
#         threading.Thread.__init__(self)
#         self.parameter_collection = parameter_collection        # save useful parameters

#         self.server = server

#         self.current_epoch = current_epoch              # global current epoch
#         self.total_epoch = total_epoch                  # global total epoch

#         self.server_lock = server_lock          # server process lock
#         self.global_stop_event = global_stop_event

#         self.W = global_W         # global model

#         self.accuracy_list = []
#         self.loss_list = []

#         self.staleness_list = []

#         self.updater_params = updater_config["params"]
#         if updater_config["method"] == "single":
#             self.update_fun = self.single_update
#         elif updater_config["method"] == "period":
#             self.update_fun = self.time_update

#     def single_model_update(self, global_W, local_W, params):
#         alpha = params["alpha"]
#         for name in global_W:
#             global_W[name].data = (1 - alpha)  * global_W[name].data.clone() + alpha * local_W[name].data.clone()

#     def single_update(self):
#         while self.current_epoch["t"] < self.total_epoch:        # if global training is going on
#             if not self.parameter_collection.decompress_queue_empty():      # if server has received model from client
#                 self.server_lock.acquire()          # lock server to update global model
#                 local_W = self.parameter_collection.pick_decompress_model()     # get local model to update
#                 local_timestamp = self.parameter_collection.pick_timestamp()        # get local timestamp to compute staleness
                
#                 alpha = self.updater_params["alpha"]                # update
#                 for name in self.W:
#                     self.W[name].data = (1 - alpha)  * self.W[name].data.clone() + alpha * local_W[name].data.clone()
                
#                 staleness = self.current_epoch["t"] - local_timestamp        # compute staleness
#                 self.staleness_list.append(staleness)
#                 print("staleness = {}\n".format(staleness))

#                 self.eval_model()

#                 self.current_epoch["t"] = self.current_epoch["t"] + 1     # update global epoch
#                 self.server_lock.release()
#                 time.sleep(0.01)
#             else:
#                 time.sleep(0.01)
#         print("Global Updater Exit.\n")
#         self.global_stop_event.set()

#     def run(self):
#         self.update_fun()

#     def eval_model(self):
#         self.server.model.eval()
#         data_loader = self.server.global_manager.test_loader
#         test_correct = 0.0
#         test_loss = 0.0
#         for data in data_loader:
#             features, labels = data
#             features = features.to(self.server.device)
#             labels = labels.to(self.server.device)
#             outputs = self.server.model(features)  # predict
#             _, id = torch.max(outputs.data, 1)
#             test_loss += self.server.loss_func(outputs, labels).item()
#             test_correct += torch.sum(id == labels.data).cpu().numpy()
#         accuracy = test_correct / len(data_loader)
#         loss = test_loss / len(data_loader)

#         self.accuracy_list.append(accuracy)
#         self.loss_list.append(loss)
#         print("Server: Global Epoch {}, Test Accuracy: {} , Test Loss: {}".format(self.current_epoch, accuracy, loss))
    
#     def get_accuracy_and_loss_list(self):
#         return self.accuracy_list, self.loss_list

#     def get_staleness_list(self):
#         return self.staleness_list
    
# class AsyncGlobalScheduler(threading.Thread):
#     def __init__(self, clients, server, schedule_config, async_client_manager,current_epoch,server_lock,global_stop_event,total_epoch):
#         threading.Thread.__init__(self)
#         self.server = server        # set server

#         self.current_epoch = current_epoch
#         self.total_epoch = total_epoch

#         self.server_lock = server_lock
#         self.global_stop_event = global_stop_event

#         self.schedule_config = schedule_config
#         self.async_client_manager = async_client_manager
#         self.parameter_collection = self.server.parameter_collection
#         self.sender = self.server.sender

#     def client_selection(self):
#         clients = self.async_client_manager.get_clients_dict()
#         selected_clients = self.schduleClass.schedule(clients)
#         return selected_clients
    
#     def run(self):
#         if self.current_epoch["t"] < self.total_epoch:        # if global training is going on
#             selected_clients = self.client_selection()      # it is likely that there are one or more clients
#             if len(selected_clients) == 0:
#                 return
#             self.server_lock.acquire()      # lock the server to transmition
#             transmit_dict = {}      # transmit dict, including global weight, timestamp of global weight
#             for selected_client in selected_clients:
#                 transmit_dict = {}
#                 tl.copy_weight(self.server.W_compress,self.server.W) # get global weight
#                 transmit_dict["weight"] = self.server.W_compress
#                 timestamp = copy.deepcopy(self.current_epoch)["t"]
#                 transmit_dict["timestamp"] = timestamp
#                 self.sender.send_to_one_client(selected_client,transmit_dict)      # send to selected client
#             self.server_lock.release()      # unlock



# class ServerSender:
#     def __init__(self,compressor_config):
#         self.compressor_config = compressor_config["downlink"]
#         self.compressor = self.get_compressor(self.compressor_config)
    
#     def get_compressor(self,compressor_config):
#         compressor_method = compressor_config["method"]
#         if compressor_method == 'topk':
#             return TopkCompressor(compressor_config["params"]["cr"])
#         elif compressor_method == 'None':
#             return NoneCompressor()
    
#     def send_to_multi_clients(self,clients,transmit_dict):
#         for cid,client in clients.items():
#             self.send_to_one_client(client,transmit_dict)

#     def send_to_one_client(self,client,transmit_dict):
#         weight = transmit_dict["weight"]      # get client model parameter dict
#         compress_model_weight = self.compress_all(weight)       # compress model weight
#         transmit_dict["weight"] = compress_model_weight     # fix the transmit_dict for compressed model weight
#         client.receive(transmit_dict)
    
#     def compress_all(self,params):
#         compressed_model_params = {}        # all compressed model params
#         for name, param in params.items():
#             compressed_param,attribute = self.compress_one(name,param)      # one compressed model params
#             compressed_model_params[name] = (compressed_param,attribute)
#         return compressed_model_params

#     def compress_one(self,name,param):
#         compressed_param,attribute = self.compressor.compress(param,name)
#         return compressed_param,attribute


# class ServerReceiver:
#     '''
#     1. receive global model and keep it
#     2. equiped with gradient compression, decompress gradient when receiving
#     '''
#     def __init__(self, compressor_config):
#         self.compressor_config = compressor_config["uplink"]
#         self.compressor = self.get_crompressor(self.compressor_config)
#         self.receive_model_params = {}      # received compressed model parameters from server

#     def get_crompressor(self,compressor_config):
#         compressor_method = compressor_config["method"]
#         if compressor_method == 'topk':
#             return TopkCompressor(compressor_config["params"]["cr"])
#         elif compressor_method == 'None':
#             return NoneCompressor()
    
#     def receive(self,model_params):
#         self.receive_model_params = model_params            # receive compress model
#         self.decompress_model_params = self.decompress_all(model_params)    # decompress
#         return self.decompress_model_params
    
#     def decompress_all(self,model_params):
#         decompress_model_params = {}
#         for name,comprssed_and_attribute in model_params.items():
#             compressed_model,attribute = comprssed_and_attribute
#             decompress_model_param = self.compressor.decompress(compressed_model,attribute)
#             decompress_model_params[name] = decompress_model_param
#         return decompress_model_params

# class ParameterCollection:
#     def __init__(self):
#         self.received_compress_queue = queue.Queue()
#         self.decompress_model = {}
#         self.decompress_model_queue = queue.Queue()   
#         self.timestamp_queue = queue.Queue()

#     def set_received_compress_model(self,received_model):
#         # self.received_compress_model = received_model
#         self.received_compress_queue.put(received_model)

#     def add_timestamp(self,timestamp):
#         self.timestamp_queue.put(timestamp)
    
#     def pick_decompress_model(self):
#         return self.decompress_model_queue.get()
    
#     def pick_timestamp(self):
#         return self.timestamp_queue.get()
    
#     def decompress_queue_empty(self):
#         return self.decompress_model_queue.empty()

#     def get_received_compress_model(self):
#         return self.received_compress_model
    
#     def set_decompress_model(self,decompress_model):
#         # self.decompress_model = decompress_model
#         self.decompress_model_queue.put(decompress_model)
    
#     def get_decompress_model(self):
#         return self.decompress_model