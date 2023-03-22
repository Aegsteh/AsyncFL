import torch
import os,sys
import copy
import time
os.chdir(sys.path[0])

import threading

from multiprocessing import Process
from model.CNN import CNN1
from model.CNN import CNN3
import random
from dataset.CustomerDataset import CustomerDataset
from dataset.utils import get_default_data_transforms
import numpy as np

from compressor.topk import TopkCompressor
from compressor import NoneCompressor

import tools.tensorTool as tl

class BaseClient(threading.Thread):
    def __init__(self,cid,dataset,client_config,compressor_config,device):
        super().__init__()
        self.cid = cid          # the id of client

        # hyperparameters
        self.epoch_num = client_config["local epoch"]      # local iteration num
        self.lr = client_config["optimizer"]["lr"]      # learning rate
        self.momentum = client_config["optimizer"]["momentum"]  # momentum
        self.batch_size = client_config["batch_size"]       # batch size
        self.delay = 1              # simulate network delay

        # dataset
        self.dataset_name = client_config["dataset"]
        self.dataset = dataset      # the dataset of client, a list with 2 elements, the first is all data, the second is all label
        self.split_train_test(proportion=0.8)
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.train_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_train, self.y_train, self.transforms_train), 
                                                        batch_size=self.batch_size,
                                                        shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval), 
                                                        batch_size=self.batch_size,
                                                        shuffle=False)   
        self.epoch_loader = iter(self.train_loader) 

        # model
        self.model_name = client_config["model"]              
        self.model = self.init_model().to(device)       # mechine learning model
        self.model_weight = self.model.state_dict()
        self.model_param_buffer = copy.deepcopy(self.model).state_dict()    # save model parameter received from server
        self.model_old = copy.deepcopy(self.model).state_dict()
        self.gradient = copy.deepcopy(self.model).state_dict()
        self.model_timestamp = 0        # timestamp, to compute staleness for server
        self.transmit_dict = {}


        # loss function
        self.loss_fun_name = client_config["loss function"]        # loss function
        self.loss_function = self.init_loss_fun()

        # optimizer
        self.optimizer_hp = client_config["optimizer"]      # optimizer
        self.optimizer = self.init_optimizer()
        
        # training device
        self.device = device            # training device (cpu or gpu)

        # receiver
        self.receiver = ClientReceiver(compressor_config)

        # sender
        self.sender = ClientSender(compressor_config)

        # multiple process valuable
        self.selected_event = threading.Event()     # indicate if the client is selected
        self.selected_event.clear()         # initialize selected as false
        self.client_lock = threading.Lock()       # lock client when training
        self.model_lock = threading.Lock()          # lock client when changing model weight

    
    def run(self):          # run the client process
        while not self.stop_event.is_set():     # if the training process is going on
            if self.selected_event.is_set():     # if the client is selected by scheduler
                # lock client
                self.client_lock.acquire()    # lock the client to prevent data in client for modifying
                # receive global model and set local model
                self.model.load_state_dict(self.model_param_buffer, strict=True)  # load weight from buffer
                # print("Client {}'s model has loaded in global epoch {}\n".format(self.cid,self.model_timestamp["t"]))
                
                # local training
                self.model_old = copy.deepcopy(self.model_weight)
                # print("Client {} is training in global epoch {}\n".format(self.cid,self.model_timestamp["t"]))
                self.update()           # local training
                print("Client {} training finishes in global epoch {}\n".format(self.cid,self.model_timestamp["t"]))
                tl.subtract_(self.gradient,self.model_weight,self.model_old)     # gradient computation

                # transmit to server (simulate network delay)
                self.transmit_dict["weight"] = self.gradient
                self.transmit_dict["timestamp"] = self.model_timestamp
                time.sleep(self.delay)      # simulate network delay
                self.send(self.transmit_dict)        # send gradient to server
                self.set_selected_event(False)      # set selected false, sympolize the client isn't on training
                self.client_lock.release()        # unlock training lock
            else:
                self.selected_event.wait()
                

    def update(self):
        self.model.train()
        train_acc = 0.0
        train_loss = 0.0
        train_num = 0
        for epoch in range(self.epoch_num):
            try: # Load new batch of data
                features, labels = next(self.epoch_loader)
                features = features.to(self.device)
                labels = labels.to(self.device)
            except: # Next epoch
                pass

            self.optimizer.zero_grad()  # set accumulate gradient to zero
            outputs = self.model(features)  # predict
            loss = self.loss_function(outputs, labels)      # compute loss
            loss.backward()                 # backward, compute gradient
            self.optimizer.step()           # update

            train_loss += loss.item() * features.size(0)         # compute total loss
            _, prediction = torch.max(outputs.data, 1)                  # get prediction label
            train_acc += torch.sum(prediction == labels.data)           # compute training accuracy
            train_num += self.train_loader.batch_size
        
        train_acc = train_acc / train_num              # compute average accuracy and loss
        train_loss = train_loss / train_num
        print("Client {}, Global Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(self.cid,self.model_timestamp["t"], train_acc, train_loss))
        return copy.deepcopy(self.model)

    def init_model(self):
        if self.model_name == 'CNN1':
            return CNN1()
        elif self.model_name == 'CNN3':
            return CNN3()
    
    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()
    
    def init_optimizer(self):
        optimizer_name = self.optimizer_hp["method"]
        if optimizer_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(),self.lr,self.momentum)
    
    def split_train_test(self,proportion): 
        # proportion is the proportion of the training set on the entire data set
        self.data = self.dataset[0]     # get raw data from dataset
        self.label = self.dataset[1]    # get label from dataset

        # package shuffle
        assert len(self.data) == len(self.label)
        randomize = np.arange(len(self.data))
        np.random.shuffle(randomize)
        data = np.array(self.data)[randomize]
        label = np.array(self.label)[randomize]

        # split train and test set
        train_num = int(proportion * len(self.data))       # the number of training samples
        self.train_num = train_num
        self.test_num = len(self.data) - train_num
        self.x_train = data[:train_num]              # the data of training set
        self.y_train = label[:train_num]            # the label of training set
        self.x_test = data[train_num:]               # the data of testing set
        self.y_test = label[train_num:]             # the label of testing set
    
    def receive(self,transmit_dict):
        self.client_lock.acquire()      # lock client
        self.set_selected_event(True)       # set selected event True: the client has been selected
        
        # self.model_lock.acquire()           # start changing model
        self.model_timestamp = transmit_dict["timestamp"]       # timestamp of global model
        
        print("Client {} has been selected in global epoch {}\n".format(self.cid,self.model_timestamp["t"]))
        
        model_weight = transmit_dict["weight"] 
        model_weight = copy.deepcopy(model_weight)    
        decompress_model_params = self.receiver.receive(model_weight)   # receive compress model from server and decompress
        self.model_param_buffer = decompress_model_params       # save decompress model into buffer
        # print("Client {}'s model has been decompressed in global epoch {}\n".format(self.cid,self.model_timestamp["t"]))
        # self.model_lock.release()           # changing model finished

        self.client_lock.release()      # unlock client

    def send(self,transmit_dict):
        transmit_dict = copy.deepcopy(transmit_dict)
        self.sender.send_to_server(self.server,transmit_dict)

    def get_model_params(self):
        return self.model.state_dict()
    
    def set_stop_event(self,stop_event):
        self.stop_event = stop_event
    
    def set_selected_event(self,bool):
        if bool == True:
            self.selected_event.set()
        else:
            self.selected_event.clear()
    
    def set_server(self,server):
        self.server = server

class ClientSender:
    '''
    1. send local model to server
    2. equiped with gradient compression, compress gradient when sending
    '''
    def __init__(self,compressor_config):
        self.compressor_config = compressor_config["uplink"]
        self.compressor = self.get_crompressor(self.compressor_config)

    def get_crompressor(self,compressor_config):
        compressor_method = compressor_config["method"]
        if compressor_method == 'topk':
            return TopkCompressor(compressor_config["params"]["cr"])
        elif compressor_method == 'None':
            return NoneCompressor()

    def send_to_server(self,server,transmit_dict):
        weight = transmit_dict["weight"]
        model_weight = weight       # get parameter dict
        compress_model_params = self.compress_all(model_weight) #  get client model parameter dict
        transmit_dict["weight"] = compress_model_params
        server.receive(transmit_dict)
    
    def compress_all(self,params):
        compressed_model_params = {}        # all compressed model params
        for name, param in params.items():
            compressed_param,attribute = self.compress_one(name,param)      # one compressed model params
            compressed_model_params[name] = (compressed_param,attribute)
        return compressed_model_params

    
    def compress_one(self,name,param):
        compressed_param,attribute = self.compressor.compress(param,name)
        return compressed_param,attribute


class ClientReceiver:
    '''
    1. receive global model and keep it
    2. equiped with gradient compression, decompress gradient when receiving
    '''
    def __init__(self, compressor_config):
        self.compressor_config = compressor_config["downlink"]
        self.compressor = self.get_crompressor(self.compressor_config)
        self.receive_model_params = {}      # received compressed model parameters from server

    
    def get_crompressor(self,compressor_config):
        compressor_method = compressor_config["method"]
        if compressor_method == 'topk':
            return TopkCompressor(compressor_config["params"]["cr"])
        elif compressor_method == 'None':
            return NoneCompressor()
    
    def receive(self,model_params):
        self.receive_model_params = model_params
        self.decompress_model_params = self.decompress_all(model_params)
        return self.decompress_model_params
    
    def decompress_all(self,model_params):
        decompress_model_params = {}
        for name,comprssed_and_attribute in model_params.items():
            compressed_model,attribute = comprssed_and_attribute
            decompress_model_param = self.compressor.decompress(compressed_model,attribute)
            decompress_model_params[name] = decompress_model_param
        return decompress_model_params