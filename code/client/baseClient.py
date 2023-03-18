import torch
import os,sys
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

class baseClient(threading.Thread):
    def __init__(self,cid,dataset,client_config,compressor_config,device):
        super().__init__()
        self.cid = cid          # the id of client

        # hyperparameters
        self.epoch_num = client_config["local epoch"]      # local iteration num
        self.lr = client_config["optimizer"]["lr"]
        self.momentum = client_config["optimizer"]["momentum"]
        self.batch_size = client_config["batch_size"]

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
        

        # model
        self.model_name = client_config["model"]              
        self.model = self.init_model().to(device)       # mechine learning model

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
    
    def run(self):          # run the client process
        self.update()

    def update(self):
        for epoch in range(self.epoch_num):
            train_acc = 0.0
            train_loss = 0.0
            self.model.train()
            for i, (features, labels) in enumerate(self.train_loader):
                # load feature and labels
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()  # set accumulate gradient to zero
                outputs = self.model(features)  # predict
                loss = self.loss_function(outputs, labels)      # compute loss
                loss.backward()                 # backward, compute gradient
                self.optimizer.step()           # update

                train_loss += loss.item() * features.size(0)         # compute total loss
                _, prediction = torch.max(outputs.data, 1)                  # get prediction label
                train_acc += torch.sum(prediction == labels.data)           # compute training accuracy
        
            train_acc = train_acc / self.train_num              # compute average accuracy and loss
            train_loss = train_loss / self.train_num
            print("Client {}, Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(self.cid,epoch, train_acc, train_loss))
    
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
    
    def receive(self,model_params):
        decompress_model_params = self.receiver.receive(model_params)
        self.model.load_state_dict(decompress_model_params)

    # def get_receiver(self):
    #     return self.receiver
    
    def get_model_params(self):
        return self.model.state_dict()

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
            return TopkCompressor(compressor_config["param"]["cr"])
        elif compressor_method == 'None':
            return NoneCompressor()


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