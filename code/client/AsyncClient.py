from model.CNN import CNN1, CNN3, VGG11s, VGG11
from tools import jsonTool
import tools.utils
import tools.tensorTool as tl
import compressor.compression_utils as comp
from compressor import NoneCompressor
from compressor.topk import TopkCompressor
import numpy as np
from dataset.utils import get_default_data_transforms
from dataset.CustomerDataset import CustomerDataset
import random
from multiprocessing import Process
import multiprocessing
import torch
import os
import sys
import copy
import traceback
import time
os.chdir(sys.path[0])


config = jsonTool.generate_config('config.json')
device = tools.utils.get_device(config["device"])


class AsyncClient:
    def __init__(self, cid, dataset, client_config, compression_config, delay, device):
        self.cid = cid          # the id of client

        # model
        self.model_name = client_config["model"]
        self.model = self.init_model()       # mechine learning model

        # model weight reference
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}     # compressed gradient
        self.dW = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}                # gradient
        # global model before local training
        self.W_old = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}                 # Error feedback

        # hyperparameters
        # local iteration num
        self.epoch_num = client_config["local epoch"]
        self.lr = client_config["optimizer"]["lr"]      # learning rate
        self.momentum = client_config["optimizer"]["momentum"]  # momentum
        self.batch_size = client_config["batch_size"]       # batch size
        self.delay = delay          # simulate network delay

        # dataset
        self.dataset_name = client_config["dataset"]
        # the dataset of client, a list with 2 elements, the first is all data, the second is all label
        self.dataset = dataset
        self.split_train_test(proportion=0.8)
        self.transforms_train, self.transforms_eval = get_default_data_transforms(
            self.dataset_name)
        self.train_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_train, self.y_train, self.transforms_train),
                                                        batch_size=self.batch_size,
                                                        shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval),
                                                       batch_size=self.batch_size,
                                                       shuffle=False)

        self.model_timestamp = 0        # timestamp, to compute staleness for server

        # loss function
        # loss function
        self.loss_fun_name = client_config["loss function"]
        self.loss_function = self.init_loss_fun()

        # optimizer
        self.optimizer_hp = client_config["optimizer"]      # optimizer
        self.optimizer = self.init_optimizer()

        # compressor
        self.compression_config = compression_config

        # training device
        self.device = device            # training device (cpu or gpu)

        # multiple process valuable
        self.selected_event = False     # indicate if the client is selected
    
    def __getstate__(self):
        """return a dict for current status"""
        state = self.__dict__.copy()
        res_keys = ['epoch_num']
        res_state = {}
        for key,value in state.items():
            if key in res_keys:
                res_state[key] = value
        return res_state

    def __setstate__(self, state):
        """return a dict for current status"""
        self.__dict__.update(state)
        return state

    def compress_weight(self, compression_config=None):
        accumulate = compression_config["params"]["error_feedback"]
        if accumulate:
            # compression with error accumulation
            tl.add(target=self.A, source=self.dW)
            tl.compress(target=self.dW_compressed, source=self.A,
                        compress_fun=comp.compression_function(compression_config))
            tl.subtract(target=self.A, source=self.dW_compressed)

        else:
            # compression without error accumulation
            tl.compress(target=self.dW_compressed, source=self.dW,
                        compress_fun=comp.compression_function(compression_config))

    def train_model(self):
        # print("Hello world")
        start_time = time.time()
        self.model.train()
        train_acc = 0.0
        train_loss = 0.0
        train_num = 0
        print(self.epoch_num)
        for epoch in range(self.epoch_num):
            print(epoch)
            try:  # Load new batch of data
                features, labels = next(self.epoch_loader)
            except:  # Next epoch
                self.epoch_loader = iter(self.train_loader)
                features, labels = next(self.epoch_loader)
            features, labels = features.to(self.device), labels.to(self.device)
            # set accumulate gradient to zero
            self.optimizer.zero_grad()
            outputs = self.model.to(self.device)(
                features)                          # predict
            loss = self.loss_function(
                outputs, labels)              # compute loss
            # backward, compute gradient
            loss.backward()
            self.optimizer.step()                                   # update

            train_loss += loss.item()                               # compute total loss
            # get prediction label
            _, prediction = torch.max(outputs.data, 1)
            # compute training accuracy
            train_acc += torch.sum(prediction == labels.data)
            train_num += self.train_loader.batch_size
            # print("******")
        # compute average accuracy and loss
        train_acc = train_acc / train_num
        train_loss = train_loss / train_num
        end_time = time.time()

        print("Client {}, Global Epoch {}, Train Accuracy: {} , Train Loss: {}, Used Time: {},cr: {}\n".format(
            self.cid, self.model_timestamp, train_acc, train_loss, end_time - start_time, self.compression_config["uplink"]["params"]["cr"]))

    def synchronize_with_server(self, server):
        self.model_timestamp = server.current_epoch
        tl.copy_weight(target=self.W, source=server.W)

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

    def init_optimizer(self):
        optimizer_name = self.optimizer_hp["method"]
        if optimizer_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(), self.lr, self.momentum)

    def split_train_test(self, proportion):
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
        # the number of training samples
        train_num = int(proportion * len(self.data))
        self.train_num = train_num
        self.test_num = len(self.data) - train_num
        self.x_train = data[:train_num]              # the data of training set
        self.y_train = label[:train_num]            # the label of training set
        self.x_test = data[train_num:]               # the data of testing set
        self.y_test = label[train_num:]             # the label of testing set

    def receive(self, transmit_dict):
        # set selected event True: the client has been selected
        self.set_selected_event(True)

        # timestamp of global model
        self.model_timestamp = transmit_dict["timestamp"]

        # print("Client {} has been selected in global epoch {}\n".format(self.cid,self.model_timestamp))

        model_weight = transmit_dict["weight"]
        # receive compress model from server and decompress
        decompress_model_weight = self.receiver.receive(model_weight)
        # save decompress model into buffer
        tl.copy_weight(self.W_buffer, decompress_model_weight)
        # print("Client {}'s model has been decompressed in global epoch {}\n".format(self.cid,self.model_timestamp["t"]))

    def send(self, server, transmit_dict):
        server.receive(transmit_dict)

    def get_model_params(self):
        return self.model.state_dict()

    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def set_selected_event(self, bool):
        self.selected_event = bool

    def set_server(self, server):
        self.server = server


def run_client(client):
    while not client.stop_event:     # if the training process is going on
        if client.selected_event:     # if the client is selected by scheduler
            # synchronize
            client.synchronize_with_server(client.server)

            # Training mode
            client.model.train()

            # W_old = W
            tl.copy_weight(client.W_old, client.W)
            # print("Client {}'s model has loaded in global epoch {}\n".format(self.cid,self.model_timestamp["t"]))

            # local training, SGD
            client.train_model()           # local training

            # dW = W - W_old
            # gradient computation
            tl.subtract_(client.dW, client.W, client.W_old)

            # compress gradient
            client.compress_weight(
                compression_config=client.compression_config["uplink"])

            # set transmit dict
            transmit_dict = {}
            transmit_dict["cid"] = client.cid
            # client gradient
            transmit_dict["client_gradient"] = client.dW_compressed
            # number of data samples
            transmit_dict["data_num"] = len(client.x_train)
            transmit_dict["timestamp"] = client.model_timestamp

            # transmit to server (simulate network delay)
            # simulate network delay
            time.sleep(client.delay *
                       client.compression_config["uplink"]["params"]["cr"])
            # send (cid,gradient,weight,timestamp) to server
            client.server.receive(transmit_dict)
            # set selected false, sympolize the client isn't on training
            client.set_selected_event(False)
    print("Client {} Exit.\n".format(client.cid))
