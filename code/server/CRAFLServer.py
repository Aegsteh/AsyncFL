import torch
from model import get_model

import threading
import queue
import copy
import time
import schedule
import multiprocessing
from multiprocessing import Manager

from compressor.topk import TopkCompressor
from compressor import NoneCompressor

from dataset.CustomerDataset import CustomerDataset
from dataset.utils import get_default_data_transforms

import server.ScheduleClass as sc

import tools.jsonTool as jsonTool
import tools.tensorTool as tl
import tools.resultTools as rt

from client.CRAFLClient import run_client
from ctypes import c_bool

import numpy as np

# load config
mode = 'crafl'
config_file = jsonTool.get_config_file(mode=mode)
config = jsonTool.generate_config(config_file)
global_config = config["global"]


def update_list(lst, num):
    if len(lst) == 0:
        lst.append(num)
    else:
        lst.append(lst[-1] + num)


class CRAFLServer:
    def __init__(self, global_config, dataset, compressor_config, clients, device):
        # mutiple processes valuable
        self.global_stop_event = False

        # global_config
        self.global_config = global_config
        self.schedule_config = global_config["schedule"]

        # device
        self.device = device

        # model
        self.model_name = global_config["model"]
        self.model = get_model(self.model_name).to(
            device)       # mechine learning model
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_cpu = {name: torch.zeros(value.shape).to(
            'cpu') for name, value in self.W.items()}          # used to transmit
        self.dW_compress = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device)
                  for name, value in self.W.items()}

        # dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset

        # global iteration
        self.current_epoch = 0       # indicate the version of global model
        self.total_epoch = global_config["epoch"]
        self.avg_local_iteration = self.global_config["local iteration"]
        self.gamma = self.global_config["gamma"]
        self.p = self.global_config["p"]
        self.tau_threshold = self.global_config["tau_threshold"]

        # loss function
        self.loss_fun_name = global_config["loss function"]
        self.loss_func = self.init_loss_fun()

        self.compressor_config = compressor_config

        # results
        self.staleness_list = []
        self.loss_list = []
        self.accuracy_list = []
        self.gradient_num_list = []
        self.communication_list = []
        self.computation_list = []
        self.time_list = []
        self.start_time = time.time()

        # global manager
        self.global_manager = CRAFLGlobalManager(clients=clients,
                                                 dataset=dataset,
                                                 global_config=global_config,
                                                 stop_event=self.global_stop_event)

    # start the whole training priod
    def start(self, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO):
        print("Start global training...\n")

        # processing pool
        client_pool = multiprocessing.Pool(
            len(self.global_manager.get_clients_list()))

        # load global model to GLOBAL_INFO
        tl.to_cpu(self.W_cpu, self.W)
        for i in range(self.global_manager.clients_num):
            GLOBAL_INFO[i] = {"weight": self.W_cpu, "timestamp": self.current_epoch,
                              "local iteration": self.avg_local_iteration,
                              "cr": self.avg_local_iteration * self.gamma}

        # Start Training
        # start all clients for global training
        self.global_manager.start_clients(
            client_pool, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO)

        # start updater and scheduler
        schedule.every(self.global_config["p"]).seconds.do(
            self.update, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO)
        while self.current_epoch < self.total_epoch:
            schedule.run_pending()

        # stop global training
        STOP_EVENT.value = True
        client_pool.join()

        # Exit
        print("Global Updater Exit.\n")

    def update(self, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO):
        if self.current_epoch >= self.total_epoch:        # if global training is going on
            return

        if not GLOBAL_QUEUE.empty():            # if server has received some gradients from clients
            client_gradients = []           # save multi local_W
            data_nums = []
            stalenesses = []
            gradient_num = 0
            computation_cost = 0
            communication_cost = 0
            while not GLOBAL_QUEUE.empty():
                # get information from client,(cid, client_gradient, data_num, timestamp)
                transmit_dict = GLOBAL_QUEUE.get()
                # cid
                cid = transmit_dict["cid"]
                # client gradient
                client_gradient = transmit_dict["client_gradient"]
                tl.to_gpu(client_gradient, client_gradient)
                # number of data samples
                data_num = transmit_dict["data_num"]
                # timestamp of client gradient
                timestamp = transmit_dict["timestamp"]
                staleness = self.current_epoch - timestamp                  # staleness
                gradient_num += 1

                # computation
                # 1 iteration computation time of cid client
                mu_cid = transmit_dict["mu"]
                # total computation time
                computation_consumption_cid = transmit_dict["computation_consumption"]

                # communication
                # time of transmit full model
                beta_cid = transmit_dict["beta"]
                # communication consumption (MB)
                communication_consumption_cid = transmit_dict["communication_consumption"]

                client_gradients.append(client_gradient)
                data_nums.append(data_num)
                stalenesses.append(staleness)
                self.staleness_list.append(staleness)

                # update computation time and communication time
                self.global_manager.computation_dict[cid].append(mu_cid)
                self.global_manager.communication_dict[cid].append(beta_cid)

                # update computation_consumption and communication_consumption
                communication_cost += communication_consumption_cid
                computation_cost += computation_consumption_cid
                # avg_mu = np.mean(
                #     [np.mean(mu_list) for mu_list in self.global_manager.computation_dict[cid]])
                # avg_beta = np.mean(
                #     [np.mean(beta_list) for beta_list in self.global_manager.communication_dict[cid]])
                # cid_k = int(self.avg_local_iteration * (avg_mu +
                #             self.gamma * avg_beta) / (mu_cid + self.gamma * beta_cid))
                cid_k = int((self.p * self.tau_threshold) / (mu_cid + self.gamma * beta_cid))
                cid_delta = self.gamma * cid_k
                self.global_manager.k_list[cid] = cid_k
                self.global_manager.delta_list[cid] = cid_delta
            
            self.time_list.append(time.time() - self.start_time)
            # update gradient num received
            update_list(self.gradient_num_list, gradient_num)
            update_list(self.computation_list, computation_cost)
            update_list(self.communication_list, communication_cost)
            # self.time_list.append(self.current_epoch * self.p)

            tl.weighted_average(target=self.dW,
                                sources=client_gradients,
                                weights=torch.Tensor(data_nums))             # global gradient
            # update global model
            tl.add(target=self.W, source=self.dW)

            # change global model
            tl.to_cpu(self.W_cpu, self.W)
            for cid in range(self.global_manager.clients_num):
                if self.global_config["adaptive"]:
                    GLOBAL_INFO[cid] = {'weight':self.W_cpu, "timestamp":self.current_epoch,
                                        'local iteration': self.global_manager.k_list[cid],
                                        'cr':self.global_manager.delta_list[cid]}
                else:
                    GLOBAL_INFO[cid] = {'weight':self.W_cpu, "timestamp":self.current_epoch,
                                        'local iteration': self.avg_local_iteration,
                                        'cr':self.avg_local_iteration * self.gamma}

            self.eval_model()

        self.current_epoch += 1
        print("Current Epoch: {}".format(self.current_epoch))

        # schedule
        participating_client_idxs = self.schedule(
            self.global_manager.clients_dict, self.schedule_config, SELECTED_EVENT)
        self.select_clients(participating_client_idxs, SELECTED_EVENT)

        # save result
        self.save_result()

    def save_result(self):
        global_acc, global_loss = self.get_accuracy_and_loss_list()
        staleness_list = self.get_staleness_list()
        dir_name = "{}_{}_{}_{}_{}_CRAFL".format(
                            global_config["model"], global_config["dataset"],
                            self.avg_local_iteration,
                            self.gamma,
                            self.tau_threshold)
        if self.global_config["adaptive"]:
            dir_name = "{}_{}_{}_{}_{}_CRAFL".format(
                            global_config["model"], global_config["dataset"],
                            self.avg_local_iteration,
                            self.gamma,
                            self.tau_threshold)
        else: 
            dir_name = "{}_{}_{}_Period".format(
                            global_config["model"], global_config["dataset"],
                            self.avg_local_iteration)
        rt.save_results(config["result"]["path"],
                        dir_name=dir_name,
                        config=config,
                        global_loss=global_loss,
                        global_acc=global_acc,
                        staleness=staleness_list,
                        gradient_num=self.gradient_num_list,
                        communication_cost=self.communication_list,
                        computation_cost=self.computation_list,
                        time=self.time_list)

    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()

    def schedule(self, clients, schedule_config, SELECTED_EVENT, **kwargs):
        participating_clients = sc.idle_schedule(
            clients, schedule_config, SELECTED_EVENT)
        return participating_clients

    def select_clients(self, participating_clients_idxs, SELECTED_EVENT):
        for idx in participating_clients_idxs:
            SELECTED_EVENT[idx] = True

    def receive(self, transmit_dict):
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
        print("Server: Global Epoch {}, Test Accuracy: {} , Test Loss: {}".format(
            self.current_epoch, accuracy, loss))

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_staleness_list(self):
        return self.staleness_list


class CRAFLGlobalManager:       # Manage clients and global information
    def __init__(self, clients, dataset, global_config, stop_event):
        # clients
        self.clients_num = len(clients)
        self.clients_list = clients
        self.clients_dict = {}
        self.computation_dict = {}
        self.communication_dict = {}
        self.k_list = []
        self.delta_list = []

        # global epoch/iteration
        self.global_epoch = global_config["epoch"]
        self.global_acc = []            # test accuracy
        self.global_loss = []           # training loss
        self.avg_local_iteration = global_config["local iteration"]
        self.avg_delta = global_config["gamma"] * self.avg_local_iteration

        self.register_clients(clients)

        # global test dataset
        self.dataset_name = global_config["dataset"]
        # the test dataset of server, a list with 2 elements, the first is all data, the second is all label
        self.dataset = dataset

        self.x_test = dataset[0]
        self.y_test = dataset[1]
        if type(self.x_test) == torch.Tensor:
            self.x_test, self.y_test = self.x_test.numpy(), self.y_test.numpy()
        elif type(self.y_test) == list:
            self.y_test = np.array(self.y_test)
        # print(self.y_test.shape)
        self.transforms_train, self.transforms_eval = get_default_data_transforms(
            self.dataset_name)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval),
                                                       batch_size=8,
                                                       shuffle=False)

        # multiple process valuable
        self.stop_event = stop_event        # False for initialization

    def find_client_by_cid(self, cid):       # find client by cid
        for client in self.clients:
            if client.cid == cid:
                return client
        return None

    def get_clients_dict(self):
        return self.clients_dict

    def get_clients_list(self):
        return self.clients_list

    def register_clients(self, clients):  # add multi-clients to server scheduler
        for client in clients:
            self.add_client(client)

    def add_client(self, client):        # add one client to server scheduler
        cid = client.cid
        if cid in self.clients_dict.keys():
            raise Exception("Client id conflict.")
        self.clients_dict[cid] = client
        self.communication_dict[cid] = []
        self.computation_dict[cid] = []
        self.k_list.append(self.avg_local_iteration)
        self.delta_list.append(self.avg_delta)

    # start all clients training
    def start_clients(self, client_pool, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO):
        clients_dict = self.get_clients_dict()
        for cid, client_thread in clients_dict.items():
            client_pool.apply_async(run_client, args=(
                client_thread, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO), error_callback=err_call_back)       # add process to process pool
        client_pool.close()
        # client_pool.join()
        print("Start all client-threads\n")

    def stop_clients(self):
        clients_list = self.get_clients_dict()      # get all clients
        for cid, client_thread in clients_list.items():
            client_thread.set_stop_event(
                self.stop_event)               # start all clients


def err_call_back(err):
    print(f'Error: {str(err)}')
