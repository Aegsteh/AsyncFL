from abc import ABC,abstractmethod
import numpy as np
import random

def schedule(clients,schedule_config):
    participating_clients = []
    if schedule_config["method"] == "idle":
        participating_clients = idle_schedule(clients,schedule_config)
    elif schedule_config["method"] == "random":
        participating_clients = random_schedule(clients,schedule_config)
    return participating_clients

def idle_schedule(clients,schedule_config,SELECTED_EVENT):
    '''
    select clients which is being idle time
    '''
    participating_client_idxs = []
    for cid,client in clients.items():
        if not SELECTED_EVENT[cid]:
            participating_client_idxs.append(cid)
    return participating_client_idxs

def random_schedule(clients,schedule_config):
    '''
    select clients randomly
    '''
    p = schedule_config["params"]["proportion"]     # the proportion of participating clients
    participating_clients = random.sample(clients,int(p * len(clients)))
    return participating_clients

class ScheduleClass(ABC):
    def __init__(self,schedule_config):
        # self.clients = clients
        self.schedule_config = schedule_config
    
    @abstractmethod
    def schedule(self):
        raise NotImplemented("Schedule method is not implemented.")

class RandomScheduleClass(ScheduleClass):
    def __init__(self, schedule_config):
        super().__init__(schedule_config)
    
    def schedule(self,clients):
        proportion = self.schedule_config["params"]["proportion"]
        scheduled_clients = random.sample(clients,proportion * len(clients))
        return scheduled_clients

class AsyncSingleScheduleClass(ScheduleClass):
    def __init__(self, schedule_config):
        super().__init__(schedule_config)
    
    def schedule(self,clients):
        selected_clients = []
        for cid,client in clients.items():      # clients is a client dict
            client.client_lock.acquire()        # visit critical variable "selected_event"
            if not client.selected_event.is_set():
                selected_clients.append(client)
            client.client_lock.release()
        return selected_clients