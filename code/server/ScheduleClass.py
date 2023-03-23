from abc import ABC,abstractmethod
import numpy as np
import random
def find_ScheduleClass(method):
    if method == "random":
        return AsyncSingleScheduleClass

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