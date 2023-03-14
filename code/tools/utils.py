import os,sys
import torch

def get_gpu_name():
    platform = sys.platform
    if platform == 'darwin':
        return 'mps'
    else:
        return 'cuda'

def get_device(device="gpu"):
    if device == "gpu":
        gpu_name = get_gpu_name()
        return torch.device(gpu_name)
    else:
        return torch.device('cpu')