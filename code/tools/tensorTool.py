import torch

def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

def copy_weight(target, source):
  for name in target:
    target[name].data = source[name].data.clone()

def average(target, sources):
  for name in target:
    target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def add(target, source):
  for name in target:
    target[name].data += source[name].data.clone()