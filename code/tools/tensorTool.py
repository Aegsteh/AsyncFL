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

def weighted_average(target, sources, weights):
  for name in target:
    summ = torch.sum(weights)
    n = len(sources)
    modify = [weight/summ*n for weight in weights]
    target[name].data = torch.mean(torch.stack([m*source[name].data for source, m in zip(sources, modify)]), dim=0).clone()

def compress(target, source, compress_fun):
  '''
  compress_fun : a function f : tensor (shape) -> tensor (shape)
  '''
  for name in target:
    target[name].data = compress_fun(source[name].data.clone())

def subtract(target, source):
  for name in target:
    target[name].data -= source[name].data.clone()

def to_cpu(target, source):
  for name in target:
    target[name] = source[name].detach().cpu().clone()

def to_gpu(target, source):
  for name in target:
    target[name] = source[name].cuda().clone()