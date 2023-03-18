from compressor import Compressor
import torch

def sparsify(tensor, compression_rate):
    tensor = tensor.flatten()   # flatten the tensor
    k = max(int(tensor.numel() * compression_rate),1) # compute k, elements greater than k-th ele will be saved
    if tensor.device.type == "mps":
        values, indices = torch.topk(tensor.cpu().abs(),k,sorted=False)
        values = values.to("mps:0")
        indices = indices.to("mps:0")
    else:
        values, indices = torch.topk(tensor.abs(),k,sorted=False)   # get topk elements' values and indices
    values = torch.gather(tensor, 0, indices)       # get all topk values
    return values, indices

def desparsify(values_and_indices,num):
    values,indices = values_and_indices
    tensor_decompress = torch.zeros(num,dtype=values.dtype,layout=values.layout,device=values.device)  # get a flatten tensor
    tensor_decompress.scatter_(0,indices,values)
    return tensor_decompress


class TopkCompressor(Compressor):
    def __init__(self, compression_rate):
        super().__init__()
        self.compression_rate = compression_rate
    
    def compress(self, tensor, name):
        values_and_indices = sparsify(tensor, self.compression_rate)        # get compressed values and indices
        num_and_size = tensor.numel(),tensor.size()
        return values_and_indices, num_and_size
    
    def decompress(self, values_and_indices, num_and_size):
        num,size = num_and_size
        tensor_decompress = desparsify(values_and_indices,num)
        tensor_decompress = tensor_decompress.view(size)
        return tensor_decompress