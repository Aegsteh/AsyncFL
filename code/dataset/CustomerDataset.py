from torch.utils.data.dataset import Dataset
import torch

class CustomerDataset(Dataset):
    '''
    A custom Dataset class for client
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms
    
    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)
    
    def __len__(self):
        return self.inputs.shape[0]