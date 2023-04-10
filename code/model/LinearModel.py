import torch.nn as nn
class logistic(nn.Module):
    def __init__(self, in_size=32 * 32 * 1, num_classes=10):
        super(logistic, self).__init__()
        self.linear = nn.Linear(in_size, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear(out)
        return out