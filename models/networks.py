import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, channel_sequence):
        super().__init__()
        nb_layers = len(channel_sequence) - 1
        self.seq = nn.Sequential()
        for i in range(nb_layers):
            self.seq.add_module(f"fc{i}", nn.Linear(channel_sequence[i], channel_sequence[i + 1]))
            if i != nb_layers - 1:
                self.seq.add_module(f"ReLU{i}", nn.ReLU(inplace=True))
        
    def forward(self, x):
        out = self.seq(x)
        return out


