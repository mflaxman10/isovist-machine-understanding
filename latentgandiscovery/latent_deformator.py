import torch
import torch.nn.functional as F
from torch import nn
import numpy as np



class LatentDeformator(nn.Module):
    def __init__(self, shift_dim, input_dim=None, out_dim=None, random_init=False, bias=True):
        super(LatentDeformator, self).__init__()
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim) #number of direction
        self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

        self.linear = nn.Linear(self.input_dim, self.out_dim, bias=bias)
        self.linear.weight.data = torch.zeros_like(self.linear.weight.data)
        if random_init:
            self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

    def forward(self, input):
        input = input.view([-1, self.input_dim]) # input dim is number of directions
        input_norm = torch.norm(input, dim=1, keepdim=True)
        out = self.linear(input)
        out = (input_norm/ torch.norm(out, dim=1, keepdim=True)) * out

        out = out.view([-1, self.shift_dim]) # gan latent space dim
        return out

