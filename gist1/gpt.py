import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

import time

import copy

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ScaledDotProductAttention(nn.Module): 
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, block_size, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        

        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)

        # self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # causal mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size))
                                .view(1, 1, block_size, block_size))
    
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        bs, T, C = q.size()

        # perform linear operation and  split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimension of  bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # causal_mask
        mask = self.causal_mask[:,:,:T,:T]

        # calculate attention
        attn = self.attention(q, k, v, mask)

        # concatenate heads and  put trough final linear layer
        concat = attn.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.dropout(self.out(concat))

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # we set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(4 * d_model, d_model)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = new_gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

# the implementation reference https://www.arxiv-vanity.com/papers/1911.03179/
class Block(nn.Module):
    def __init__(self, d_model, heads, block_size, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = CausalMultiHeadAttention(heads, d_model, block_size)
        self.ff = FeedForward(d_model)


    def forward(self, x):
        # normalize
        x2 = self.norm_1(x)
        # compute self attention
        x2 = self.attn(x2, x2, x2)
        # residual
        x = x + x2
        # normalize
        x2= self.norm_2(x)
        # positionwise feed forward network
        x2 = self.ff(x2)
        # residual
        x = x + x2
        return x

# layer multiplier
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module)for i in range(N)])

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, block_size=80, dropout=0.1):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Parameter(torch.zeros(1, block_size, d_model)) 
        self.dropout = nn.Dropout(dropout)
        self.layers = get_clones(Block(d_model, heads, block_size), N)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, src):
        b, t = src.size()
        tok_emb = self.embed(src)
        position_embeddings = self.pe[:, :t, :] 
        x = tok_emb + position_embeddings
        x = self.dropout(x)
        x = self.norm(x)
        for i in range(self.N):
            x = self.layers[i](x)
        x = self.norm(x)
        x = self.out(x)
        return x


class Scheduler(_LRScheduler):
    def __init__(self, optimizer, dim_embed, warmpup_steps, last_epoch=-1, verbose=False):       
        self.dim_embed = dim_embed
        self.warmup_steps = warmpup_steps
        self.num_param_groups = len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        lr = self.dim_embed**(-0.5) * min(self._step_count**(-0.5),self._step_count * self.warmup_steps**(-1.5))
        return [lr] * self.num_param_groups
