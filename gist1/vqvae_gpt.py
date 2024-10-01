import torch
import torch.nn as nn
import torch.nn.functional as F
from gist1.gpt import GPT
from gist1.vqvae import VQVAE

from utils.misc import save_params, load_params
import os
import time


class VQVAETransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vqvae = self.load_vqvae(args)
        self.transformer = self.load_transformer(args)
        # self.sos_token = self.get_sos_token(args)
        self.pkeep = args['pkeep']
        self.vqvae_vocab_size = args['vocab_size']
        self.loc_vocab_size = args['loc_vocab_size']
        self.block_size = args['block_size']

    def load_vqvae(self, args):
        # VQVAE_path = args['vqvae_checkpoint']
        VQVAE_cfg = args['vqvae_cfg']
        cfg = load_params(VQVAE_cfg)
        seed= cfg['seed']
        torch.manual_seed(seed)
        num_hiddens = cfg['num_hiddens']
        num_residual_layers = cfg['num_residual_layers']
        num_residual_hiddens = cfg['num_residual_hiddens']
        num_embeddings = cfg['num_embeddings']
        latent_dim = cfg['latent_dim']
        commitment_cost = cfg['commitment_cost']
        decay = cfg['decay']
        model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                num_embeddings, latent_dim, commitment_cost,
                decay)
        # model.load_state_dict(torch.load(VQVAE_path))
        # model = model.eval()
        
        # update args from vqvae cfg
        args['vocab_size'] = num_embeddings

        return model

    def load_vqvae_weight(self, args):
        VQVAE_path = args['vqvae_checkpoint']
        self.vqvae.load_state_dict(torch.load(VQVAE_path))
        self.vqvae.eval()

    def load_transformer(self, args):
        seed= args['seed']
        torch.manual_seed(seed)
        latent_dim = args['latent_dim']
        heads = args['heads']
        N = args['N']
        block_size = args['block_size']
        vocab_size = args['vocab_size'] + args['loc_vocab_size']
        model = GPT(vocab_size, latent_dim, N, heads, block_size)
        return model
    
    @torch.no_grad()
    def encode_to_z(self, x):
        quantized, indices = self.vqvae.encode(x)
        indices = indices.view(quantized.shape[0], -1)
        return quantized, indices

    @ torch.no_grad()
    def z_to_isovist(self, indices):
        indices[indices > self.vqvae_vocab_size-1] = self.vqvae_vocab_size-1
        embedding_dim = self.vqvae.vq.embedding_dim
        ix_to_vectors = self.vqvae.vq.embedding(indices).reshape(indices.shape[0], -1, embedding_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 2, 1)
        isovist = self.vqvae.decode(ix_to_vectors)
        return isovist
    
    def loc_to_indices(self, x):
        starting_index = self.vqvae_vocab_size
        indices = x.long() + starting_index
        return indices
    
    def indices_to_loc(self, indices):
        starting_index = self.vqvae_vocab_size
        locs = indices - starting_index
        locs[locs < 0] = 0
        locs[locs > (self.loc_vocab_size-1)] = self.loc_vocab_size-1
        return locs
    
    def seq_encode(self, locs, isovists):
        # BSW
        indices_seq = []
        # indices_loc = []
        for i in range(isovists.shape[1]): # iterate trought the sequence
            loc = locs[:, i].unsqueeze(1) # BL
            indices_seq.append(self.loc_to_indices(loc))
            isovist = isovists[:, i, :].unsqueeze(1) # BCW
            _, indices = self.encode_to_z(isovist)
            indices_seq.append(indices)
        indices = torch.cat(indices_seq, dim=1)
        return indices


    def forward(self, indices):
        device = indices.device
        # indices = self.seq_encode(locs, isovists)


        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(indices.shape, device=device))
            mask = mask.round().to(dtype=torch.int64)
            random_indices = torch.randint_like(indices,  self.vqvae_vocab_size) # doesn't include sos token
            new_indices = mask*indices + (1-mask)*random_indices
        else:
            new_indices = indices


        target = indices[:, 1:]


        logits =  self.transformer(new_indices[:, :-1])


        return logits, target

    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out


   
    def sample(self, x, steps, temp=1.0, top_k=100, seed=None, step_size=17, zeroing=False):
        is_train = False
        if self.transformer.training == True:
            is_train = True    
        self.transformer.eval()
        block_size = self.block_size
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        for k in range(steps):
            if x.size(1) < block_size:
                x_cond = x
            else:
                remain = step_size - (x.size(1) % step_size)
                x_cond = x[:, -(block_size-remain):]  # crop context if needed
                if zeroing:
                    x_cond = x_cond.clone()
                    x_cond[:, 0] = self.vqvae_vocab_size
            logits = self.transformer(x_cond)
            logits = logits[:, -1, :] / temp

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim = -1)

            ix = torch.multinomial(probs, num_samples=1, generator=generator)

            x = torch.cat((x, ix), dim=1)

        if is_train == True:
            self.transformer.train()

        return x
    

    def get_loc(self, ploc, dir):
        if dir == 0:
            loc = ploc
        elif dir == 1:
            loc = (ploc[0]+1, ploc[1])
        elif dir == 2:
            loc = (ploc[0]+1, ploc[1]+1)
        elif dir == 3:
            loc = (ploc[0], ploc[1]+1)
        elif dir == 4:
            loc = (ploc[0]-1, ploc[1]+1)
        elif dir == 5:
            loc = (ploc[0]-1, ploc[1])
        elif dir == 6:
            loc = (ploc[0]-1, ploc[1]-1)
        elif dir == 7:
            loc = (ploc[0], ploc[1]-1)
        elif dir == 8:
            loc = (ploc[0]+1, ploc[1]-1)
        else:
            raise NameError('Direction unknown')
        return loc
            
    
    def init_loc(self, x, step_size):
        device = x.device
        loc_dict = {}
        loc = None
        cached_loc = None
        if x.shape[1] > 1:
            steps = x.shape[1] -1
            for k in range(steps):
                if k % step_size == 0:
                    dir = x[:,k].detach().item() - self.vqvae_vocab_size
                    if dir == 0:
                        loc = (0, 0) # init loc
                    else:
                        loc = self.get_loc(loc, dir) # getloc
                    loc_dict[loc] = torch.empty(1,0).long().to(device)
                    cached_loc = loc
                else:
                    ix = x[:,[k]]
                    loc_dict[cached_loc]  = torch.cat((loc_dict[cached_loc], ix), dim = 1)
        # print(loc_dict)
        return loc_dict, loc
    
    def sample_memorized(self, x, steps, temp=1.0, top_k=100, seed=None, step_size=17, zeroing=False):
        device = x.device
        loc_dict, loc = self.init_loc(x, step_size)
        is_train = False
        if self.transformer.training == True:
            is_train = True    
        self.transformer.eval()
        block_size = self.block_size
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        is_visited = False
        cache_counter = 0
        # loc = None
        for k in range(steps):
            # check directionality
            if k % step_size == 0:
                dir = x[:,-1].detach().item() - self.vqvae_vocab_size
                if dir == 0:
                    is_visited = False
                    loc = (0, 0) # init loc
                    loc_dict[loc] = torch.empty(1,0).long().to(device)
                else:
                    loc = self.get_loc(loc, dir) # getloc
                    if loc in loc_dict:
                        is_visited = True
                        cache_counter = 0
                    else:
                        is_visited = False
                        loc_dict[loc] = torch.empty(1,0).long().to(device)


            if x.size(1) < block_size:
                x_cond = x
            else:
                remain = step_size - (x.size(1) % step_size)
                x_cond = x[:, -(block_size-remain):]  # crop context if needed
                if zeroing:
                    x_cond = x_cond.clone()
                    x_cond[:, 0] = self.vqvae_vocab_size

            if is_visited == False:
                logits = self.transformer(x_cond)
                logits = logits[:, -1, :] / temp

                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)

                probs = F.softmax(logits, dim = -1)
                ix = torch.multinomial(probs, num_samples=1, generator=generator)
                # print('this shouldnt')
                loc_dict[loc] = torch.cat((loc_dict[loc], ix), dim = 1)
            else:
                if cache_counter == 31: #reaching end of latent code
                    is_visited = False 
                ix = loc_dict[loc][:,[cache_counter]]
                # print(ix)
                cache_counter += 1

            x = torch.cat((x, ix), dim=1)


        if is_train == True:
            self.transformer.train()

        return x


    

    