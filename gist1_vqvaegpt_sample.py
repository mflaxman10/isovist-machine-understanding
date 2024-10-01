import os
from os.path import join
import torch
from utils.isoutil import *
from utils.misc import load_params, save_params
import json
from PIL import Image
import numpy as np
from numpy.random import default_rng
from gist1.vqvae_gpt import VQVAETransformer

from torch.utils.tensorboard import SummaryWriter

import argparse

import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

scaler = torch.cuda.amp.GradScaler()

cuda = True
device = torch.device("cuda" if cuda else "cpu")
print(device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='vae_conf/dev.json', type=str)
    return parser.parse_args()

def read_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def get_folders(cfg):
    root = cfg['root']
    folder = cfg['folder']
    training_path = join(root, folder)
    return training_path

def create_folders(cfg):
    training_path = get_folders(cfg)
    if not os.path.isdir(training_path):
        os.makedirs(training_path)
    return training_path


def get_vqvae_transformer(cfg_init):
    TransformerPath = cfg_init['checkpoint']
    model_folder = os.path.abspath(os.path.join(TransformerPath, os.pardir, os.pardir))
    cfg = load_params(os.path.join(model_folder, 'param.json'))

    transformer = VQVAETransformer(cfg)
    transformer.load_state_dict(torch.load(TransformerPath))
    transformer = transformer.to(device)
    transformer.eval()
    print(f"Load  {TransformerPath}")
    return transformer, cfg


def split_indices(indices, loc_len=1, isovist_len=32):
    seg_length = loc_len + isovist_len
    batch_size = indices.shape[0]
    splits = indices.reshape(batch_size, -1, seg_length) # BS(L+I)
    ilocs, iisovists = torch.split(splits, [loc_len, isovist_len], dim=2) # BSL , BSI
    return ilocs, iisovists


def generate_and_save_images(model, sample_folder, cfg, sample, seq_length=8, top_k=50, figsize=8, lim=4, alpha=0.015, seed=None):
    model_loc_start_idx = model.vqvae_vocab_size
    loc_dim = cfg['loc_dim']
    isovist_latent_dim = cfg['isovist_latent_dim']
    sample_num = 1
    start_indices = torch.ones((sample_num, 1)).long().to("cuda") * model_loc_start_idx
    steps = seq_length * (loc_dim + isovist_latent_dim) - loc_dim # loc dim + latent
    
    rng = default_rng(seed)
    seeds = rng.choice(999999, size=sample, replace=False)
    print(seeds)
    for image_num, seed in enumerate(seeds):
        seed = int(seed)
        print(image_num, seed)
        sample_indices = model.sample(start_indices, steps=steps, top_k=top_k, seed=seed)

        ilocs, iisovists = split_indices(sample_indices, loc_len=loc_dim, isovist_len=isovist_latent_dim)


        locs = []
        sampled_isovists = []
        for i in range(iisovists.shape[1]): 
            iloc = ilocs[:, i, :]
            locs.append(model.indices_to_loc(iloc).detach().cpu().numpy()) # S X BL
            iisovist = iisovists[:, i, :] # BI
            sampled_isovists.append(model.z_to_isovist(iisovist).detach().cpu().numpy()) # S X BCW

        locs = np.stack(locs, axis=1)
        sampled_isovists = np.stack(sampled_isovists, axis=1) #BSCW



        figs = []
        for i in range(sample_num):
            loc = locs[i]
            sampled_isovist = sampled_isovists[i]
            sampled_isovist = np.squeeze(sampled_isovist, axis=1)
            figs.append(plot_isovist_sequence_grid(loc, sampled_isovist, figsize=(figsize, figsize), center=True, lim=lim, alpha=alpha))

        im = Image.fromarray(figs[0].transpose((1, 2, 0)))
        im.save(join(sample_folder, f'sample_{seed}_{top_k}_{seq_length}.jpg'))

           

if __name__ == '__main__':
    opts = get_args()
    cfg_init = read_config(opts.config)
    training_path  = create_folders(cfg_init)
    sample_dir = training_path
    save_params(cfg_init, training_path)
    transformer, cfg = get_vqvae_transformer(cfg_init)
    sample_num = cfg_init["sample_num"]
    seq_length = cfg_init["seq_length"]
    figsize = cfg_init["figsize"]
    topk = cfg_init["topk"]
    lim = cfg_init["lim"]
    alpha = cfg_init["alpha"]
    seed = cfg_init["seed"]
    generate_and_save_images(transformer, sample_dir, cfg, sample_num, seq_length=seq_length, top_k=topk, figsize=figsize, lim=lim, alpha=alpha, seed=seed)