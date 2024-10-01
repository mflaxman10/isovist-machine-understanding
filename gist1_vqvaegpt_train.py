import os
import shutil
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from utils.dataload import SequenceSamplerCubicasa, ToTensor
from tqdm import tqdm
from utils.isoutil import *
from utils.misc import save_params, write
import json
from PIL import Image
import numpy as np
# from vae.isovistvae_mse import VAE, vae_loss
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
    parser.add_argument('--config', default='gist1/conf/gist1_gpt_base_500ep.json', type=str)
    return parser.parse_args()

def read_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def get_folders(cfg):
    root = cfg['root']
    folder = cfg['folder']
    training_path = join(root, folder)
    sample_folder = join(training_path, 'samples')
    ckpt_folder = join(training_path, 'checkpoints')
    log_folder = join(training_path, 'logs')
    vqvae_folder = join(training_path, 'vqvae')
    config_path = join(training_path, 'config.json')
    return training_path, config_path, sample_folder, ckpt_folder, log_folder, vqvae_folder

def create_folders(cfg):
    training_path, config_path, *folders = get_folders(cfg)
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    return training_path, config_path, *folders

def copy_vqvae(cfg):
    _, _, _, _, _, vqvae_dir = get_folders(cfg)
    vqvae_ckpt = cfg['vqvae']
    ckpt_name = os.path.basename(vqvae_ckpt)
    model_folder = os.path.abspath(os.path.join(vqvae_ckpt, os.pardir, os.pardir))
    param_path = os.path.join(model_folder, 'param.json')
    ckpt_dest = os.path.join(vqvae_dir, ckpt_name).replace(os.sep, '/')
    param_dest = os.path.join(vqvae_dir, 'param.json').replace(os.sep, '/')
    shutil.copy(vqvae_ckpt, ckpt_dest)
    shutil.copy(param_path, param_dest)
    cfg['vqvae_checkpoint'] = ckpt_dest
    cfg['vqvae_cfg'] = param_dest
    

def get_vqvae_transformer(cfg):
    #set seed for reproducibility
    transformer = VQVAETransformer(cfg)
    transformer.load_vqvae_weight(cfg)
    transformer = transformer.to(device)
    return transformer

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def get_dataset(cfg):
    root = cfg['data_root']
    batch_size = cfg['batch_size']
    seq_num = cfg['seq_num']
    seq_length = cfg['seq_length']
    p = cfg['p']
    q = cfg['q']

    transform = [ToTensor()]
    transform = transforms.Compose(transform)

    train_dataset = SequenceSamplerCubicasa(root=root, seq_length=seq_length, seq_num=seq_num,  p=p, q=q, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0)
    
    # cfg['seq_length'] = train_dataset.get_seq_length()

    return train_loader, train_dataset

def set_constant_sample(cfg, test_loader):
    torch.manual_seed(cfg['seed'])
    sample_num = cfg['sample_num']
    batch_size = cfg['batch_size']
    assert batch_size >= sample_num
    dataiter = iter(test_loader)
    sample, _ = dataiter.next()
    test_sample = sample[:sample_num]
    return test_sample


def split_indices(indices, loc_len=1, isovist_len=32):
    seg_length = loc_len + isovist_len
    batch_size = indices.shape[0]
    splits = indices.reshape(batch_size, -1, seg_length) # BS(L+I)
    ilocs, iisovists = torch.split(splits, [loc_len, isovist_len], dim=2) # BSL , BSI
    return ilocs, iisovists



def generate_and_save_images(model, iter_num, sample_num, sample_folder, cfg):
    model_loc_start_idx = model.vqvae_vocab_size
    block_seq_length = cfg['block_seq_length']
    loc_dim = cfg['loc_dim']
    isovist_latent_dim = cfg['isovist_latent_dim']
    start_indices = torch.ones((sample_num, 1)).long().to("cuda") * model_loc_start_idx
    steps = block_seq_length * (loc_dim + isovist_latent_dim) - loc_dim # loc dim + latent

    sample_indices = model.sample(start_indices, steps=steps, top_k=50)

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

    # samples = sampled_isovists.detach().cpu().numpy()

    figs = []
    for i in range(sample_num):
        loc = locs[i]
        sampled_isovist = sampled_isovists[i]
        sampled_isovist = np.squeeze(sampled_isovist, axis=1)
        # figs.append(plot_isovist_boundary_numpy(sample, sample, figsize=(2,2)))
        figs.append(plot_isovist_sequence_grid(loc, sampled_isovist, figsize=(4, 4), center=True))

    figs = torch.tensor(figs, dtype=torch.float)
    im = torchvision.utils.make_grid(figs, normalize=True, range=(0, 255), nrow=4)
    im = Image.fromarray(np.uint8(np.transpose(im.numpy(), (1, 2, 0))*255))
    im.save(join(sample_folder, f'sample_{iter_num:03}.jpg'))


def save_test_sample(test_sample, training_path):
    figs = []
    for isovist in test_sample:
        isovist = isovist
        figs.append(plot_isovist_boundary_numpy(np.squeeze(isovist), np.squeeze(isovist), figsize=(2,2)))

    figs = torch.tensor(figs, dtype=torch.float)
    im = torchvision.utils.make_grid(figs, normalize=True, range=(0, 255), nrow=4)
    im = Image.fromarray(np.uint8(np.transpose(im.numpy(), (1, 2, 0))*255))
    im.save(join(training_path, 'sample_dataset.jpg'))


def configure_optimizers(model, learning_rate):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add("pe")

    param_dict = {pn: p for pn, p in model.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    return optimizer


def load_checkpoint(transformer, cfg):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is not None and checkpoint is not False:
        if os.path.isfile(checkpoint):
            transformer.load_state_dict(torch.load(checkpoint))
            transformer.to(device)
            print(f"Resume from {checkpoint}")
        else:
            raise Exception("The checkpoint is not found")
    else:
        print("Checkpoint is not defined, train from scratch")

    


def train(transformer, cfg):
    lr = cfg['learning_rate']
    epochs = cfg['epochs']
    print_freq = cfg['print_freq']
    img_freq = cfg['img_freq']
    save_freq = cfg['save_freq']
    block_seq_length = cfg['block_seq_length']
    seq_length = cfg['seq_length']
    training_path, config_path, sample_dir, ckpt_dir, log_dir, vqvae_dir  = get_folders(cfg)


    #dataset
    train_loader, _ = get_dataset(cfg)
    # test_sample=set_constant_sample(cfg, test_loader)
    # save_test_sample(test_sample, training_path)

    # data_variance = np.var(train_dataset.data)

    #logger
    tb_writer = SummaryWriter(log_dir)

    optimizer = configure_optimizers(transformer.transformer, lr)

    transformer.train()

    step = 0

    # indices crop
    gap = seq_length - block_seq_length

    iter_total = len(train_loader)
    for epoch in range(epochs):

        
        for i, batch in enumerate(train_loader):
            locs, isovists = batch
            _, _, s, p = isovists.size()
            locs = locs.view(-1, s)
            isovists = isovists.view(-1 ,s, p)
            # locs = locs.to(device)
            # isovists = isovists.to(device)
            # predict

            #sampling indices
            b, _ = locs.size()
            rand_idx = np.random.randint(gap+1, size=b)
            locs_tmp = []
            isovists_tmp = []
            for j, idx in enumerate(rand_idx):
                locs_tmp.append(locs[j, idx:idx+block_seq_length])
                isovists_tmp.append(isovists[j, idx:idx+block_seq_length, :])
            locs = torch.stack(locs_tmp).to(device)
            isovists = torch.stack(isovists_tmp).to(device)
            indices = transformer.seq_encode(locs, isovists)
            logits , targets = transformer(indices)
            
            # loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            # backpropagation
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            # one step of the optmizer (using the gradients from backpropagation)
            # optimizer.step()
            scaler.step(optimizer)

            

            scaler.update()

            if not ((i+1) % print_freq):
                write(f'[{epoch+1}/{epochs}] : iter [{i+1}/{iter_total}] transformer_loss: {loss.item()}')
                tb_writer.add_scalar('transformer_loss', loss.item(), step)
            step += 1

            
        if not ((epoch+1) % img_freq):
                sample_num = 4
                generate_and_save_images(transformer, epoch+1, sample_num, sample_dir, cfg)
            
        if not ((epoch+1) % save_freq):
            torch.save(transformer.state_dict(),join(ckpt_dir, f'{epoch+1:03}_gist.pth'))

            



if __name__ == '__main__':
    opts = get_args()
    cfg = read_config(opts.config)
    training_path, sample_dir, ckpt_dir, log_dir, config_path, vqvae_dir = create_folders(cfg)
    copy_vqvae(cfg)
    save_params(cfg, training_path)
    transformer = get_vqvae_transformer(cfg)
    load_checkpoint(transformer, cfg)
    train(transformer, cfg)