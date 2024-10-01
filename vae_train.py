import os
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from utils.dataload import SingleIsovistsCubicasa, ToTanhRange, ToTensor, RandomRotate, RandomFlip
from tqdm import tqdm
from utils.isoutil import *
from utils.misc import save_params
import json
from PIL import Image
import numpy as np
from vae.isovistvae_mse import VAE, vae_loss

from torch.utils.tensorboard import SummaryWriter

import argparse

cuda = True
device = torch.device("cuda" if cuda else "cpu")
print(device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='vae/conf/vae_1000k.json', type=str)
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
    config_path = join(training_path, 'config.json')
    return training_path, sample_folder, ckpt_folder, log_folder, config_path

def create_folders(cfg):
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    if not os.path.isdir(sample_folder):
        os.makedirs(sample_folder)
    if not os.path.isdir(ckpt_folder):
        os.makedirs(ckpt_folder)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)
    return training_path, sample_folder, ckpt_folder, log_folder, config_path

def get_vae(cfg):
    #set seed for reproducibility
    seed= cfg['seed']
    torch.manual_seed(seed)
    latent_dim = cfg['latent_dim']
    starting_filters = cfg['starting_filters']
    vae = VAE(latent_dim=latent_dim, starting_filters=starting_filters)
    vae = vae.to(device)
    return vae

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def get_dataset(cfg):
    root = cfg['data_root']
    batch_size = cfg['batch_size']
    rotate = cfg['rotate']
    flip = cfg['flip']

    transform = [ToTensor()]
    if rotate:
        transform.append(RandomRotate(256))
    if flip:
        transform.append(RandomFlip())

    transform = transforms.Compose(transform)

    train_dataset = SingleIsovistsCubicasa(root=root, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0)
    train_dl_iter = cycle(train_loader)

    test_dataset = SingleIsovistsCubicasa(root=root, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
    return train_dl_iter, test_loader

def set_constant_sample(cfg, test_loader):
    torch.manual_seed(cfg['seed'])
    sample_num = cfg['sample_num']
    batch_size = cfg['batch_size']
    assert batch_size >= sample_num
    dataiter = iter(test_loader)
    sample, _ = next(dataiter)
    test_sample = sample[:sample_num]
    return test_sample


def generate_and_save_images(model, iter_num, sample, sample_folder):
    model.eval()
    with torch.no_grad():
        sample_ = sample.to(device)
        recons, latent_mu, latent_logvar = model(sample_)
        recons = recons.detach().cpu().numpy()
        sample = sample.detach().cpu().numpy()

        figs = []
        for i, recon in enumerate(recons):
            recon = np.squeeze(recon)
            boundary = np.squeeze(sample[i])
            figs.append(plot_isovist_boundary_numpy(recon, boundary, figsize=(2,2)))
        figs = torch.tensor(figs, dtype=torch.float)
        im = torchvision.utils.make_grid(figs, normalize=True, range=(0, 255), nrow=4)
        im = Image.fromarray(np.uint8(np.transpose(im.numpy(), (1, 2, 0))*255))
        im.save(join(sample_folder, f'vae_recon_{iter_num:07}.jpg'))
    model.train()


def save_test_sample(test_sample, training_path):
    figs = []
    for isovist in test_sample:
        isovist = isovist
        figs.append(plot_isovist_boundary_numpy(np.squeeze(isovist), np.squeeze(isovist), figsize=(2,2)))

    figs = torch.tensor(figs, dtype=torch.float)
    im = torchvision.utils.make_grid(figs, normalize=True, range=(0, 255), nrow=4)
    im = Image.fromarray(np.uint8(np.transpose(im.numpy(), (1, 2, 0))*255))
    im.save(join(training_path, 'sample_dataset.jpg'))
    


def train(vae, cfg):
    lr = cfg['learning_rate']
    beta = cfg['beta']
    iterations = cfg['iterations']
    img_freq = cfg['img_freq']
    save_freq = cfg['save_freq']
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)


    #dataset
    train_dl_iter, test_loader = get_dataset(cfg)
    test_sample=set_constant_sample(cfg, test_loader)
    save_test_sample(test_sample, training_path)

    #logger
    tb_writer = SummaryWriter(log_folder)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    
    for i in range(1, iterations+1):
        x, _ = next(train_dl_iter)
        x = x.to(device)
        # vae reconstruction
        x_recon, latent_mu, latent_logvar = vae(x)
        # reconstruction error
        loss, recons_loss, kl_div = vae_loss(x_recon, x, latent_mu, latent_logvar, beta)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()

        tb_writer.add_scalar('loss', loss.item(), i)
        tb_writer.add_scalar('recons_loss', recons_loss.item(), i)
        tb_writer.add_scalar('kl_div', kl_div.item(), i)

        if not (i % 50):
            print(f'[{i}/{iterations}] loss: {loss.item()} recons_loss: {recons_loss.item()} kl_div: {kl_div.item()}')
        
        if not (i % img_freq):
            generate_and_save_images(vae, i, test_sample, sample_folder)
        
        if not (i % save_freq):
            torch.save(vae.state_dict(),join(ckpt_folder, f'{i:07}_vae.pth'))



if __name__ == '__main__':
    opts = get_args()
    cfg = read_config(opts.config)
    training_path, sample_folder, ckpt_folder, log_folder, config_path = create_folders(cfg)
    save_params(cfg, training_path)
    vae = get_vae(cfg)
    train(vae, cfg)