import os
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.autograd import grad
from utils.dataload import SingleIsovistsCubicasa, ToTanhRange, ToTensor, RandomRotate, RandomFlip
from tqdm import tqdm
from utils.isoutil import *
from utils.misc import save_images, save_params
import json
from PIL import Image
import numpy as np
from IPython.display import clear_output

from progan1d.model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter

import argparse


cuda = True
device = torch.device("cuda" if cuda else "cpu")
print(device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='progan1d/progan1d_1200k.json', type=str)
    return parser.parse_args()

def read_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def set_constant_sample(cfg):
    torch.manual_seed(cfg['seed'])
    sample_num = cfg['sample_num']
    latent_dim = cfg['latent_dim']
    fixed_noise = torch.randn(sample_num, latent_dim, device=device)
    return fixed_noise

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

def accumulate(model1, model2, decay=0.995):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def init_GAN(cfg):
    n_channel = cfg['n_channel']
    latent_dim = cfg['latent_dim']
    pixel_norm = cfg['pixel_norm']
    tanh = cfg['tanh']
    kernel_size = cfg.get('kernel_size', 3)
    padding_mode = cfg.get('padding_mode', 'zeros')
    generator = Generator(in_channel=n_channel, input_code_dim=latent_dim, pixel_norm=pixel_norm, tanh=tanh, kernel_size=kernel_size, pad_mode=padding_mode).to(device)
    discriminator = Discriminator(feat_dim=n_channel,  kernel_size=kernel_size, pad_mode=padding_mode).to(device)
    g_running = Generator(in_channel=n_channel, input_code_dim=latent_dim, pixel_norm=pixel_norm, tanh=tanh,  kernel_size=kernel_size, pad_mode=padding_mode).to(device)
    g_running.eval()
    accumulate(g_running, generator, 0)
    return generator, discriminator, g_running

def isovist_loader(cfg):
    def loader(transform):
        root = cfg['data_root']
        num_workers = cfg['num_workers']
        batch_size = cfg['batch_size']
        data = SingleIsovistsCubicasa(root=root, train=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=num_workers)
        return data_loader
    return loader

def sample_data(dataloader, cfg):
    rotate = cfg['rotate']
    flip = cfg['flip']
    transform_list = [ToTensor()]
    if rotate:
        transform_list.append(RandomRotate(256))
    if flip:
        transform_list.append(RandomFlip())
    transform_list.extend([ToTanhRange()])
    transform = transforms.Compose(transform_list)
    loader = dataloader(transform)

    return loader

def cycle(dl):
    while True:
        for data in dl:
            yield data
        

def train(generator, discriminator, g_running, cfg):
    # cfg
    beta1 = cfg['beta1']
    lr = cfg['learning_rate']
    step = cfg['step']
    total_iter = cfg['total_iter']
    n_critic = cfg['n_critic']
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    latent_dim = cfg['latent_dim']
    img_freq = cfg['img_freq']
    save_freq = cfg['save_freq']


    # fixed samples
    fixed_noise = set_constant_sample(cfg)

    # set optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.99))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.99))

    # progoress bar
    total_iter_remain = total_iter - (total_iter//6)*(step-1)
    pbar = tqdm(range(total_iter_remain))

    # logger
    tb_writer = SummaryWriter(log_folder)

    # train init
    alpha = 0
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    iteration = 0

    # dataset
    loader = isovist_loader(cfg)
    data_loader = sample_data(loader, cfg)
    dataset = cycle(data_loader)


    grow_iter_limit = total_iter

    for i in pbar:
        discriminator.zero_grad()
        alpha = min(1, (2/(grow_iter_limit//6)) * iteration)
        if iteration > grow_iter_limit//6:
            alpha = 0
            iteration = 0
            step += 1
            if step > 6:
                alpha = 1
                step = 6
        
        
        real_isovist, label = next(dataset)


        iteration += 1

        if i > grow_iter_limit:
            alpha = 1

        res = 4 * 2**step

         ### 1. train Discriminator
        b_size = real_isovist.size(0)
        real_isovist = real_isovist.to(device)
        real_isovist = F.interpolate(real_isovist, res, mode='linear', align_corners=False)
        real_predict = discriminator(real_isovist, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        # sample input data: vector for Generator
        gen_z = torch.randn(b_size, latent_dim).to(device)
        fake_isovist = generator(gen_z, step=step, alpha=alpha)
        fake_predict = discriminator(fake_isovist.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        ### gradient penalty for D
        eps = torch.rand(b_size, 1, 1).to(device)
        x_hat = eps * real_isovist.data + (1 - eps) * fake_isovist.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss = grad_penalty.item()
        d_loss= (real_predict - fake_predict).item()
        d_optimizer.step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()
            predict = discriminator(fake_isovist, step=step, alpha=alpha)
            loss = -predict.mean()
            # gen_loss_val += loss.item()

            loss.backward()
            g_loss = loss.item()
            g_optimizer.step()
            accumulate(g_running, generator)

        ### Logging
        if (i + 1) % img_freq == 0 or i==total_iter-1:
            with torch.no_grad():
                isovists = g_running(fixed_noise, step=step, alpha=alpha).data.cpu()
                isovists = isovists*0.5+0.5
                save_images(isovists, i+1, 'gen', sample_folder)

        tb_writer.add_scalar('G_loss', g_loss, i)
        tb_writer.add_scalar('D_loss', d_loss, i)
        tb_writer.add_scalar('grad_loss', grad_loss, i)

        # saving models
        if (i+1) % save_freq == 0 or i==total_iter-1:
            try:
                dir_path = join(ckpt_folder, f'{i + 1:07}')
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(g_running.state_dict(),join(dir_path, f'gen_ema.pth'))
                torch.save(discriminator.state_dict(),join(dir_path, f'disc.pth'))
                torch.save(generator.state_dict(),join(dir_path, f'gen.pth'))
            except:
                print('error saving models')
                pass
        

if __name__=='__main__':
    opts = get_args()
    cfg = read_config(opts.config)
    # Working folder
    training_path, sample_folder, ckpt_folder, log_folder, config_path = create_folders(cfg)
    save_params(cfg, training_path)
    generator, discriminator, g_running = init_GAN(cfg)
    train(generator, discriminator, g_running, cfg)
