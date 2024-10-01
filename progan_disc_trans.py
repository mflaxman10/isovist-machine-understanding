import os
from os.path import join


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.autograd import grad
from utils.dataload import SingleIsovistsCubicasa
from tqdm import tqdm
from utils.isoutil import *
from utils.misc import save_params, save_images
import json
from PIL import Image
import numpy as np
from IPython.display import clear_output

from progan1d.model import Generator, Discriminator, Encoder
from aae.isovistaae_upsample import Encoder as AAEncoder


from torch.utils.tensorboard import SummaryWriter

import argparse


cuda = True
device = torch.device("cuda" if cuda else "cpu")
print(device)


class Perceptual(nn.Module):
    def __init__(self, encoder, requires_grad=False):
        super(Perceptual, self).__init__()
        encoder_conv_stack = encoder.conv_stack
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), encoder_conv_stack[x])
        for x in range(3,6):
            self.slice2.add_module(str(x), encoder_conv_stack[x])
        for x in range(6,9):
            self.slice3.add_module(str(x), encoder_conv_stack[x])
        for x in range(9,15):
            self.slice4.add_module(str(x), encoder_conv_stack[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu1_2 = h
        h = self.slice3(h)
        h_relu1_3 = h
        h = self.slice4(h)
        h_relu1_4 = h
        return h_relu1_1, h_relu1_2, h_relu1_3, h_relu1_4

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='progan1d/conf/disc_trans.json', type=str)
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


def get_perceptual_model(cfg):
    perceptual_model = os.path.abspath(cfg['perceptual_model'])
    model_folder = os.path.abspath(os.path.join(perceptual_model, os.pardir, os.pardir))
    print(perceptual_model)
    print(model_folder)
    encoder_param = read_config(os.path.join(model_folder, 'param.json'))
    latent_dim = encoder_param['latent_dim']
    starting_filters = encoder_param['starting_filters']
    encoder = AAEncoder(latent_dim=latent_dim, starting_filters=starting_filters).to(device)
    encoder.load_state_dict(torch.load(perceptual_model,map_location=device))
    encoder.eval()
    perceptual = Perceptual(encoder).to(device)
    perceptual.eval()
    return perceptual

def get_generator_encoder(cfg):
    model_checkpoint = os.path.abspath(cfg['progan_model'])
    disc_model = os.path.join(model_checkpoint, 'disc.pth')
    gen_model = os.path.join(model_checkpoint, 'gen_ema.pth')
    model_folder = os.path.abspath(os.path.join(model_checkpoint, os.pardir, os.pardir))
    progan_param = read_config(os.path.join(model_folder, 'param.json'))
    n_channel = progan_param['n_channel']
    latent_dim = progan_param['latent_dim']
    pixel_norm = progan_param['pixel_norm']
    tanh = progan_param['tanh']
    netG = Generator(in_channel=n_channel, input_code_dim=latent_dim, pixel_norm=pixel_norm, tanh=tanh).to(device)
    netG.load_state_dict(torch.load(gen_model,map_location=device))
    netD = Discriminator(feat_dim=n_channel).to(device)
    netD.load_state_dict(torch.load(disc_model,map_location=device))
    netE = Encoder(feat_dim=n_channel, latent_dim=latent_dim).to(device)
    toggle_grad(netD,False)
    toggle_grad(netE,False)
    paraDict = dict(netD.named_parameters()) # pre_model weight dict
    for i,j in netE.named_parameters():
        if i in paraDict.keys():
            w = paraDict[i]
            j.copy_(w)
    toggle_grad(netE,True)
    del netD
    return netG, netE, latent_dim


def loss_function(syn_isovist, isovist, perceptual):
    MSE_loss = nn.MSELoss(reduction="mean")
    r0, r1, r2, r3 = perceptual(isovist*0.5+0.5)
    syn0, syn1, syn2, syn3 = perceptual(syn_isovist*0.5+0.5)

    per_loss = 0
    per_loss += MSE_loss(syn0,r0)
    per_loss += MSE_loss(syn1,r1)
    per_loss += MSE_loss(syn2,r2)
    per_loss += MSE_loss(syn3,r3)
    return per_loss/4.0

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def train(perceptual, netG, netE, latent_dim, cfg):
    lr = cfg['learning_rate']
    beta1 = cfg['beta1']
    beta2 = cfg['beta2']
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    epochs = cfg['epochs']
    save_freq = cfg['save_freq']
    batch_size = cfg['batch_size']
    per_const = cfg['per_const']
    l2_const = cfg['l2_const']
    l2z_const = cfg['l2z_const']
    optimizer = torch.optim.Adam(netE.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-8)

    loss_l2 = nn.MSELoss()


    tb_writer = SummaryWriter(log_folder)

    iteration = 0
    for epoch in range(epochs):
        losses_per_avg = 0
        losses_l2_avg = 0
        loss_l2z_avg = 0

        for i in tqdm(range(5000)):
            z = torch.randn(batch_size, latent_dim).to(device)
            with torch.no_grad():
                x = netG(z, step=6, alpha=1)
            z_ = netE(x.detach(), step=6, alpha=1)
            x_ = netG(z_, step=6, alpha=1)
            optimizer.zero_grad()
            # loss = loss_l1(x, x_) + 0.01*loss_l2(z, z_)
            per_loss = loss_function(x_, x, perceptual)
            l2_loss = loss_l2(x, x_)
            l2z_loss = loss_l2(z, z_)
            loss = per_const*per_loss + l2_const*l2_loss + l2z_const*l2z_loss
            loss.backward()
            optimizer.step()
            losses_per_avg += per_loss.detach().item()
            losses_l2_avg += l2_loss.detach().item()
            loss_l2z_avg += l2z_loss.detach().item()
            # losses_per.append(per_loss.item())
            # losses_l2.append(l2_loss.item())
            # losses_l2z.append(l2z_loss)
            tb_writer.add_scalar('perceptual_loss', per_loss.detach().item(), iteration)
            tb_writer.add_scalar('l2_loss', l2_loss.detach().item(), iteration)
            tb_writer.add_scalar('z_l2_loss', l2z_loss.detach().item(), iteration)
            iteration += 1
        # Output training stats
        losses_per_avg /= (i+1)
        losses_l2_avg /= (i+1)
        loss_l2z_avg /= (i+1)
        print(f'[{epoch+1}/{epochs}]\tlosses_per_avg: {losses_per_avg:.4f}\tlosses_l2_avg: {losses_l2_avg:.4f}\tloss_l2z_avg {loss_l2z_avg:.4f}')
        x = x.detach().cpu().numpy()*0.5+0.5
        x_ = x_.detach().cpu().numpy()*0.5+0.5
        save_images(x, iteration, 'syn', sample_folder)
        save_images(x_, iteration, 'rec', sample_folder)
        if (epoch+1)%save_freq == 0:
            torch.save(netE.state_dict(),join(ckpt_folder, f'{iteration:06}_encoder.pth'))



if __name__ == '__main__':
    opts = get_args()
    cfg = read_config(opts.config)
    perceptual = get_perceptual_model(cfg)
    netG, netE, latent_dim = get_generator_encoder(cfg)
    training_path, sample_folder, ckpt_folder, log_folder, config_path = create_folders(cfg)
    save_params(cfg, training_path)
    train(perceptual, netG, netE, latent_dim, cfg)
