import os
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from utils.dataload import SingleIsovistsCubicasa, ToTensor
from tqdm import tqdm
from utils.isoutil import *
from utils.misc import save_params
import json
from PIL import Image
import numpy as np
from aae.isovistaae_upsample import Decoder, Encoder, Discriminator
import itertools

from torch.utils.tensorboard import SummaryWriter


import argparse


cuda = True
device = torch.device("cuda" if cuda else "cpu")
print(device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='aae/conf/aae_50epoch.json', type=str)
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

def init_models(cfg):
    #set seed for reproducibility
    seed= cfg['seed']
    torch.manual_seed(seed)
    latent_dim = cfg['latent_dim']
    starting_filters = cfg['starting_filters']
    disc_feats = cfg['disc_feats']
    encoder = Encoder(latent_dim=latent_dim, starting_filters=starting_filters)
    encoder = encoder.to(device)
    decoder = Decoder(latent_dim=latent_dim, starting_filters=starting_filters)
    decoder = decoder.to(device)
    discriminator = Discriminator(latent_dim=latent_dim, starting_nets=disc_feats)
    discriminator.to(device)
    return encoder, decoder, discriminator

def load_dataset(cfg):
    data_root = cfg['data_root']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    transform = transforms.Compose([ToTensor()])
    train_dataset = SingleIsovistsCubicasa(root=data_root, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)
    test_dataset = SingleIsovistsCubicasa(root=data_root, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
    return train_loader, test_loader

def set_constant_sample(cfg, test_loader):
    torch.manual_seed(cfg['seed'])
    sample_num = cfg['sample_num']
    batch_size = cfg['batch_size']
    assert batch_size >= sample_num
    dataiter = iter(test_loader)
    sample, _ = dataiter.next()
    test_sample = sample[:sample_num]
    return test_sample

def generate_and_save_images(encoder, decoder, epoch, sample, sample_folder):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        sample_ = sample.to(device)
        z_ = encoder(sample_)
        recons = decoder(z_)
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
        im.save(join(sample_folder, f'aae_recon_{epoch+1:05}.jpg'))
    encoder.train()
    decoder.train()

def save_test_sample(test_sample, training_path):
    figs = []
    for isovist in test_sample:
        isovist = isovist
        figs.append(plot_isovist_boundary_numpy(np.squeeze(isovist), np.squeeze(isovist), figsize=(2,2)))

    figs = torch.tensor(figs, dtype=torch.float)
    im = torchvision.utils.make_grid(figs, normalize=True, range=(0, 255), nrow=4)
    im = Image.fromarray(np.uint8(np.transpose(im.numpy(), (1, 2, 0))*255))
    im.save(join(training_path, 'sample_dataset.jpg'))

def train(encoder, decoder, discriminator, cfg):
    lr = cfg['learning_rate']
    beta1 = cfg['beta1']
    beta2 = cfg['beta2']
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    epochs = cfg['epochs']
    save_freq = cfg['save_freq']
    latent_dim = cfg['latent_dim']

    encoder.train()
    decoder.train()
    discriminator.train()
    # loss
    adversarial_loss = nn.BCELoss().to(device)
    reconstruction_loss = nn.BCELoss().to(device)

    # optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # load_dataset
    train_loader, test_loader = load_dataset(cfg)
    test_sample=set_constant_sample(cfg, test_loader)
    save_test_sample(test_sample, training_path)

    # logger
    tb_writer = SummaryWriter(log_folder)

    step = 0
    for epoch in range(epochs):
        for x, _ in tqdm(train_loader):
            x = x.to(device)

            valid = torch.ones((x.size()[0], 1), device=device)
            fake = torch.zeros((x.size()[0], 1), device=device)

            # 1) reconstruction + generator loss
            optimizer_G.zero_grad()
            fake_z = encoder(x)
            decoded_x = decoder(fake_z)
            validity_fake_z = discriminator(fake_z)
            adv_loss = adversarial_loss(validity_fake_z, valid)
            recon_loss = reconstruction_loss(decoded_x, x)
            G_loss = 0.001*adv_loss + 0.999*recon_loss
            G_loss.backward()
            optimizer_G.step()

            # 2) discriminator loss
            optimizer_D.zero_grad()
            real_z = torch.randn(x.size()[0], latent_dim, device=device)
            real_loss = adversarial_loss(discriminator(real_z), valid)
            fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
            D_loss = 0.5*(real_loss + fake_loss)
            D_loss.backward()
            optimizer_D.step()

            tb_writer.add_scalar('adversarial_loss', adv_loss.item(), step)
            tb_writer.add_scalar('recon_loss', recon_loss.item(), step)
            tb_writer.add_scalar('disc_loss', D_loss.item(), step)

            step += 1

        print(f'Epoch [{epoch+1} / {epochs}] d_loss: {D_loss.item()}, adv_loss: {adv_loss.item()}, recon_loss: {recon_loss.item()}')
        generate_and_save_images(encoder, decoder, epoch, test_sample, sample_folder)

        if (epoch+1)%save_freq == 0:
            torch.save(encoder.state_dict(),join(ckpt_folder, f'{epoch+1:05}_encoder.pth'))
            torch.save(decoder.state_dict(),join(ckpt_folder, f'{epoch+1:05}_decoder.pth'))
            torch.save(discriminator.state_dict(),join(ckpt_folder, f'{epoch+1:05}_disc.pth'))


if __name__=='__main__':
    opts = get_args()
    cfg = read_config(opts.config)
    # Working folder
    training_path, sample_folder, ckpt_folder, log_folder, config_path = create_folders(cfg)
    save_params(cfg, training_path)
    encoder, decoder, discriminator = init_models(cfg)
    train(encoder, decoder, discriminator, cfg)
