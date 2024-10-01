import os
from os.path import join

import torch
import torch.nn as nn


from tqdm import tqdm
from utils.isoutil import *
from utils.misc import save_images, save_params, MeanTracker
import json
from PIL import Image
import numpy as np
from IPython.display import clear_output

from progan1d.model import Generator
from latentgandiscovery.latent_deformator import LatentDeformator
from latentgandiscovery.latentshiftpredictor import LatentShiftPredictor
from latentgandiscovery.visualization import make_interpolation_chart
from torch.utils.tensorboard import SummaryWriter

import argparse


cuda = True
device = torch.device("cuda" if cuda else "cpu")

print(device)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='latentgandiscovery/latentgandisc_16dir.json', type=str)
    return parser.parse_args()

def read_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config


def get_generator(cfg):
    model_checkpoint = os.path.abspath(cfg['progan_model'])
    gen_model = os.path.join(model_checkpoint, 'gen_ema.pth')
    model_folder = os.path.abspath(os.path.join(model_checkpoint, os.pardir, os.pardir))
    progan_param = read_config(os.path.join(model_folder, 'param.json'))
    n_channel = progan_param['n_channel']
    latent_dim = progan_param['latent_dim']
    pixel_norm = progan_param['pixel_norm']
    tanh = progan_param['tanh']
    netG = Generator(in_channel=n_channel, input_code_dim=latent_dim, pixel_norm=pixel_norm, tanh=tanh).to(device)
    netG.load_state_dict(torch.load(gen_model,map_location=device))
    return netG

def get_deformator(cfg, netG):
    deformator = LatentDeformator(shift_dim= netG.input_dim,
                                  input_dim=cfg['directions_count'],
                                  out_dim=cfg['max_latent_dim'],
                                  random_init=cfg['deformator_random_init']).to(device)
    return deformator

def make_shifts(input_dim, cfg):
    target_indices = torch.randint(
        0, cfg['directions_count'], [cfg['batch_size']], device=device)

    shifts = 2.0 * torch.rand(target_indices.shape, device=device) - 1.0

    shifts = cfg['shift_scale'] * shifts
    min_shift = cfg['min_shift']
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > -min_shift) & (shifts < 0)] = -min_shift

    try:
        input_dim[0]
        input_dim = list(input_dim)
    except Exception:
        input_dim = [input_dim]

    z_shift = torch.zeros([cfg['batch_size']] + input_dim, device='cuda')
    for i, (index, val) in enumerate(zip(target_indices, shifts)):
        z_shift[i][index] += val

    return target_indices, shifts, z_shift


def get_shift_predictor(dim, size):
    return LatentShiftPredictor(dim, size).to(device)


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

def start_from_checkpoint(deformator, shift_predictor, checkpoint):
    step = 1
    if os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint)
        step = state_dict['step']
        deformator.load_state_dict(state_dict['deformator'])
        shift_predictor.load_state_dict(state_dict['shift_predictor'])
        print('starting from step {}'.format(step))
    return step

def fig_to_image(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = Image.fromarray(im)
    return im

def log_train(cfg, tb_writer, step, should_print=True, stats=()):
    if should_print:
        out_text = '{}% [step {:06d}]'.format(int(100 * step / cfg['n_steps']), step)
        for named_value in stats:
            out_text += (' | {}: {:.2f}'.format(*named_value))
        print(out_text)
    for named_value in stats:
        tb_writer.add_scalar(named_value[0], named_value[1], step)

def log_interpolation(G, deformator, step, fixed_test_noise, cfg):
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    noise = torch.randn(1, G.input_dim).to(device)
    if fixed_test_noise is None:
        fixed_test_noise = noise.clone()
    for z, prefix in zip([noise, fixed_test_noise], ['rand', 'fixed']):
        fig = make_interpolation_chart(
            G, deformator, z=z, shifts_r=3.0 * cfg['shift_scale'], shifts_count=6, dims_count=16,
            dpi=300, figsize=(9,16))

        img = fig_to_image(fig)
        plt.close()
        img.save(os.path.join(sample_folder, '{}_{:06d}.jpg'.format(prefix, step)))

def save_checkpoint(deformator, shift_predictor, step, cfg):
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    state_dict = {
        'step': step,
        'deformator': deformator.state_dict(),
        'shift_predictor': shift_predictor.state_dict(),
    }
    checkpoint = os.path.join(training_path, 'checkpoint.pt')
    torch.save(state_dict, checkpoint)

def save_models(deformator, shift_predictor, step, cfg):
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    torch.save(deformator.state_dict(),
               os.path.join(ckpt_folder, 'deformator_{:06d}.pt'.format(step)))
    torch.save(shift_predictor.state_dict(),
               os.path.join(ckpt_folder, 'shift_predictor_{:06d}.pt'.format(step)))


def log(tb_writer, G, deformator, shift_predictor, step, avgs, cfg, **kwargs):
    fixed_test_noise = kwargs['fixed_test_noise']
    if (step+1) % cfg['steps_per_log'] == 0:
        log_train(cfg, tb_writer, step+1, True, [avg.flush() for avg in avgs])
    if (step+1) % cfg['steps_per_img_log'] == 0:
        log_interpolation(G, deformator, step+1, fixed_test_noise, cfg)
    if (step+1) % cfg['steps_per_backup'] == 0 and step > 0:
        save_checkpoint(deformator, shift_predictor, step+1, cfg)
    if (step+1) % cfg['steps_per_save'] == 0 and step > 0:
        save_models(deformator, shift_predictor, step+1, cfg)


def train(G, deformator, shift_predictor, cfg):
    #config
    training_path, sample_folder, ckpt_folder, log_folder, config_path = get_folders(cfg)
    n_steps = cfg['n_steps']
    batch_size = cfg['batch_size']
    label_weight = cfg['label_weight']
    shift_weight = cfg['shift_weight']

    G.to(device).eval()
    deformator.to(device).train()
    shift_predictor.to(device).train()

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=cfg['deformator_lr'])
    shift_predictor_opt = torch.optim.Adam(
        shift_predictor.parameters(), lr=cfg['shift_predictor_lr'])

    # loss
    cross_entropy = nn.CrossEntropyLoss()

    # logger
    tb_writer = SummaryWriter(log_folder)

    avgs = MeanTracker('percent'), MeanTracker('loss'), MeanTracker('direction_loss'),\
           MeanTracker('shift_loss')
    avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss = avgs

    checkpoint = os.path.join(training_path, 'checkpoint.pt')
    recovered_step = start_from_checkpoint(deformator, shift_predictor, checkpoint) - 1

    # fixed_noise
    fixed_test_noise = torch.randn(1, G.input_dim).to(device)

    for step in range(recovered_step, n_steps, 1):
        G.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z = torch.randn(batch_size, G.input_dim).to(device)
        target_indices, shifts, basis_shift = make_shifts(deformator.input_dim, cfg)

        # Deformation
        shift = deformator(basis_shift)

        imgs = G(z, step=6, alpha=1) * 0.5 + 0.5
        imgs_shifted = G(z+shift, step=6, alpha=1) * 0.5 + 0.5

        logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
        logit_loss = label_weight * cross_entropy(logits, target_indices)
        shift_loss = shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

        # total loss
        loss = logit_loss + shift_loss
        loss.backward()

        deformator_opt.step()
        shift_predictor_opt.step()

        # update statistics
        # update statistics trackers
        avg_correct_percent.add(torch.mean(
                (torch.argmax(logits, dim=1) == target_indices).to(torch.float32)).detach())
        avg_loss.add(loss.item())
        avg_label_loss.add(logit_loss.item())
        avg_shift_loss.add(shift_loss)

        log(tb_writer, G, deformator, shift_predictor, step, avgs, cfg, fixed_test_noise=fixed_test_noise)


if __name__=='__main__':
    opts = get_args()
    cfg = read_config(opts.config)
    torch.manual_seed(cfg['seed'])
    training_path, sample_folder, ckpt_folder, log_folder, config_path = create_folders(cfg)
    save_params(cfg, training_path)
    netG= get_generator(cfg)
    deformator = get_deformator(cfg, netG)
    shift_predictor = get_shift_predictor(deformator.input_dim, cfg['shift_predictor_size'])
    train(netG, deformator, shift_predictor, cfg)
