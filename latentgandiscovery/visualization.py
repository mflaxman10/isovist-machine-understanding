import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from utils.isoutil import plot_isovist_numpy
from PIL import Image


def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec

@torch.no_grad()
def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False):
    shifted_images = []
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        if deformator is not None:
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
        else:
            latent_shift = one_hot(G.dim_shift, shift, dim).cuda()
        shifted_image = G(z+latent_shift, step=6, alpha=1).cpu()[0] * 0.5 + 0.5
        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)
        shifted_images.append(shifted_image)
    return shifted_images

@torch.no_grad()
def make_interpolation_chart(G, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, **kwargs):
    with_deformation = deformator is not None
    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()
    z = z if z is not None else torch.randn(1, G.input_dim).cuda()

    original_img = G(z, step=6, alpha=1).cpu()[0] * 0.5 + 0.5
    imgs = [[original_img]]
    if dims is None:
        dims = range(dims_count)
    for i in dims:
        imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator)) # list in list

    rows_count = len(imgs)
    fig, axs = plt.subplots(rows_count, **kwargs)

    if texts is None:
        texts = [''] + list(dims)
    for ax, shifts_imgs, text in zip(axs, imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        figs = []
        for shifted_img in shifts_imgs:
            shifted_img = np.squeeze(shifted_img)
            figs.append(plot_isovist_numpy(shifted_img, figsize=(1,1)))
        nrow = 2 * shifts_count + 1
        figs = torch.tensor(figs, dtype=torch.float)
        im = torchvision.utils.make_grid(figs, normalize=True, range=(0, 255), nrow=nrow)
        im = Image.fromarray(np.uint8(np.transpose(im.numpy(), (1, 2, 0))*255))
        ax.imshow(im)
        ax.text(-40, 20, str(text), fontsize=6, ha='right')
        fig.tight_layout()

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig


def add_border(tensor):
    border = 3
    for ch in range(tensor.shape[0]):
        color = 1.0 if ch == 0 else -1
        tensor[ch, :border, :] = color
        tensor[ch, -border:,] = color
        tensor[ch, :, :border] = color
        tensor[ch, :, -border:] = color
    return tensor
