import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, latent_dim=16, starting_filters=16):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv_stack = nn.Sequential(
            # 256 - 128
            nn.Conv1d(1, starting_filters, kernel_size=3, stride=2,
                      padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters),
            nn.LeakyReLU(),
            # 128 - 64
            nn.Conv1d(starting_filters, starting_filters*2, kernel_size=3, stride=2,
                      padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters*2),
            nn.LeakyReLU(),
            # 64 - 32
            nn.Conv1d(starting_filters*2, starting_filters*2**2, kernel_size=3, stride=2,
                      padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters*2**2),
            nn.LeakyReLU(),
            # 32 - 16
            nn.Conv1d(starting_filters*2**2, starting_filters*2**3, kernel_size=3, stride=2,
                      padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters*2**3),
            nn.LeakyReLU(),
            # 16 - 8
            nn.Conv1d(starting_filters*2**3, starting_filters*2**4, kernel_size=3, stride=2,
                      padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters*2**4),
            nn.LeakyReLU(),
            # flatten
            nn.Flatten())
        self.fc_mu = nn.Linear(starting_filters*2**4*8, latent_dim)
        self.fc_var = nn.Linear(starting_filters*2**4*8, latent_dim)

    def forward(self, x):
        x = self.conv_stack(x)
        mean = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mean, log_var

class Decoder(nn.Module):

    def __init__(self, latent_dim=16, starting_filters=16):
        super(Decoder, self).__init__()
        self.starting_filters=starting_filters
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, starting_filters*2**4*8)
        self.deconv_stack = nn.Sequential(
            # 8 - 16
            nn.ConvTranspose1d(starting_filters*2**4, starting_filters*2**3, kernel_size=3,
                                stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(starting_filters*2**3),
            nn.LeakyReLU(),
            # 16 - 32
            nn.ConvTranspose1d(starting_filters*2**3, starting_filters*2**2, kernel_size=3,
                                stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(starting_filters*2**2),
            nn.LeakyReLU(),
            # 32 - 64
            nn.ConvTranspose1d(starting_filters*2**2, starting_filters*2, kernel_size=3,
                                stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(starting_filters*2),
            nn.LeakyReLU(),
            # 64 - 128
            nn.ConvTranspose1d(starting_filters*2, starting_filters, kernel_size=3,
                                stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(starting_filters),
            nn.LeakyReLU(),
            # 128 - 256
            nn.ConvTranspose1d(starting_filters, starting_filters, kernel_size=3,
                                stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(starting_filters),
            nn.LeakyReLU(),

            nn.Conv1d(starting_filters, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.starting_filters*2**4, 8)
        x = self.deconv_stack(x)
        return x

class VAE(nn.Module):

    def __init__(self, latent_dim=16, starting_filters=16):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, starting_filters)
        self.decoder = Decoder(latent_dim, starting_filters)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_recon = self.decoder(z)
        return x_recon, mean, log_var

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recons_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kl_div= torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    loss = recons_loss + beta*kl_div
    return loss, recons_loss, kl_div
