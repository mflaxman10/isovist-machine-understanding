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
        self.linear = nn.Linear(starting_filters*2**4*8, latent_dim)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim=16, starting_filters=16):
        super(Decoder, self).__init__()
        self.starting_filters=starting_filters
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, starting_filters*2**4*8)
        self.deconv_stack = nn.Sequential(
            # 8 - 16
            nn.Upsample(scale_factor=2.0, mode='linear'),
            nn.Conv1d(starting_filters*2**4, starting_filters*2**3, kernel_size=3,
                                stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters*2**3),
            nn.LeakyReLU(),
            # 16 - 32
            nn.Upsample(scale_factor=2.0, mode='linear'),
            nn.Conv1d(starting_filters*2**3, starting_filters*2**2, kernel_size=3,
                                stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters*2**2),
            nn.LeakyReLU(),
            # 32 - 64
            nn.Upsample(scale_factor=2.0, mode='linear'),
            nn.Conv1d(starting_filters*2**2, starting_filters*2, kernel_size=3,
                                stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters*2),
            nn.LeakyReLU(),
            # 64 - 128
            nn.Upsample(scale_factor=2.0, mode='linear'),
            nn.Conv1d(starting_filters*2, starting_filters, kernel_size=3,
                                stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters),
            nn.LeakyReLU(),
            # 128 - 256
            nn.Upsample(scale_factor=2.0, mode='linear'),
            nn.Conv1d(starting_filters, starting_filters, kernel_size=3,
                                stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(starting_filters),
            nn.LeakyReLU(),

            nn.Conv1d(starting_filters, 1, kernel_size=3, padding=1, padding_mode='circular'),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.starting_filters*2**4, 8)
        x = self.deconv_stack(x)
        return x
