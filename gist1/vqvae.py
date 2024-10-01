# reference https://github.com/zalandoresearch/pytorch-vq-vae
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert input from BCW -> BWC
        inputs = inputs.permut(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BWC -> BCW
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        self.commitment_cost = commitment_cost

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs):
        #convert inputs from BCW -> BWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)

            # laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = self.ema_cluster_size + self.epsilon / (n + self.num_embeddings * self.epsilon * n)
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # straight trough estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BWC -> BCW
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encoding_indices


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddnes, num_residual_hiddens):
        super().__init__()
        self.block = nn.Sequential( nn.ReLU(inplace=True),
                                    nn.Conv1d(  in_channels=in_channels,
                                                out_channels=num_residual_hiddens,
                                                kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular'),
                                    nn.ReLU(inplace=True),
                                    nn.Conv1d(in_channels=num_residual_hiddens,
                                    out_channels=num_hiddnes,
                                    kernel_size=1, stride=1, bias=False)
                                    )
    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                    for _ in range(self.num_residual_layers)])
    
    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # 256 -> 128
        self.conv_1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2, padding=1, padding_mode='circular')
        # 128 -> 64
        self.conv_2 = nn.Conv1d(in_channels=num_hiddens//2,
                                out_channels=num_hiddens,
                                kernel_size=4,
                                stride=2, padding=1, padding_mode='circular')
        # 64 -> 32
        self.conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=4,
                                stride=2, padding=1, padding_mode='circular')
        # 32 -> 16
        self.conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=4,
                                stride=2, padding=1, padding_mode='circular')
        self.conv_final = nn.Conv1d(in_channels=num_hiddens,
                                    out_channels=num_hiddens,
                                    kernel_size=3,
                                    stride=1, padding=1, padding_mode='circular')
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_hiddens=num_residual_hiddens,
                                            num_residual_layers=num_residual_layers)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.relu(x)

        x = self.conv_3(x)
        x = F.relu(x)

        x = self.conv_4(x)
        x = F.relu(x)

        x = self.conv_final(x)
        x = self.residual_stack(x)
        
        return x

    
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.conv_init = nn.Conv1d( in_channels=in_channels,
                                    out_channels=num_hiddens,
                                    kernel_size=3,
                                    stride=1, padding=1)
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)
        
        # 16 -> 32
        self.conv_trans_0 = nn.ConvTranspose1d( in_channels=num_hiddens,
                                                out_channels=num_hiddens,
                                                kernel_size=4,
                                                stride=2, padding=1)

        # 32 -> 64
        self.conv_trans_1 = nn.ConvTranspose1d( in_channels=num_hiddens,
                                                out_channels=num_hiddens,
                                                kernel_size=4,
                                                stride=2, padding=1)
        # 64 -> 128
        self.conv_trans_2 = nn.ConvTranspose1d( in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1)
        # 128 -> 256
        self.conv_trans_3 = nn.ConvTranspose1d( in_channels=num_hiddens//2,
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)
    
    def forward(self, inputs):
        x = self.conv_init(inputs)

        x = self.residual_stack(x)

        x = self.conv_trans_0(x)
        x = F.relu(x)

        x = self.conv_trans_1(x)
        x = F.relu(x)

        x = self.conv_trans_2(x)
        x = F.relu(x)

        return self.conv_trans_3(x)


class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                        embedding_dim, commitment_cost, decay=0):
        super().__init__()
        self.encoder = Encoder( 1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.pre_vq_conv = nn.Conv1d(   in_channels=num_hiddens,
                                        out_channels=embedding_dim,
                                        kernel_size=1,
                                        stride=1)
        
        if decay > 0.0:
            self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = Decoder( embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
    def encode(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        _, quantized, _, encoding_indices = self.vq(z)

        return quantized, encoding_indices

    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)

        loss, quantized, perplexity, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        
        return loss, x_recon, perplexity




