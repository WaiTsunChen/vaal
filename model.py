import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3,img_size=128):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.image_size = img_size
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),
        #     nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(True),
        #     # View((-1, 1024*2*2)),                                 # B, 1024*4*4
        #     View((-1, 1024*6*6)),
        # )

        # self.fc_mu = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        # self.fc_logvar = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        # self.fc_mu = nn.Linear(1024*6*6, z_dim)                           
        # self.fc_logvar = nn.Linear(1024*6*6, z_dim)
        # self.decoder = nn.Sequential(
        #     # nn.Linear(z_dim, 1024*4*4),                           # B, 1024*8*8
        #     # View((-1, 1024, 4, 4)),                               # B, 1024,  8,  8
        #     nn.Linear(z_dim, 1024*12*12),                          
        #     View((-1, 1024, 12, 12)),
        #     nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        #     nn.Sigmoid()
        # )

        modules = []
        hidden_dims = [64, 128, 256, 512]

        conv_factor = self.image_size // 2 ** len(hidden_dims)
        conv_factor = conv_factor
        in_channels = nc
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                View((-1,hidden_dims[-1] * conv_factor**2)))
        )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * conv_factor**2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * conv_factor**2, z_dim)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(z_dim, hidden_dims[-1] * conv_factor**2),
                View((-1,hidden_dims[-1], conv_factor, conv_factor))
            )
        )

        hidden_dims.reverse()
        for i in range(len(hidden_dims) -1 ):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        # add last Decoder layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    hidden_dims[-1],
                    out_channels= 3,
                    kernel_size= 3,
                    padding= 1),
                nn.Sigmoid())
        )
        
        self.decoder = nn.Sequential(*modules)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        # elif torch.bakends.mps.is_available():
        #     stds, epsilon = stds.to('mps'), epsilon.to('mps')
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
