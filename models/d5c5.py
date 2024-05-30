import torch
import torch.nn as nn
from packaging import version
from utils.math import complex_abs
# from utils.transform import normalize, unnormalize

from utils.fourier import fft2c as fft2c
from utils.fourier import ifft2c as ifft2c



def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator
    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4:  # input is 2D
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5:  # input is 3D
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = fft2c(x)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_rec = ifft2c(out)

        if x.dim() == 4:
            x_rec = x_rec.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_rec = x_rec.permute(0, 4, 2, 3, 1)

        return x_rec


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block(n_ch, nd, nf=64, ks=3, dilation=1, bn=False, nl='relu', conv_dim=2, n_out=None):
    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf, nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd - 2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class D5C5(nn.Module):
    def __init__(self, nc, nd, n_channels):
        super(D5C5, self).__init__()
        self.nc = nc
        self.nd = nd
        self.n_channels = n_channels

        # print('Creating D{}C{}'.format(self.nd, self.nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for i in range(self.nc):
            conv_blocks.append(conv_layer(self.n_channels, self.nd))
            dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self, image, kspace, mask, mean_value=None, std_value=None):

        for i in range(self.nc):

            mean = torch.mean(image, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (b*n_coils, 1, 1, 1)
            std = torch.std(image, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (b*n_coils, 1, 1, 1)
            image = (image - mean) / std # (b*n_coils, c=2, h, w)

            x_cnn = self.conv_blocks[i](image) + image # (b*n_coils, c=2, h, w)
            x_cnn = x_cnn * std + mean # (b*n_coils, c=2, h, w)

            image = self.dcs[i].perform(x_cnn, kspace, mask) # (b*n_coils, c=2, h, w)

        return image