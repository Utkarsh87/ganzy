import torch
import torch.nn as nn


def deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    batch_norm=True,
):
    """Transposed-convolutional layer, with optional batch-norm"""
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
    )
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    batch_norm=True,
    init_zero_weights=False,
):
    """Convolutional layer, with optional batch-norm"""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    if init_zero_weights:
        conv_layer.weight.data = (
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
        )
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(Generator, self).__init__()

        self.deconv1 = deconv(
            in_channels=noise_size,
            out_channels=conv_dim * 4,
            kernel_size=4,
            padding=0,
        )
        self.deconv2 = deconv(
            in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4
        )
        self.deconv3 = deconv(
            in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4
        )
        self.deconv4 = deconv(
            in_channels=conv_dim,
            out_channels=3,
            kernel_size=4,
            batch_norm=False,
        )

    def forward(self, z):
        """Generates an image given a sample of random noise.
        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1
        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x32x32
        """

        out = torch.relu(self.deconv1(z))
        out = torch.relu(self.deconv2(out))
        out = torch.relu(self.deconv3(out))
        out = torch.tanh(self.deconv4(out))

        return out


class Discriminator(nn.Module):
    """
    Defines the architecture of the discriminator network.
    Note: Both discriminators D_X and D_Y have the same architecture.
    """

    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        self.conv1 = conv(
            in_channels=3, out_channels=conv_dim, kernel_size=4
        )  # 32x32x3 --> 16x16x32
        self.conv2 = conv(
            in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4
        )  # 16x16x32 --> 8x8x64
        self.conv3 = conv(
            in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4
        )  # 8x8x64 --> 4x4x128
        self.conv4 = conv(
            in_channels=conv_dim * 4,
            out_channels=1,
            kernel_size=4,
            stride=4,
            padding=0,
            batch_norm=False,
        )  # 4x4x128 --> 1x1x1(real/fake label)

    def forward(self, x):

        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = torch.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = torch.sigmoid(out)

        return out
