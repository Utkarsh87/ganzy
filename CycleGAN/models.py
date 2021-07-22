import torch
import torch.nn as nn


def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    batch_norm=True,
    init_zero_weights=False,
):
    """Creates a convolutional layer, with optional batch normalization."""
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


def deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    batch_norm=True,
):
    """Creates a transposed-convolutional layer, with optional batch normalization."""
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
    )
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class Generator(nn.Module):
    """
    Defines the architecture of the generator network.
    Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(Generator, self).__init__()

        # 1. Define the encoder part of the generator (extracts features from the input image)
        self.conv1 = conv(
            in_channels=3,
            out_channels=conv_dim,
            kernel_size=4,
            init_zero_weights=init_zero_weights,
        )
        self.conv2 = conv(
            in_channels=conv_dim,
            out_channels=conv_dim * 2,
            kernel_size=4,
            init_zero_weights=init_zero_weights,
        )

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim * 2)

        # 3. Define the decoder part of the generator (builds up the output image from features)
        self.deconv1 = deconv(
            in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4
        )
        self.deconv2 = deconv(
            in_channels=conv_dim,
            out_channels=3,
            kernel_size=4,
            batch_norm=False,
        )

    def forward(self, x):
        """
        Generate output image conditioned on input image
        Input
        -----
            x: BS x 3 x 32 x 32
        Output
        ------
            out: BS x 3 x 32 x 32
        """

        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))

        out = torch.relu(self.resnet_block(out))

        out = torch.relu(self.deconv1(out))
        out = torch.tanh(self.deconv2(out))

        return out


class Discriminator(nn.Module):
    """
    Define architecture of the discriminator
    Note: Both discriminators D_X and D_Y have the same architecture
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
