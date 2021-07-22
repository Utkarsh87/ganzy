import os
import imageio
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from models import Generator, Discriminator

SEED = 11
CUDA = torch.cuda.is_available()
np.random.seed(SEED)
torch.cuda.manual_seed(SEED) if CUDA else torch.manual_seed(SEED)


CYLW = "\033[93m"
CYLW2 = "\033[33m"
CEND = "\033[0m"
BOLD = "\033[1m"
RULE_STRING = 80 * "-"
LINEBREAK = f"{CYLW}{BOLD}{RULE_STRING}{CEND}"


def to_var(x):
    """Convert numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Convert variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Create a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_model(model, model_name):
    """Print model information"""
    print(LINEBREAK)
    print(f"\t\t\t\t{CYLW2}{BOLD}{model_name}{CEND}")
    print(LINEBREAK)
    print(model)
    print(LINEBREAK)


def sample_noise(dim, batch_size):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def build_discriminator(opts):
    D = Discriminator(conv_dim=opts.conv_dim)

    print_model(D, "DISCRIMINATOR")

    if torch.cuda.is_available():
        D.cuda()

    return D


def build_generator(opts):
    G = Generator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)

    print_model(G, "GENERATOR")

    if torch.cuda.is_available():
        G.cuda()

    return G


def checkpoint(iteration, G, D, opts):
    """Save the parameters of the generator G and discriminator D"""
    G_path = os.path.join(opts.checkpoint_dir, "G.pkl")
    D_path = os.path.join(opts.checkpoint_dir, "D.pkl")
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def create_image_grid(array, ncols=None):
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w, :
            ] = array[i * ncols + j].transpose(1, 2, 0)

    return result.squeeze() if channels == 1 else result


def save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = to_data(generated_images)

    grid = create_image_grid(generated_images)

    path = os.path.join(opts.sample_dir, "sample-{:06d}.png".format(iteration))
    imageio.imwrite(path, grid)
    print("Saved {}".format(path))


def create_parser():
    """Creates a parser for command-line arguments."""
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="The side length N to convert images to NxN.",
    )
    parser.add_argument("--conv_dim", type=int, default=32)
    parser.add_argument("--noise_size", type=int, default=100)

    # Training hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The number of images in a batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of threads to use for the DataLoader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="The learning rate (default 0.0003)",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    # Data sources
    parser.add_argument(
        "--emoji",
        type=str,
        default="Apple",
        choices=["Apple", "Facebook", "Windows"],
        help="Choose the type of emojis to generate.",
    )

    # Directories and checkpoint/sample iterations
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_vanilla")
    parser.add_argument("--sample_dir", type=str, default="./samples_vanilla")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=200)
    parser.add_argument("--checkpoint_every", type=int, default=400)

    return parser
