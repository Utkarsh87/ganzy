import os
import argparse
import imageio
import numpy as np
import torch
from torch.autograd import Variable
from models import Generator, Discriminator

SEED = 11
CUDA = torch.cuda.is_available()
np.random.seed(SEED)
torch.cuda.manual_seed(SEED) if CUDA else torch.manual_seed(SEED)


CRED = "\033[91m"
CGRN = "\033[92m"
CYLW = "\033[93m"
CVLT = "\033[94m"

CEND = "\033[0m"
BOLD = "\033[1m"

RULE_STRING = 80 * "-"
THICK_RULE_STRING = 80 * "="
LINEBREAK = f"{CGRN}{BOLD}{RULE_STRING}{CEND}"
THICK_LINEBREAK = f"{CGRN}{BOLD}{THICK_RULE_STRING}{CEND}"


def to_var(x):
    """Convert numpy to variable"""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Convert variable to numpy"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Create a directory if it does not already exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_model(model, model_name):
    """Print model information"""
    print(LINEBREAK)
    print(f"{CYLW}{BOLD}{model_name}{CEND}".center(80))
    print(LINEBREAK)
    print(model)
    print(LINEBREAK)


def print_opts(opts):
    """Print all CL args"""
    print(THICK_LINEBREAK)
    print(f"{CYLW}{BOLD}Opts{CEND}".center(80))
    print(LINEBREAK)

    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))

    print(THICK_LINEBREAK)


def create_parser():
    """Create a parser for CL args"""
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="The side length N to convert images to NxN.",
    )
    parser.add_argument("--g_conv_dim", type=int, default=32)
    parser.add_argument("--d_conv_dim", type=int, default=32)
    parser.add_argument(
        "--use_cycle_consistency_loss",
        action="store_true",
        default=False,
        help="Choose whether to include the cycle consistency term in the loss.",
    )
    parser.add_argument(
        "--init_zero_weights",
        action="store_true",
        default=False,
        help="Choose whether to initialize the generator conv weights to 0 (implements the identity function).",
    )

    # Training hyper-parameters
    parser.add_argument(
        "--train_iters",
        type=int,
        default=1000,
        help="The number of training iterations to run (you can Ctrl-C out earlier if you want).",
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=1,
        help="The number of generator cycles(i.e. number of X->Y->X and Y->X->Y cycles per discriminator update)",
    )
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
        "--X",
        type=str,
        default="Apple",
        choices=["Apple", "Windows"],
        help="Choose the type of images for domain X.",
    )
    parser.add_argument(
        "--Y",
        type=str,
        default="Windows",
        choices=["Apple", "Windows"],
        help="Choose the type of images for domain Y.",
    )

    # Saving directories and checkpoint/sample iterations
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_cyclegan")
    parser.add_argument("--sample_dir1", type=str, default="samples_X-Y")
    parser.add_argument("--sample_dir2", type=str, default="samples_Y-X")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=1000)

    return parser


def build_discriminator(opts, disc_name: str):
    D = Discriminator(conv_dim=opts.d_conv_dim)

    print_model(D, disc_name)

    D.cuda() if torch.cuda.is_available() else None

    return D


def build_generator(opts, gen_name: str):
    G = Generator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)

    print_model(G, gen_name)

    G.cuda() if torch.cuda.is_available() else None

    return G


def create_model(opts):
    """Builds the generators and discriminators."""
    G_XtoY = build_generator(opts, gen_name="G_XtoY")
    G_YtoX = build_generator(opts, gen_name="G_YtoX")
    D_X = build_discriminator(opts, disc_name="D_X")
    D_Y = build_discriminator(opts, disc_name="D_Y")

    return G_XtoY, G_YtoX, D_X, D_Y


def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints."""
    G_XtoY_path = os.path.join(opts.load, "G_XtoY.pkl")
    G_YtoX_path = os.path.join(opts.load, "G_YtoX.pkl")
    D_X_path = os.path.join(opts.load, "D_X.pkl")
    D_Y_path = os.path.join(opts.load, "D_Y.pkl")

    G_XtoY = Generator(
        conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights
    )
    G_YtoX = Generator(
        conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights
    )
    D_X = Discriminator(conv_dim=opts.d_conv_dim)
    D_Y = Discriminator(conv_dim=opts.d_conv_dim)

    G_XtoY.load_state_dict(
        torch.load(G_XtoY_path, map_location=lambda storage, loc: storage)
    )
    G_YtoX.load_state_dict(
        torch.load(G_YtoX_path, map_location=lambda storage, loc: storage)
    )
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print("Models moved to GPU.")

    return G_XtoY, G_YtoX, D_X, D_Y


def checkpoint(G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Save params of G_YtoX, G_XtoY, D_X and D_Y"""
    G_XtoY_path = os.path.join(opts.checkpoint_dir, "G_XtoY.pkl")
    G_YtoX_path = os.path.join(opts.checkpoint_dir, "G_YtoX.pkl")
    D_X_path = os.path.join(opts.checkpoint_dir, "D_X.pkl")
    D_Y_path = os.path.join(opts.checkpoint_dir, "D_Y.pkl")
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def merge_images(sources, targets, opts, k=10):
    """
    Create grid consisting of pairs of columns; where each pair
    first column: source images
    second column: images generated by CycleGAN from corresponding images in first column
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row * h, row * w * 2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i * h : (i + 1) * h, (j * 2) * h : (j * 2 + 1) * h] = s
        merged[:, i * h : (i + 1) * h, (j * 2 + 1) * h : (j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)


def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """
    Save samples from both generators X->Y and Y->X
    Generate test results on a fixed set of images(test set results)
    """
    fake_X, fake_Y = G_YtoX(fixed_Y), G_XtoY(fixed_X)

    X, fake_X = to_data(fixed_X), to_data(fake_X)
    Y, fake_Y = to_data(fixed_Y), to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir1, "sample-{:06d}.png".format(iteration))
    imageio.imwrite(path, merged)
    # print("Saved {}".format(path))

    merged = merge_images(Y, fake_X, opts)
    path = os.path.join(opts.sample_dir2, "sample-{:06d}.png".format(iteration))
    imageio.imwrite(path, merged)
    # print("Saved {}".format(path))
