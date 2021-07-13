import argparse
import warnings

warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.optim as optim

# Local imports
import utils
from dataloader import get_emoji_loader


def train(train_dataloader, opts):
    """
    Run training loop
    * Save checkpoints every opts.checkpoint_every iterations
    * Save generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G = utils.build_generator(opts)
    D = utils.build_discriminator(opts)
    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # Generate fixed noise for sampling from the generator
    fixed_noise = utils.sample_noise(
        opts.noise_size, opts.batch_size
    )  # batch_size x noise_size x 1 x 1

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)

    for epoch in range(opts.num_epochs):

        for batch in train_dataloader:

            real_images, labels = batch
            real_images, labels = (
                utils.to_var(real_images),
                utils.to_var(labels).long().squeeze(),
            )

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            d_optimizer.zero_grad()

            D_real_loss = torch.mean((D.forward(real_images) - 1) ** 2) / 2
            noise = utils.sample_noise(opts.noise_size, opts.batch_size)
            fake_images = G.forward(noise)
            D_fake_loss = torch.mean((D.forward(fake_images)) ** 2) / 2
            D_total_loss = D_real_loss + D_fake_loss
            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            for _ in range(3):
                noise = utils.sample_noise(opts.noise_size, opts.batch_size)
                fake_images = G.forward(noise)
                G_loss = torch.mean((D.forward(fake_images) - 1) ** 2)
                G_loss.backward()
                g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print(
                    "Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}".format(
                        iteration,
                        total_train_iters,
                        D_real_loss.data,
                        D_fake_loss.data,
                        G_loss.data,
                    )
                )

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                utils.save_samples(G, fixed_noise, iteration, opts)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                utils.checkpoint(iteration, G, D, opts)

            iteration += 1


def main(opts):
    """Load data; create checkpoint and sample dirs; train"""

    # Create a dataloader for the training images
    train_dataloader, _ = get_emoji_loader(opts.emoji, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    train(train_dataloader, opts)


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
        "--batch_size", type=int, default=16, help="The number of images in a batch."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of threads to use for the DataLoader.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="The learning rate (default 0.0003)"
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


if __name__ == "__main__":

    parser = create_parser()
    opts = parser.parse_args()
    print(opts)
    main(opts)
