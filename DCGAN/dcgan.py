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

            for _ in range(3):
                g_optimizer.zero_grad()
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


if __name__ == "__main__":

    parser = utils.create_parser()
    opts = parser.parse_args()
    print(opts)
    main(opts)
