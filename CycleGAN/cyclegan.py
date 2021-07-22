import numpy as np
import torch
import torch.optim as optim
import utils
from dataloader import get_emoji_loader


def train(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
    """
    Run training loop
    * Save checkpoints every opts.checkpoint_every iterations
    * Save generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = (
        utils.load_checkpoint(opts) if opts.load else utils.create_model(opts)
    )

    g_params = list(G_XtoY.parameters()) + list(
        G_YtoX.parameters()
    )  # Get generator parameters

    d_params = list(D_X.parameters()) + list(
        D_Y.parameters()
    )  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = utils.to_var(test_iter_X.next()[0])
    fixed_Y = utils.to_var(test_iter_Y.next()[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    for iteration in range(1, opts.train_iters + 1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, labels_X = iter_X.next()
        images_X, labels_X = (
            utils.to_var(images_X),
            utils.to_var(labels_X).long().squeeze(),
        )

        images_Y, labels_Y = iter_Y.next()
        images_Y, labels_Y = (
            utils.to_var(images_Y),
            utils.to_var(labels_Y).long().squeeze(),
        )

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        # Train with real images
        d_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        D_X_loss = torch.mean((D_X.forward(images_X) - 1) ** 2)
        D_Y_loss = torch.mean((D_Y.forward(images_Y) - 1) ** 2)

        d_real_loss = D_X_loss + D_Y_loss
        d_real_loss.backward()
        d_optimizer.step()

        # Train with fake images
        d_optimizer.zero_grad()

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX.forward(images_Y)

        # 3. Compute the loss for D_X
        D_X_loss = torch.mean(D_X.forward(fake_X) ** 2)

        # 4. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY.forward(images_X)

        # 5. Compute the loss for D_Y
        D_Y_loss = torch.mean(D_Y.forward(fake_Y) ** 2)

        d_fake_loss = D_X_loss + D_Y_loss
        d_fake_loss.backward()
        d_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        for _ in range(opts.num_cycles):

            #########################################
            ##             Y-->X-->Y CYCLE         ##
            #########################################

            g_optimizer.zero_grad()

            # 1. Generate fake images that look like domain X based on real images in domain Y
            fake_X = G_YtoX.forward(images_Y)

            # 2. Compute the generator loss based on domain X
            g_loss = torch.mean((D_X.forward(fake_X) - 1) ** 2)

            if opts.use_cycle_consistency_loss:
                reconstructed_Y = G_XtoY(fake_X)
                # 3. Compute the cycle consistency loss (the reconstruction loss)
                cycle_consistency_loss = torch.mean((images_Y - reconstructed_Y) ** 2)
                g_loss += cycle_consistency_loss

            g_loss.backward()
            g_optimizer.step()

            #########################################
            ##             X--Y-->X CYCLE          ##
            #########################################

            g_optimizer.zero_grad()

            # 1. Generate fake images that look like domain Y based on real images in domain X
            fake_Y = G_XtoY.forward(images_X)

            # 2. Compute the generator loss based on domain Y
            g_loss = torch.mean((D_Y.forward(fake_Y) - 1) ** 2)

            if opts.use_cycle_consistency_loss:
                reconstructed_X = G_YtoX(fake_Y)
                # 3. Compute the cycle consistency loss (the reconstruction loss)
                cycle_consistency_loss = torch.mean((images_X - reconstructed_X) ** 2)
                g_loss += cycle_consistency_loss

            g_loss.backward()
            g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print(
                "\033[92m \033[1m Iteration [{:5d}/{:5d}] \033[0m"
                " | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | "
                "d_fake_loss: {:6.4f} | g_loss: {:6.4f}".format(
                    iteration,
                    opts.train_iters,
                    d_real_loss.data,
                    D_Y_loss.data,
                    D_X_loss.data,
                    d_fake_loss.data,
                    g_loss.data,
                )
            )

        # Save the generated samples
        if iteration % opts.sample_every == 0:
            utils.save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            utils.checkpoint(G_XtoY, G_YtoX, D_X, D_Y, opts)


def main(opts):
    """Load data, create checkpoint and sample dirs; train"""

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type=opts.X, opts=opts)
    dataloader_Y, test_dataloader_Y = get_emoji_loader(emoji_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir1)
    utils.create_dir(opts.sample_dir2)

    # Start training
    train(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts)


if __name__ == "__main__":

    parser = utils.create_parser()
    opts = parser.parse_args()

    if opts.use_cycle_consistency_loss:
        opts.sample_dir1 = "samples_X-Y_cycle"
        opts.sample_dir2 = "samples_Y-X_cycle"

    if opts.load:
        opts.sample_dir1 = "{}_pretrained".format(opts.sample_dir1)
        opts.sample_dir2 = "{}_pretrained".format(opts.sample_dir2)
        opts.sample_every = 1000

    utils.print_opts(opts)
    main(opts)
