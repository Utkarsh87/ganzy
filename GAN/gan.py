# IMPORTS
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam

import os
if not os.path.exists('images'):
    os.makedirs('images')

from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
plt.style.use('fivethirtyeight')
plt.rc('grid', color='k', linestyle='--')
plt.rc('xtick', direction='out', color='black')
plt.rc('ytick', direction='out', color='black')
plt.rc('axes', facecolor='#E6E6E6', edgecolor='gray', axisbelow=True, grid=True)



# NETWORK ARGUEMENTS/HYPERPARAMETERS
class args():
  lr = 0.0002
  beta_1 = 0.5
  image_shape = (28, 28, 1)
  latent_dim = 100
  nodes = 256 # base number of neurons in a dense layer

  epochs = 45000
  batch_size = 128
  sample_interval = 300 # interval after which generated images are saved



# GAN 
class GAN():
  def __init__(self):
    self.img_shape = args.image_shape
    self.latent_dim = args.latent_dim

    optimizer = Adam(args.lr, args.beta_1)

    # Build and compile the discriminator
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

    # Build the generator
    self.generator = self.build_generator()

    # The generator takes noise as input and generates images
    z = Input(shape=(self.latent_dim,)) # white_noise
    img = self.generator(z)

    # For the combined model we will only train the generator
    self.discriminator.trainable = False

    verdict = self.discriminator(img)

    # combined model (stack the generator and discriminator)
    self.combined = Model(z, verdict)
    self.combined.compile(loss='binary_crossentropy',
                          optimizer=optimizer)



  def build_generator(self):
    '''
    Generator:
    inputs: white noise image
    outputs: generated image
    '''

    model = Sequential()

    model.add(Dense(args.nodes, input_dim=self.latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(2*args.nodes))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(4*args.nodes))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod(self.img_shape), activation='tanh'))
    model.add(Reshape(self.img_shape))

    print()
    print(30*"#" + " Generator " + 30*"#")
    model.summary()
    print(70*"#")
    print()

    noise = Input(shape=(self.latent_dim,))
    img = model(noise)

    return Model(noise, img) # generate image from noise


  def build_discriminator(self):
    '''
    Discriminator: 
    inputs: generated images
    outputs: verdict on the image(real/fake)
    '''

    model = Sequential()

    model.add(Flatten(input_shape=self.img_shape)) # take image as input to dense layer

    model.add(Dense(2*args.nodes))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(args.nodes))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(1, activation='sigmoid')) # one output neuron to classify as real/fake

    print()
    print(30*"#" + " Discriminator " + 30*"#")
    model.summary()
    print(70*"#")
    print()

    img = Input(shape=self.img_shape)
    verdict = model(img)

    return Model(img, verdict) # classify generated image as real/fake


  def train(self, epochs, batch_size=128, sample_interval=50):

    # Load the dataset
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # Rescale -1 to 1
    x_train = (x_train/127.5) - 1.
    x_train = np.expand_dims(x_train, axis=3)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # lists to store discriminator loss and accuracy and generator loss
    dis_loss = []
    dis_acc = []
    gen_loss = []

    for epoch in range(1, epochs+1):

      # ---------------------
      #  Train Discriminator
      # ---------------------

      # Select a random batch of images
      index = np.random.randint(0, x_train.shape[0], batch_size)
      images = x_train[index]

      noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

      # Generate a batch of new images from the generator
      gen_images = self.generator.predict(noise)

      # loss update
      # train_on_batch(data, target)
      d_loss_real = self.discriminator.train_on_batch(images, valid)
      # for real loss: data is the image from the dataset, and all those
      # images are valid, hence the target is all 1s

      d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
      # for fake loss: data is the image from the generator, and all those
      # images are fake, hence the target is all 0s

      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      # ---------------------
      #  Train Generator
      # ---------------------

      noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

      # Train the generator (to have the discriminator label samples as valid)
      g_loss = self.combined.train_on_batch(noise, valid)

      # Print the progress every 300 epochs
      if (epoch % 300 == 0):
        print ("[Epoch: %d] [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        # store discriminator and generator loss and accuracy metrics
        dis_loss.append(d_loss[0])
        dis_acc.append(100*d_loss[1])
        gen_loss.append(g_loss)

      # If at save interval => save generated image samples
      if (epoch % sample_interval == 0):
        self.sample_images(epoch)
    
    # plot the loss and accuracy of the discriminator and generator networks
    epochs = range(1, args.epochs+1)
    
    plt.plot(epochs, gen_loss, "-b", label="Generator Loss")
    plt.plot(epochs, dis_loss, "-r", label="Discriminator Loss")
    plt.legend(loc="upper left")
    plt.title("Generator and Discriminator Loss")
    plt.savefig("images/loss.png")
    plt.close()

    plt.plot(epochs, dis_acc)
    plt.title("Discriminator Accuracy")
    plt.savefig("images/acc.png")
    plt.close()


  def sample_images(self, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    gen_images = self.generator.predict(noise)

    # Rescale images 0 - 1
    gen_images = 0.5 * gen_images + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
      for j in range(c):
        axs[i,j].imshow(gen_images[cnt, :,:,0], cmap='gray')
        axs[i,j].axis('off')
        cnt += 1
    fig.savefig(f'images/{epoch}.png')
    plt.close()


if __name__ == '__main__':
  gan = GAN()
  gan.train(epochs=args.epochs, batch_size=args.batch_size, sample_interval=args.sample_interval)