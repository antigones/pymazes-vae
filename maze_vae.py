import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import os

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a maze."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def read_images(dir):
    working_dir = os.curdir + dir
    out = []
    for filename in os.listdir(working_dir):
        with open(working_dir+filename, 'r'): # open in readonly mode
            img = imageio.imread(working_dir+filename)
            out.append(np.asarray(img))
    return np.asarray(out)


def sample_to_image(pred: np.ndarray):
    o = pred.copy()
    o[o>0.8] = 255
    o[o<=0.8] = 0

    im = Image.fromarray(o)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    imageio.imsave("output.gif", im)


def plot_latent_space(decoder, maze_size=28, n=30, scale=15):
    # display a n*n 2D manifold of mazes
    figure = np.zeros((maze_size * n, maze_size * n))
    # We will sample n points within [-scale, scale] standard deviations
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            maze = x_decoded[0].reshape(maze_size, maze_size)
            figure[i * maze_size: (i + 1) * maze_size,
                j * maze_size: (j + 1) * maze_size] = maze

    plt.figure(figsize=(maze_size, maze_size))
    plt.imshow(figure)
    plt.show()
   

def get_prediction(decoder, maze_size, n=30, scale = 1.0):

    # get a point in latent space, to be decoded
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    xi = np.random.choice(grid_x)
    yi = np.random.choice(grid_y)

    z_sample = np.array([[xi, yi]])
    x_decoded = decoder.predict(z_sample)
    maze = x_decoded[0].reshape(maze_size, maze_size)
    return maze

def main():

    # build the encoder
    
    # size = 28
    size = 36*3 # maze edge * 3
    original_dim = size * size
    intermediate_dim = 64
    latent_dim = 2
    
    inputs = keras.Input(shape=(original_dim))
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # build the decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")
    decoder.summary()

    # train the VAE
    x_train = read_images("\\imgs\\train\\")
    x_test = read_images("\\imgs\\test\\")
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


    # vae = VAE(encoder, decoder)
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')


    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer=keras.optimizers.Adam())

    checkpoint_filepath = os.curdir + '\\checkpoints\\'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True)

    vae.fit(x_train, x_train,  epochs=300, batch_size=128, callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss')])
    vae.load_weights(checkpoint_filepath)
    scale = 100
    plot_latent_space(decoder, maze_size = size, n=8, scale=scale)

    pred = get_prediction(decoder, maze_size=size, n=8, scale=scale)
    print(pred)
    sample_to_image(pred)
    

if __name__ == '__main__':
    main()