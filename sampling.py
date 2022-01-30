import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):

    """
    This is the reparametrization trick: inputs data points are mapped to a normal multivariate distribution
    Uses (z_mean, z_log_var) to sample z, the vector encoding a maze
    with formula: z = z_mean + sigma * epsilon
    where sigma = exp(z_log_var/2)
    epsilon is a random variable, with a very low value (adds stochasticity to the system)
    z_mean, z_log_var are learneable parameters
    https://towardsdatascience.com/reparameterization-trick-126062cfd3c3
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

