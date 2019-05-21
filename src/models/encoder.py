from __future__ import absolute_import

import warnings
from math import log2

from keras.layers import Input, BatchNormalization, Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.models import Model

from src.common import sample_layer

warnings.filterwarnings('ignore')


class DeepMindEncoder:
    def __init__(self, input_shape=64, latent_dim=10):
        self._input_shape = input_shape
        self._z_dim = latent_dim
        self.__conv_func_reps = int(log2(input_shape/4))

    @staticmethod
    def conv_func(input_layer):
        x = Convolution2D(filters=32, kernel_size=4, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build(self, vae_gamma, vae_capacity):
        init = x = Input((self._input_shape, self._input_shape, 3))

        for _ in range(self.__conv_func_reps):
            x = self.conv_func(x)

        x = Flatten()(x)

        x = Dense(units=256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(units=256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(units=self._z_dim * 2, activation='linear')(x)
        # Beta-VAE implementation as follows

        z_mean = Dense(units=self._z_dim, activation='linear', name='sample_mean')(x)
        z_log_var = Dense(units=self._z_dim, activation='linear', name='sample_log_var')(x)
        embed_layer = sample_layer.SampleLayer(gamma=vae_gamma, capacity=vae_capacity, name='sampling_layer')([z_mean,
                                                                                                               z_log_var])
        return init, embed_layer

    def create_encoder(self, vae_gamma, vae_capacity):
        init, embed_layer = self.build(vae_gamma=vae_gamma, vae_capacity=vae_capacity)
        model = Model(inputs=[init], outputs=[embed_layer])
        return init, model
