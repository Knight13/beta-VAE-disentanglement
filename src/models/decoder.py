from math import log2
from keras.layers import Convolution2D, UpSampling2D, Deconvolution2D, Reshape, Lambda, BatchNormalization, \
    Dense, Activation


class DeepMindDecoder:
    def __init__(self, decoder_input, latent_dim, output_shape):
        self._dec_input = decoder_input
        self._latent_dim = latent_dim
        self.__deconv_func_reps = int(log2(output_shape/4))

    @staticmethod
    def deconv_func(input_layer, pad_mode='same'):
        x = Deconvolution2D(32, 4, strides=2, padding=pad_mode)(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build(self, encoder_output):
        x = Dense(units=256)(encoder_output)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(units=256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(units=512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Reshape((-1, 1, 512), input_shape=(512,))(x)

        for i in range(self.__deconv_func_reps):
            if i == 0:
                x = self.deconv_func(x, pad_mode='valid')
            else:
                x = self.deconv_func(x)
        x = Deconvolution2D(3, 4, strides=2, padding='same', activation='tanh')(x)
        decoded = x
        return decoded

    def create_decoder(self):
        input_dec = Lambda(lambda x: x, name='decoder_inp')(self._dec_input)

        return self.build(encoder_output=input_dec)
