from keras.layers import Convolution2D, UpSampling2D, Deconvolution2D, Reshape, Lambda, BatchNormalization, \
    Dense, Activation


class DeepMindDecoder:
    def __init__(self, decoder_input, latent_dim, output_shape):
        self._dec_input = decoder_input
        self._latent_dim = latent_dim
        self.__dewconv_func_reps = int(output_shape / 16)

    @staticmethod
    def deconv_func(input_layer):
        x = Deconvolution2D(32, 4, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build(self, encoder_output):
        x = Reshape((self._latent_dim,), input_shape=(self._latent_dim,))(encoder_output)
        x = Dense(units=256, activation='linear', name='decoder_inp')(x)
        x = Dense(units=256, activation='linear')(x)
        x = Reshape((-1, 1, 256), input_shape=(256,))(x)

        x = UpSampling2D((4, 4))(x)

        for _ in range(self.__dewconv_func_reps):
            x = self.deconv_func(x)

        x = Convolution2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        decoded = x
        return decoded

    def create_decoder(self):
        input_dec = Lambda(lambda x: x)(self._dec_input)

        return self.build(encoder_output=input_dec)
