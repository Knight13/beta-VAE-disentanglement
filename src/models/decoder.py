from keras.layers import Convolution2D, UpSampling2D, Deconvolution2D, Reshape, Lambda, BatchNormalization, \
    Dense


class DeepMindDecoder:
    def __init__(self, decoder_input, output_shape):
        self._dec_input = decoder_input
        self.__dewconv_func_reps = int(output_shape / 16)

    @staticmethod
    def deconv_func(input_layer):
        x = Deconvolution2D(32, 4, strides=2, padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        return x

    def build(self, encoder_output):
        x = Dense(units=256, activation='linear', name='decoder_inp')(encoder_output)
        x = Dense(units=256, activation='linear')(x)
        x = Reshape((-1, 1, 256), input_shape=(256,))(x)

        x = UpSampling2D((4, 4))(x)

        for _ in range(self.__dewconv_func_reps):
            x = self.deconv_func(x)

        x = Convolution2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        x = BatchNormalization()(x)

        decoded = x
        return decoded

    def create_decoder(self):
        input_dec = Lambda(lambda x: x)(self._dec_input)

        return self.build(encoder_output=input_dec)
