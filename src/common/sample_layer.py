from keras.engine import Layer
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects


class SampleLayer(Layer):
    def __init__(self, gamma, capacity, name, **kwargs):
        super(SampleLayer, self).__init__(**kwargs)
        self.gamma = gamma
        self.max_capacity = capacity
        self.name = name

    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)
        self.built = True

    def call(self, layer_inputs, **kwargs):
        if len(layer_inputs) != 2:
            raise Exception('input layers must be a list: mean and stddev')
        if len(K.int_shape(layer_inputs[0])) != 2 or len(K.int_shape(layer_inputs[1])) != 2:
            raise Exception('input shape is not a vector [batchSize, latentSize]')

        mean = layer_inputs[0]
        log_var = layer_inputs[1]

        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]

        latent_loss = -0.5 * (1 + log_var - K.square(mean) - K.exp(log_var))
        latent_loss = K.sum(latent_loss, axis=1, keepdims=True)
        latent_loss = K.mean(latent_loss)
        latent_loss = self.gamma * K.abs(latent_loss - self.max_capacity)

        latent_loss = K.reshape(latent_loss, [1, 1])

        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        layer_output = mean + K.exp(0.5 * log_var) * epsilon

        self.add_loss(losses=[latent_loss], inputs=[layer_inputs])

        return layer_output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'gamma': self.gamma,
            'capacity': self.max_capacity,
            'name': self.name
        }
        base_config = super(SampleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'SampleLayer': SampleLayer})
