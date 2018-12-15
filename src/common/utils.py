from __future__ import absolute_import

import os

import numpy as np
from PIL import Image
from keras import optimizers
from keras.callbacks import Callback
from keras.models import Model

from src.models import encoder


def pre_process_input(image_array):
    x = np.asarray(image_array)
    y = x - 0.5
    return y


def post_process_output(generated_image):
    generated_image[generated_image > 0.5] = 0.5
    generated_image[generated_image < -0.5] = -0.5
    gen_image = np.uint8((generated_image + 0.5) * 255)
    return gen_image


def batch_gen(generator_object):
    while True:
        data = generator_object.next()
        yield [pre_process_input(data[0])], [pre_process_input(data[0])]


def get_encoder(encoder_name, image_size, bottleneck, vae_gamma):
    if encoder_name == 'deepmind_enc':
        enc = encoder.DeepMindEncoder(input_shape=image_size, latent_dim=bottleneck)
        enc_input, enc_model = enc.create_encoder(vae_gamma=vae_gamma, vae_capacity=0)
    else:
        raise NotImplementedError
    return enc_input, enc_model


def select_optimizer(optimizer, base_learning_rate):
    if optimizer == 'SGD':
        optimizer = optimizers.SGD(lr=base_learning_rate, momentum=0.9)
    elif optimizer == 'RMSPROP':
        optimizer = optimizers.RMSprop(lr=base_learning_rate)
    elif optimizer == 'ADAGRAD':
        optimizer = optimizers.Adagrad(lr=base_learning_rate)
    elif optimizer == 'ADADELTA':
        optimizer = optimizers.Adadelta()
    elif optimizer == 'ADAM':
        optimizer = optimizers.Adam(lr=base_learning_rate)
    else:
        raise ImportError
    return optimizer


class GenerateImage(Callback):
    def __init__(self, test_image_folder, target_dir, image_shape, encoder_name):
        self.test = []
        self.gen_images = []
        self.target_dir = target_dir
        self.encoder_name = encoder_name
        for image in os.listdir(test_image_folder):
            image = Image.open(os.path.join(test_image_folder, image))
            image = image.resize((image_shape, image_shape), Image.NEAREST)
            image = np.asarray(image)
            image = image / 255.
            image = pre_process_input(image)
            self.test.append(image)
        super(Callback).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.gen_images = []

    def on_epoch_end(self, epoch, logs=None):
        for image_data in self.test:
            gen_image = self.model.predict(np.array([image_data]))
            gen_image = post_process_output(gen_image)
            gen_image = gen_image[0]
            self.gen_images.append(Image.fromarray(gen_image))

        folder = "epoch_" + str(epoch)
        epoch_dir = os.path.join(self.target_dir, folder)

        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)
        for idx in range(len(self.gen_images)):
            file_name = self.encoder_name + "_img_" + str(idx) + ".png"
            image = self.gen_images[idx]
            image.save(os.path.join(epoch_dir, file_name))
        del self.gen_images
        print('Generated images saved in ', self.target_dir)


class CapacityIncrease(Callback):
    def __init__(self, max_capacity, max_epochs):
        self.max_capacity = max_capacity
        self.current_capacity = 0
        self.max_epochs = max_epochs
        super(Callback).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.max_epochs:
            self.current_capacity = self.max_capacity * epoch / self.max_epochs
        else:
            self.current_capacity = self.max_capacity
        print("Updated vae capacity param: ", self.current_capacity)
        self.model.get_layer('sampling_layer').max_capacity = self.current_capacity


class SplitModel:
    def __init__(self, parent_model):
        self.__parent = parent_model

    def get_layer_idx_by_name(self, layername):
        for idx, layer in enumerate(self.__parent.layers):
            if layer.name == layername:
                return idx

    def split_model(self, start, end):
        confs = self.__parent.get_config()
        weights = {l.name: l.get_weights() for l in self.__parent.layers}
        # split model
        kept_layers = set()
        for i, l in enumerate(confs['layers']):
            if i == 0:
                confs['layers'][0]['config']['batch_input_shape'] = self.__parent.layers[start].input_shape
            elif i < start or i > end:
                continue
            kept_layers.add(l['name'])
        # filter layers
        layers = [l for l in confs['layers'] if l['name'] in kept_layers]
        layers[1]['inbound_nodes'][0][0][0] = layers[0]['name']
        # set conf
        confs['layers'] = layers
        confs['input_layers'][0][0] = layers[0]['name']
        confs['output_layers'][0][0] = layers[-1]['name']
        # create new model
        newModel = Model.from_config(confs)
        for l in newModel.layers:
            l.set_weights(weights[l.name])
        return newModel
