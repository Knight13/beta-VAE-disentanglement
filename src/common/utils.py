from __future__ import absolute_import
import os

from src.models import encoder
import numpy as np
from PIL import Image
from keras import optimizers
from keras.callbacks import Callback


def batch_gen(generator_object):
    while True:
        data = generator_object.next()
        yield [data[0]], [data[0]]


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
    def __init__(self, test_image_folder, target_dir, action, image_shape, encoder_name):
        self.test = []
        self.gen_images = []
        self.target_dir = target_dir
        self.action = action
        self.encoder_name = encoder_name
        for image in os.listdir(test_image_folder):
            image = Image.open(os.path.join(test_image_folder, image))
            image = image.resize((image_shape, image_shape), Image.NEAREST)
            image = np.asarray(image) / 255.
            image = pre_process(image, self.action)
            self.test.append(image)
        super(Callback).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.gen_images = []

    def on_epoch_end(self, epoch, logs=None):
        for image_data in self.test:
            gen_image = self.model.predict(np.array([image_data]))
            if self.action == 'simple':
                gen_image[gen_image > 0.5] = 0.5
                gen_image[gen_image < -0.5] = -0.5
                gen_image = np.uint8((gen_image + 0.5) * 255)
                gen_image = gen_image[0]

            elif self.action == 'mean_centered':
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
