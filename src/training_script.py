from __future__ import absolute_import
from src.common import utils
from src.models import decoder
from src.common import sample_layer

import sys
import argparse
import os
import math
from tensorflow import set_random_seed
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import time
from numpy.random import seed

def main(args):
    # ToDO :  Write a function to select optimizer
    optimizer = utils.select_optimizer(optimizer=args.optimizer, base_learning_rate=args.base_learning_rate)

    # ToDo: Create data generators (Training/ Validation)
    print("Creating data generators.......")
    data_gen = ImageDataGenerator(
        rescale=1 / 255.,
        validation_split=args.val_split)

    train_generator = data_gen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        color_mode='grayscale',
        batch_size=args.train_batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = data_gen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        color_mode='grayscale',
        batch_size=args.val_batch_size,
        class_mode='categorical',
        subset='validation')

    training_steps = math.ceil(train_generator.n / args.train_batch_size)
    validation_steps = math.ceil(validation_generator.n / args.val_batch_size)

    print('Steps per epoch in training: ', training_steps)
    print('Steps per epoch in validation: ', validation_steps)

    train_generator = utils.batch_gen(generator_object=train_generator)

    validation_generator = utils.batch_gen(generator_object=validation_generator)
    # ToDo: Create Generic Callbacks (Tensorboard/ Modelcheckpoint)

    # ToDo: Create Custom Callbacks to generate images from test set at the end of each epoch / linearly increase C.

    # ToDo: Build Model

    # ToDO: Train model using fit generator
    pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # ToDo: Add args to pass in default arguments

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
