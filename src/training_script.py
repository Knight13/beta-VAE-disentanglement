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
    print("Creating callbacks.......")
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=args.decay_factor,
                                    mode='min', patience=args.scheduler_epoch, min_lr=1e-010)

    tb = TensorBoard(log_dir=args.graph_dir, histogram_freq=0, write_graph=True, write_images=True)

    timestamp = time.strftime("%d%m%Y", time.localtime())

    save_file = os.path.join(args.save_dir, str(args.encoder_type + '_vae' + timestamp + '.hdf5'))
    checkpoint = ModelCheckpoint(filepath=save_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # ToDo: Create Custom Callbacks to generate images from test set at the end of each epoch / linearly increase C.
    generator_cb = utils.GenerateImage(test_image_folder=args.test_image_folder, target_dir=args.target_image_dir,
                                       image_shape=args.image_size, encoder_name=args.encoder_type)

    capacity_cb = utils.CapacityIncrease(max_capacity=args.capacity, max_epochs=args.num_epochs)

    # ToDo: Build Model
    if args.pretrained_model is not None:
        print("Loading pre-trained model.......")
        model = load_model(args.pretrained_model, custom_objects={'SampleLayer': sample_layer.SampleLayer})

    else:
        print("Building model.......")

        [enc_input, enc_model] = utils.get_encoder(encoder_name=args.encoder_type, image_size=args.image_size,
                                                   bottleneck=args.bottleneck, vae_gamma=args.vae_gamma)
        z = enc_model.output

        dec = decoder.DeepMindDecoder(decoder_input=z, output_shape=args.image_size)
        dec_output = dec.create_decoder()

        model = Model(inputs=[enc_input], outputs=[dec_output])

    print(model.summary())

    arch_pdf = args.encoder_type + '_model.pdf'

    if arch_pdf not in os.listdir(args.arch_dir):
        path = os.path.join(args.arch_dir, arch_pdf)
        plot_model(model, path)

    model.compile(optimizer=optimizer, loss=['binary_crossentropy'])

    # ToDO: Train model using fit generator
    pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # ToDo: Add args to pass in default arguments

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
