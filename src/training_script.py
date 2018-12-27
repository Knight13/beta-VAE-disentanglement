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


seed(1990)
set_random_seed(1990)


def main(args):
    optimizer = utils.select_optimizer(optimizer=args.optimizer, base_learning_rate=args.base_learning_rate)

    print("Creating data generators.......")
    data_gen = ImageDataGenerator(
        rescale=1 / 255.,
        validation_split=args.val_split)

    train_generator = data_gen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.train_batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = data_gen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.val_batch_size,
        class_mode='categorical',
        subset='validation')

    training_steps = math.ceil(train_generator.n / args.train_batch_size)
    validation_steps = math.ceil(validation_generator.n / args.val_batch_size)

    print('Steps per epoch in training: ', training_steps)
    print('Steps per epoch in validation: ', validation_steps)

    training_generator = utils.batch_gen(generator_object=train_generator)

    validation_generator = utils.batch_gen(generator_object=validation_generator)

    print("Creating callbacks.......")
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=args.decay_factor,
                                    mode='min', patience=args.scheduler_epoch, min_lr=1e-010)

    tb = TensorBoard(log_dir=args.graph_dir, histogram_freq=0, write_graph=True, write_images=True)

    timestamp = time.strftime("%d%m%Y", time.localtime())

    save_file = os.path.join(args.save_dir, str(args.encoder_type + '_vae' + timestamp + '.hdf5'))
    checkpoint = ModelCheckpoint(filepath=save_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    generator_cb = utils.GenerateImage(test_image_folder=args.test_image_folder, target_dir=args.gen_image_dir,
                                       image_shape=args.image_size, encoder_name=args.encoder_type)

    capacity_cb = utils.CapacityIncrease(max_capacity=args.capacity, max_epochs=args.max_epochs)

    if args.pretrained_model is not None:
        print("Loading pre-trained model.......")
        model = load_model(args.pretrained_model, custom_objects={'SampleLayer': sample_layer.SampleLayer})

    else:
        print("Building model.......")

        [enc_input, enc_model] = utils.get_encoder(encoder_name=args.encoder_type, image_size=args.image_size,
                                                   bottleneck=args.bottleneck, vae_gamma=args.vae_gamma)
        z = enc_model.output

        dec = decoder.DeepMindDecoder(decoder_input=z, latent_dim=args.bottleneck, output_shape=args.image_size)
        dec_output = dec.create_decoder()

        model = Model(inputs=[enc_input], outputs=[dec_output])

    print(model.summary())

    arch_pdf = args.encoder_type + '_model.pdf'

    if arch_pdf not in os.listdir(args.arch_dir):
        path = os.path.join(args.arch_dir, arch_pdf)
        plot_model(model, path)

    model.compile(optimizer=optimizer, loss=['mean_squared_error'])

    model.fit_generator(training_generator, steps_per_epoch=training_steps, epochs=args.num_epochs,
                        validation_data=next(validation_generator), validation_steps=validation_steps,
                        callbacks=[tb, checkpoint, generator_cb, capacity_cb, lr_schedule],
                        workers=args.num_workers, use_multiprocessing=args.multi_process)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder_type', type=str, choices=['deepmind_enc'],
                        help='The encoder architecture to use', default='deepmind_enc')

    parser.add_argument('--data_dir', type=str,
                        help='Data directory.',
                        default=None)
    
    parser.add_argument('--graph_dir', type=str,
                        help='The directory to write the training graphs.',
                        default='../graphs/Celeb_A/')

    parser.add_argument('--arch_dir', type=str,
                        help='The directory to write the model architectures in pdf format.',
                        default='architectures/')

    parser.add_argument('--save_dir', type=str,
                        help='The directory to save the trained model.',
                        default='../saved_models/')

    parser.add_argument('--pretrained_model', type=str,
                        help='The directory to load the pre-trained model.',
                        default=None)

    parser.add_argument('--test_image_folder', type=str,
                        help='The directory of the test images.',
                        default='../test_images/')

    parser.add_argument('--gen_image_dir', type=str,
                        help='The directory to save the generated images at the end of each epoch.',
                        default='../generated_images/')

    parser.add_argument('--bottleneck', type=int,
                        help='The size of the embedding layers.',
                        default=32)

    parser.add_argument('--val_split', type=float,
                        help='The percentage of generated_data in the validation set',
                        default=0.2)

    parser.add_argument('--image_size', type=int,
                        help='Size of the input image. Only (64, 64) is supported.',
                        default=64)

    parser.add_argument('--train_batch_size', type=int,
                        help='Batch size for training.',
                        default=64)

    parser.add_argument('--val_batch_size', type=int,
                        help='Batch size for validation.',
                        default=64)

    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'SGD'],
                        help='The optimization algorithm to use', default='ADAM')

    parser.add_argument('--base_learning_rate', type=float,
                        help='The base learning rate for the model.',
                        default=5e-4)

    parser.add_argument('--num_epochs', type=int,
                        help='The total number of epochs for training.',
                        default=200)

    parser.add_argument('--scheduler_epoch', type=int,
                        help='The number of epochs to wait for the val loss to improve.',
                        default=10)

    parser.add_argument('--decay_factor', type=float,
                        help='The learning rate decay factor.',
                        default=0.1)

    parser.add_argument('--vae_gamma', type=float,
                        help='The vae regularizer.',
                        default=1000)

    parser.add_argument('--capacity', type=float,
                        help='The latent space capacity.',
                        default=50.0)

    parser.add_argument('--max_epochs', type=float,
                        help='The maximum epoch to linearly increase the vae capacity.',
                        default=100)

    parser.add_argument('--num_workers', type=float,
                        help='The number of workers to use during training.',
                        default=8)

    parser.add_argument('--multi_process', type=bool,
                        help='Use multi-processing for dit generator during training.',
                        default=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
