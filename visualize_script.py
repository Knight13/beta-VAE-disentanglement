import argparse
import sys
import os

import imageio
import numpy as np
from PIL import Image

from conifg import enc_model, dec_model
from src.common import utils


def main(args):
    save_folder = os.path.join(args.save_dir, args.test_image.split('.')[0].split('/')[-1])

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    test_image = Image.open(args.test_image)
    enc_inp = test_image.resize((args.image_size, args.image_size), Image.NEAREST)
    enc_inp = np.array([np.asarray(enc_inp) / 255.])
    enc_inp = utils.pre_process_input(enc_inp)

    enc_out = enc_model.predict(enc_inp)

    traversal_range = np.arange(args.start_range, args.end_range, 1 / float(args.traversal_steps))

    dec_inp = list(enc_out.ravel())

    print('Generating phase....')
    for emb_dim in range(len(dec_inp)):
        print('Traversing embedding dimension: {}'.format(emb_dim + 1))
        images = []
        inp = dec_inp
        for val in list(traversal_range):
            inp[emb_dim] = val
            inp_arr = np.array(inp)
            inp_arr = inp_arr.reshape((-1, len(dec_inp)))
            dec_out = dec_model.predict(inp_arr)
            dec_out = utils.post_process_output(dec_out)
            dec_out = dec_out[0]
            images.append(dec_out)
        name = 'z_' + str(emb_dim + 1) + '.gif'
        full_name_path = os.path.join(save_folder, name)
        imageio.mimsave(full_name_path, images)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_image', type=str,
                        help='The image to produce the traversals on.',
                        default=None)

    parser.add_argument('--start_range', type=int,
                        help='The starting point of the traversal range.',
                        default=-3)

    parser.add_argument('--end_range', type=int,
                        help='The ending point of the traversal range.',
                        default=3)

    parser.add_argument('--traversal_steps', type=float,
                        help='The number of steps in the traversal range.',
                        default=5)

    parser.add_argument('--image_size', type=int,
                        help='Size of the input image. Only (64, 64) is supported.',
                        default=64)

    parser.add_argument('--save_dir', type=str,
                        help='The directory to save the generated gifs.',
                        default='generated_gifs')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
