from conifg import enc_model, dec_model
import imageio
import os
import argparse
import sys


def main(args):
    pass


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
                        default=100)

    parser.add_argument('--image_size', type=int,
                        help='Size of the input image. Only (64, 64) is supported.',
                        default=64)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
