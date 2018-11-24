import argparse
import sys


def main(args):
    # ToDO :  Write a function to select optimizer

    # ToDo: Create data generators (Training/ Validation)

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
