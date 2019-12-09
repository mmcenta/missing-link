import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Split a dataset into training and validation set.')

parser.add_argument('--input_file', nargs=1, required=True,
                    help='file containing the dataset to be split')

parser.add_argument('--train_output_file', nargs=1, default='./data/train.txt',
                    help='file that will contain the')

parser.add_argument('--val_output_file', nargs=1, default='./data/val.txt',
                    help='name of the file containing the graph embeddings')

parser.add_argument('--val_size', nargs=1, type=float, default=0.2,
                    help='proportion of the dataset that will be reserved for validation')

parser.add_argument('--random_state', nargs=1, type=int, default=42,
                    help='seed used for the split')


def _save_lines(lines, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading dataset...")

    lines = []
    with open(args.input_file[0], "r") as f:
        lines = f.readlines()

    print("Done.\nSplitting into train and val then saving...")

    train_lines, val_lines = train_test_split(lines, test_size=args.val_size[0], random_state=args.random_state[0])

    _save_lines(train_lines, args.train_output_file[0])
    _save_lines(val_lines, args.val_output_file[0])

    print('Done')
