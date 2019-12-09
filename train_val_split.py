import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Split a dataset into training and validation set.')

parser.add_argument('--input_file', required=True,
                    help='file containing the dataset to be split')

parser.add_argument('--train_output_file', default='./data/train.txt',
                    help='file that will contain the')

parser.add_argument('--val_output_file', default='./data/val.txt',
                    help='name of the file containing the graph embeddings')

parser.add_argument('--val_size', type=float, default=0.2,
                    help='proportion of the dataset that will be reserved for validation')

parser.add_argument('--random_state', type=int, default=42,
                    help='seed used for the split')


def _save_lines(lines, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading dataset...")

    lines = []
    with open(args.input_file, "r") as f:
        lines = f.readlines()

    print("Done.\nSplitting into train and val then saving...")

    train_lines, val_lines = train_test_split(lines, test_size=args.val_size, random_state=args.random_state)

    _save_lines(train_lines, args.train_output_file)
    _save_lines(val_lines, args.val_output_file)

    print('Done')
