import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Generate the edge list of the graph represented by a dataset.')

parser.add_argument('--input_file', nargs=1, required=True,
                    help='file containing the dataset')


def _count_files(dir_path):
    return len([d for d in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, d))])


if __name__ == "__main__":
    args = parser.parse_args()

    print("Generating adjacency list...")

    num_nodes = _count_files("./data/node_information/text")

    adjlist = [list() for _ in range(num_nodes)]
    with open(args.input_file[0], "r") as f:
        for line in f:
            line = line.split()
            if line[2] == '1':
                src, tgt = int(line[0]), int(line[1])
                adjlist[src].append(str(tgt))


    with open("./data/adjlist.txt", "w") as f:
        for node, adj in enumerate(adjlist):
            line = " ".join([str(node)] + adjlist[node])
            f.write(line + '\n')

    print("Done.")