import os
import pickle
import numpy as np


def save_embeddings_from_array(embeddings, filepath, header=True):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        if header:
            num_nodes, dim = embeddings.shape
            f.write(str(num_nodes) + " " + str(dim) + '\n')
        for idx, vector in enumerate(embeddings):
            f.write(" ".join([str(idx)] + list(map(str, vector))) + '\n')


def save_embeddings_from_dict(embeddings, filepath, header=True):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        if header:
            num_nodes, dim = len(embeddings.keys()), len(next(iter(embeddings.values())))
            f.write(str(num_nodes) + " " + str(dim) + '\n')
        for key, vector in embeddings.items():
            f.write(" ".join([str(key)] + list(map(str, vector))) + '\n')


def load_embeddings(filepath, key_transform=None, vector_transform=float, header=True):
    embeddings = dict()
    with open(filepath, "r") as f:
        if header:
            num_nodes, dim = tuple(map(int, f.readline().split()))
        for line in f:
            line = line.split()
            if len(line) < dim:
                print(len(line), dim)
            key = line[0] if key_transform is None else key_transform(line[0])
            vector = np.array(list(map(vector_transform, line[1:])))
            embeddings[key] = vector
    return embeddings
