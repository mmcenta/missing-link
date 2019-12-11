import os
import csv
import numpy as np
from tqdm import trange

def save_dataset(name, X, y=None):
    os.makedirs('/home/matheuscenta/missing-link/data/datasets', exist_ok=True)

    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".data", "w") as f:
        out = csv.writer(f)
        for i in trange(X.shape[0]):
            out.writerow(X[i])

    if y is not None:
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".data", "w") as f:
            out = csv.writer(f)
            for i in trange(y.shape[0]):
               out.writerow(y[i])


def load_dataset(name):
    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".data", "r") as f:
        f.readline()
        X = []
        for line in f:
            X.append(list(map(float, line.split())))
        X = np.array(X)

    y = None
    if os.path.isfile("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".data"):
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".data", "r") as f:
            f.readline()
            y = []
            for line in f:
                y.append(list(map(int, line.split())))
            y = np.array(y)

    return X, y
