import os
import tqdm
import numpy as np


def save_dataset(name, X, y=None):
    os.makedirs('/home/matheuscenta/missing-link/data/datasets', exist_ok=True)

    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".data", "w") as f:
        f.write(str(X.shape[0]) + " " + str(X.shape[1]) + "\n")
        for i in tqdm(range(len(X.shape[0]))):
            f.write(" ".join(list(map(str, X[i]))) + "\n")

    if y is not None:
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".data", "wb") as f:
           f.write(str(y.shape[0]) + '\n')
           for i in tqdm(range(len(y.shape[0]))):
               f.write(" ".join(list(map(str, y[i]))) + "\n")


def load_dataset(name):
    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".data", "rb") as f:
        f.readline()
        X = []
        for line in f:
            X.append(list(map(float, line.split())))
        X = np.array(X)

    y = None
    if os.path.isfile("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".data"):
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".data", "rb") as f:
            f.readline()
            y = []
            for line in f:
                y.append(list(map(int, line.split())))
            y = np.array(y)

    return X, y
