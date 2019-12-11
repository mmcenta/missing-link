import os
import numpy as np

def save_dataset(name, X, y=None):
    os.makedirs('/home/matheuscenta/missing-link/data/datasets', exist_ok=True)
    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".npy", "wb") as f:
        np.save(f, X)
    if y is not None:
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".npy", "wb") as f:
            np.save(f, y)


def load_dataset(name):
    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".npy", "rb") as f:
        X = np.load(f)

    y = None
    if os.path.isfile("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".npy"):
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".npy", "rb") as f:
            y = np.load(f)

    return X, y
