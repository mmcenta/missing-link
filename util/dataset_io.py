import os
import pickle
import numpy as np


def save_dataset(name, X, y=None):
    os.makedirs('/home/matheuscenta/missing-link/data/datasets', exist_ok=True)
    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".pickle", "wb") as f:
        pickle.dump(X, f)
    if y is not None:
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".pickle", "wb") as f:
            pickle.dump(y, f)

def load_dataset(name):
    with open("/home/matheuscenta/missing-link/data/datasets/X_" + name + ".pickle", "rb") as f:
        X = pickle.load(f)
    y = None
    if os.path.isfile("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".pickle"):
        with open("/home/matheuscenta/missing-link/data/datasets/y_" + name + ".pickle", "rb") as f:
            y = pickle.load(f)
    return X, y
