import pickle
import numpy as np

def load_dataset(name):
    with open("/home/matheuscenta/missing-link/data/X_" + name + ".pickle", "rb") as f:
        X = pickle.load(f)
    with open("/home/matheuscenta/missing-link/data/y_" + name + ".pickle", "rb") as f:
        y = pickle.load(f)
    return X, y

def save_dataset(name, X, y):
    with open("./data/X_" + name + ".pickle", "wb") as f:
        pickle.dump(X, f)
    with open("./data/y_" + name + ".pickle", "wb") as f:
        pickle.dump(y, f)
