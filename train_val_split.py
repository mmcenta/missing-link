import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Load training data...")
    
    lines = []
    with open("./data/training.txt", "r") as f:
        lines = f.readlines()

    print("Finished loading data.")

    print("Performing train-val split and saving...")
    train_lines, val_lines = train_test_split(lines, test_size=0.2)

    with open("./data/train.txt", "w") as f:
        f.writelines(train_lines)

    with open("./data/val.txt", "w") as f:
        f.writelines(val_lines)
