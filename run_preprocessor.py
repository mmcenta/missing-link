import numpy as np
from preprocessor.preprocessor import Preprocessor

if __name__ == "__main__":
    # p = Preprocessor()
    # p.preprocess()

    print("Generating edge list...")

    labels, edges = [], []
    with open("./data/training.txt", "r") as f:
        for line in f:
            line = line.split()
            labels.append(line[2])
            if line[2] == '1':
                edges.append([line[0], line[1]])
    
    with open("./data/edges.txt", "w") as f:
        for e in edges:
            f.write(e[0] + ' ' + e[1] + '\n')
    
    with open("./data/labels.txt", "w") as f:
        for l in labels:
            f.write(l + '\n')

    print("Finished generating edge list.")
