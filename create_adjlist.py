import os
import numpy as np
from sklearn.model_selection import train_test_split


def _count_files(dir_path):
    return len([d for d in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, d))])


if __name__ == "__main__":
    print("Generating adjacency list...")

    num_nodes = _count_files("./data/node_information/text")

    adjlist = [list() for _ in range(num_nodes)]
    with open("./data/train.txt", "r") as f:
        for line in f:
            line = line.split()
            if line[2] == '1':
                src, tgt = int(line[0]), int(line[1])
                adjlist[src].append(str(tgt))
            
    
    with open("./data/adjlist.txt", "w") as f:
        for node, adj in enumerate(adjlist):
            line = " ".join([str(node)] + adjlist[node])
            f.write(line + '\n')
    
    print("Finished generating adjacency list.")    