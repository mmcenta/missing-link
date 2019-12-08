import pickle
import numpy as np
from util.io import save_dataset

def transform_data(filepath, text_emb, graph_emb):
    X, y = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.split()
            src, tgt = int(line[0]), int(line[1])
            X.append(np.concatenate([text_emb[src], graph_emb.get(src, np.zeros((dim,))),
                                     text_emb[tgt], graph_emb.get(tgt, np.zeros((dim,)))], axis=None))
            y.append(int(line[2]))
    return np.array(X), np.array(y).ravel()


if __name__ == "__main__":
    # Load text embeddigns
    with open("./data/node_information/reduced_tfidf_emb.pickle", "rb") as f:
        text_emb = pickle.load(f)
    
    # Load graph embeddings
    with open("./data/node_information/deepwalk.embeddings", "r") as f:
        num_nodes, dim = tuple(map(int, f.readline().split()))

        graph_emb = dict()
        for line in f:
            line = line.split()
            node = int(line[0])
            emb = np.array(list(map(float, line[1:])))
            graph_emb[node] = emb
    
    # Transform datasets
    X_train, y_train = transform_data("./data/train.txt", text_emb, graph_emb)
    X_val, y_val = transform_data("./data/val.txt", text_emb, graph_emb) 
    X_test, y_test = transform_data("./data/testing.txt", text_emb, graph_emb)

    # Save datasets
    save_dataset("train", X_train, y_train)
    save_dataset("val", X_val, y_val)
    save_dataset("test", X_test, y_test)
