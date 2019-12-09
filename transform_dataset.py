import pickle
import argparse
import numpy as np
from util.dataset_io import save_dataset

parser = argparse.ArgumentParser(description='Transform the original dataset to use the precalculated embeddings.')

parser.add_argument('--input_file', nargs=1, required=True,
                    help='file containing the dataset to be transformed')

parser.add_argument('--output_name', nargs=1, required=True,
                    help='name of the transformed dataset')

parser.add_argument('--text_embeddings_file', nargs=1, default='./data/node_information/reduced_tfidf_emb.pickle',
                    help='name of the file containing the text embeddings')

parser.add_argument('--graph_embeddings_file', nargs=1, default='./data/node_information/train_deepwalk.embeddings',
                    help='name of the file containing the graph embeddings')


if __name__ == "__main__":
    args = parser.parse_args()

    # Load text embeddigns
    print(args.text_embeddings_file, args.graph_embeddings_file, args.input_file, args.output_name)
    with open(args.text_embeddings_file[0], "rb") as f:
        text_emb = pickle.load(f)

    # Load graph embeddings
    with open(args.graph_embeddings_file[0], "r") as f:
        num_nodes, dim = tuple(map(int, f.readline().split()))
        graph_emb = dict()
        for line in f:
            line = line.split()
            node = int(line[0])
            emb = np.array(list(map(float, line[1:])))
            graph_emb[node] = emb

    # Transform dataset
    X, y = [], []
    with open(args.input_file[0], "r") as f:
        for line in f:
            line = line.split()
            src, tgt = int(line[0]), int(line[1])
            X.append(np.concatenate([text_emb[src], graph_emb.get(src, np.zeros((dim,))),
                                     text_emb[tgt], graph_emb.get(tgt, np.zeros((dim,)))], axis=None))
            if len(line) >= 3:
                y.append(int(line[2]))
    X, y = np.array(X), np.array(y).ravel() if len(y) > 0 else None

    # Save datasets
    save_dataset(args.output_name[0], X, y)
