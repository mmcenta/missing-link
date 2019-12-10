import pickle
import argparse
import numpy as np
from util.dataset_io import save_dataset
from util.embeddings_io import load_embeddings

parser = argparse.ArgumentParser(description='Transform the original dataset to use the precalculated embeddings.')

parser.add_argument('--input_file', required=True,
                    help='file containing the dataset to be transformed')

parser.add_argument('--output_name', required=True,
                    help='name of the transformed dataset')

parser.add_argument('--text_embeddings_file', default='./data/node_information/reduced_tfidf.embeddings',
                    help='name of the file containing the text embeddings')

parser.add_argument('--graph_embeddings_file', default='./data/node_information/train_deepwalk.embeddings',
                    help='name of the file containing the graph embeddings')


if __name__ == "__main__":
    args = parser.parse_args()

    # Load text embeddigns
    text_emb = load_embeddings(args.text_embeddings_file, key_transform=int)

    # Load graph embeddings
    graph_emb = load_embeddings(args.graph_embeddings_file, key_transform=int, header=True)

    # Transform dataset
    X, y = [], []
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.split()
            src, tgt = int(line[0]), int(line[1])
            X.append(np.concatenate([text_emb[src], graph_emb[src],
                                     text_emb[tgt], graph_emb[tgt]], axis=None))
            if len(line) >= 3:
                y.append(int(line[2]))
    X, y = np.array(X), np.array(y).ravel() if len(y) > 0 else None

    # Save datasets
    save_dataset(args.output_name, X, y)
