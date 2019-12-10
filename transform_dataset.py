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

parser.add_argument('--graph_embeddings_file', default=True,
                    help='name of the file containing the graph embeddings')

parser.add_argument('--text_embeddings_file',
                    help='name of the file containing the text embeddings')


def _get_vector(key, embeddings, vec_shape):
    if key in embeddings:
        return embeddings[key]
    return np.random.normal(size=vec_shape)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load graph embeddings
    graph_emb = load_embeddings(args.graph_embeddings_file, key_transform=int)

    # Load text embeddigns
    if args.text_embeddings_file is not None:
        text_emb = load_embeddings(args.text_embeddings_file, key_transform=int)

    # Transform dataset
    shape = next(iter(text_emb.values())).shape

    X, y = [], []
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.split()
            src, tgt = int(line[0]), int(line[1])

            src_embedding = _get_vector(src, graph_emb, shape)
            tgt_embedding = _get_vector(tgt, graph_emb, shape)
            if args.text_embeddings_file is not None:
                src_embedding = np.concatenate([src_embedding, _get_vector(src, text_emb, shape)], axis=None)
                tgt_embedding = np.concatenate([tgt_embedding, _get_vector(tgt, text_emb, shape)], axis=None)

            # Use the Hadamard operator to combine the two embeddings
            X.append(np.multiply(src_embedding, tgt_embedding))
            if len(line) >= 3:
                y.append(int(line[2]))
    X, y = np.array(X), np.array(y).ravel() if len(y) > 0 else None

    # Save datasets
    save_dataset(args.output_name, X, y)
