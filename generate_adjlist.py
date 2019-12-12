import os
import argparse
import pickle
import numpy as np
from networkx import DiGraph
from gensim.models import KeyedVectors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from tqdm import trange

from util.embeddings_io import load_embeddings


parser = argparse.ArgumentParser(description='Generate the edge list of the graph represented by a dataset.')

parser.add_argument('--input_file', required=True,
                    help='file containing the dataset')

parser.add_argument('--output_file', default='./data/adjlist.txt',
                    help='file that will hold the resulting adjacency list')

parser.add_argument('--embeddings_file',
                    help='file containing the similarity embeddings between nodes')

parser.add_argument('--tfidf_file',
                    help='file containing the tfidf sparse vectors of each document')

parser.add_argument('--num_potential_links', type=int, default=5,
                    help='number of potential links to be added to the graph')


def _count_files(dir_path):
    return len([d for d in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, d))])


if __name__ == "__main__":
    args = parser.parse_args()

    print("Generating adjacency list...")

    num_nodes = _count_files("./data/node_information/text")

    # Create directed graph
    G = DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Read input graph
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.split()
            if line[2] == '1':
                src, tgt = int(line[0]), int(line[1])
                G.add_edge(src, tgt)

    if args.embeddings_file is not None:
        # If a similarity embeddings were provided, link the num_potential_links
        # nearest neighbours to each node

        kv = KeyedVectors.load_word2vec_format(args.embeddings_file)
        for node in trange(num_nodes):
            potential_links = [int(pair[0]) for pair in kv.most_similar(positive=[str(node)],
                                                                        topn=args.num_potential_links)]
            for adj in potential_links:
                G.add_edge(node, adj)

    if args.tfidf_file is not None:
        with open(args.tfidf_file, "rb") as f:
            embeddings = pickle.load(f)
        print(len(embeddings))
        seen = set()
        for x in embeddings:
            if len(x) not in seen:
                print(len(x))
                seen.add(len(x))

        normalize(embeddings, copy=False)
        tree = KDTree(embeddings)
        for node in trange(num_nodes):
            potential_links = tree.query(embeddings[node], k=args.num_potential_links, return_distance=False)
            for adj in potential_links:
                G.add_edge(node, adj)


    # Save the adjacency list
    with open(args.output_file, "w") as f:
        adjlist = G.adjacency()
        for node, adj in adjlist:
            line = " ".join([str(node)] + list(map(str, sorted(adj.keys()))))
            f.write(line + '\n')

    print("Done.")