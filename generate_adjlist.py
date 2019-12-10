import os
import argparse
import numpy as np
from heapq import heappush, heappop
from networkx import DiGraph
from sklearn.metrics.pairwise import cosine_similarity

from util.embeddings_io import load_embeddings


parser = argparse.ArgumentParser(description='Generate the edge list of the graph represented by a dataset.')

parser.add_argument('--input_file', required=True,
                    help='file containing the dataset')

parser.add_argument('--embeddings_file',
                    help='file containing the similarity embeddings between nodes')

parser.add_argument('--num_potential_links', type=int, default=3,
                    help='number of potential links to be added to the graph')


def _count_files(dir_path):
    return len([d for d in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, d))])


def _k_nearest_neighbours(k, node, num_nodes, embeddings):
    heap = []

    for adj in range(num_nodes):
        if adj == node:
            continue

        dist = cosine_similarity(embeddings[node], embeddings[adj]).asscalar()
        if len(heap) == 0 or dist < -heap[0][0]:
            heappush(heap, (-dist, adj))
            if len(heap) > k:
                heappop(heap)

    heap.sort()
    return [pair[1] for pair in heap]


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

        embeddings = load_embeddings(args.embeddings_file, key_transform=int)

        for node in range(num_nodes):
            potential_links = _k_nearest_neighbours(args.num_potential_links,
                                                    node, num_nodes, embeddings)
            for adj in potential_links:
                G.add_edge(node, adj)

    # Save the adjacency list
    with open("./data/adjlist.txt", "w") as f:
        adjlist = G.adjacency()
        for node, adj in adjlist:
            line = " ".join([str(node)] + list(sorted(adj.keys())))
            f.write(line + '\n')

    print("Done.")