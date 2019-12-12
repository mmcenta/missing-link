with open("./data/node_information/doc2vec.embeddings", "r") as f:
    seen = set()
    num_nodes, dim = tuple(map(int, f.readline().split()))
    print(num_nodes, dim)
    for line in f:
        l = len(line.split())
        if l not in seen:
            print(l)
            seen.add(l)