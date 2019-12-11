from util.embeddings_io import load_embeddings, save_embeddings_from_dict

embeddings = load_embeddings('./data/node_information/doc2vec.embeddings')

for key in embeddings.keys():
    x = embeddings[key]
    del embeddings[key]
    embeddings[int(key[-1])] = x

save_embeddings_from_dict(embeddings, './data/node_information/doc2vec_test.embeddings')