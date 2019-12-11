from util.embeddings_io import load_embeddings, save_embeddings_from_dict

embeddings = load_embeddings('./data/node_information/doc2vec.embeddings')

fixed = dict()

for key, vector in embeddings.items():
    key = int(key.split('*dt_')[1])
    fixed[key] = vector

save_embeddings_from_dict(fixed, './data/node_information/doc2vec_test.embeddings')