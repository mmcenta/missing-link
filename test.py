import csv
import pickle
import numpy as np
from xgboost.sklearn import XGBClassifier


if __name__ == "__main__":
    # Load text embeddigns
    with open("./data/node_information/reduced_tfidf_emb.pickle", "rb") as f:
        text_emb = pickle.load(f)
    
    # Load graph (deepwalk) embeddings
    with open("./data/node_information/deepwalk.embeddings", "r") as f:
        num_nodes, dim = tuple(map(int, f.readline().split()))

        graph_emb = dict()
        for line in f:
            line = line.split()
            node = int(line[0])
            emb = np.array(list(map(float, line[1:])))
            graph_emb[node] = emb
    
    # Load testing data
    X= []
    with open("./data/testing.txt", "r") as f:
        for line in f:
            line = line.split()
            X.append([int(line[0]), int(line[1])])
    X= np.array(X)

    # Transform data using the embeddings
    transformed = []
    for x in X:
        src, tgt = tuple(map(int, x))
        x_transformed = np.concatenate([text_emb[src], graph_emb.get(src, np.zeros((dim,))),
                                        text_emb[tgt], graph_emb.get(tgt, np.zeros((dim,)))], axis=None)
        transformed.append(x_transformed)
    X = np.array(transformed)

    # Load model
    with open('./models/base.pickle', 'rb') as f:
        model = pickle.load(f)

    # Predict
    predictions = model.predict(X)

    predictions = enumerate(predictions)
    # Write the output in the format required by Kaggle
    with open("xgb_predictions.csv", "w") as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id','predicted'])
        for idx, label in predictions:
            csv_out.writerow([idx, int(label)]) 
