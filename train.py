import pickle
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV


params = {
    'max_depth': [2, 3, 5],
    'num_estimators': [10, 100, 500],
    'learning_rate': [0.01, 0.1, 0.3, 1.0],
    'reg_alpha': [0, 1.0]
}


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
    
    # Load training data
    X, y = [], []
    with open("./data/training.txt", "r") as f:
        for line in f:
            line = line.split()
            X.append([int(line[0]), int(line[1])])
            y.append([int(line[2])])
    X, y = np.array(X), np.array(y)

    # Transform data using the embeddings
    transformed = []
    for x in X:
        src, tgt = tuple(map(int, x))
        x_transformed = np.concatenate([text_emb[src], graph_emb[src],
                                        text_emb[src], graph_emb[src]], axis=None)
        transformed.append(x_transformed)
    X = np.array(transformed)

    print('Begin training...')

    model = XGBClassifier(n_workers=16)
    rsearch = GridSearchCV(model, params, cv=3, verbose=2)

    rsearch.fit(X, y)

    print('Best Parameters:\n' + str(rsearch.best_params_))
    with open('.models/basic_xbb.pickle', 'wb') as f:
        pickle.dump(f, rsearch.best_estimator_)
