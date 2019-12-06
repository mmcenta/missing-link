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
    with open("./data/node_information/reduced_tfidf_emb.pickle", "rb") as f:
        text_emb = pickle.load(f)
    
    with open("./data/node_information/deepwalk.embeddings", "r") as f:
        num_nodes, dim = tuple(map(int, f.readline().split()))

        graph_emb = np.zeros((num_nodes, dim))
        for line in f:
            line = line.split()
            node = int(line[0])
            emb = np.array(list(map(float, line[1:])))
            graph_emb[node, :] = emb
    
    X = np.concatenate(text_emb, graph_emb, axis=1)

    y = []
    with open("./data/labels.txt", "r") as f:
        for line in f:
            y.append(float(line))
    y = np.array(y)

    model = XGBClassifier(n_workers=15)
    rsearch = GridSearchCV(model, params, cv=3, verbose=2)

    rsearch.fit(X, y)

    print('Best Parameters:\n' + str(rsearch.best_params_))
    with open('.models/basic_xbb.pickle', 'wb') as f:
        pickle.dump(f, rsearch.best_estimator_)
