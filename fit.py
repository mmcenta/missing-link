import csv
import pickle
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from feature_extractors.text import TextFeatureExtractor
from feature_extractors.graph import GraphFeatureExtractor


transformers = [
    ('graph_feats', GraphFeatureExtractor()),
    # ('text_feats', TextFeatureExtractor())
]
    
pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformers, n_jobs=4)),
    ('classifier', XGBClassifier(n_jobs=4, tree_method='gpu_hist'))
])


if __name__ == "__main__":
    X, y = [], []
    with open("./data/training.txt", "r") as f:
        for line in f:
            line = line.split()
            X.append([int(line[0]), int(line[1])])
            y.append(int(line[2]))
    X = np.array(X)
    y = np.array(y)

    pipeline.fit(X, y)
    print(pipeline.score(X, y))

    X_test = []
    with open("./data/testing.txt") as f:
        for line in f:
            line = line.split()
            X_test.append([int(line[0]), int(line[1])])
    predictions = enumerate(pipeline.predict(X_test))

    with open("first_predictions.csv", "w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(['id','predicted'])
        for row in predictions:
            csv_out.writerow(row) 

    with open('.models/basic_xbb.pickle', 'wb') as f:
        pickle.dump(f, pipeline)
