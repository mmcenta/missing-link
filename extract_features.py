import csv
import pickle
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from feature_extractors.text import TextFeatureExtractor
from feature_extractors.graph import GraphFeatureExtractor


transformers = [
    ('graph_feats', GraphFeatureExtractor()),
    ('text_feats', TextFeatureExtractor())
]
    
pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformers, n_jobs=4))
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

    X = pipeline.fit_transform(X, y)

    with open("./data/X_train.txt", "w") as f:
        for x in X:
            line = " ".join(list(map(str, x))) + "\n"
            f.write(line)

    with open("./data/y_train", "w") as f:
        for e in y:
            f.write(str(e) + "\n")
