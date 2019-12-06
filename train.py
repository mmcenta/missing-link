import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, GridSearchCV

from feature_extractors.text import TextFeatureExtractor
from feature_extractors.graph import GraphFeatureExtractor


transformers = [
    ('graph_feats', GraphFeatureExtractor()),
    ('text_feats', TextFeatureExtractor())
]
    
pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformers, n_jobs=4)),
    ('classifier', XGBClassifier(n_jobs=4))
])
    
params = {
    'classifier__max_depth': [2, 3, 5, 7, 10],
    'classifier__num_estimators': [10, 100, 500, 700, 1000],
    'classifier__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 1.0],
    'classifier__reg_alpha': [0, 1.0]
}


if __name__ == "__main__":
    X, y = [], []
    with open("./data/training.txt", "r") as f:
        for line in f:
            line = line.split()
            X.append([int(line[0]), int(line[1])])
            y.append(int(line[2]))
    X = np.array(X)
    y = np.array(y)

    rsearch = GridSearchCV(pipeline, params, cv=5, verbose=2)

    rsearch.fit(X, y)
