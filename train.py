import pickle
import numpy as np
import xgboost as xgb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def XGBCallback(env):
    tune.track.log(**dict(env.evaluation_result_list))


def get_train_fn(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.10)
    train_set = xgb.DMatrix(train_X, label=train_y)
    val_set = xgb.DMatrix(val_X, label=val_y)

    def train(config):
        model = xgb.cv(config, train_set, nfold=3, early_stopping_rounds=10, metrics=['error', 'auc'])
        preds = model.predict(val_set)
        pred_labels = np.rint(preds)
        tune.track.log(
            mean_accuracy=accuracy_score(val_y, pred_labels),
            done=True)


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
    X, y = np.array(X), np.array(y).ravel()

    # Transform data using the embeddings
    transformed = []
    for x in X:
        src, tgt = tuple(map(int, x))
        x_transformed = np.concatenate([text_emb[src], graph_emb.get(src, np.zeros((dim,))),
                                        text_emb[tgt], graph_emb.get(tgt, np.zeros((dim,)))], axis=None)
        transformed.append(x_transformed)
    X = np.array(transformed)

    print('Begin training...')

    train_fn = get_train_fn(X, y)

    nthread = 4
    config = {
        "verbosity": 2,
        "nthread": nthread,
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "eval_metric": ["auc", "ams@0", "logloss"],
        "max_depth": 3,
        "eta": 0.01,
        "gamma": 1,
        "colsample_bytree": 1,
        "grow_policy": tune.choice(["depthwise", "lossguide"]),
        "num_parallel_tree": tune.choice(1, 10, 100, 1000)
    }

    analysis = tune.run(
                train_fn,
                resources_per_trial={"cpu": nthread},
                config=config,
                num_samples=2,
                scheduler=ASHAScheduler(metric="eval-logloss", mode="min"))

    print("Best config is", analysis.get_best_config(metric="auc"))