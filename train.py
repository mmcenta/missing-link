import pickle
import numpy as np
import xgboost as xgb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def XGBCallback(env):
    tune.track.log(**dict(env.evaluation_result_list))


def load_dataset():
    with open("/home/matheuscenta/missing-link/data/X_train.pickle", "rb") as f:
        X = pickle.load(f)
    with open("/home/matheuscenta/missing-link/data/y_train.pickle", "rb") as f:
        y = pickle.load(f)
    return X, y

if __name__ == "__main__":
    param = {
        "verbosity": 1,
        "nthread": nthread,
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "eval_metric": ["auc", "ams@0", "logloss"],
        "max_depth": 3,
        "eta": 0.01,
        "gamma": 1,
        "colsample_bytree": 1,
        "grow_policy": "depthwise", # "lossguide",
        "num_parallel_tree": 10,
    }
    X, y = load_dataset()

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.10)
    train_set = xgb.DMatrix(train_X, label=train_y)
    val_set = xgb.DMatrix(val_X, label=val_y)

    model = xgb.train(param, train_set, early_stopping_rounds=10, evals=[(val_set, "eval")], callbacks=[XGBCallback])
    preds = model.predict(val_set)
    pred_labels = np.rint(preds)
    print("accuracy:", accuracy_score(val_y, pred_labels))


    