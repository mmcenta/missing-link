import pickle
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_dataset():
    with open("/home/matheuscenta/missing-link/data/X_train.pickle", "rb") as f:
        X = pickle.load(f)
    with open("/home/matheuscenta/missing-link/data/y_train.pickle", "rb") as f:
        y = pickle.load(f)
    return X, y

if __name__ == "__main__":
    param = {
        "verbosity": 1,
        "nthread": 16,
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "eval_metric": ["error"],
        "max_depth": 3,
        "eta": 0.01,
        "gamma": 1,
        "colsample_bytree": 1,
        "grow_policy": "depthwise", # "lossguide",
        "num_parallel_tree": 100,
    }
    X, y = load_dataset()

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.10)

    model = XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.01,  
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=100, 
                          reg_alpha = 0.3,
                          max_depth=4, 
                          gamma=10)
    model.fit(X, y)

    preds = model.predict(val_X)
    pred_labels = np.rint(preds)
    print("accuracy:", accuracy_score(val_y, pred_labels))

    with open("./models/base.pickle", "wb") as f:
        pickle.dump(model, f)
