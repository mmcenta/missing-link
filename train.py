import pickle
import argparse
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from util.io import load_dataset

parser = argparse.ArgumentParser(description='Train a xgboost model.')
parser.add_argument('--model_name', nargs=1, default='base',
                    help='the name of the model for saving')
parser.add_argument('--n_estimators', nargs=1, default=100,
                    help='the number of trees to be trained')
parser.add_argument('--gpu', action='store_true',
                    help="wheter to use a gpu when training")

if __name__ == "__main__":
    args = parser.parse_args()

    X_train, y_train = load_dataset("train")
    X_val, y_val = load_dataset("val")

    if not args.gpu:
        model = XGBClassifier(silent=False, 
                              scale_pos_weight=1,
                              learning_rate=0.01,  
                              colsample_bytree=0.4,
                              subsample=0.8,
                              objective='binary:logistic', 
                              n_estimators=args.n_estimators, 
                              reg_alpha=0.3,
                              max_depth=4, 
                              gamma=1)
    else:
        model = XGBClassifier(tree_method="gpu_hist",
                              gpu_id=0,
                              silent=False, 
                              scale_pos_weight=1,
                              learning_rate=0.01,  
                              colsample_bytree=0.4,
                              subsample=0.8,
                              objective='binary:logistic', 
                              n_estimators=args.n_estimators, 
                              reg_alpha=0.3,
                              max_depth=4, 
                              gamma=1)
    print(type(X_train), X_train.shape)
    print(type(y_train), y_train.shape)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    pred_labels = np.rint(preds)
    print("accuracy:", accuracy_score(y_val, pred_labels))

    with open("./models/" + args.model_name + ".pickle", "wb") as f:
        pickle.dump(model, f)
