import pickle
import argparse
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

from util.dataset_io import load_dataset

parser = argparse.ArgumentParser(description='Train a xgboost model.')

parser.add_argument('--train_name', required=True,
                    help='name of the training dataset')

parser.add_argument('--val_name',
                    help='name of the validation dataset')

parser.add_argument('--model_name', default='base',
                    help='name of the model for saving')

parser.add_argument('--n_estimators', type=int, default=100,
                    help='number of trees to fit')

parser.add_argument('--gpu', action='store_true',
                    help='whether to use a gpu when training')

parser.add_argument('--max_depth', type=int, default=3,
                    help='maximum tree depth for base learners')

parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='boosting learning rate')

parser.add_argument('--subsample', type=float, default=0.8,
                    help='subsample ratio of the training instance')

parser.add_argument('--colsample_bytree', type=float, default=0.8,
                    help='subsample ratio of columns when constructing each tree')

parser.add_argument('--gamma', type=float, default=0,
                    help='minimum loss reduction required to make a further partition on a leaf node of the tree')

parser.add_argument('--scale_pos_weight', type=float, default=1.0)

parser.add_argument('--min_child_weight', type=int, default=1)

parser.add_argument('--reg_alpha', type=float, default=0.0)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading training datataset...")

    X_train, y_train = load_dataset(args.train_name)

    print("Done.\nTraining...")

    if not args.gpu:
        model = XGBClassifier(silent=False,
                              scale_pos_weight=args.scale_pos_weight,
                              min_child_weight=args.min_child_weight,
                              learning_rate=args.learning_rate,
                              colsample_bytree=args.colsample_bytree,
                              subsample=args.subsample,
                              objective='binary:logistic',
                              n_estimators=args.n_estimators,
                              reg_alpha=args.reg_alpha,
                              max_depth=args.max_depth,
                              gamma=args.gamma)
    else:
        model = XGBClassifier(tree_method="gpu_hist",
                              gpu_id=0,
                              silent=False,
                              scale_pos_weight=args.scale_pos_weight,
                              min_child_weight=args.min_child_weight,
                              learning_rate=args.learning_rate,
                              colsample_bytree=args.colsample_bytree,
                              subsample=args.subsample,
                              objective='binary:logistic',
                              n_estimators=args.n_estimators,
                              reg_alpha=args.reg_alpha,
                              max_depth=args.max_depth,
                              gamma=args.gamma)
    model.fit(X_train, y_train)

    print("Done.\nSaving...")

    with open("./models/" + args.model_name + ".pickle", "wb") as f:
        pickle.dump(model, f)

    print("Done.")

    if args.val_name is not None:
        print("Validating...")

        preds = model.predict(X_train)
        pred_labels = np.rint(preds)
        print("\ttrain accuracy:", accuracy_score(y_train, pred_labels))

        X_val, y_val = load_dataset(args.val_name)
        preds = model.predict(X_val)
        pred_labels = np.rint(preds)
        print("\tval accuracy:", accuracy_score(y_val, pred_labels))

        print("Done.")
