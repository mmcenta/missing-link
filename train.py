import pickle
import argparse
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

from util.io import load_dataset

parser = argparse.ArgumentParser(description='Train a xgboost model.')

parser.add_argument('--train_name', nargs=1, required=True,
                    help='name of the training dataset')

parser.add_argument('--val_name', nargs=1,
                    help='name of the validation dataset')

parser.add_argument('--model_name', nargs=1, default='base',
                    help='name of the model for saving')

parser.add_argument('--n_estimators', nargs=1, type=int, default=100,
                    help='number of trees to fit')

parser.add_argument('--gpu', action='store_true',
                    help='whether to use a gpu when training')

parser.add_argument('--max_depth', nargs=1, type=int, default=3,
                    help='maximum tree depth for base learners')

parser.add_argument('--learning_rate', nargs=1, type=float, default=0.01,
                    help='boosting learning rate')

parser.add_argument('--subsample', nargs=1, type=float, default=0.8,
                    help='subsample ratio of the training instance')

parser.add_argument('--colsample_bytree', nargs=1, type=float, default=0.8,
                    help='subsample ratio of columns when constructing each tree')

parser.add_argument('--gamma', nargs=1, type=float, default=0,
                    help='minimum loss reduction required to make a further partition on a leaf node of the tree')


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading training datataset...")

    X_train, y_train = load_dataset(args.train_name[0])

    print("Done.\nTraining...")

    if not args.gpu:
        model = XGBClassifier(silent=False,
                              scale_pos_weight=1,
                              learning_rate=args.learning_rate[0],
                              colsample_bytree=args.colsample_bytree[0],
                              subsample=args.subsmaple[0],
                              objective='binary:logistic',
                              n_estimators=args.n_estimators[0],
                              reg_alpha=0.3,
                              max_depth=args.max_depth[0],
                              gamma=args.gamma[0])
    else:
        model = XGBClassifier(tree_method="gpu_hist",
                              gpu_id=0,
                              silent=False,
                              scale_pos_weight=1,
                              learning_rate=args.learning_rate[0],
                              colsample_bytree=args.colsample_bytree[0],
                              subsample=args.subsmaple[0],
                              objective='binary:logistic',
                              n_estimators=args.n_estimators[0],
                              reg_alpha=0.3,
                              max_depth=args.max_depth[0],
                              gamma=args.gamma[0])
    model.fit(X_train, y_train)

    print("Done.\nSaving...")

    with open("./models/" + args.model_name[0] + ".pickle", "wb") as f:
        pickle.dump(model, f)

    print("Done.")

    if args.val_name is not None:
        print("Validating...")

        X_val, y_val = load_dataset(args.val_name[0])
        preds = model.predict(X_val)
        pred_labels = np.rint(preds)
        print("\taccuracy:", accuracy_score(y_val, pred_labels))

        print("Done.")
