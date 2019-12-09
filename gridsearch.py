import pickle
import argparse
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from util.io import load_dataset

parser = argparse.ArgumentParser(description='Perform grid search on the hyperparameters given.')

parser.add_argument('--train_name', required=True,
                    help='name of the training dataset')

parser.add_argument('--search_space_file', required=True,
                    help='file specifying the search space')

parser.add_argument('--val_name',
                    help='name of the validation dataset')

parser.add_argument('--model_name', default='base',
                    help='name of the best model for saving')

parser.add_argument('--gpu', action='store_true',
                    help='whether to use a gpu when training')

parser.add_argument('--jobs', type=int, default=1,
                    help='number of jobs to run in parallel')


param_type = {
    "n_estimators": int,
    "max_depth": int,
    "learning_rate": float,
    "colsample_bytree": float,
    "subsample": float,
    "gamma": float
}


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading training datataset...")

    X_train, y_train = load_dataset(args.train_name)

    print("Done.\nLoad search space...")

    search_space = dict()
    with open(args.search_space_file, "r") as f:
        for line in f.readlines():
            line = line.split()

            param = line[0]
            if param not in param_type:
                raise ValueError("Parameter not recognized.")
            options = list(map(param_type[param], line[1:]))
            search_space[param] = options

    print("Done.\nBegin grid search...")

    if not args.gpu:
        model = XGBClassifier(silent=False,
                              scale_pos_weight=1,
                              objective='binary:logistic',
                              reg_alpha=0.3)
    else:
        model = XGBClassifier(tree_method="gpu_hist",
                              gpu_id=0,
                              silent=False,
                              scale_pos_weight=1,
                              objective='binary:logistic',
                              reg_alpha=0.3)

    gsearch = GridSearchCV(model, search_space, scoring='accuracy', n_jobs=args.jobs)
    gsearch.fit(X_train, y_train)

    print("Done.")

    print("\tbest accuracy:", str(gsearch.best_score_))
    print("\tbest params:\n", str(gsearch.best_params_))

    print("Saving best model...")

    with open("./models/" + args.model_name + ".pickle", "wb") as f:
        pickle.dump(gsearch.best_estimator_, f)

    print("Done.")

    if args.val_name is not None:
        print("Validating...")

        X_val, y_val = load_dataset(args.val_name)
        preds = model.predict(X_val)
        pred_labels = np.rint(preds)
        print("\taccuracy:", accuracy_score(y_val, pred_labels))

        print("Done.")
