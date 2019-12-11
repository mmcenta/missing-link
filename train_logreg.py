import pickle
import argparse
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from util.dataset_io import load_dataset

parser = argparse.ArgumentParser(description='Train a xgboost model.')

parser.add_argument('--train_name', required=True,
                    help='name of the training dataset')

parser.add_argument('--val_name',
                    help='name of the validation dataset')

parser.add_argument('--model_name', default='base',
                    help='name of the model for saving')

parser.add_argument('--C', type=float, default=1.0,
                    help='inverse regularization strength')


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading training datataset...")

    X_train, y_train = load_dataset(args.train_name)

    print("Done.\nTraining...")


    model = LogisticRegression(verbose=1, C=args.C, n_jobs=4, solver='saga')
    model.fit(X_train, y_train)

    print("Done.\nSaving...")

    with open("./models/" + args.model_name + ".pickle", "wb") as f:
        pickle.dump(model, f)

    print("Done.")

    if args.val_name is not None:
        print("Validating...")

        X_val, y_val = load_dataset(args.val_name)
        preds = model.predict(X_val)
        pred_labels = np.rint(preds)
        print("\taccuracy:", accuracy_score(y_val, pred_labels))

        print("Done.")
