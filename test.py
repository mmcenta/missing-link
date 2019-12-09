import csv
import pickle
import argparse
import numpy as np
from xgboost.sklearn import XGBClassifier

from util.dataset_io import load_dataset


parser = argparse.ArgumentParser(description='Output predictions of the given model to a CSV file.')

parser.add_argument('--model_file', args=1, required=True,
                    help='file containing the model to make predictions')

parser.add_argument('--input_name', nargs=1, required=True,
                    help='name of the test dataset')

parser.add_argument('--output', nargs=1, default='submission.csv',
                    help='')


if __name__ == "__main__":
    args = parser.parse_args()

    # Load testing data
    X, _ = load_dataset(args.input_name[0])

    # Load model
    with open(args.model_file[0], 'rb') as f:
        model = pickle.load(f)

    # Predict
    predictions = model.predict(X)

    # Write the output in the format required by Kaggle
    predictions = enumerate(predictions)
    with open("xgb_predictions.csv", "w") as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id','predicted'])
        for idx, label in predictions:
            csv_out.writerow([idx, int(label)])
