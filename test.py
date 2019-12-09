import csv
import pickle
import argparse
import numpy as np
from xgboost.sklearn import XGBClassifier

from util.dataset_io import load_dataset


parser = argparse.ArgumentParser(description='Output predictions of the given model to a CSV file.')

parser.add_argument('--model_file', required=True,
                    help='file containing the model to make predictions')

parser.add_argument('--input_name', required=True,
                    help='name of the test dataset')

parser.add_argument('--output_file', default='submission.csv',
                    help='file which will contain the predictions')


if __name__ == "__main__":
    args = parser.parse_args()

    # Load testing data
    X, _ = load_dataset(args.input_name)

    # Load model
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    # Predict
    predictions = model.predict(X)

    # Write the output in the format required by Kaggle
    predictions = enumerate(predictions)
    with open(args.output_file, "w") as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id','predicted'])
        for idx, label in predictions:
            csv_out.writerow([idx, int(label)])
