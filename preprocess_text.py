import argparse
import numpy as np
from preprocessor.preprocessor import Preprocessor


parser = argparse.ArgumentParser(description='Extract all features from text and save them.')

parser.add_argument('--use_bert', action='store_true',
                    help='if set, bert embeddings will be calculated')

if __name__ == "__main__":
    args = parser.parse_args()

    p = Preprocessor(use_bert=args.use_bert)
    p.preprocess()
