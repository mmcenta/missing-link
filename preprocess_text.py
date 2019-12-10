import argparse
import numpy as np
from preprocessor.preprocessor import Preprocessor


parser = argparse.ArgumentParser(description='Extract all features from text and save them.')

parser.add_argument('--use_tfidf', action='store_true',
                    help='if set, tf-idf embeddings will be calculated')

parser.add_argument('--use_bert', action='store_true',
                    help='if set, bert embeddings will be calculated')

parser.add_argument('--use_doc2vec', action='store_true',
                    help='if set, doc2vec embeddings will be calculated')

if __name__ == "__main__":
    args = parser.parse_args()

    p = Preprocessor(use_tfidf=args.use_tfidf,
                     use_bert=args.use_bert,
                     use_doc2vec=args.use_doc2vec)
    p.preprocess()
