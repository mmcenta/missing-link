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

parser.add_argument('--representation_size', type=int, default=256,
                    help='the dimension of the embedding vectors')

if __name__ == "__main__":
    args = parser.parse_args()

    p = Preprocessor(representation_size=args.representation_size,
                     use_tfidf=args.use_tfidf,
                     use_bert=args.use_bert,
                     use_doc2vec=args.use_doc2vec)
    p.preprocess()
