import os
import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity


def load_files(dir_path):
    def _count_files(dir_path):
        return len([d for d in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, d))])

    def _get_file_string(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    file_text = []
    num_files = _count_files(dir_path)
    for i in range(num_files):
        filepath = os.path.join(dir_path, str(i) + ".txt")
        file_text.append(_get_file_string(filepath))

    return file_text


def load_object(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class UrlMatcher:
    def __init__(self, vocab):
        self.vocab = set(vocab)

    def match_all_urls(self, urls, tokens):
        matches = []
        for url in urls:
            words = url.split(":/-_.")
            for w in words:
                for i in range(len(url)):
                    for j in range(i, len(url)):
                        subword = w[i:j + 1]
                        if (subword in self.vocab
                            and subword in tokens):
                            matches.append(subword)
        return matches
    

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, data_path="./data"):
        self.data_path = data_path

    def fit(self, X, y):
        return self

    def transform(self, X):
        tokens_path = os.path.join(self.data_path, "tokens")
        tokens = load_files(tokens_path)

        urls_path = os.path.join(self.data_path, "urls")
        urls = load_files(urls)
        urls = [u.split() for u in urls]

        vocab_path = os.path.join(self.data_path, "tfidf_vocab.pickle")
        vocab = load_object(vocab_path)

        embs_path = os.path.join(self.data_path, "tfidf_embeddings.pickle")
        embs = load_object(embs_path)

        rembs_path = os.path.join(self.data_path, "reduced_tfidf_embeddings.pickle")
        rembs = load_object(rembs_path)

        url_matcher = UrlMatcher(vocab)

        features = list()
        for x in X:
            f = []
            u, v = x[0], x[1]
            url_matches = url_matcher.match_all_urls(urls[u], tokens[v])

            f.append(cosine_similarity(embs[u], embs[v]))
            f.append(cosine_similarity(rembs[u], rembs[v]))
            f.append(len(url_matches))
            f = np.array(f)

            features.append(f)

        return np.array(features)
