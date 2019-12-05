import os
import pickle
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .tokenizer import FullTokenizer

def load_all_files(data_filepath):
    def _count_files(dir_path):
        return len([d for d in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, d))])

    def _get_file_string(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    print('Begin loading files...')

    file_text = []
    num_files = _count_files(data_filepath)
    for i in tqdm(range(num_files)):
        filepath = os.path.join(data_filepath, str(i) + ".txt")
        file_text.append(_get_file_string(filepath))

    print('Finished loading files.')
    return file_text


def save_object(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class Preprocessor:
    def __init__(self, data_filepath='./data'):
        self.DATA_PATH = data_filepath
        self.NODES_PATH = os.path.join(self.DATA_PATH, "node_information")
        self.TEXT_PATH = os.path.join(self.NODES_PATH, "text")
        self.TOKENS_PATH = os.path.join(self.NODES_PATH, "tokens")
        self.URLS_PATH = os.path.join(self.NODES_PATH, "urls")
        
        # Variables that store different parts of the processed documents
        self.file_text = load_all_files(self.TEXT_PATH)
        self.file_tokens = None
        self.file_urls = None
        self.tfidf_embeddings = None

        # Elements
        self.tokenizer = FullTokenizer()
        self.tfidf_vectorizer = TfidfVectorizer(input='content',
                                                encoding='utf-8',
                                                lowercase=True)
        self.tsvd = TruncatedSVD(n_components=256)

    def tokenize_all(self):
        print('Begin tokenizing files...')

        # Tokenize each file
        self.file_urls = []
        self.file_tokens = []
        for file in tqdm(self.file_text):
            tokens, urls = self.tokenizer.tokenize(file)
            self.file_tokens.append(tokens)
            self.file_urls.append(urls)

        print('Finished tokenizing files.\nBegin saving...')
        
        # Make tokens and urls directories
        os.makedirs(self.TOKENS_PATH, exist_ok=True)
        os.makedirs(self.URLS_PATH, exist_ok=True)

        # Save tokens and urls
        for i in tqdm(range(len(self.file_text))):
            tokens_file = os.path.join(self.TOKENS_PATH, str(i) + ".txt")
            urls_file = os.path.join(self.URLS_PATH, str(i) + ".txt")
            with open(tokens_file, 'w', encoding='utf-8') as tf:
                tf.write('\n'.join(self.file_tokens[i]))
            with open(urls_file, 'w', encoding='utf-8') as uf:
                uf.write('\n'.join(self.file_urls[i]))
        
        print('Finished saving.')

    def tfidf_vectorize(self):
        print('Begin tf-idf vectorization...')
        tokens = ['\n'.join(t) for t in self.file_tokens]

        # Fit the vectorizer and get the vocab
        self.tfidf_vectorizer.fit(tokens)
        self.tfidf_vocab = self.tfidf_vectorizer.get_feature_names()

        # Apply the vectorizer to the text
        self.tfidf_embeddings = self.tfidf_vectorizer.transform(tokens)

        print('Finished tf-ifd vectorization.\nBegin saving...')

        # Save the vectorizer, the vocab and the embeddings
        vectorizer_file = os.path.join(self.NODES_PATH,
                                       "tfidf_vectorizer.pickle")
        save_object(self.tfidf_vectorizer, vectorizer_file)
        
        vocab_file = os.path.join(self.NODES_PATH,
                                  "tfidf_vocab.pickle")
        save_object(self.tfidf_vocab, vocab_file)

        emb_file = os.path.join(self.NODES_PATH,
                                "tfidf_embeddings.pickle")
        save_object(self.tfidf_embeddings, emb_file)

        print('Finished saving.')

    def reduce_tfidf_embeddings(self):
        print('Begin dimentionality reduction on tf-idf embeddings...')

        # Train an incremental PCA algorithm on the sparse data
        sparce_embeddings = self.tsvd.fit_transform(self.tfidf_embeddings)
        self.reduce_tfidf_embeddings = sparce_embeddings.toarray()

        print('Finished dimentionality reduction on tf-idf embeddings.')
        print('Begin saving...')

        # Save the reduced embeddings and the IPCA object
        tsvd_file = os.path.join(self.NODES_PATH, "tsvd.pickle")
        save_object(self.tsvd, tsvd_file)

        reduced_emb_file = os.path.join(self.NODES_PATH, "reduced_tfidf_emb.pickle")
        save_object(self.reduced_tfidf_embeddings, reduced_emb_file)

        print('Finished saving.')

    def preprocess(self):
        self.tokenize_all()
        self.tfidf_vectorize()
        self.reduce_tfidf_embeddings()
