import os
import pickle
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from .tokenizer import FullTokenizer
from .bert import BertVectorizer
from .doc2vec import Doc2VecVectorizer
from util.embeddings_io import save_embeddings_from_array


def load_all_files(data_filepath):
    def _count_files(dir_path):
        return len([d for d in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, d))])

    def _get_file_string(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    print('Loading files...')

    file_text = []
    num_files = _count_files(data_filepath)
    for i in tqdm(range(num_files)):
        filepath = os.path.join(data_filepath, str(i) + ".txt")
        file_text.append(_get_file_string(filepath))

    print('Done.')
    return file_text


def save_object(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class Preprocessor:
    def __init__(self, data_filepath='./data', representation_size=256, use_tfidf=False, use_doc2vec=False):
        self.DATA_PATH = data_filepath
        self.NODES_PATH = os.path.join(self.DATA_PATH, "node_information")
        self.TEXT_PATH = os.path.join(self.NODES_PATH, "text")
        self.TOKENS_PATH = os.path.join(self.NODES_PATH, "tokens")
        self.URLS_PATH = os.path.join(self.NODES_PATH, "urls")
        self.CONTEXTS_PATH = os.path.join(self.NODES_PATH, "contexts")

        self.use_tfidf = use_tfidf
        self.use_doc2vec = use_doc2vec

        # Variables that store different parts of the processed documents
        self.file_text = load_all_files(self.TEXT_PATH)

        # Preprocessing objects
        self.tokenizer = FullTokenizer()
        self.tfidf_vectorizer = TfidfVectorizer(input='content',
                                                encoding='utf-8',
                                                lowercase=True)
        self.tfidf_vocab = None
        self.tsvd = TruncatedSVD(n_components=representation_size)
        self.bert_vectorizer = BertVectorizer()
        self.pca = PCA(n_components=representation_size)
        self.doc2vec_vectorizer = Doc2VecVectorizer(n_components=representation_size)


    def tokenize_all(self, file_text):
        print('Tokenizing files...')

        # Tokenize each file
        file_urls = []
        file_tokens = []
        file_contexts = []
        for file in tqdm(file_text):
            tokens, urls, contexts = self.tokenizer.tokenize(file)
            file_tokens.append(tokens)
            file_urls.append(urls)
            file_contexts.append(contexts)

        print('Done.\nSaving...')

        # Make tokens and urls directories
        os.makedirs(self.TOKENS_PATH, exist_ok=True)
        os.makedirs(self.URLS_PATH, exist_ok=True)
        os.makedirs(self.CONTEXTS_PATH, exist_ok=True)

        # Save tokens and urls
        for i in tqdm(range(len(self.file_text))):
            tokens_file = os.path.join(self.TOKENS_PATH, str(i) + ".txt")
            with open(tokens_file, 'w', encoding='utf-8') as tf:
                tf.write('\n'.join(file_tokens[i]))

            urls_file = os.path.join(self.URLS_PATH, str(i) + ".txt")
            with open(urls_file, 'w', encoding='utf-8') as uf:
                uf.write('\n'.join(file_urls[i
            contexts_file = os.path.join(self.CONTEXTS_PATH, str(i) + ".txt")
            with open(contexts_file, 'w', encoding='utf-8') as cf:
                cf.write('\n'.join(file_contexts[i]))

        print('Done.')

        return file_tokens, file_urls

    def tfidf_vectorize(self, file_tokens):
        print('Perform tf-idf vectorization...')
        docs = ['\n'.join(t) for t in file_tokens]

        # Fit the vectorizer and get the vocab
        self.tfidf_vectorizer.fit(docs)
        self.tfidf_vocab = self.tfidf_vectorizer.get_feature_names()

        # Apply the vectorizer to the text
        tfidf_embeddings = self.tfidf_vectorizer.transform(docs)

        print('Done.\nSaving...', end='')

        # Save  the embeddings
        emb_file = os.path.join(self.NODES_PATH,
                                "tfidf_embeddings.pickle")
        save_object(tfidf_embeddings, emb_file)

        print('Done.')

        return tfidf_embeddings

    def reduce_sparse_embeddings(self, sparse_embeddings):
        print('Perform dimentionality reduction on sparse embeddings...')

        # Train an truncated SVD algorithm on the sparse data
        reduced_embeddings = self.tsvd.fit_transform(sparse_embeddings)

        print('Done.\nSaving...')

        # Save the reduced embeddings
        reduced_emb_file = os.path.join(self.NODES_PATH, "reduced_tfidf.embeddings")
        save_embeddings_from_array(reduced_embeddings, reduced_emb_file)

        print('Done.')

        return reduced_embeddings

    def reduce_embeddings(self, full_embeddings):
        print('Perform dimensionality reduction on embeddings...')

        # Train a PCA vectorizer on the full embeddings
        reduced_embeddings = self.pca.fit_transform(full_embeddings)

        print('Done. Saving...')

        # Save reduced embeddings
        emb_file = os.path.join(self.NODES_PATH, "reduced_bert.embeddings")
        save_embeddings_from_array(reduced_embeddings, emb_file)

    def doc2vec_vectorize(self, file_tokens):
        print('Perform doc2vec vectorization and save...')

        emb_file = os.path.join(self.NODES_PATH, "doc2vec.embeddings")
        self.doc2vec_vectorizer.transform_save(file_tokens, emb_file)

        print('Done.')

    def preprocess(self):
        file_tokens, _ = self.tokenize_all(self.file_text)
        if self.use_tfidf:
            self.reduce_sparse_embeddings(self.tfidf_vectorize(file_tokens))
        if self.use_bert:
            self.reduce_embeddings(self.bert_vectorize(file_tokens))
        if self.use_doc2vec:
            self.doc2vec_vectorize(file_tokens)
