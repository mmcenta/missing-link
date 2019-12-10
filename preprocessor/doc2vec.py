from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Doc2VecVectorizer:
    def __init__(self, n_components=256, window_size=10):
        self.model = Doc2Vec(dm=1,
                             hs=1,
                             vector_size=n_components,
                             min_count=2,
                             alpha=0.21,
                             min_alpha=0.01,
                             epochs=10,
                             workers=4)

    def transform_save(self, file_tokens, filepath):
        docs = [TaggedDocument(" ".join(t), [i]) for i, t in enumerate(file_tokens)]

        self.model.build_vocab(docs)
        self.model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
        self.model.save_word2vec_format(filepath, doctag_vec=True, word_vec=False, binary=False)
