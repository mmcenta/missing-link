import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
    

class GraphFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        G = nx.Graph()
        edge_to_idx = {}
        for i, x in enumerate(X):
            G.add_edge(x[0], x[1])
            edge_to_idx[(x[0], x[1])] = i

        feature_iters = [
            nx.resource_allocation_index(G),
            nx.jaccard_coefficient(G),
            nx.adamic_adar_index(G),
            nx.preferential_attachment(G),
            nx.cn_soundarajan_hopcroft(G),
            nx.ra_index_soundarajan_hopcroft(G),
            nx.within_inter_cluster(G)
        ]

        features = np.zeros((X.shape[0], len(feature_iters)))
        for i, it in enumerate(feature_iters):
            for u, v, p in it:
                if (u, v) in edge_to_idx:
                    idx = edge_to_idx[(u, v)]
                    features[idx, i] = p
        
        return features
