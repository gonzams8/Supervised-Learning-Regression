"""
Custom transformers used in the feature pipeline.
"""
from __future__ import annotations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

__all__ = ["ClusterSimilarity", "RatioTransformer"]


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Replace lat/lon with similarities to K‑means centroids."""
    def __init__(self, n_clusters: int = 10, gamma: float = 1.0, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        ).fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        X = check_array(X)
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"cluster_{i}_sim" for i in range(self.n_clusters)]


class RatioTransformer(BaseEstimator, TransformerMixin):
    """Compute element‑wise ratio col0 / col1."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X)
        return X[:, [0]] / X[:, [1]]

    def get_feature_names_out(self, names=None):
        return ["ratio"]
