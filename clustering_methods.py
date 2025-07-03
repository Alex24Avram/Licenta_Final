import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigvalsh

def estimate_clusters(eigenvalues, max_clusters=5):
    gaps = np.diff(eigenvalues[:max_clusters + 1])
    return int(np.argmax(gaps) + 1)


def diarize_kmeans(features, n_clusters, gt_labels=None):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features, 30)
    pca = PCA(n_components=n_components)
    X_red = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_red)

    if gt_labels is not None:
        max_gt = max([l for l in gt_labels if l != -1], default=-1) + 1
        contingency = np.zeros((n_clusters, max_gt), dtype=int)
        for i, gt in enumerate(gt_labels):
            if gt != -1:
                contingency[labels[i], gt] += 1
        row_ind, col_ind = linear_sum_assignment(-contingency)
        mapping = {row: col for row, col in zip(row_ind, col_ind)}
        labels = np.array([mapping.get(l, l) for l in labels])

    return labels


def diarize_spectral(features, gt_labels=None):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features, 30)
    pca = PCA(n_components=n_components)
    X_red = pca.fit_transform(X)

    connectivity = kneighbors_graph(X_red, n_neighbors=min(n_samples - 1, 10), include_self=False)

    L, _ = laplacian(connectivity, normed=True, return_diag=True)
    evals = eigvalsh(L.toarray())

    est_clusters = estimate_clusters(evals, max_clusters=5)

    sc = SpectralClustering(
        n_clusters=est_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = sc.fit_predict(connectivity.toarray())

    if gt_labels is not None and len(set([l for l in gt_labels if l != -1])) > 1:
        max_gt = max([l for l in gt_labels if l != -1]) + 1
        contingency = np.zeros((est_clusters, max_gt), dtype=int)
        for i, gt in enumerate(gt_labels):
            if gt != -1:
                contingency[labels[i], gt] += 1
        row_ind, col_ind = linear_sum_assignment(-contingency)
        mapping = {row: col for row, col in zip(row_ind, col_ind)}
        labels = np.array([mapping.get(l, l) for l in labels])

    return labels
