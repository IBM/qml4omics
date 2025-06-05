# ====== Scikit-learn imports ======

from sklearn.decomposition import PCA

def compute_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca