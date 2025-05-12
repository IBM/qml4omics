# ====== Scikit-learn imports ======

from sklearn.decomposition import NMF

def compute_nmf(X, n_components=None, max_iter=200):
    nmf = NMF(n_components=n_components)
    nmf.fit(X)
    return nmf