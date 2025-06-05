# ====== Embedding functions imports ======
from sklearn.decomposition import PCA 
from sklearn.decomposition import NMF
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
#from umap import UMAP

def get_embeddings(embedding, X_train, X_test, n_neighbors=30, n_components=None, method=None):
    assert n_components <= X_train.shape[1], "number of components greater than number of feature in the dataset"
    if 'none' == embedding:
        return X_train, X_test
    else:
        embedding_model = None
        if 'pca' == embedding:
            embedding_model = PCA(
                                n_components=n_components)
        elif 'nmf' == embedding:
            embedding_model = NMF(
                                n_components=n_components)
        elif 'lle' == embedding:
            if method==None: 
                embedding_model = LocallyLinearEmbedding(
                                    n_neighbors=n_neighbors,
                                    n_components=n_components, 
                                    method='standard')   
            else: 
                embedding_model = LocallyLinearEmbedding(
                                    n_neighbors=n_neighbors,
                                    n_components=n_components, 
                                    method='modified')
        elif 'isomap' == embedding: 
            embedding_model = Isomap(
                                n_neighbors=n_neighbors,
                                n_components=n_components, 
                                )
        elif 'spectral' == embedding: 
            embedding_model = SpectralEmbedding(
                                n_components=n_components, 
                                eigen_solver="arpack")
        elif 'umap' == embedding: 
            embedding_model = UMAP(
                                n_neighbors=n_neighbors,
                                n_components=n_components, 
                                )

        X_train = embedding_model.fit_transform(X_train)
        X_test = embedding_model.transform(X_test)
    
    return X_train, X_test