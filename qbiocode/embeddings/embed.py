# ====== Embedding functions imports ======
from sklearn.decomposition import PCA 
from sklearn.decomposition import NMF
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from umap import UMAP

def get_embeddings(embedding: str, X_train, X_test, n_neighbors=30, n_components=None, method=None):

    """This function applies the specified embedding technique to the training and test datasets.

    Args:
        embedding (str): The embedding technique to use. Options are 'none', 'pca', 'nmf', 'lle', 'isomap', 'spectral', or 'umap'.
        X_train (array-like): The training dataset.
        X_test (array-like): The test dataset.
        n_neighbors (int, optional): Number of neighbors for certain embeddings. Defaults to 30.
        n_components (int, optional): Number of components for the embedding. If None, it defaults to the number of features in X_train.
        method (str, optional): Method for Locally Linear Embedding. Defaults to None.
        
    Returns:
        tuple: Transformed training and test datasets.
    """

    embedding = embedding.lower()    
    valid_modes = ['none', 'pca', 'lle', 'isomap', 'spectral', 'umap', 'nmf']
    if embedding not in valid_modes:
        raise ValueError(f"Invalid mode: {embedding}. Mode must be one of {valid_modes}")


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