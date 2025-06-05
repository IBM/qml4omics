from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import pandas as pd


X, y = make_classification(n_samples = [50,350,50], 
n_features = [5,20,5], # min/max/step
n_informative = [0,16,4], # min/max/step
n_redundant= [0,16,4], # min/max/step
n_classes=[2,4,6], # absolute
n_clusters_per_class=[1,2,3], # absolute
weights = [0.5, 0.9, 0.7] # spread out over n_classes
)
print(X.shape)
print(y.shape)
# Create DataFrame with features as columns
dataset = pd.DataFrame(X)
# give custom names to the features
dataset.columns = ['X1', 'X2', 'X3', 'X4', 'X5']
# Now add the label as a column
dataset['y'] = y

dataset.info()
