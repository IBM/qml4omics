import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json

np.random.seed(42)

def make_spirals(n_samples=5000, n_classes=2, noise=0.3, dim=3):
    """Generates an N-dimensional dataset of spirals."""

    X = []
    y = []

    for i in range(n_classes):
        t = np.linspace(0, 4 * np.pi, n_samples // n_classes)
        x = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
        y_ = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
        z = t + np.random.normal(0, noise, n_samples // n_classes)
        if dim==3:
            X.append(np.column_stack([x, y_, z])) # any new dimensions need to be added to this list
        
        # to add more dimensions, apparently you would just keep adding 't' variable from above, to each new dimension, 
        # as seen below. The question is, how can we iteratively do this while maintaining the binary classification
        # that this for loop is creating? 
        # nesting a loop iterating over the number of dimensions doesn't really work from what I'm seeing. so far
        # However, manually adding repeats of the same 3Ds, does work, as seen below -- is this correct?
        
    # for j in range(dim-3): # for anything above the first 3D
        if dim==6:
            new_d1 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d2 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d3 = t + np.random.normal(0, noise, n_samples // n_classes)
            X.append(np.column_stack([x, y_, z, new_d1, new_d2, new_d3])) # any new dimensions need to be added to this list
        if dim==9:
            new_d1 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d2 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d3 = t + np.random.normal(0, noise, n_samples // n_classes)
            new_d4 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d5 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d6 = t + np.random.normal(0, noise, n_samples // n_classes)
            X.append(np.column_stack([x, y_, z, new_d1, new_d2, new_d3, new_d4, new_d5, new_d6])) # any new dimensions need to be added to this list
        if dim==12:
            new_d1 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d2 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d3 = t + np.random.normal(0, noise, n_samples // n_classes)
            new_d4 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d5 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d6 = t + np.random.normal(0, noise, n_samples // n_classes)
            new_d7 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d8 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d9 = t + np.random.normal(0, noise, n_samples // n_classes)
            X.append(np.column_stack([x, y_, z, new_d1, new_d2, new_d3, new_d4, new_d5, new_d6, new_d7, new_d8, new_d9]))
        y.extend([i] * (n_samples // n_classes))

    return np.vstack(X), np.array(y)


# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 50))
N_CLASSES = [2]
NOISE = [0.3, 0.6, 0.9]
DIM = [3, 6, 9, 12]

def my_make_spirals(
    n_s,
    n_c, 
    n_n,
    n_d,
):
        
    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_s, n_c, n_n, n_d]))
    print(configurations)
    print(len(configurations))
    count_configs = 1

    dataset_config = {}

      # populate all the configs with the corresponding argument values
    for n_s, n_c, n_n, n_d in configurations:
            config = "samples={}, classes={}, noise={}, dimensions={}".format(
                n_s, n_c, n_n, n_d
            )
            print(count_configs)
            
            X, y = make_spirals(
                n_samples=n_s,
                n_classes=n_c,
                noise=n_n,
                dim=n_d
            )
            print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
            dataset = pd.DataFrame(X)
            dataset['class'] = y
            with open('spirals_data/dataset_config.json', 'w') as outfile:
                dataset_config.update({'spirals_data-{}.csv'.format(count_configs):
                {'n_samples': n_s,
                'noise': n_n,
                'dimensions': n_d
                }})  
                json.dump(dataset_config, outfile, indent=4) 
            new_dataset = dataset.to_csv('spirals_data/spirals_data-{}.csv'.format(count_configs), index=False)
            
            # plot the last 3 dimensions in each case
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, n_d-3], X[:, n_d-2],X[:, n_d-1], c=y, cmap='viridis')
            plt.savefig('spirals_data/spirals_data-{}.png'.format(count_configs))
            count_configs += 1
            #print(X.shape)
            #print(y.shape)
    return

my_make_spirals(
    n_s=N_SAMPLES,
    n_c=N_CLASSES,
    n_n=NOISE,
    n_d=DIM
)



# dim=12
# X, y = make_spirals(dim=dim)
# print(len(y))
# print(X[:, [2, 5]])

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, dim-7], X[:, dim-6], X[:, dim-5], s=5, c=y)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, dim-5], X[:, dim-4], X[:, dim-3], s=5, c=y)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, dim-4], X[:, dim-3], X[:, dim-2], s=5, c=y)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, dim-3], X[:, dim-2], X[:, dim-1], s=5, c=y)
# plt.show()

## do this if you have more than 3D and need to reduce in order to visualize
# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], s=5, c=y)
# plt.show()
