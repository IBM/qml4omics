from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import json
import itertools
import os


dataset_config = {}

np.random.seed(42)

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 50))
N_FEATURES = list(range(10,60,10))
N_INFORMATIVE = list(range(2,8,4))
N_REDUNDANT = list(range(2,8,4))
N_CLASSES = list(range(2, 4, 6))
N_CLUSTERS_PER_CLASS = list(range(1, 2, 3))
WEIGHTS = [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]

def my_make_classification(
    n_samples, 
    n_features, 
    n_informative, 
    n_redundant,
    n_classes,
    n_clusters_per_class,
    weights,
    save_path=None
):
    print("Generating classes dataset...")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_samples, n_features, n_informative, n_redundant, n_classes, n_clusters_per_class, weights]))
    count_configs = 1

    # populate all the configs with the corresponding argument values
    for n_s, n_f, n_i, n_r, n_cla, n_clu, weights in configurations:
            if (n_i + n_r) <= n_f:
                config = "n_samples={}, n_features={}, n_informative={}, n_redundant={}, n_classes={}, n_clusters_per_class={}, weights={}".format(
                    n_s, n_f, n_i, n_r, n_cla, n_clu, weights
                )
                # print(count_configs)
        
                
            # iteratively run the function for each combination of arguments
                X, y = make_classification(
                    n_samples=n_s,
                    n_features=n_f,
                    n_informative=n_i,
                    n_redundant=n_r,
                    n_classes=n_cla,
                    n_clusters_per_class=n_clu,
                    weights=weights,
                )
                # print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
                dataset = pd.DataFrame(X)
                dataset['class'] = y
                with open( os.path.join( save_path, 'dataset_config.json' ), 'w') as outfile:
                    dataset_config.update({'hd_data-{}.csv'.format(count_configs):
                    {'n_samples': n_s,
                    'n_features': n_f,
                    'n_informative': n_i, 
                    'n_redundant': n_r, 
                    'n_classes': n_cla, 
                    'n_clusters_per_class': n_clu, 
                    'weights': weights}})  
                    json.dump(dataset_config, outfile, indent=4) 
                new_dataset = dataset.to_csv( os.path.join( save_path, 'class_data-{}.csv'.format(count_configs)), index=False)
                count_configs += 1  
                # print(X.shape)
                # print(y.shape)
    return
                