from sklearn.datasets import make_s_curve
import pandas as pd
import numpy as np
import itertools
import json
import os

np.random.seed(42)

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 20))
NOISE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def my_make_s_curve(
    n_samples=N_SAMPLES, 
    noise=NOISE,
    save_path=None
):
    """
    This function generates a series of 'S-curve' data sets.  It uses itertools to generate a range of input arguments 
    to pass into the sklearn make_s_curve function.  A data set is generated for each set of input arguments, which allows 
    the function to produce a variety of different S-curves based on varying number of samples (n_samples) and noise.
    
    Args:
        n_samples: list of integers
            The number of sample points on the Swiss Roll.
        noise: list of floats
            The standard deviation of the gaussian noise.

    Returns:
        df (pandas.DataFrame): Dataset in pandas dataframe with samples in the first column, features in the middle columns, and 
                               labels in the last column.  
    """
    print("Generating S Curve dataset...")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_samples, noise]))
    # print(configurations)
    # print(len(configurations))
    count_configs = 1

    dataset_config = {}

    # populate all the configs with the corresponding argument values
    for n_s, n_n in configurations:
            config = "n_samples={}, noise={}".format(
                n_s, n_n,
            )
            # print(count_configs)
    
            
        # iteratively run the function for each combination of arguments
            X, y = make_s_curve(
                n_samples=n_s,
                noise=n_n,
            )
            # print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
            dataset = pd.DataFrame(X)
            dataset['class'] = y
            with open( os.path.join( save_path, 'dataset_config.json' ), 'w') as outfile:
                dataset_config.update({'s_curve_data-{}.csv'.format(count_configs):
                {'n_samples': n_s,
                'noise': n_n}})  
                json.dump(dataset_config, outfile, indent=4) 
            new_dataset = dataset.to_csv( os.path.join( save_path, 's_curve_data-{}.csv'.format(count_configs)), index=False)
            count_configs += 1
            # print(X.shape)
            # print(y.shape)
    return
                
