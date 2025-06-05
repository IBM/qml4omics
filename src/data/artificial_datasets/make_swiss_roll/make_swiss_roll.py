from sklearn.datasets import make_swiss_roll
import pandas as pd
import numpy as np
import itertools
import json


np.random.seed(42)

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 20))
NOISE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HOLE = [True, False]

def my_make_swiss_roll(
    n_samples, 
    noise,
    hole
):
    """
    This function generates a series of 'swiss roll' data sets.  It uses itertools to generate a range of input arguments 
    to pass into the sklearn make_swiss_roll function.  A data set is generated for each set of input arguments, which allows 
    the function to produce a variety of different swiss rolls based on varying number of samples (n_samples), noise, and whether or not
    the swiss roll has a 'hole' in it.
    
    Args:
        n_samples: list of integers
            The number of sample points on the Swiss Roll.
        noise: list of floats
            The standard deviation of the gaussian noise.
        hole: list of bools
            If True generates the swiss roll with hole dataset.

    Returns:
        df (pandas.DataFrame): Dataset in pandas dataframe with samples in the first column, features in the middle columns, and 
                               labels in the last column.    
    """
    
    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_samples, noise, hole]))
    print(configurations)
    print(len(configurations))
    count_configs = 1

    dataset_config = {}

    # populate all the configs with the corresponding argument values
    for n_s, n_n, n_h in configurations:
            config = "n_samples={}, noise={}, hole={}".format(
                n_s, n_n, n_h
            )
            print(count_configs)
    
            
        # iteratively run the function for each combination of arguments
            X, y = make_swiss_roll(
                n_samples=n_s,
                noise=n_n,
                hole=n_h
            )
            print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
            dataset = pd.DataFrame(X)
            dataset['class'] = y
            with open('swiss_roll_data/dataset_config.json', 'w') as outfile:
                dataset_config.update({'swiss_roll_data-{}.csv'.format(count_configs):
                {'n_samples': n_s,
                'noise': n_n,
                'hole': n_h}})  
                json.dump(dataset_config, outfile, indent=4) 
            new_dataset = dataset.to_csv('swiss_roll_data/swiss_roll_data-{}.csv'.format(count_configs), index=False)
            count_configs += 1
            print(X.shape)
            print(y.shape)
    return
                

my_make_swiss_roll(
    n_samples=N_SAMPLES,
    noise=NOISE,
    hole=HOLE
)
