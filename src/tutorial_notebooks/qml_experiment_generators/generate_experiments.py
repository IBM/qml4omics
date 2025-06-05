import os, re, sys
import yaml
import pandas as pd
import numpy as np

# QMethods
qmethods = [ 'qnn', 'vqc', 'qsvc' ]
reps = [1,2]
optimizers = ['COBYLA', 'SPSA' ]
entanglements = ['linear', 'full' ]
feature_maps = ['Z', 'ZZ']
ansatz_types = ['amp', 'esu2']
n_components = [5,10]
Cs =  [0.1, 1, 10]
max_iters = [100,500]
embeddings = ['pca', 'nmf', 'none']

# paths
dir_home = re.sub( 'qml4omics.*', 'qml4omics', os.getcwd() )
dir_config = os.path.join( dir_home, 'configs' )
file_template_config = os.path.join( dir_config, 'config.yaml' )
dir_config_new = os.path.join( dir_config, 'configs_qml_gridsearch' )
used_data_files = os.path.join( dir_config_new, 'used_data_files.csv' )


if not os.path.exists(dir_config_new):
    os.mkdir(dir_config_new)


#############    
import itertools
p = [qmethods, reps, optimizers, entanglements, feature_maps, ansatz_types, n_components, Cs, max_iters, embeddings]

p_c = pd.DataFrame(list(itertools.product(*p)),
                   columns=[
                       'method',
                       'reps',
                       'local_optimizer',
                       'entanglement',
                       'feature_map',
                       'ansatz_type',
                       'n_components',
                       'C',
                       'max_iter',
                       'embedding'
                   ])
p_c.loc[ p_c['method'].isin( ['qnn','vqc']), 'C' ] = 1
p_c.loc[ p_c['method'].isin( ['qsvc']), 'ansatz_type' ] = 'amp'
p_c.loc[ p_c['method'].isin( ['qsvc']), 'max_iter' ] = 100
p_c.loc[ p_c['method'].isin( ['qsvc']), 'local_optimizer' ] = 'COBYLA'

p_c = p_c.drop_duplicates()
p_c = p_c[~((p_c['n_components'] >= 10) & (p_c['max_iter'] < 500))]
p_c = p_c[~((p_c['reps'] > 1) & (p_c['n_components'] <= 10))] # for small feature space, you may not need more than 1 layer of the ansatz
print(p_c)

############
idx = 1 
cfg_orig = yaml.safe_load( open( file_template_config, 'r+' ) )

if os.path.exists( used_data_files ):
    used_files = pd.read_csv(used_data_files)
else:
    used_files = []

# Data Files
for dir_data in [
    #os.path.join( dir_home, 'data', 'tutorial_test_data', 'higher_dim_datasets'),
    os.path.join( dir_home, 'data', 'tutorial_test_data', 'lower_dim_datasets')

]:
    files = [ fl for fl in os.listdir(dir_data) if 'csv' in fl ]
    files.sort()
    # remove files previously used    
    files = list(set(files).difference(set(used_files)) )

    
    # Use all files
    files = list(np.random.choice( files, int(len(files)*1)))
    # Or use a random sample of 10% of files
    # files = list(np.random.choice( files, int(len(files)*1)))
    used_files = used_files + files
    p_c_t = p_c.copy()
    if ('moons' in dir_data) | ('circles' in dir_data):
        p_c_t = p_c_t[ p_c_t['embedding'] == 'none' ]
    else:
        p_c_t = p_c_t[ p_c_t['embedding'] != 'none' ]

    count = 1
    for ix, row in p_c_t.iterrows():
        for fl in files:
            file_yaml = os.path.join( dir_config_new, 'exp_' + str(idx) + '.yaml')
            key = row['method'] + '_' + re.sub( '.csv', '', fl )
            cfg = cfg_orig.copy()
            cfg['yaml'] = file_yaml
            cfg['model'] = [row['method']]
            cfg['file_dataset'] = str(fl)
            cfg['folder_path'] = re.sub( 'data/', '', dir_data )
            cfg['hydra']['run']['dir'] = os.path.join( 'results', 'qmlgridsearch_' + key )
            
            ###
            # This block looks at the input dataset and ensures the embedding method placed in the yaml
            # is 'none' if n_components is equal to or greater than the original number of features 
            # in the input data set.  This should help avoid redundant yaml files, since using either of 
            # the embedding methods (pca, nmf) in this scenario is essentially the same as choosing 'none', thus
            # leading to the same experiment being run, even though the yaml file might be different. 
            fl_df=pd.read_csv(dir_data+'/'+fl)
            orig_features = fl_df.shape[1]-1
            if row['n_components'] >= orig_features:
                #print(orig_features, row['n_components'])
                cfg['embeddings'] = [str('none')]
                count += 1
            else:
                cfg['embeddings'] = [row['embedding']]
            
            cfg['n_components'] = row['n_components']
            ###
            
            cfg[row['method']+'_args']['reps'] = row['reps']
            cfg[row['method']+'_args']['entanglement'] = row['entanglement']
            cfg[row['method']+'_args']['encoding'] = row['feature_map']
            
            if row['method'] != 'qsvc':
                cfg[row['method']+'_args']['ansatz_type'] = row['ansatz_type']
                cfg[row['method']+'_args']['maxiter'] = row['max_iter']
            else:
                cfg[row['method']+'_args']['C'] = row['C']
                cfg[row['method']+'_args']['local_optimizer'] = row['local_optimizer']
                            
            yaml.dump( cfg, open( file_yaml, 'w'), default_flow_style= False )   
            idx += 1

pd.Series(used_files).to_csv( used_data_files, index = False)

idx-1

