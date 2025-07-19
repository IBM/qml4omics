# ====== Base class imports ======
import os, json
import pandas as pd
import subprocess
import yaml, glob
from datetime import datetime, timezone
import time

# ======= Parallelization =======
from joblib import Parallel, delayed

# ======= checkpointing =========
from qbiocode import checkpoint_restart

# time stamp for output file
output_folder_timestamp= datetime.now(timezone.utc).strftime("%Y-%m-%d_%H_%M_%S_%f")
data_type = 'test_data'
beg_time = time.time() 
configfile = 'configs/basic_config.yaml'
input_data_path = 'data/tutorial_test_data/lower_dim_datasets'
# shutil.copy(configfile, 'temp_config.yaml')


def run_job(data_file):
    """This function runs the qbiocode-profiler.py script with a given data file and updates the YAML configuration file accordingly.
    It modifies the configuration file to include the data file name, timestamp, and data type, and then executes the profiling script with the updated configuration.
    This function is designed to be used in a batch processing context, where multiple datasets are processed in parallel. 

    Args:
        data_file (str): The name of the data file to be processed.
    Returns:
        None
    """

    ## edit YAML    

    # Read the YAML file
    
    with open(configfile, "r+") as yaml_file:
        data = yaml.safe_load(yaml_file)
        # add timestamp to output dir key of config file
        data['timestamp'] = output_folder_timestamp
        data['data_type'] = data_type

    # Modify the entry
    data["file_dataset"] = data_file

    # Write the updated data back to the file
    config_name = 'config_'+data_type+'_'+output_folder_timestamp+'__'+data_file.replace('.csv','').replace('.txt','')+'.yaml'
    with open('configs/'+config_name, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    
    commands = ["python", "qbiocode-profiler.py", "--config-name="+config_name]
    subprocess.run(commands) 

def main():
    """Main function to run the qbiocode-profiler.py in batch mode. It sets up the environment, processes datasets in parallel, and collects results.
    This function is designed to handle multiple datasets efficiently, allowing for parallel processing of machine learning methods and datasets.
    
    Args:
        None
    Returns:
        None
    """  

    n_jobs = 1
    current_dir = os.getcwd()
    path_to_input = os.path.join(current_dir, input_data_path)
    
    ###########################################################################
    # # This is a quick checkpointing function. Uncomment lines below in order to resume from last run.
    # # See documentation for 'checkpoint_restart' function to see how it works. 
    # previous_results_dir = 'results/batch_2024_12_14_17_56_52_448313' # this is the path to the previous output folder
    # completed_files = checkpoint_restart(previous_results_dir) # this list tells us what which datasets in path_to_input were previously fully processed in the run
    # # resume job starting from the datasets in path_to_input that did not complete
    # results = Parallel(n_jobs=n_jobs)(delayed(run_job)(file) for file in os.listdir(path_to_input) if file.endswith('csv') and file not in completed_files)
    ###########################################################################
    
    # comment line below if you are using the checkpoint approach above.
    results = Parallel(n_jobs=n_jobs)(delayed(run_job)(file) for file in os.listdir(path_to_input) if file.endswith('csv')) 
    
    #collect results

    #delete config and folders (?)
    final_model_results = pd.DataFrame()
    final_rde_results = pd.DataFrame()
    for file in os.listdir(path_to_input):
        print(file)
        if file.endswith('csv'):
                indv_results = ('results/'+data_type+'_batch_'+output_folder_timestamp+'/dataset='+file+'/ModelResults.csv')
                if os.path.isfile(indv_results):
                    model_results = pd.read_csv(indv_results, index_col=0)
                    final_model_results = pd.concat([final_model_results, model_results])
                    rde =  pd.read_csv('results/'+data_type+'_batch_'+output_folder_timestamp+'/dataset='+file+'/RawDataEvaluation.csv', index_col=0)
                    final_rde_results = pd.concat([final_rde_results, rde])
                    for f in glob.glob('configs/config_{}'.format(output_folder_timestamp)+'*'):
                        os.remove(f)
        
                # else:
                #     print('+'*5)
                #     print('ERROR found for file:', file)
                #     print('+'*5)  

    final_model_results.to_csv('results/'+data_type+'_batch_'+output_folder_timestamp+'/ModelResults.csv')
    final_rde_results.to_csv('results/'+data_type+'_batch_'+output_folder_timestamp+'/RawDataEvaluation.csv')
    total_time = (time.time() - beg_time)/3600
    print('Total run time of program in batch mode is:', round(total_time, 2), 'hours')

    return None
if __name__ == "__main__":
   main()

