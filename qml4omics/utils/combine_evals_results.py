import os
import pandas as pd   

## The purpose of this script is to provide a quick way to combine data frames once a job had to be restarted
## You can run this script anywhere, as long as you're pointing to the right directories, although it might make most
## sense to run this in the output folder where the job was resumed from.

 # Do this to find out which datasets were complete
def track_progress(input_dataset_dir, current_results_dir):
    """
    This function checks the current results directory for completed datasets and compares them to the input dataset directory.
    It prints out the number of completed datasets and how many input datasets are left to process.
    It assumes that a dataset is considered complete if a specific file ('RawDataEvaluation.csv') exists in the dataset's results directory, since producing
    this file is the last step in the pipeline.
    
    Args:
        input_dataset_dir (str): Path to the directory containing input datasets.
        current_results_dir (str): Path to the directory containing outputs of the current job.

    Returns:
        None
    """
    completed_files = []
    file_name = 'RawDataEvaluation.csv' # if this file was produced, then this dataset was fully processed
    for entry in os.scandir(current_results_dir):
        if entry.is_dir():
            for file in os.listdir(entry):
                if file == file_name:
                    completed_files.append(entry.name[8:])
    print('The completed datasets are: {}'.format(completed_files))

    num_input_datasets =[]
    for file in os.listdir(input_dataset_dir):
        if file.endswith('csv'):
            num_input_datasets.append(file)
    print('You have finished running program on {} out of a total of {} input datasets.'.format(len(completed_files), len(num_input_datasets)))
    print('You have {} input datasets left before program finishes.'.format(len(num_input_datasets)-len(completed_files)))
    return
input_dataset_dir = 'path to dir containing input datasets your current job is iterating over'
current_results_dir = 'path to dir containing outputs of current job'
track_progress(input_dataset_dir=input_dataset_dir, current_results_dir=current_results_dir)

def combine_results(prev_results_dir, recent_results_dir):
    """
    This function combines the results of a previous run of a job with the results of a recent run.
    It reads CSV files from both directories, concatenates them, and saves the combined dataframes to new CSV files.
    
    Args:
        prev_results_dir (str): Path to the directory where the previous job stopped prematurely.
        recent_results_dir (str): Path to the directory where the job was resumed and presumably ran to completion.

    Returns:
        new_combined_eval_df (pd.DataFrame): Combined dataframe of evaluations.
        new_combined_result_df (pd.DataFrame): Combined dataframe of model results.
    """
    ## instantiate lists from previous run
    eval_dfs = []
    previous_combined_eval_df = []
    results_dfs = []
    previous_combined_result_df = []

    #start grabbing all the individual csv files from previous run
    for entry in os.scandir(prev_results_dir):
        if entry.is_dir():
            for file in os.listdir(entry):
                if file.startswith('Raw'):
                    eval_csv_files = os.path.join(entry, file)
                    eval_dfs.append(eval_csv_files)
                if file.startswith('Model'):
                    results_csv_files = os.path.join(entry, file)
                    results_dfs.append(results_csv_files)
            
    # Read and append dataframes for previous Raw Data Evals
    for evalfile in eval_dfs:
        df1 = pd.read_csv(evalfile)
        previous_combined_eval_df.append(df1)

    # Read and append dataframes for previous Model Results
    for resultsfile in results_dfs:
        df2 = pd.read_csv(resultsfile)
        previous_combined_result_df.append(df2)

    # Concatenate all previous dataframes
    concat_previous_eval_df = pd.concat(previous_combined_eval_df, ignore_index=True)
    concat_previous_eval_df.to_csv('RawDataEvaluation_previous.csv')
    concat_previous_result_df = pd.concat(previous_combined_result_df, ignore_index=True)
    concat_previous_result_df.to_csv('ModelResults_previous.csv')

    # Concatenate previous and new dataframes (from folder where job was restarted and presumably ran to completion)
    for file in os.listdir(recent_results_dir):
        if file.startswith('Raw'):
            recent_eval_csv_file = os.path.join(recent_results_dir, file)
            recent_eval_df = pd.read_csv(recent_eval_csv_file, index_col=0)
            recent_eval_df.reset_index(drop=True, inplace=True)
        if file.startswith('Model'):
            recent_results_csv_file = os.path.join(recent_results_dir, file)
            recent_results_df = pd.read_csv(recent_results_csv_file, index_col=0)
            recent_results_df.reset_index(drop=True, inplace=True)
    print(recent_eval_df.shape[1])
    print(concat_previous_eval_df.shape[1])

    # Final combined data evaluations
    new_combined_eval_df = pd.concat([concat_previous_eval_df, recent_eval_df])                             
    new_combined_eval_df.reset_index(drop=True, inplace=True)
    new_combined_eval_df.to_csv('RawDataEvaluation_Combined.csv') 

    # Final combined Model Results 
    new_combined_result_df = pd.concat([concat_previous_result_df, recent_results_df])
    new_combined_result_df.reset_index(drop=True, inplace=True)
    new_combined_result_df.to_csv('ModelResults_Combined.csv')
    return new_combined_eval_df, new_combined_result_df

# prev_results_dir = 'path to dir where job stopped prematurely'
# recent_results_dir = 'path to dir where job was resumed'
# combine_results(prev_results_dir=prev_results_dir, recent_results_dir=recent_results_dir)    

    