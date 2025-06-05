import os

def checkpoint_restart(previous_results_dir):

    """ 
    This is a hacky sort of checkpointing snippet
    Do this to start this run from the files that were not complete in the previous run.
    It will look into the user specified previous output folder (prev_results_dir)
    referencing the input data set path, and find cases where a dataset 
    didn't complete and resume from those, based on whether or not RawDataEvaluation.csv
    was produced inside that particular dataset's output folder.
    Keep in mind, this is now producing a different output folder, so you will likely have to combine 
    the final *.csv files from the previous and current output folder.
    
    Args:
        previous_results_dir (path): This is the path to the output folder for the job(s) that was prematurely
                                    stopped.
    Returns: 
        completed_files (list): contains a list of datasets for which the jobs were completed. This can then be used in
                                the job running script (either qmlbench.py or run_databatch.py) to compare to the list
                                of datasets in the job's input path, and resume the job starting from the datasets that 
                                are not common to both lists.
    """
     
    completed_files = []
    file_name = 'RawDataEvaluation.csv' # if this file was produced, then this dataset was fully processed
    for entry in os.scandir(previous_results_dir):
        if entry.is_dir():
            for file in os.listdir(entry):
                if file == file_name:
                    completed_files.append(entry.name[8:])
    # print(completed_files, len(completed_files))
    return(completed_files)
