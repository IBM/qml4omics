import os
import filecmp
import itertools

def find_duplicate_files(directory):
    ''' 
    This function scans the specified directory for files and compares their content.
    It identifies files that have identical content, even if they have different names.
    It returns a list of tuples, where each tuple contains the paths of two duplicate files.
    If no duplicates are found, it returns an empty list.
    This function reads the content of each file line by line, ignoring empty lines,
    and compares the sorted content of each pair of files to determine if they are duplicates.

    This could have many practical uses, including finding duplicate config.yaml files, which
    can be useful when running a slurm script that is iterating over the yaml files in the config folder.
    For example, When generating the qml gridsearch configs (via the generate_experiments notebook), 
    it is possible to generate duplicate configs, which would cause redundant runs.  This function can help identify those duplicates, 
    allowing the user to remove them before running the slurm script.

    Args:
        directory (str): The path to the directory to search for duplicate files.

    Returns:
        duplicates (list): A list of tuples, where each tuple contains the paths of two duplicate files.
    If no duplicates are found, an empty list is returned.
    Example:
        >>> find_duplicate_files("configs/configs_qml_gridsearch/")
        [('configs/configs_qml_gridsearch/config1.yaml', 'configs/configs_qml_gridsearch/config2.yaml')]
    Example:
        >>> find_duplicate_files("configs/configs_qml_gridsearch/")
        []
    '''

    files = []
    for file in os.scandir(os.path.join(directory)):
            files.append(os.path.join(file))

    duplicates = []
    for file1, file2 in itertools.combinations(files, 2):
        content1 = sorted([line for line in open(file1).readlines() if line.strip()])
        content2 = sorted([line for line in open(file2).readlines() if line.strip()])
        same = True
        for line1, line2 in zip(content1,content2):
            if line1 != line2:
                same = False
                break  
        if same:
            duplicates.append((file1, file2))
    return duplicates

if __name__ == "__main__":
    directory_to_search = "configs/configs_qml_gridsearch/"  # Replace with the path to the directory you want to search
    duplicate_files = find_duplicate_files(directory_to_search)

    if duplicate_files:
        print("Duplicate files found:")
        for file1, file2 in duplicate_files:
            print(f"- {file1} and {file2}")
    else:
        print("No duplicate files found.")