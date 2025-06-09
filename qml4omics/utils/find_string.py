import os

#os.path.join('configs', 'configs_CCC_grid')
def find_string_in_file(file_path, search_string):
    '''This function searches for a specific string in all files within a given directory.
    It prints the names of files that contain the string and counts how many files contain it.
    This can be useful for quickly identifying configurations or settings in multiple files,
    such as when checking for specific parameters in configuration files or logs.
    
    Args:
        file_path (str): The path to the directory containing the files to search.
        search_string (str): The string to search for in the files.

    Returns:
        None
    '''
    nc_filecount = 0
    file_count = []
    for file in os.scandir(os.path.join(file_path)):
            #print(file)
            #if file.is_file():
            with open(file, 'r') as fl:
                for line in fl:
                    if search_string in line:
                        print('this {} contains {}'.format(file, search_string))
                        nc_filecount += 1
                        
            file_count.append(file)
            
    print(len(file_count))
    print('there are {} files with {}'.format(nc_filecount, search_string))
    return
                    
                    # return True
        #return False

file_path = 'configs/configs_qml_gridsearch' 
search_string = 'embeddings: none' 

find_string_in_file(file_path, search_string)


