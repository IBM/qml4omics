import os
import filecmp
import itertools

def find_duplicate_files(directory):
    '''Find duplicate files in a directory based on their content.  This can be used to find duplicate config.yaml files, for example.'''

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