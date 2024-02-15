import os
import csv

# Define the path to the root folder
root_folder = 'Data/ohsumed-first-20000-docs/training'

# Define the path to the output CSV file
output_file = 'train_csv.csv'

# Create a set to store the folder names
folder_set = set()

# Loop through all the subfolders and files in the root folder
for root, dirs, files in os.walk(root_folder):
    # Loop through all the files in the current subfolder
    for file in files:
        # Get the full path of the file
        file_path = os.path.join(root, file)
        # Get the list of folder names that the file is in
        folder_names = root.split(os.sep)
        # Read the contents of the file
        with open(file_path, 'r') as f:
            file_contents = f.read()
        # Add the folder names to the set
        folder_set.update(folder_names)

# Convert the set of folder names to a sorted list
folder_list = sorted(list(folder_set))

# Create a dictionary to store the file contents and folder names
file_dict = {}

# Loop through all the subfolders and files in the root folder
for root, dirs, files in os.walk(root_folder):
    # Loop through all the files in the current subfolder
    for file in files:
        # Get the full path of the file
        file_path = os.path.join(root, file)
        # Get the list of folder names that the file is in
        folder_names = root.split(os.sep)
        # Read the contents of the file
        with open(file_path, 'r') as f:
            file_contents = f.read()
        # Create a list of zeros for the folder columns
        folder_columns = [0] * len(folder_list)
        # Set the value of the folder column to 1 if the file is in the folder
        for folder_name in folder_names:
            if folder_name in folder_list:
                folder_index = folder_list.index(folder_name)
                folder_columns[folder_index] = 1
        # Check if the file contents are already in the dictionary
        if file_contents in file_dict:
            # If the file contents are already in the dictionary, update the folder columns
            file_dict[file_contents] = [max(file_dict[file_contents][i], folder_columns[i]) for i in range(len(folder_list))]
        else:
            # If the file contents are not in the dictionary, add them with the folder columns
            file_dict[file_contents] = folder_columns

# Convert the dictionary to a list of lists
file_list = [[k] + v for k, v in file_dict.items()]

# Write the list of file contents and folder names to the output CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['File Contents'] + folder_list)
    writer.writerows(file_list)