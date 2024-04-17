import os
import shutil

import pandas as pd

data_base_dir = 'D:/ICBT/cloth2/data/'
images_dir = os.path.join(data_base_dir, 'images/img/')
partition_file_path = os.path.join(data_base_dir, 'annotation/list_eval_partition.txt')

df = pd.read_csv(partition_file_path, sep='\\s+', skiprows=2, names=['image_name', 'evaluation_status'])

# print(df.head())

splits_dir = os.path.join(data_base_dir, 'images')
os.makedirs(os.path.join(splits_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(splits_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(splits_dir, 'test', 'images'), exist_ok=True)


def organize_dataset(dataframe, source_dir, target_dir):
    # Go through each file in the dataframe and move it to the corresponding split directory
    for _, row in dataframe.iterrows():
        source_path = os.path.join(source_dir, row['image_name'])
        destination_path = os.path.join(target_dir, row['evaluation_status'], 'images', row['image_name'])

        # Move the file
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
        else:
            print(f"Warning: {source_path} does not exist and will not be moved.")


# Run the dataset organization
organize_dataset(df, images_dir, splits_dir)

# Output the result
print("Dataset splitting complete.")
