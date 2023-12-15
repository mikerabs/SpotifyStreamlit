import os
import pandas as pd

path = ''

folder_name = path + 'spotify_files'  # Replace with the actual folder path

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_name) if file.endswith('.csv')]

# Initialize an empty DataFrame
merged_df = pd.DataFrame()

# Iterate over each CSV file and vertically merge them
for file in csv_files:
    file_path = os.path.join(folder_name, file)
    df = pd.read_csv(file_path)
    merged_df = pd.concat([merged_df, df], axis=0)

# Reset the index of the merged DataFrame
merged_df.reset_index(drop=True, inplace=True)

# Write merged_df to a CSV file
merged_df.to_csv(path + 'data.csv', index=False)


