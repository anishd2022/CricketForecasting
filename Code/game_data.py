# Take .json ball by ball data, convert to pandas df, and move to SQL database

# Import necessary libraries
from helper_functions import convert_game_json_to_ballbyball_df
import pandas as pd
import os
import pickle


# path to JSON files:
folder_path = "Data/all_json/"

# Initialize an empty list to store DataFrames
df_list = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Only process JSON files
        file_path = os.path.join(folder_path, filename)
        df = convert_game_json_to_ballbyball_df(file_path)
        print(f"processed dataframe for {filename}")
        if df is not None:  # Ensure only valid DataFrames are added
            df_list.append(df)

# Now df_list contains DataFrames, one for each match
print(f"Processed {len(df_list)} DataFrames.")


# Path where you want to save the pickle file
pickle_file_path = "Data/all_ball_by_ball/all_games_ballbyball_data.pkl"

# Save the df_list to the pickle file
with open(pickle_file_path, 'wb') as f:
    pickle.dump(df_list, f)
    
print(f"df_list saved to {pickle_file_path}")
