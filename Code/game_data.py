# Take .json ball by ball data for all matches of a given format and convert it to .csv files saved in the Data folder

# Import necessary libraries
from helper_functions import convert_game_json_to_ballbyball_df
import pandas as pd
import os
import json

# path to JSON files:
folder_path = "Data/all_json/"
# set match type:
match_type = "T20"
total_games = 0
# Iterate over all files in the folder:
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Only process JSON files
        file_path = os.path.join(folder_path, filename)
        
        # Extract game ID (filename without .json)
        game_id = os.path.splitext(filename)[0]
        
        # Read JSON file to extract the date
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Extract date from the "info" -> "dates" field
        game_date = data["info"]["dates"][0]  # Assuming there's always one date
        formatted_date = game_date.replace("-", "")  # Convert to YYYYMMDD
        
        df = convert_game_json_to_ballbyball_df(file_path, match_format=match_type)
        if df is not None:
            # define output .csv path:
            output_filename = f"{formatted_date}_{game_id}.csv"
            output_path = os.path.join("Data/all_T20ball_by_ball/", output_filename)
            
            df.to_csv(output_path, index=False)
            print(f"Processed dataframe for {filename} -> {output_filename}")
            total_games += 1
        

# print out how many games you processed
print(f"Processed {total_games} DataFrames.")

# example usage:
"""
df = convert_game_json_to_ballbyball_df("Data/all_json/336023.json", match_format="T20")
print(df)
df.to_csv("sample_2.csv", index=False)
"""


