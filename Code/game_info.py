# go through all .json game files in the Data/all_json folder and then create a dataframe that gives out info for each game
# store this dataframe in a .csv file

# import necessary libraries:
import os
from helper_functions import convert_game_json_to_game_info_df
import pandas as pd


# Path to all .json files:
folder_path = "Data/all_json/"

total_games = 0

# Initialize an empty DataFrame to store all game info
all_games_df = pd.DataFrame()

# iterate over all games:
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Only process JSON files
        # extract relative filepath
        file_path = os.path.join(folder_path, filename)
        
        game_row = convert_game_json_to_game_info_df(file_path)
        
        # Append to master DataFrame
        all_games_df = pd.concat([all_games_df, game_row], ignore_index=True)
        
        total_games += 1
        print(total_games)
        
# save output df to appropriate folder:
output_csv_path = "Data/game_data/all_game_info.csv"
all_games_df.to_csv(output_csv_path, index=False)

print(f"Successfully processed {total_games} games.")
print(f"Summary CSV saved to: {output_csv_path}")