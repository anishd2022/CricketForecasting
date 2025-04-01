# Take .json ball by ball data, convert to pandas df, and move to SQL database

# Import necessary libraries
from helper_functions import convert_game_json_to_ballbyball_df


# Example usage
filename = "Data/all_json/336023.json"
df = convert_game_json_to_ballbyball_df(filename)
print(df)
