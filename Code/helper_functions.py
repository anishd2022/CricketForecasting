# Helper functions for rest of code

# Import necessary libraries:
from sqlalchemy import create_engine
import json
import pandas as pd
import os

# Helper function to move pandas dataframe from python to mysql database:
# Parameters:
#   data: pandas dataframe that follows the structure of the sql_table you intend to append it to
#   sql_table_name: name of the SQL table where you want to move your pandas df to
#   if_exists: can take 3 values --> 'append' (default), 'fail' (raises an error if table alr exists), 'replace' (drops table and creates a new one)
# Output:
#   data seen in mySQL database
def move_from_python_to_sql(data, sql_table_name, if_exists='append'):
    connection_string = "mysql+pymysql://root:root@localhost:8889/CricketData"
    engine = create_engine(connection_string)
    # Insert rows into the database
    try:
        data.to_sql(sql_table_name, con=engine, if_exists=if_exists, index=False)
        print("Rows successfully inserted into the database!")
    except Exception as e:
        print(f"Error inserting rows: {e}")
        
        

# Function to convert a .json game file to a pandas df showing simple ball by ball data
# Parameters:
#   filename: string --> relative path to the JSON file
#   match_format: string --> default (T20), specifies which match format to parse...will not process data if not matching match format
# Output:
#   pandas dataframe containing ball by ball data
def convert_game_json_to_ballbyball_df(filename, match_format="T20"):
    """
    Converts a JSON file containing ball-by-ball cricket data into a Pandas DataFrame.
    
    :param filename: str, path to the JSON file
    :return: pd.DataFrame, DataFrame containing ball-by-ball data
    """
    # Load JSON file
    with open(filename, "r") as file:
        data = json.load(file)

    # Check if match type is T20, only process if so
    match_type = data["info"].get("match_type", "Unknown")
    print(match_type)
    if match_type != match_format:
        print(f"Match type is {match_type}, not {match_format}. Skipping processing.")
        return None
    
    # Get player registry for IDs
    player_registry = data["info"].get("registry", {}).get("people", {})
    
    rows = []

    # Process innings
    for inning in data["innings"]:
        batting_team = inning["team"]  # The team that is batting
        ball_number = 1  # Reset ball counter at the start of innings
        total_team_runs = 0
        total_team_wickets = 0
        
        for over in inning["overs"]:
            for delivery in over["deliveries"]:
                # Update team total runs and wickets
                total_team_runs += delivery["runs"]["total"]
                
                # extract wickets information:
                wicket = 0 # default to no wicket
                wicket_type = None
                if "wickets" in delivery:
                    for w in delivery["wickets"]:
                        wicket = 1  # A wicket was taken
                        wicket_type = w.get("kind", "Unknown")
                        total_team_wickets += 1 # increment total wicket count
                
                # extract player IDs:
                striker = delivery["batter"]
                bowler = delivery["bowler"]
                nonstriker = delivery["non_striker"]
                striker_ID = player_registry.get(striker, "Unknown")
                bowler_ID = player_registry.get(bowler, "Unknown")
                nonstriker_ID = player_registry.get(nonstriker, "Unknown")
                
                # Extract runs
                batter_runs = delivery["runs"].get("batter", 0)
                extras_runs = delivery["runs"].get("extras", 0)
                total_delivery_runs = delivery["runs"].get("total", 0)

                # Append row data
                rows.append({
                    "batting_team": batting_team,
                    "ball_number": ball_number,
                    "total_team_runs": total_team_runs,  # Cumulative runs
                    "total_team_wickets": total_team_wickets,  # Cumulative wickets
                    "striker_ID": striker_ID,
                    "bowler_ID": bowler_ID,
                    "nonstriker_ID": nonstriker_ID,
                    "batter_runs": batter_runs,
                    "extras_runs": extras_runs,
                    "total_delivery_runs": total_delivery_runs,
                    "wicket": wicket,
                    "wicket_type": wicket_type
                })

                ball_number += 1  # Increment ball number

    # Convert to Pandas DataFrame
    df = pd.DataFrame(rows)

    return df



# Function to convert a .json game file into a pandas dataframe of one observation which includes info about the game
# Params:
#   filename: the .json file to extract info from
# Return:
    # pandas df: containing game info like date, game_id, batting_1st_team, batting_2nd_team, venue, format, officials
def convert_game_json_to_game_info_df(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extract the game ID from the filename (e.g., "3344.json" → "3344")
    game_id = os.path.splitext(os.path.basename(filename))[0]
    
    # Extract game date and convert to YYYYMMDD format
    game_date_raw = data["info"]["dates"][0]
    game_date = game_date_raw.replace("-", "")
    
    # Extract teams
    batting_first_team = data["innings"][0]["team"]
    all_teams = data["info"]["teams"]
    bowling_first_team = [team for team in all_teams if team != batting_first_team][0] 

    # Venue and match type
    venue = data["info"]["venue"]
    match_type = data["info"]["match_type"]
    
    # Umpires (safe fallback in case keys are missing)
    officials = data["info"].get("officials", {})
    umpires = officials.get("umpires", ["Unknown", "Unknown"])
    umpire_1 = umpires[0] if len(umpires) > 0 else "Unknown"
    umpire_2 = umpires[1] if len(umpires) > 1 else "Unknown"
    
    # Construct the DataFrame
    row = {
        "game_id": game_id,
        "game_date": game_date,
        "batting_first_team": batting_first_team,
        "bowling_first_team": bowling_first_team,
        "venue": venue,
        "match_type": match_type,
        "umpire_1": umpire_1,
        "umpire_2": umpire_2
    }
    
    return pd.DataFrame([row])
