# implement multi dimensional robust synthetic control (mRSC) algorithm:

# import necessary libraries:
import pandas as pd
import numpy as np
from sql_tables import Match, MatchFormat, BallByBall, initialize_db_params, Session
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text


engine = 0
Session = 0
# Load environment variables
def initialize_db_params():
    print("Initialize params")
    global engine
    global Session
    load_dotenv()
    username = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_USER")
    password = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_PW")
    host = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_HOST")
    port = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_PORT")
    database = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_DBNAME")
    
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}', echo=False)
    Session = sessionmaker(bind=engine)

# Params:
#   pandas df: ball by ball data for a given team-inning containing columns game_id, inning, ball_number, 
#              total_runs, total_wickets, extras_runs, extras_type         
# Output:
#   pandas df restructured: ball by ball data for a given team-inning containing columns game_id, inning, ball_number
#                           total_runs, total_wickets
#              the rows that contain values of "wides" or "noballs" will merge with the row before them. The 
#              total_team runs will have the extras runs added to it, and so will the total_wickets, if a wicket happened on
#              that wide or no ball. This ensures that a max of 120 rows will be there for a T20 game each inning, since
#              we won't be counting illegal deliveries as a ball anymore. 
def merge_extras_for_ball_by_ball_for_specific_game_inning(df):
    df = df.copy()
    illegal_rows = df[df['extras_type'].isin(['wides', 'noballs'])].index
    for idx in illegal_rows:
        if idx == 0:
            continue
        df.loc[idx - 1, 'total_team_runs'] += df.loc[idx, 'extras_runs']
        df.loc[idx - 1, 'total_team_wickets'] += df.loc[idx, 'total_team_wickets'] - df.loc[idx - 1, 'total_team_wickets']
    df = df[~df['extras_type'].isin(['wides', 'noballs'])].reset_index(drop=True)
    return df[['game_id', 'inning', 'ball_number', 'total_team_runs', 'total_team_wickets']]


# Helper function: get all ball-by-ball data for a specific game
def get_game_data(session, game_id):
    q = (
        session.query(
            BallByBall.game_id,
            BallByBall.inning,
            BallByBall.ball_number,
            BallByBall.total_team_runs,
            BallByBall.total_team_wickets,
            BallByBall.extras_runs,
            BallByBall.extras_type
        )
        .filter(BallByBall.game_id == game_id)
    )
    return pd.read_sql(q.statement, session.bind)



# Params:
#   session: the session connecting to the mySQL database
#   match_format: defualts to "T20", but can also be "ODI", "ODM", "IT20"
# Output:
# A 3 dimensional tensor
#   dimensions (3): metrics (total_runs, total_wickets), time (ball_number), unit (team-inning)
def get_tensor_for_all_games_of_a_format(session, match_format="T20"):
    format_id = session.query(MatchFormat.id).filter(MatchFormat.match_format == match_format).scalar()
    game_ids = session.query(Match.game_id).filter(Match.format == format_id).all()
    game_ids = [g[0] for g in game_ids]

    merged = []
    for idx, gid in enumerate(game_ids, start=1):
        print(f"🔍 Processing game {idx} out of {len(game_ids)}: game_id = {gid}")
        df = get_game_data(session, gid)
        if df['inning'].nunique() > 2:
            print(f"ℹ️  Found game with super overs (more than 2 innings): game_id = {gid}. Skipping super over data.")
        df = df[df['inning'].isin([1, 2])]  # Exclude super overs
        df['unit_id'] = df['game_id'].astype(str) + "_inn" + df['inning'].astype(str)
        for unit in df['unit_id'].unique():
            sub_df = df[df['unit_id'] == unit].sort_values(by='ball_number').reset_index(drop=True)
            cleaned = merge_extras_for_ball_by_ball_for_specific_game_inning(sub_df)
            if len(cleaned) > 120:
                print(f"⚠️⚠️⚠️⚠️⚠️  Unit {unit} has more than 120 rows: {len(cleaned)} rows")
            cleaned['unit_id'] = unit
            merged.append(cleaned)

    cleaned_df = pd.concat(merged, ignore_index=True)
    units = cleaned_df['unit_id'].unique()
    unit_map = {uid: idx for idx, uid in enumerate(units)}
    cleaned_df['unit_idx'] = cleaned_df['unit_id'].map(unit_map)
    max_ball_number = cleaned_df['ball_number'].max()

    tensor = np.full((len(units), max_ball_number, 2), np.nan)
    for row in cleaned_df.itertuples():
        i = row.unit_idx
        t = row.ball_number - 1
        tensor[i, t, 0] = row.total_team_runs
        tensor[i, t, 1] = row.total_team_wickets

    return tensor, units, ['total_team_runs', 'total_team_wickets']















# MAIN:
def main():
    # initialize db parameters
    initialize_db_params()
    # start session and connect to database
    session = Session()
    session.execute(text('SELECT 1'))
    # print that database connection was successful:
    print("✅ Database session started successfully.")
    
    # create tensor of all data 
    tensor, units, metrics = get_tensor_for_all_games_of_a_format(session)
    print("✅ Tensor shape:", tensor.shape)
    print("🔹 Sample units:", units[:5])
    print("🔹 Metrics:", metrics)
    
    return 0


if __name__ == "__main__":
    main()