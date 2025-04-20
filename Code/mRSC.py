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
    # find the format id associated with the match format
    format_id = session.query(MatchFormat.id).filter(MatchFormat.match_format == match_format).scalar()
    # get all game ids associated with that match format
    game_ids = session.query(Match.game_id).filter(Match.format == format_id).all()
    game_ids = [g[0] for g in game_ids]
    
    # get max balls per inning:
    if match_format == "T20" or match_format == "IT20":
        max_balls_allowed = 120
    elif match_format == "ODI" or match_format == "ODM":
        max_balls_allowed = 300
    
    merged = []
    # loop over each game
    for idx, gid in enumerate(game_ids, start=1):
        print(f"🔍 Processing game {idx} out of {len(game_ids)}: game_id = {gid}")
        # get ball by ball data for this specific game_id
        df = get_game_data(session, gid)

        # if there was a super over, only consider the actual game, not the super over
        if df['inning'].nunique() > 2:
            print(f"ℹ️ Found game with super overs (more than 2 innings): game_id = {gid}. Skipping super over data.")
        df = df[df['inning'].isin([1, 2])]  # Exclude super overs
        # label each unit with the format of {game_id}_inn{number}
        df['unit_id'] = df['game_id'].astype(str) + "_inn" + df['inning'].astype(str)
        # loop over each inning in a specific game:
        for unit in df['unit_id'].unique():
            sub_df = df[df['unit_id'] == unit].sort_values(by='ball_number').reset_index(drop=True)
            cleaned = merge_extras_for_ball_by_ball_for_specific_game_inning(sub_df)
            
            # if rows still more than max balls for that format
            if len(cleaned) > max_balls_allowed:
                print(f"⚠️ Unit {unit} has more than {max_balls_allowed} rows: {len(cleaned)} rows — trimming to 120.")
                cleaned = cleaned.iloc[:max_balls_allowed]
            
            # if innings is shorter than max balls, pad with NANs:
            if len(cleaned) < max_balls_allowed:
                pad_len = max_balls_allowed - len(cleaned)
                padding = pd.DataFrame({
                    'game_id': [gid] * pad_len,
                    'inning': [sub_df['inning'].iloc[0]] * pad_len,
                    'ball_number': [np.nan] * pad_len,
                    'total_team_runs': [np.nan] * pad_len,
                    'total_team_wickets': [np.nan] * pad_len,
                    'unit_id': [unit] * pad_len
                })
                cleaned = pd.concat([cleaned, padding], ignore_index=True)
            
            cleaned['unit_id'] = unit
            merged.append(cleaned)

    cleaned_df = pd.concat(merged, ignore_index=True)
    units = cleaned_df['unit_id'].unique()
    unit_map = {uid: idx for idx, uid in enumerate(units)}
    cleaned_df['unit_idx'] = cleaned_df['unit_id'].map(unit_map)
    max_ball_number = cleaned_df.groupby('unit_id').size().max()

    tensor = np.full((len(units), max_ball_number, 2), np.nan)
    for row in cleaned_df.itertuples():
        i = row.unit_idx
        t = row.Index % 120
        tensor[i, t, 0] = row.total_team_runs
        tensor[i, t, 1] = row.total_team_wickets

    # Save tensor to file for future use
    np.savez_compressed("t20_tensor_data.npz", tensor=tensor, units=units, metrics=['total_team_runs', 'total_team_wickets'])
    
    return tensor, units, ['total_team_runs', 'total_team_wickets']



# Constructs: Z (donor tensor) and X1 (treatment unit's pre-intervention data): 
# Params:
#   tensor: numpy array of 3 dimensions with shape (N units, T balls, K metrics)
#   units: array of unit ids (in context of cricket, these are innings) (length N)
#   metrics: list of metric names (length K)
#   intervention_ball: the ball at which the intervention takes place (default = 90 balls)
#   target_unit: the unit which we want to forecast for (default inning = 0)
# Returns:
#   Z: numpy array of shape (N-1, K*T)
#   X1: numpy array of shape (1, K*T0)
#   unit_ids: donor unit identifiers
def get_Z_and_X1(tensor, units, metrics, intervention_ball=90, target_unit=0):
    # assign values to N, T, K based on the shape of the tensor
    N, T, K = tensor.shape
    # set intervention ball as T0:
    T0 = intervention_ball
    
    # extract X1 for the specified treatment unit:
    X1_tensor = tensor[target_unit, :T0, :]  # shape (T0, K)
    X1 = X1_tensor.T.reshape(1, -1)  # reshape to (1, K*T0)  (this concatenates the K metrics into 1 row)
        # X1 is now a row vector with length K*T0
    
    # Step 2: Prepare donor matrix Z using all other units (exclude target_unit):
    Z_list = []
    # for each unit (inning) N
    for i in range(N):
        # skip if i is equal to the target unit, since we want to leave that out
        if i == target_unit:
            continue
        # take transpose of ith unit from tensor and then flatten the matrix row wise to get dimensions K * T
        Z_i = tensor[i].T.reshape(-1)
        Z_list.append(Z_i)
    Z = np.stack(Z_list)  # shape (N-1, K*T)
    
    # set donor unit ids to be all units except the target unit
    donor_unit_ids = np.delete(units, target_unit)
    # return output
    return Z, X1, donor_unit_ids











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
    '''
    tensor, units, metrics = get_tensor_for_all_games_of_a_format(session)
    '''
    
    # load the saved T20 ball by ball data tensor:
    data = np.load("t20_tensor_data.npz", allow_pickle=True)
    tensor = data["tensor"]
    units = data["units"]
    metrics = data["metrics"]
    
    # print out tensor characteristics
    print("✅ Tensor shape:", tensor.shape)
    print("🔹 Sample units:", units[:5])
    print("🔹 Metrics:", metrics)
    
    # create X1 and Z matrices:
    Z, X1, donor_unit_ids = get_Z_and_X1(tensor, units, metrics, intervention_ball=60, target_unit=0)
    # print out matrices characteristics:
    print("Shape of Z: ", Z.shape)
    print("Shape of X1: ", X1.shape)

    # return 0
    return 0


if __name__ == "__main__":
    main()
    
