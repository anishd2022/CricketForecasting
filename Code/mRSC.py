# implement multi dimensional robust synthetic control (mRSC) algorithm:

# import necessary libraries:
import pandas as pd
import numpy as np
from sql_tables import Match, MatchFormat, BallByBall, initialize_db_params, Session
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import optuna
import random
from numpy.linalg import solve


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



def get_tensor_for_all_games_of_a_format_include_inning(session, match_format="T20"):
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

    tensor = np.full((len(units), max_ball_number, 3), np.nan)
    for row in cleaned_df.itertuples():
        i = row.unit_idx
        t = row.Index % 120
        tensor[i, t, 0] = row.total_team_runs
        tensor[i, t, 1] = row.total_team_wickets
        tensor[i, t, 2] = row.inning

    # Save tensor to file for future use
    np.savez_compressed("t20_tensor_data_with_inning.npz", tensor=tensor, units=units, metrics=['total_team_runs', 'total_team_wickets', 'inning'])
    
    return tensor, units, ['total_team_runs', 'total_team_wickets', 'inning']




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



# Helper function to filter out rows in Z (units) that contain any missing values
def filter_units_with_no_missing(Z, unit_ids=None):
    """
    Filters Z to keep only rows (units) with no NaN entries.

    Params:
    - Z: numpy array of shape (N-1, K*T)
    - unit_ids: optional list/array of unit identifiers of length N-1

    Returns:
    - Z_filtered: numpy array with only complete rows
    - kept_indices: indices of rows kept (w.r.t. original Z)
    - filtered_unit_ids: filtered unit_ids (if provided)
    """
    # Identify complete rows (no NaNs)
    mask = ~np.isnan(Z).any(axis=1)
    Z_filtered = Z[mask]
    kept_indices = np.where(mask)[0]

    if unit_ids is not None:
        filtered_unit_ids = np.array(unit_ids)[mask]
    else:
        filtered_unit_ids = None

    print(f"✅ Retained {Z_filtered.shape[0]} out of {Z.shape[0]} units with no missing values.")

    return Z_filtered, kept_indices, filtered_unit_ids



# Params:
#   Z: numpy array of shape (N-1, K*T)
#   lambda_thresh: singular value threshold (λ)
# Output:
#   Mc: denoised version of Z with same shape
#   S: indices of retained singular values
#   singular_values: full array of singular values (for diagnostics)
def denoise_Z_using_svd(Z, lambda_thresh):
    # filter Z to only have units with no missing values:
    Z_filtered, kept_indices, filtered_unit_ids = filter_units_with_no_missing(Z)
    # compute observed fraction of values in Z:
    prop_missing = np.isnan(Z_filtered).sum() / Z_filtered.size
    observed_fraction = 1.0 - prop_missing
    
    # print out Z_filtered characteristics:
    print("Shape of filtered Z: ", Z_filtered.shape)
    
    # perform SVD:
    U, s, VT = np.linalg.svd(Z_filtered, full_matrices=False)
    
    # keep only singular values >= lambda threshold:
    S = []
    for i, singular_value in enumerate(s):
        if singular_value >= lambda_thresh:
            S.append(i)
    print(f"🔍 Retained {len(S)} singular values out of {len(s)} above threshold λ = {lambda_thresh}")
    
    # reconstruct Z using only top singular values (hard thresholding):
    # start by creating an zero matrix with same dimensions as Z filtered:
    Mc = np.zeros_like(Z_filtered)
    # adds each rank-1 component into Mc:
    for i in S:
        Mc += s[i] * np.outer(U[:, i], VT[i, :])
    
    # Rescale by observed fraction if Z was sparse/masked
    Mc = Mc / observed_fraction
    
    # return output
    return Mc, S, s


# Construct McT0 from Mc:
def construct_McT0(Mc, K, T0):
    """
    Extract McT0 (pre-intervention donor data)
    Mc: numpy array (N-1, K*T)
    Returns:
    - McT0: (N-1, K*T0), taking T0 time steps from each of K metrics
    """
    N_minus_1, KT = Mc.shape
    T = KT // K
    indices = []
    for k in range(K):
        start = k * T
        end = start + T0
        indices.extend(range(start, end))
    McT0 = Mc[:, indices]
    return McT0


# Create delta matrix given weights for each metric:
# Params:
#   weights: a vector of weights corresponding to each metric
#   T0: the intervention ball number
# Output:
#   a diagonal matrix delta whose diagonal values are the weights (each repeated T0 times)
def create_delta_matrix(weights, T0):
    repeated_weights = np.repeat(weights, T0)
    return np.diag(repeated_weights)

# apply metric weights using diagonal matrix Δ:
# Params: 
#   McT0: the denoised donor matrix pre-intervention
#   X1: the tretment unit pre-intervention
#   delta: the weighting matrix
# Output:
#   a weighted McT0 matrix and a weighted X1 matrix
def apply_metric_weights(McT0, X1, delta):
    McT0_weighted = McT0 @ delta
    X1_weighted = X1 @ delta
    return McT0_weighted, X1_weighted
    
    


# Solve the weighted least squares regression:
# we want to put the appropriate weights on each donor unit such that the error is minimized and we can create
# an accurate counterfactual.
# Params:
#   McT0_weighted: (N-1, K*T0) numpy array
#   X1_weighted: (1, K*T0) numpy array
# Output:
#   beta_hat: (N-1,) vector of optimal weights
def solve_weighted_least_squares(McT0_weighted, X1_weighted):
    # Transpose the donor matrix to shape (K*T0, N-1)
    A = McT0_weighted.T  # shape: (K*T0, N-1)
    # transpose the pre-intervention treatment data to shape (K*T0, 1)
    b = X1_weighted.T    # shape: (K*T0, 1)
    
    # solve least squares:
    beta_hat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Flatten to 1D array for convenience
    return beta_hat.ravel()


# Params:
#   Mc: numpy array of shape (N-1, K*T), denoised donor matrix
#   beta_hat: vector of shape (N-1,), estimated weights from WLS
#   K: number of metrics (e.g., 2 for [total_runs, total_wickets])
# Output:
#   counterfactuals: array of shape (T, K), synthetic control time series
#                    for the treatment unit for each metric.
def reconstruct_counterfactual(Mc, beta_hat, K):
    # find total T based on matrix dimensions:
    N_minus_1, KT = Mc.shape
    T = KT // K
    # initialize empty counterfactuals matrix:
    counterfactuals = []
    
    # for each metric:
    for k in range(K):
        Mc_k = Mc[:, k*T:(k+1)*T]  # Extract Mc^(k): shape (N-1, T)
        Mc1_k = beta_hat @ Mc_k    # Weighted avg: shape (T,)
        counterfactuals.append(Mc1_k)   # add to the counterfactuals

    # stack counterfactuals per metric to give 2D matrix structure:
    return np.stack(counterfactuals, axis=1)  # shape (T, K)
    


# Plot true vs estimated counterfactual values for each metric for the treatment unit:
# Params:
#   counterfactual: numpy array (T, K)
#   tensor: original tensor data (N, T, K)
#   treatment_unit: index of treated unit
#   metrics: list of metric names (length K)
# Output:
#   Plot
def plot_counterfactual_vs_truth(counterfactual, tensor, treatment_unit, metrics):
    true_values = tensor[treatment_unit]  # shape (T, K)
    T, K = true_values.shape
    
    # for each metric...
    for k in range(K):
        plt.figure(figsize=(10, 4))
        plt.plot(range(T), true_values[:, k], label="True", linewidth=2)
        plt.plot(range(T), counterfactual[:, k], label="Counterfactual", linestyle='--', linewidth=2)
        plt.xlabel("Ball Number")
        plt.ylabel(metrics[k])
        plt.title(f"Treatment Unit ({treatment_unit}): {metrics[k]}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Function to compute post-intervention MSE for a given counterfactual and treatment unit
# Params:
# - counterfactual: array of shape (T, K)
# - true_tensor: original tensor of shape (N, T, K)
# - treatment_unit: index of treated unit
# - intervention_ball: the ball at which the intervention starts (T0)
# Returns:
# - list of MSEs for each metric (length K)
def compute_post_intervention_mse(counterfactual, true_tensor, treatment_unit, intervention_ball):
    T, K = counterfactual.shape
    mse_list = []
    true_post = true_tensor[treatment_unit, intervention_ball:, :]
    pred_post = counterfactual[intervention_ball:, :]
    valid_metric_found = False

    for k in range(K):
        valid_mask = ~np.isnan(true_post[:, k])
        if np.any(valid_mask):
            valid_metric_found = True
            mse = np.mean((pred_post[valid_mask, k] - true_post[valid_mask, k]) ** 2)
            mse_list.append(mse)
        else:
            print(f"⚠️ No valid post-intervention data for metric {k} in unit {treatment_unit}")
            mse_list.append(np.nan)

    if not valid_metric_found:
        return None  # signal that this trial is unusable

    return mse_list


# Function to optimize hyperparameters using Optuna
# use random subset of units to optimize on for each trial:
# Params:
# - tensor, units, metrics: as before
# - intervention_ball, treatment_unit: as before
# Output:
# - best (lambda, weight) and associated MSEs
# this tries to find the parameters that minimize the mean of the MSE of runs
def optuna_objective(trial, tensor, units, metrics, intervention_ball, max_units=50):
    lam = trial.suggest_float("lambda", 1.0, 30.0)
    weight_run = trial.suggest_float("weight_run", 0.0000000001, 10000000000)
    weight_wicket = trial.suggest_float("weight_wicket", 0.0000000001, 10000000000)
    weights = np.array([weight_run, weight_wicket])

    total_mse_across_units = []

    sampled_units = random.sample(range(len(units)), min(max_units, len(units)))

    for idx, treatment_unit in enumerate(sampled_units):
        try:
            print(f"🔁 [{idx + 1}/{len(sampled_units)}] Evaluating treatment unit {treatment_unit}")
            Z, X1, _ = get_Z_and_X1(tensor, units, metrics, intervention_ball, treatment_unit)
            Mc, _, _ = denoise_Z_using_svd(Z, lambda_thresh=lam)
            McT0 = construct_McT0(Mc, K=len(metrics), T0=intervention_ball)
            delta = create_delta_matrix(weights=weights, T0=intervention_ball)
            McT0_w, X1_w = apply_metric_weights(McT0, X1, delta)
            beta_vec = solve_weighted_least_squares(McT0_w, X1_w)
            counterfactual = reconstruct_counterfactual(Mc, beta_hat=beta_vec, K=len(metrics))
            mse_list = compute_post_intervention_mse(counterfactual, tensor, treatment_unit, intervention_ball)
            if mse_list is None or np.isnan(mse_list).any():
                print(f"⚠️ Skipping unit {treatment_unit} due to invalid MSE")
                continue
            run_mse = mse_list[0]
            print(f"   ➤ MSE (runs only) for unit {treatment_unit}: {run_mse:.4f}")
            total_mse_across_units.append(run_mse)
        except Exception as e:
            print(f"⚠️ Skipping unit {treatment_unit} due to error: {e}")
            continue  # skip units with failure

    final_mse = np.mean(total_mse_across_units) if total_mse_across_units else float("inf")
    print(f"🎯 Trial completed with λ={lam:.2f}, weights={[weight_run, weight_wicket]} → MSE={final_mse:.4f}")
    return final_mse


# runs hyperparameter search to find best hyperparams:
def run_optuna_search(tensor, units, metrics, intervention_ball, n_trials=20, max_units=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, tensor, units, metrics, intervention_ball, max_units), n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    print("\n✅ Best Parameters from Optuna:")
    print(f"Lambda: {best_params['lambda']:.4f}")
    print(f"Weights: [Run: {best_params['weight_run']:.4f}, Wicket: {best_params['weight_wicket']:.4f}]")
    print(f"Best Avg MSE across units: {best_value:.4f}")

    return best_params, best_value





























# MAIN:
def main():
    '''
    # initialize db parameters
    initialize_db_params()
    # start session and connect to database
    session = Session()
    session.execute(text('SELECT 1'))
    # print that database connection was successful:
    print("✅ Database session started successfully.")
    
    # create tensor of all data 
    tensor, units, metrics = get_tensor_for_all_games_of_a_format_include_inning(session)
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
    
    
    
    # set intervention ball and target unit:
    intervention_ball_number = 60
    treatment_unit = 3476
    # how much weight you want to give to [runs, wickets]
    weights_vector = np.array([1, 1])
    # weights_vector_2 = np.array([1, 1])
    # lambda threshold
    lambda_threshold = 7
    
    # create X1 and Z matrices:
    Z, X1, donor_unit_ids = get_Z_and_X1(tensor, units, metrics, intervention_ball=intervention_ball_number, target_unit=treatment_unit)
    # print out matrices characteristics:
    print("Shape of Z: ", Z.shape)
    print("Shape of X1: ", X1.shape)
    print(Z[46, :])

    # de-noise the Z matrix using singular value decomposition (SVD):
    Mc, S, s = denoise_Z_using_svd(Z, lambda_thresh=lambda_threshold)
    print("Shape of Mc: ", Mc.shape)
    print(Mc[46, :])
    
    
    # plot singular values:
    '''
    plt.plot(s[1:], marker='o')
    plt.title("Singular Value Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.show()
    '''
    
    
    # construct McT0 from Mc:
    McT0 = construct_McT0(Mc, K=len(metrics), T0=intervention_ball_number)
    print("Shape of McT0: ", McT0.shape)
    print(McT0[46, :])
    
    # create delta matrix from the weights:
    delta = create_delta_matrix(weights=weights_vector, T0=intervention_ball_number)
    # delta_2 = create_delta_matrix(weights=weights_vector_2, T0=intervention_ball_number)
    print("Shape of delta matrix: ", delta.shape)
    print(delta)
    
    # apply delta matrix onto McT0 and X1:
    McT0_weighted, X1_weighted = apply_metric_weights(McT0, X1, delta)
    # McT0_weighted_2, X1_weighted_2 = apply_metric_weights(McT0, X1, delta_2)
    print("Shape of McT0 weighted: ", McT0_weighted.shape)
    print("Shape of X1 weighted: ", X1_weighted.shape)
    # print(McT0_weighted[46, :])
    
    # get beta vector:
    beta_vec = solve_weighted_least_squares(McT0_weighted, X1_weighted)
    # beta_vec_2 = solve_weighted_least_squares(McT0_weighted_2, X1_weighted_2)
    # print("Shape of beta vector: ", beta_vec.shape)
    # print(beta_vec)
    # print("diff:", np.linalg.norm(beta_vec - beta_vec_2))
    # get counterfactual:
    counterfactual = reconstruct_counterfactual(Mc, beta_hat=beta_vec, K=len(metrics))
    # print("Shape of counterfactual: ", counterfactual.shape)
    # print(counterfactual)
    
    
    # plot counterfactual estimates vs truth:
    plot_counterfactual_vs_truth(counterfactual, tensor, treatment_unit, metrics)
    
    
    # compute post intervention MSE for a given unit / intervention ball:
    mse = compute_post_intervention_mse(counterfactual, tensor, treatment_unit, intervention_ball_number)
    # print(mse)
    
    
    '''
    # TRY TUNING PARAMS:
    best_params, best_avg_mse = run_optuna_search(tensor, units, metrics, intervention_ball_number, n_trials=50, max_units=50)
    print("Best params: ", best_params)
    print("Best avg mse: ", best_avg_mse)
    '''
    
    # return 0
    return 0


if __name__ == "__main__":
    main()
    



'''
✅ Best Parameters from Optuna:
Lambda: 7.7112
Weights: [Run: 1021525777.7835, Wicket: 2635021527.6955]
Best Avg MSE across units: 79.8973
Best params:  {'lambda': 7.711245476560286, 'weight_run': 1021525777.7834821, 'weight_wicket': 2635021527.695471}
Best total mse:  79.89730086875244
'''



'''
# set intervention ball and target unit:
    intervention_ball_number = 60
    treatment_unit = 697
    # how much weight you want to give to [runs, wickets]
    weights_vector = np.array([1.0493, 1.9604])
    # lambda threshold
    lambda_threshold = 3.0848
'''
# The beta vector does not change despite changing the weights_vector...
# weights_vector = np.array([1.0493, 1.9604])
# [ 0.00075673 -0.00032352 -0.00023451 ...  0.00215572 -0.00025168
#  -0.00087367]

# weights_vector = np.array([1.0493, 100.9604])
# [ 0.00075673 -0.00032352 -0.00023451 ...  0.00215572 -0.00025168
#  -0.00087367]