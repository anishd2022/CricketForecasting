# implement multiple time series model:

# start with linear regression:
# predicting runs scored in over 12 after knowing runs scored in overs 4, 6, 8, 10 gives R^2 of 0.92
# predicting runs scored in over 15 after knowing runs scored in overs 4, 6, 8, 10 gives R^2 of 0.82

# now try ARIMA model:

# dimensions are [innings, balls, metrics]  [N, T, K]

# import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter



# FUNCTION:
def build_linear_regression_model(tensor, feature_balls, target_ball):
    """
    Builds and evaluates a linear regression model.

    Params:
    - tensor: numpy array of shape (N, 120, 2)
    - feature_balls: list of ball numbers (e.g., [24, 36, 48, 60])
    - target_ball: ball number to predict (e.g., 90)

    Returns:
    - model: trained LinearRegression model
    - r2: R-squared score on the training data
    """
    X = []
    y = []

    for unit_idx in range(tensor.shape[0]):
        total_runs = tensor[unit_idx, :, 0]  # cumulative runs

        # Skip innings if NaNs at important indices
        indices = [b-1 for b in feature_balls + [target_ball]]
        if np.any(np.isnan(total_runs[indices])):
            continue

        features = [total_runs[b-1] for b in feature_balls]
        target = total_runs[target_ball-1]

        X.append(features)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Prepared dataset: {X.shape[0]} examples")

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    print(f"\n📈 Model R^2 Score: {r2:.4f}")

    print("\nModel Coefficients:")
    for idx, coef in enumerate(model.coef_):
        print(f"  Coef for Ball {feature_balls[idx]}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    return model, r2



def arima_predict_post_intervention(tensor, unit_index=0, intervention_ball=60, arima_order=(1,0,1), plot=True):
    """
    Fit ARIMA on data before intervention_ball, forecast after it, and compare to actuals.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    cumulative_runs = tensor[unit_index, :, 0]
    cumulative_runs = cumulative_runs[~np.isnan(cumulative_runs)]

    if len(cumulative_runs) <= intervention_ball:
        print("⚠️ Not enough post-intervention data.")
        return None, None

    # Use per-ball runs
    per_ball_runs = np.diff(cumulative_runs)
    pre = per_ball_runs[:intervention_ball - 1]  # exclude the ball before intervention

    # Fit ARIMA on pre-intervention data
    model = ARIMA(pre, order=arima_order)
    model_fit = model.fit()

    # Forecast for post-intervention duration
    forecast_horizon = len(per_ball_runs) - (intervention_ball - 1)
    forecast = model_fit.forecast(steps=forecast_horizon)

    # Reconstruct predicted cumulative run trajectory
    predicted_post = np.cumsum(forecast) + cumulative_runs[intervention_ball - 1]

    # Get true cumulative for comparison
    actual_post = cumulative_runs[intervention_ball:]

    # Compute R²
    valid = ~np.isnan(actual_post)
    r2 = r2_score(actual_post[valid], predicted_post[valid])

    # Plot if needed
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(cumulative_runs, label="Actual", linewidth=2)
        plt.plot(range(intervention_ball, intervention_ball + len(predicted_post)), predicted_post,
                 label="ARIMA Forecast", linestyle='--', linewidth=2)
        plt.axvline(intervention_ball, color='red', linestyle=':', label="Intervention Ball")
        plt.title(f"ARIMA Forecast Post-Intervention | R² = {r2:.3f}")
        plt.xlabel("Ball Number")
        plt.ylabel("Cumulative Runs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return model_fit, r2



def evaluate_arima_rmse_across_units(tensor, intervention_ball=60, arima_order=(2, 1, 2), max_units=1000, verbose=True):
    """
    Loops through multiple innings and evaluates average post-intervention ARIMA RMSE.
    Also plots a histogram of RMSE values.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    import numpy as np

    try:
        from tqdm import tqdm
        iterator = tqdm(range(min(max_units, tensor.shape[0])), desc="Evaluating ARIMA RMSE")
    except ImportError:
        iterator = range(min(max_units, tensor.shape[0]))

    rmse_list = []

    for i in iterator:
        try:
            cumulative_runs = tensor[i, :, 0]
            cumulative_runs = cumulative_runs[~np.isnan(cumulative_runs)]
            if len(cumulative_runs) <= intervention_ball:
                continue

            per_ball_runs = np.diff(cumulative_runs)
            pre = per_ball_runs[:intervention_ball - 1]

            # Fit ARIMA
            model = ARIMA(pre, order=arima_order)
            model_fit = model.fit()

            forecast_horizon = len(per_ball_runs) - (intervention_ball - 1)
            forecast = model_fit.forecast(steps=forecast_horizon)
            predicted_post = np.cumsum(forecast) + cumulative_runs[intervention_ball - 1]
            actual_post = cumulative_runs[intervention_ball:]

            valid = ~np.isnan(actual_post)
            if np.any(valid):
                mse = mean_squared_error(actual_post[valid], predicted_post[valid])
                rmse = np.sqrt(mse)
                rmse_list.append(rmse) 

        except Exception as e:
            if verbose:
                print(f"⚠️ Error on unit {i}: {e}")
            continue

    if not rmse_list:
        print("❌ No valid innings processed. Returning NaN.")
        return float("nan"), []

    avg_rmse = np.mean(rmse_list)
    print(f"\n✅ Evaluated {len(rmse_list)} valid innings.")
    print(f"📉 Average ARIMA RMSE across innings: {avg_rmse:.4f}")

    # Plot histogram
    bin_edges = np.linspace(0, 60, 61)
    plt.figure(figsize=(10, 5))
    plt.hist(rmse_list, bins=bin_edges, edgecolor='black', color='lightcoral')
    plt.axvline(avg_rmse, color='blue', linestyle='--', linewidth=2, label=f"Mean RMSE = {avg_rmse:.2f}")
    plt.xlim(0, 60)
    plt.title("Distribution of ARIMA Post-Intervention RMSE Scores")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return avg_rmse, rmse_list



def get_time_series_for_specific_inning(tensor, inning_number, end_ball):
    """
    Returns the time series for a specific inning up to and including end_ball.
    Shape: (end_ball + 1, K)
    """
    return tensor[inning_number, :end_ball + 1, :]


def get_distance_between_two_time_series(time_series_1, time_series_2, lambda_wicket=100):
    """
    Computes the scaled Euclidean distance between two innings.
    Assumes the first column is runs and the second column is wickets.
    λ (lambda_wicket) is the weight on wicket differences.
    """
    # Extract run and wicket trajectories
    runs_1, wickets_1 = time_series_1[:, 0], time_series_1[:, 1]
    runs_2, wickets_2 = time_series_2[:, 0], time_series_2[:, 1]

    # Compute squared differences
    run_diff_sq = (runs_1 - runs_2) ** 2
    wicket_diff_sq = (wickets_1 - wickets_2) ** 2

    # Weighted sum
    total_distance = np.sum(run_diff_sq + lambda_wicket * wicket_diff_sq)
    return total_distance


def get_knn_next_ball_deltas(tensor, treatment_idx, intervention_ball, k_neighbors=5, lambda_wicket=100):
    """
    Finds k-nearest neighbors to a treatment unit up to intervention_ball.
    Returns the list of [run_delta, wicket_delta] at ball (intervention_ball + 1) for each neighbor.
    """
    N, T, K = tensor.shape
    assert intervention_ball + 1 < T, "Intervention ball too close to end of innings"

    treatment_ts = get_time_series_for_specific_inning(tensor, treatment_idx, intervention_ball)
    
    distances = []
    for donor_idx in range(N):
        if donor_idx == treatment_idx:
            continue  # skip self
        donor_ts = get_time_series_for_specific_inning(tensor, donor_idx, intervention_ball)
        dist = get_distance_between_two_time_series(treatment_ts, donor_ts, lambda_wicket)
        distances.append((donor_idx, dist))

    # Sort by distance and get top K
    distances.sort(key=lambda x: x[1])
    top_k_donors = [idx for idx, _ in distances[:k_neighbors]]

    # Get deltas at ball T0+1
    deltas = []
    for idx in top_k_donors:
        run_delta = tensor[idx, intervention_ball + 1, 0] - tensor[idx, intervention_ball, 0]
        wicket_delta = tensor[idx, intervention_ball + 1, 1] - tensor[idx, intervention_ball, 1]

        if np.isnan(run_delta) or np.isnan(wicket_delta):
            continue  # skip invalid deltas

        deltas.append([float(run_delta), float(wicket_delta)])

    return deltas



def get_knn_next_ball_deltas_with_custom_history(tensor, current_history, treatment_idx, current_ball, k_neighbors=5, lambda_wicket=100):
    """
    Uses current_history (T × 2) to match with donor innings up to the same ball number.
    Returns K nearest neighbor [run_delta, wicket_delta] at next ball.
    """
    N, T, K = tensor.shape
    assert current_ball + 1 < T

    distances = []
    for donor_idx in range(N):
        if donor_idx == treatment_idx:
            continue

        donor_ts = tensor[donor_idx, :current_ball + 1, :]
        dist = get_distance_between_two_time_series(current_history, donor_ts, lambda_wicket)
        distances.append((donor_idx, dist))

    # Sort and get top K
    distances.sort(key=lambda x: x[1])
    top_k_donors = [idx for idx, _ in distances[:k_neighbors]]

    # Get deltas
    deltas = []
    for idx in top_k_donors:
        run_delta = tensor[idx, current_ball + 1, 0] - tensor[idx, current_ball, 0]
        wicket_delta = tensor[idx, current_ball + 1, 1] - tensor[idx, current_ball, 1]

        if np.isnan(run_delta) or np.isnan(wicket_delta):
            continue  # skip invalid deltas

        deltas.append([float(run_delta), float(wicket_delta)])

    print(deltas)

    return deltas



def simulate_counterfactual_inning(tensor, treatment_idx, intervention_ball, k_neighbors=5, lambda_wicket=100, seed=None):
    """
    Simulates a counterfactual continuation of an inning using sequential KNN sampling.
    Stops at 120 balls or 10 wickets.

    Returns:
        counterfactual: np.array of shape (T - T0, 2), where each row is [cumulative_runs, cumulative_wickets]
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, K = tensor.shape
    assert K >= 2, "Tensor must include at least two metrics: runs and wickets"
    assert intervention_ball + 1 < T, "Intervention point too close to end of game"

    # Start with observed part
    treatment_ts = get_time_series_for_specific_inning(tensor, treatment_idx, intervention_ball)
    counterfactual = []

    # Get starting point: last observed cumulative runs and wickets
    current_runs = treatment_ts[-1, 0]
    current_wickets = treatment_ts[-1, 1]

    current_history = treatment_ts.copy()

    ball = intervention_ball + 1

    while ball < T and current_wickets < 10:
        # Get next KNN deltas
        neighbors = get_knn_next_ball_deltas_with_custom_history(
            tensor, current_history, treatment_idx, ball - 1, k_neighbors, lambda_wicket
        )

        # Sample one neighbor delta
        run_delta, wicket_delta = neighbors[np.random.choice(len(neighbors))]

        # Update current state
        current_runs += run_delta
        current_wickets += wicket_delta
        counterfactual.append([current_runs, current_wickets])

        # Append to history for next KNN round
        new_row = np.array([[current_runs, current_wickets]])
        current_history = np.vstack([current_history, new_row])

        ball += 1

    return np.array(counterfactual)



def plot_simulation_vs_truth(tensor, treatment_idx, intervention_ball, counterfactual):
    """
    Plots the actual trajectory of an inning vs. a simulated counterfactual trajectory.
    
    Parameters:
    - tensor: np.array of shape (N, T, K)
    - treatment_idx: index of the inning to analyze
    - intervention_ball: integer (T₀), the intervention point
    - counterfactual: np.array of shape (T - T₀ - 1, 2), simulated [runs, wickets]
    """

    true_trajectory = tensor[treatment_idx, :, :]  # (120, 2)

    # Build full simulated trajectory aligned with true one
    simulated_trajectory = np.full_like(true_trajectory, np.nan)
    simulated_trajectory[:intervention_ball + 1] = true_trajectory[:intervention_ball + 1]
    sim_len = min(len(counterfactual), true_trajectory.shape[0] - (intervention_ball + 1))
    simulated_trajectory[intervention_ball + 1:intervention_ball + 1 + sim_len] = counterfactual[:sim_len]

    balls = np.arange(true_trajectory.shape[0])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Runs
    axes[0].plot(balls, true_trajectory[:, 0], label="True Runs", color="black", linewidth=2)
    axes[0].plot(balls, simulated_trajectory[:, 0], label="Simulated Runs", linestyle="--", color="red", linewidth=2)
    axes[0].axvline(intervention_ball, color="blue", linestyle=":", label="Intervention")
    axes[0].set_ylabel("Cumulative Runs")
    axes[0].legend()
    axes[0].grid(True)

    # Wickets
    axes[1].plot(balls, true_trajectory[:, 1], label="True Wickets", color="black", linewidth=2)
    axes[1].plot(balls, simulated_trajectory[:, 1], label="Simulated Wickets", linestyle="--", color="red", linewidth=2)
    axes[1].axvline(intervention_ball, color="blue", linestyle=":", label="Intervention")
    axes[1].set_ylabel("Cumulative Wickets")
    axes[1].set_xlabel("Ball Number")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"True vs Simulated Trajectory (Unit {treatment_idx})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_multiple_simulations_vs_truth(tensor, treatment_idx, intervention_ball,
                                       simulate_func, num_sims=10, k_neighbors=50,
                                       lambda_wicket=100):
    """
    Plots the true trajectory vs. multiple simulated counterfactuals.

    Parameters:
    - tensor: np.array of shape (N, T, K)
    - treatment_idx: index of the inning
    - intervention_ball: integer (T₀)
    - simulate_func: function that generates a counterfactual trajectory
    - num_sims: number of simulations to run
    - k_neighbors: number of neighbors for KNN
    - lambda_wicket: weight for wicket distance in KNN
    """

    true_trajectory = tensor[treatment_idx, :, :]
    T = true_trajectory.shape[0]
    balls = np.arange(T)

    # Initialize the figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot true trajectory
    axes[0].plot(balls, true_trajectory[:, 0], color="black", linewidth=2, label="True Runs")
    axes[1].plot(balls, true_trajectory[:, 1], color="black", linewidth=2, label="True Wickets")

    for sim_num in range(num_sims):
        cf_traj = simulate_func(tensor, treatment_idx, intervention_ball,
                                k_neighbors=k_neighbors, lambda_wicket=lambda_wicket, seed=None)

        # Align counterfactual with full 120-ball timeline
        sim_traj = np.full_like(true_trajectory, np.nan)
        sim_traj[:intervention_ball + 1] = true_trajectory[:intervention_ball + 1]
        sim_len = min(len(cf_traj), T - (intervention_ball + 1))
        sim_traj[intervention_ball + 1:intervention_ball + 1 + sim_len] = cf_traj[:sim_len]

        # Plot simulated lines
        axes[0].plot(balls, sim_traj[:, 0], linestyle="--", alpha=0.5, label=f"Sim {sim_num + 1}" if sim_num < 1 else None)
        axes[1].plot(balls, sim_traj[:, 1], linestyle="--", alpha=0.5)

    # Styling
    axes[0].axvline(intervention_ball, color="blue", linestyle=":", label="Intervention")
    axes[1].axvline(intervention_ball, color="blue", linestyle=":")

    axes[0].set_ylabel("Cumulative Runs")
    axes[1].set_ylabel("Cumulative Wickets")
    axes[1].set_xlabel("Ball Number")

    axes[0].legend()
    axes[1].legend(["True Wickets"], loc="upper left")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"True vs {num_sims} Simulated Trajectories (Unit {treatment_idx})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()






















# MAIN:
def main():
    
    # load the saved T20 ball by ball data tensor:
    data = np.load("t20_tensor_data.npz", allow_pickle=True)
    tensor = data["tensor"]
    units = data["units"]
    metrics = data["metrics"]
    
    # print out tensor characteristics
    print("Tensor shape:", tensor.shape)
    print("Sample units:", units[:5])
    print("Metrics:", metrics)
    
    '''
    # dimensions are [innings, balls, metrics]
    inning_data = get_time_series_for_specific_inning(tensor, 1, 120)
    print(inning_data)
    '''
    
    # Call the model builder function:
    feature_balls = [24, 36, 48, 60]  # balls after 4, 6, 8, and 10 overs
    target_ball = 90                  # predict score after 15 overs

    unit = 1883
    intervention = 60
    arima_order = (2, 1, 2)
    lambda_wicket = 20
    knn = 50
    num_sims=3
    
    
    # sample_deltas = get_knn_next_ball_deltas(tensor, 1, intervention, knn, lambda_wicket)
    # print(sample_deltas)
    
    # cf_traj = simulate_counterfactual_inning(tensor, treatment_idx=unit, intervention_ball=intervention, 
    #                                          k_neighbors=knn, lambda_wicket=lambda_wicket, seed=None)
    # print(cf_traj)
    
    # plot_simulation_vs_truth(tensor, unit, intervention, cf_traj)
    
    plot_multiple_simulations_vs_truth(
        tensor,
        treatment_idx=unit,
        intervention_ball=intervention,
        simulate_func=simulate_counterfactual_inning,
        num_sims=num_sims,
        k_neighbors=knn,
        lambda_wicket=lambda_wicket
    )
    
    '''
    # run linear regression model to get baseline value of R-squared:
    model, r2 = build_linear_regression_model(tensor, feature_balls, target_ball)
    '''
    
    '''
    # now try running arima model:
    arima_model, arima_r2 = arima_predict_post_intervention(tensor, unit_index=unit, intervention_ball=intervention, arima_order=arima_order)
    print(f"📊 ARIMA R²: {arima_r2:.4f}")
    '''
    
    '''
    # Evaluate ARIMA model across first N innings
    avg_r2, r2_list = evaluate_arima_rmse_across_units(tensor, intervention_ball=intervention, arima_order=arima_order, max_units=1000)
    '''
    
    return 0



if __name__ == "__main__":
    main()
    
    

# ✅
# 🔹
# 📄