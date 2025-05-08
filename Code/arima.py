# import necessary libraries:
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def extract_arima_features(series, order=(2,1,2)):
    """
    Fit ARIMA and extract coefficients and residual stats as features.
    series: 1D array of cumulative runs
    """
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        params = model_fit.params
        resid = model_fit.resid
        features = {
            "ar1": params.get("ar.L1", 0),
            "ar2": params.get("ar.L2", 0),
            "ma1": params.get("ma.L1", 0),
            "ma2": params.get("ma.L2", 0),
            "sigma2": params.get("sigma2", 0),
            "resid_mean": np.mean(resid),
            "resid_std": np.std(resid)
        }
    except Exception as e:
        # fallback in case ARIMA fails
        features = {
            "ar1": 0, "ar2": 0,
            "ma1": 0, "ma2": 0,
            "sigma2": 0,
            "resid_mean": 0,
            "resid_std": 0
        }
    return features

def build_global_dataset(tensor, intervention_ball=60, arima_order=(2,1,2), max_innings=500):
    """
    Builds a dataset of features and targets for up to `max_innings` innings.
    """
    X = []
    y = []

    total_innings = min(tensor.shape[0], max_innings)

    for i in range(total_innings):
        runs = tensor[i, :, 0]
        wickets = tensor[i, :, 1]

        if np.isnan(runs[intervention_ball]) or np.isnan(runs[119]):
            continue

        pre = runs[:intervention_ball + 1]
        post = runs[intervention_ball + 1:120]
        run_target = post[-1] - post[0]

        features = extract_arima_features(pre, order=arima_order)
        features.update({
            "mean_runs": np.mean(pre),
            "std_runs": np.std(pre),
            "start_runs": pre[0],
            "end_runs": pre[-1],
            "start_wickets": wickets[intervention_ball]
        })

        X.append(features)
        y.append(run_target)

        # Progress feedback
        if i % 100 == 0:
            print(f"Processed {i} / {total_innings} innings")

    return pd.DataFrame(X), np.array(y)



def predict_final_score_for_unit(tensor, unit_idx, model, intervention_ball=60, arima_order=(2,1,2)):
    """
    Predicts final total score of a single inning using the global model.
    Returns (predicted_final_score, actual_final_score, cumulative_run_series)
    """
    runs = tensor[unit_idx, :, 0]
    wickets = tensor[unit_idx, :, 1]

    if np.isnan(runs[intervention_ball]) or np.isnan(runs[119]):
        return None  # Skip incomplete innings

    pre = runs[:intervention_ball + 1]
    features = extract_arima_features(pre, order=arima_order)
    features.update({
        "mean_runs": np.mean(pre),
        "std_runs": np.std(pre),
        "start_runs": pre[0],
        "end_runs": pre[-1],
        "start_wickets": wickets[intervention_ball]
    })

    X_unit = pd.DataFrame([features])
    predicted_remaining = model.predict(X_unit)[0]
    predicted_final = pre[-1] + predicted_remaining

    actual_final = runs[119]
    return predicted_final, actual_final, runs


def plot_counterfactual_vs_actual(tensor, unit_idx, model, intervention_ball=60, arima_order=(2,1,2)):
    result = predict_final_score_for_unit(tensor, unit_idx, model, intervention_ball, arima_order)
    if result is None:
        print("⚠️ Inning is incomplete or invalid.")
        return

    predicted_final, actual_final, runs = result

    # Create predicted trajectory: keep true values up to T0, then flat line to predicted final
    pred_trajectory = runs.copy()
    slope = (predicted_final - runs[intervention_ball]) / (119 - intervention_ball)
    for t in range(intervention_ball + 1, 120):
        pred_trajectory[t] = pred_trajectory[t - 1] + slope

    # Plot
    plt.plot(range(120), runs, label="Actual Trajectory", linewidth=2)
    plt.plot(range(120), pred_trajectory, label="Predicted (Counterfactual)", linestyle='--', linewidth=2)
    plt.axvline(intervention_ball, color='gray', linestyle=':', label=f"Intervention (Ball {intervention_ball})")

    plt.title(f"Inning {unit_idx}: Actual vs Predicted Final Score\nPredicted = {predicted_final:.1f}, Actual = {actual_final:.1f}")
    plt.xlabel("Ball Number")
    plt.ylabel("Cumulative Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





























# MAIN:
def main():
    
    # set params:
    intervention = 60
    random_state = None
    max_innings = 500
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Load your tensor (already done earlier)
    data = np.load("t20_tensor_data.npz", allow_pickle=True)
    tensor = data["tensor"]

    # Build dataset
    X_df, y = build_global_dataset(tensor, intervention_ball=intervention, arima_order=(2,1,2), max_innings=max_innings)
    print("✅ Dataset shape:", X_df.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=random_state)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"🎯 Mean Absolute Error: {mae:.2f}")

    '''
    # Optional: feature importances
    importances = model.feature_importances_
    sns.barplot(x=importances, y=X_df.columns)
    plt.title("Feature Importance")
    plt.show()
    '''
    
    plot_counterfactual_vs_actual(tensor, unit_idx=1, model=model, intervention_ball=60)


if __name__ == "__main__":
    main()