import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ann import ANN
from pso import PSO

# --- 1. Data Loading and Preprocessing ---
def load_and_prepare_data(url):
    df = pd.read_csv(url)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y, scaler_X

# --- 2. ANN-PSO Coupling and Evaluation ---
def create_objective_function(ann_instance, X_train, y_train):
    """Creates the objective function for PSO to minimize (MAE)."""
    def objective_function(params_vector):
        # The ANN now handles decoding the full vector internally
        ann_instance.set_params_from_vector(params_vector)
        predictions = ann_instance.forward(X_train)
        error = np.mean(np.abs(predictions - y_train))
        return error
    return objective_function

def evaluate_on_test_set(ann_instance, best_params, X_test, y_test, scaler_y):
    """Evaluates the final trained ANN on the unseen test set."""
    # Configure the ANN with the best found parameters (including activations)
    ann_instance.set_params_from_vector(best_params)
    
    predictions_scaled = ann_instance.forward(X_test)
    predictions_orig = scaler_y.inverse_transform(predictions_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    mae = np.mean(np.abs(predictions_orig - y_test_orig))
    
    # Report the chosen activation functions
    chosen_activations = [func.__name__ for func in ann_instance.activations if func is not None]
    print(f"    - Discovered Activations: {chosen_activations}")
    
    return mae

# --- 3. Experimental Setup ---
def run_single_experiment(config):
    """Runs a single PSO-ANN training and evaluation trial."""
    X_train, X_test, y_train, y_test, scaler_y, _ = data
    
    # Setup ANN without pre-defined activations
    ann = ANN(layer_sizes=config['ann_layers'])
    
    # Get total number of parameters, now including activation choices
    num_params = ann.get_total_params(include_activations=True)

    objective_func = create_objective_function(ann, X_train, y_train)
    
    pso = PSO(
        objective_function=objective_func,
        dim=num_params,
        swarm_size=config['swarm_size'],
        num_iterations=config['iterations'],
        param_bounds=(-2, 2) # Wider bounds to accommodate activation choices
    )

    best_params, _ = pso.optimize()
    test_mae = evaluate_on_test_set(ann, best_params, X_test, y_test, scaler_y)
    return test_mae

# --- Main Execution ---
if __name__ == "__main__":
    # Download the CSV and place it in the same directory as 'concrete_data.csv'
    LOCAL_DATA_PATH = 'concrete_data.csv'
    data = load_and_prepare_data(LOCAL_DATA_PATH)
    
    NUM_RUNS = 10 # Recommended: 10, use a lower number (e.g., 3) for faster testing
    
    print("--- Starting Experimental Investigation (with Activation Search) ---")

    # We will now use a single, powerful architecture and let PSO find the best activations.
    deep_architecture = [8, 16, 16, 1] # Two hidden layers

    maes = []
    print(f"\n--- Optimizing Deep Architecture: {deep_architecture} ---")
    print(f"--- PSO will attempt to find the best activation function for each of the 2 hidden layers ---")
    
    for i in range(NUM_RUNS):
        print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
        config = {
            'ann_layers': deep_architecture, 
            'swarm_size': 30, # Slightly larger swarm for a more complex search space
            'iterations': 25
        }
        mae = run_single_experiment(config)
        maes.append(mae)
        print(f"    - Run {i+1} Test MAE: {mae:.4f}")

    print("\n" + "="*50)
    print("                FINAL RESULTS")
    print("="*50)
    print(f"Architecture: {deep_architecture}")
    print(f"Average MAE over {NUM_RUNS} runs: {np.mean(maes):.4f}")
    print(f"Standard Deviation of MAE: {np.std(maes):.4f}")
    print("="*50)
