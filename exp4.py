# exp_boundary.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ann import ANN  # Requires ann.py
from pso import PSO  # Requires modified pso.py

# --- Boilerplate Functions (Data, Coupling, Evaluation) ---
# (These are identical to your other experiment files)
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
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y

def create_objective_function(ann_instance, X_train, y_train):
    def objective_function(params_vector):
        ann_instance.set_params_from_vector(params_vector)
        predictions = ann_instance.forward(X_train)
        error = np.mean(np.abs(predictions - y_train))
        return error
    return objective_function

def evaluate_on_test_set(ann_instance, best_params, X_test, y_test, scaler_y):
    ann_instance.set_params_from_vector(best_params)
    predictions_scaled = ann_instance.forward(X_test)
    predictions_orig = scaler_y.inverse_transform(predictions_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)
    mae = np.mean(np.abs(predictions_orig - y_test_orig))
    return mae

def run_single_experiment(config):
    X_train, X_test, y_train, y_test, scaler_y = data
    
    ann = ANN(layer_sizes=config['ann_layers'], activations=config['activations'])
    num_params = ann.get_total_params()
    objective_func = create_objective_function(ann, X_train, y_train)
    
    # Pass all coefficients AND boundary strategy to PSO
    pso = PSO(
        objective_function=objective_func,
        dim=num_params,
        swarm_size=config['swarm_size'],
        num_iterations=config['iterations'],
        w=config['w'],
        c1=config['c1'],
        c2=config['c2'],
        boundary_strategy=config['boundary_strategy'] # <--- PASSING THE STRATEGY
    )

    best_params, _ = pso.optimize()
    test_mae = evaluate_on_test_set(ann, best_params, X_test, y_test, scaler_y)
    return test_mae

# --- Main Execution for Boundary Handling Experiment ---
if __name__ == "__main__":
    LOCAL_DATA_PATH = 'concrete_data.csv'
    data = load_and_prepare_data(LOCAL_DATA_PATH)
    NUM_RUNS = 10 # As recommended [cite: 115]

    print("--- 'Going Further' Experiment: Investigating Boundary Handling ---")

    # Fixed settings from previous experiments
    FIXED_LAYERS = [8, 16, 1]
    FIXED_ACTIVATIONS = [ANN.tanh, None]
    FIXED_SWARM_SIZE = 25
    FIXED_ITERATIONS = 20
    FIXED_W = 0.7
    FIXED_C1 = 1.5
    FIXED_C2 = 1.5
    
    boundary_strategies = ["clamp", "reflect", "random"]

    for strategy in boundary_strategies:
        maes = []
        print(f"\n--- Testing Strategy: '{strategy}' ---")
        for i in range(NUM_RUNS):
            print(f"  Run {i+1}/{NUM_RUNS}...")
            config = {
                'ann_layers': FIXED_LAYERS,
                'activations': FIXED_ACTIVATIONS,
                'swarm_size': FIXED_SWARM_SIZE,
                'iterations': FIXED_ITERATIONS,
                'w': FIXED_W,
                'c1': FIXED_C1,
                'c2': FIXED_C2,
                'boundary_strategy': strategy # <--- KEY VARIABLE
            }
            mae = run_single_experiment(config)
            maes.append(mae)
        print(f"Result for '{strategy}': Avg MAE = {np.mean(maes):.4f}, Std Dev = {np.std(maes):.4f}")