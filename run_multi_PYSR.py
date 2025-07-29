import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from ginnlp.de_learn_network import eql_model_v3_multioutput, eql_opt, get_multioutput_sympy_expr
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import PySR, with better error handling
PYSR_AVAILABLE = False
try:
    import pysr
    PYSR_AVAILABLE = True
    print("PySR imported successfully!")
except ImportError:
    print("PySR not found. Attempting to install...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "pysr"])
        import pysr
        PYSR_AVAILABLE = True
        print("PySR installed and imported!")
    except Exception as e:
        print(f"PySR installation failed: {e}")
        print("Continuing without PySR - will only run GINN model")
        PYSR_AVAILABLE = False
except Exception as e:
    print(f"PySR import failed: {e}")
    print("This is likely due to Julia not being installed or permission issues.")
    print("Continuing without PySR - will only run GINN model")
    PYSR_AVAILABLE = False

def weighted_multi_task_loss(task_weights):
    """Custom loss function with manual weights for each task."""
    def loss(y_true, y_pred):
        weights = tf.constant(task_weights, dtype=tf.float32)
        mse_0 = tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])
        mse_1 = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
        weighted_mse_0 = weights[0] * mse_0
        weighted_mse_1 = weights[1] * mse_1
        total_loss = weighted_mse_0 + weighted_mse_1
        return total_loss
    return loss

def train_ginn_model(X_scaled, Y_scaled, input_size, num_outputs=2):
    """Train GINN model and extract equations."""
    print("\n" + "="*60)
    print("STEP 1: TRAINING GINN MODEL")
    print("="*60)
    
    # Model configuration
    ln_blocks = (4, 4)          # 2 shared layers: both with 4 PTA blocks
    lin_blocks = (1, 1)         # Must match ln_blocks
    output_ln_blocks = 2        # 2 PTA blocks per output for simpler equations
    
    decay_steps = 1000
    init_lr = 0.01
    opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)
    
    model = eql_model_v3_multioutput(
        input_size=input_size,
        opt=opt,
        ln_blocks=ln_blocks,
        lin_blocks=lin_blocks,
        output_ln_blocks=output_ln_blocks,
        num_outputs=num_outputs,
        compile=False,
        l1_reg=1e-3,
        l2_reg=1e-3,
        output_l1_reg=0.2,
        output_l2_reg=0.1
    )
    
    # Compile with custom weighted loss
    task_weights = [0.2, 0.8]
    custom_loss = weighted_multi_task_loss(task_weights)
    model.compile(optimizer=opt, loss=custom_loss, metrics=['mean_squared_error', 'mean_absolute_percentage_error'])
    
    # Train model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=100,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        [X_scaled[:, i].reshape(-1, 1) for i in range(input_size)],
        [Y_scaled[:, i].reshape(-1, 1) for i in range(num_outputs)],
        epochs=10000,
        batch_size=32,
        validation_split=0.2,
        verbose=2,
        callbacks=[early_stopping]
    )
    
    # Get predictions
    predictions_scaled = model.predict([X_scaled[:, i].reshape(-1, 1) for i in range(input_size)])
    
    # Post-process for simpler equations
    threshold = 0.01
    for layer in model.layers:
        if 'output_' in layer.name:
            weights = layer.get_weights()
            if len(weights) > 0:
                kernel = weights[0]
                kernel_mask = np.abs(kernel) < threshold
                kernel[kernel_mask] = 0.0
                weights[0] = kernel
                if len(weights) > 1:
                    bias = weights[1]
                    bias_mask = np.abs(bias) < threshold
                    bias[bias_mask] = 0.0
                    weights[1] = bias
                layer.set_weights(weights)
    
    # Extract GINN equations
    print("\n=== GINN EXTRACTED EQUATIONS ===")
    get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3)
    
    return model, predictions_scaled, history

def run_pysr_regression(X, Y, feature_names, target_names):
    """Run PySR to find simpler symbolic expressions."""
    if not PYSR_AVAILABLE:
        print("\n" + "="*60)
        print("STEP 2: PYSR NOT AVAILABLE")
        print("="*60)
        print("PySR is not available due to Julia installation issues.")
        print("To use PySR, you need to:")
        print("1. Install Julia from https://julialang.org/downloads/")
        print("2. Add Julia to your system PATH")
        print("3. Run: pip install pysr")
        print("\nContinuing with GINN-only analysis...")
        return [], None
    
    print("\n" + "="*60)
    print("STEP 2: RUNNING PYSR SYMBOLIC REGRESSION")
    print("="*60)
    
    pysr_equations = []
    pysr_predictions = []
    
    for i, target_name in enumerate(target_names):
        print(f"\n--- PySR for {target_name} ---")
        
        # Configure PySR
        model = pysr.PySRRegressor(
            model_selection="best",  # Use the best model
            niterations=40,         # Number of iterations
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt", "abs"],
            loss="loss(x, y) = (x - y)^2",  # MSE loss
            maxsize=25,             # Maximum equation size
            batching=True,          # Use batching for speed
            batch_size=50,          # Batch size
            warm_start=True,        # Warm start from previous runs
            julia_kwargs={"optimization": 3, "extra_precision": 10},
            progress=True,
            random_state=42
        )
        
        # Fit PySR model
        model.fit(X, Y[:, i])
        
        # Get the best equation
        best_equation = model.equations_.iloc[0]
        print(f"Best PySR equation for {target_name}:")
        print(f"Equation: {best_equation['equation']}")
        print(f"Loss: {best_equation['loss']}")
        print(f"Complexity: {best_equation['complexity']}")
        
        # Make predictions
        y_pred = model.predict(X)
        
        pysr_equations.append(best_equation)
        pysr_predictions.append(y_pred)
    
    return pysr_equations, np.column_stack(pysr_predictions)

def evaluate_equations(Y_true, ginn_pred, pysr_pred=None, scaler_Y=None):
    """Evaluate and compare GINN vs PySR equations."""
    print("\n" + "="*60)
    print("STEP 3: EVALUATING EQUATIONS")
    print("="*60)
    
    results = {}
    
    # Evaluate GINN on scaled data
    for i in range(2):
        # GINN metrics
        ginn_mse = mean_squared_error(Y_true[:, i], ginn_pred[i].flatten())
        ginn_mape = mean_absolute_percentage_error(Y_true[:, i], ginn_pred[i].flatten()) * 100
        
        results[f'Y{i+1}_GINN'] = {'MSE': ginn_mse, 'MAPE': ginn_mape}
        
        print(f"\nTarget {i+1} - GINN Results:")
        print(f"MSE: {ginn_mse:.6f}")
        print(f"MAPE: {ginn_mape:.2f}%")
        
        # PySR metrics if available
        if pysr_pred is not None:
            pysr_mse = mean_squared_error(Y_true[:, i], pysr_pred[:, i])
            pysr_mape = mean_absolute_percentage_error(Y_true[:, i], pysr_pred[:, i]) * 100
            
            results[f'Y{i+1}_PySR'] = {'MSE': pysr_mse, 'MAPE': pysr_mape}
            
            print(f"Target {i+1} - PySR Results:")
            print(f"MSE: {pysr_mse:.6f}")
            print(f"MAPE: {pysr_mape:.2f}%")
            
            # Determine which is better
            if pysr_mse < ginn_mse:
                print(f"✓ PySR performs better for Target {i+1}")
            else:
                print(f"✓ GINN performs better for Target {i+1}")
    
    # Evaluate on original scale if scaler provided
    if scaler_Y is not None:
        print(f"\n=== ORIGINAL SCALE COMPARISON ===")
        Y_original = scaler_Y.inverse_transform(Y_true)
        ginn_original = scaler_Y.inverse_transform(np.column_stack(ginn_pred))
        
        for i in range(2):
            ginn_mse_orig = mean_squared_error(Y_original[:, i], ginn_original[:, i])
            ginn_mape_orig = mean_absolute_percentage_error(Y_original[:, i], ginn_original[:, i]) * 100
            
            print(f"\nTarget {i+1} (Original Scale) - GINN:")
            print(f"MSE: {ginn_mse_orig:.4f}")
            print(f"MAPE: {ginn_mape_orig:.2f}%")
            
            if pysr_pred is not None:
                pysr_original = scaler_Y.inverse_transform(pysr_pred)
                pysr_mse_orig = mean_squared_error(Y_original[:, i], pysr_original[:, i])
                pysr_mape_orig = mean_absolute_percentage_error(Y_original[:, i], pysr_original[:, i]) * 100
                
                print(f"Target {i+1} (Original Scale) - PySR:")
                print(f"MSE: {pysr_mse_orig:.4f}")
                print(f"MAPE: {pysr_mape_orig:.2f}%")
    
    return results

def main():
    """Main workflow: GINN + PySR integration."""
    print("="*80)
    print("GINN + PYSR INTEGRATION: SYMBOLIC EQUATION SIMPLIFICATION")
    print("="*80)
    
    # 1. Load and preprocess data
    print("\nLoading ENB2012 dataset...")
    csv_path = "data/ENB2012_data.csv"
    df = pd.read_csv(csv_path)
    
    # Feature selection
    num_features = 8  # Change this to 5, 6, 7, or 8
    print(f"Using first {num_features} features (X1-X{num_features})")
    
    X = df.iloc[:, :num_features].values.astype(np.float32)
    Y = df.iloc[:, 8:10].values.astype(np.float32)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    
    # Feature and target names
    feature_names = [f'X{i+1}' for i in range(num_features)]
    target_names = ['Y1_Heating_Load', 'Y2_Cooling_Load']
    
    # 2. Apply scaling (avoid zeros for GINN)
    print("\nApplying MinMaxScaler (avoiding zeros)...")
    scaler_X = MinMaxScaler(feature_range=(0.1, 10.0))  # Safer range to avoid zeros
    scaler_Y = MinMaxScaler(feature_range=(0.1, 10.0))  # Safer range to avoid zeros
    
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)
    
    print("Scaled input ranges:")
    for i in range(num_features):
        print(f"X{i+1}: {X_scaled[:, i].min():.6f} to {X_scaled[:, i].max():.6f}")
    
    # 3. Train GINN model
    ginn_model, ginn_predictions, ginn_history = train_ginn_model(
        X_scaled, Y_scaled, num_features
    )
    
    # 4. Run PySR regression (if available)
    pysr_equations, pysr_predictions = run_pysr_regression(
        X_scaled, Y_scaled, feature_names, target_names
    )
    
    # 5. Compare results
    results = evaluate_equations(Y_scaled, ginn_predictions, pysr_predictions, scaler_Y)
    
    # 6. Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\nDataset: ENB2012 with {num_features} features (X1-X{num_features})")
    print(f"Scaling: MinMaxScaler [1e-6, 1.0] to avoid zeros")
    
    if pysr_equations:
        print(f"\nPySR Equations Found:")
        for i, eq in enumerate(pysr_equations):
            print(f"Y{i+1}: {eq['equation']}")
            print(f"  - Loss: {eq['loss']:.6f}")
            print(f"  - Complexity: {eq['complexity']}")
        
        print(f"\nRecommendation:")
        print("Use PySR equations if they achieve similar/better accuracy with lower complexity.")
        print("Use GINN equations if they achieve significantly better accuracy.")
    else:
        print(f"\nPySR was not available.")
        print("GINN equations have been extracted and evaluated.")
        print("To use PySR for equation simplification, install Julia and PySR.")
    
    # Save results
    np.save('ginn_predictions_scaled.npy', np.column_stack(ginn_predictions))
    print(f"\nGINN predictions saved to: ginn_predictions_scaled.npy")
    
    if pysr_predictions is not None:
        np.save('pysr_predictions_scaled.npy', pysr_predictions)
        print(f"PySR predictions saved to: pysr_predictions_scaled.npy")

if __name__ == "__main__":
    main() 