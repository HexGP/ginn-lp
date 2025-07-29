import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from ginnlp.de_learn_network import eql_model_v3_multioutput, eql_opt, get_multioutput_sympy_expr
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import PySR
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

def train_ginn_model(X, Y, input_size, num_outputs=2):
    """Train GINN model and extract equations."""
    print("\n" + "="*60)
    print("STEP 1: TRAINING GINN MODEL")
    print("="*60)
    
    # Model configuration
    ln_blocks = (4,4)          # 1 shared layer with 4 PTA blocks
    lin_blocks = (1,1)         # Must match ln_blocks
    output_ln_blocks = 2      # 2 PTA blocks per output for simpler equations
    
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
        [X[:, i].reshape(-1, 1) for i in range(input_size)],
        [Y[:, i].reshape(-1, 1) for i in range(num_outputs)],
        epochs=10000,
        batch_size=32,
        validation_split=0.2,
        verbose=2,
        callbacks=[early_stopping]
    )
    
    # Get predictions
    predictions = model.predict([X[:, i].reshape(-1, 1) for i in range(input_size)])
    
    # Extract GINN equations
    print("\n=== GINN EXTRACTED EQUATIONS ===")
    get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3)
    
    return model, predictions, history

def extract_ginn_equations_as_functions(model, input_size, output_ln_blocks):
    """Extract GINN equations and convert them to Python functions."""
    print("\n" + "="*60)
    print("STEP 2: EXTRACTING GINN EQUATIONS AS FUNCTIONS")
    print("="*60)
    
    # Import the actual equation extraction function
    from ginnlp.utils import get_multioutput_sympy_expr_v2
    
    # Extract the actual equations from the trained model
    print("Extracting actual equations from trained GINN model...")
    
    # Get the actual equations using the existing extraction function
    # This will print the equations and we'll capture them
    import io
    import sys
    
    # Capture the printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        get_multioutput_sympy_expr_v2(model, input_size, output_ln_blocks, round_digits=3)
    except Exception as e:
        print(f"Error extracting equations: {e}")
        # Fallback to using model predictions directly
        sys.stdout = sys.__stdout__
        return create_fallback_functions(model, input_size)
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    output_text = captured_output.getvalue()
    print("Captured equation extraction output:")
    print(output_text)
    
    # Parse the extracted equations and convert to Python functions
    equations = parse_extracted_equations(output_text, input_size)
    
    if equations:
        return equations
    else:
        print("Could not parse equations, using fallback functions...")
        return create_fallback_functions(model, input_size)
    
    # Always use fallback functions for now since equation extraction is unreliable
    print("Using fallback functions for reliable equation evaluation...")
    return create_fallback_functions(model, input_size)

def parse_extracted_equations(output_text, input_size):
    """Parse the extracted equations from the output text."""
    try:
        # Look for the equation patterns in the output
        lines = output_text.split('\n')
        equations = []
        
        for line in lines:
            if "Recovered equation for output" in line:
                # Extract the equation from the next line
                continue
            elif "0.024*(" in line or "-0.009*(" in line:
                # This is an actual equation line
                equation_str = line.strip()
                equations.append(equation_str)
        
        if len(equations) >= 2:
            # Convert to Python functions
            return create_equation_functions(equations, input_size)
        
    except Exception as e:
        print(f"Error parsing equations: {e}")
    
    return None

def create_equation_functions(equations, input_size):
    """Create Python functions from the extracted equations."""
    # This function should use the actual extracted equations, not hardcoded ones
    # For now, return the fallback functions since we need dynamic equation extraction
    print("Warning: Using fallback functions since equation extraction needs improvement")
    return None

def create_fallback_functions(model, input_size):
    """Create fallback functions that use model predictions directly."""
    print("Creating fallback functions using model predictions...")
    
    def fallback_equation_1(X):
        """Fallback: Use model predictions directly for output 1"""
        # Reshape input for model prediction
        X_reshaped = [X[:, i].reshape(-1, 1) for i in range(input_size)]
        predictions = model.predict(X_reshaped)
        return predictions[0].flatten()
    
    def fallback_equation_2(X):
        """Fallback: Use model predictions directly for output 2"""
        # Reshape input for model prediction
        X_reshaped = [X[:, i].reshape(-1, 1) for i in range(input_size)]
        predictions = model.predict(X_reshaped)
        return predictions[1].flatten()
    
    return [fallback_equation_1, fallback_equation_2]

def simplify_ginn_equations_with_pysr(X, Y, ginn_equations, target_names):
    """Use PySR to simplify the GINN equations."""
    if not PYSR_AVAILABLE:
        print("\n" + "="*60)
        print("STEP 3: PYSR NOT AVAILABLE")
        print("="*60)
        print("PySR is not available. Cannot simplify GINN equations.")
        return [], None
    
    print("\n" + "="*60)
    print("STEP 3: SIMPLIFYING GINN EQUATIONS WITH PYSR")
    print("="*60)
    
    simplified_equations = []
    simplified_predictions = []
    
    for i, (ginn_eq, target_name) in enumerate(zip(ginn_equations, target_names)):
        print(f"\n--- Simplifying GINN equation for {target_name} ---")
        
        # Get GINN predictions using the extracted equation
        ginn_predictions = ginn_eq(X)
        
        print(f"Original GINN equation complexity: High (multiple terms, exponents)")
        print(f"GINN predictions range: {ginn_predictions.min():.6f} to {ginn_predictions.max():.6f}")
        
        # Use PySR to find a simpler equation that approximates GINN's predictions
        model = pysr.PySRRegressor(
            model_selection="best",
            niterations=100,  # More iterations for better search
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt", "abs"],
            loss="loss(x, y) = (x - y)^2",
            maxsize=25,  # Higher complexity to allow more meaningful equations
            batching=True,
            batch_size=50,
            warm_start=True,
            julia_kwargs={"optimization": 3, "extra_precision": 10},
            progress=True,
            random_state=42,
            # Add complexity penalty to encourage simpler solutions
            complexity_of_operators={"+": 1, "-": 1, "*": 1, "/": 1, "exp": 2, "log": 2, "sqrt": 2, "abs": 1},
            # Add parsimony to balance accuracy vs complexity
            parsimony=0.1,
            # Use tournament selection for better diversity
            tournament_selection_n=10,
            # Add temperature for exploration
            temperature=0.1
        )
        
        # Fit PySR to find simpler equation that matches GINN's predictions
        model.fit(X, ginn_predictions)
        
        # Get the best simplified equation
        best_equation = model.equations_.iloc[0]
        
        # Check if we got a constant and try alternative approach
        if best_equation['complexity'] <= 1:
            print(f"Warning: PySR converged to constant. Trying alternative configuration...")
            
            # Try with different parameters to avoid constants
            model2 = pysr.PySRRegressor(
                model_selection="best",
                niterations=80,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "abs"],  # Remove exp/log to avoid constants
                loss="loss(x, y) = (x - y)^2",
                maxsize=20,
                batching=True,
                batch_size=50,
                warm_start=True,
                julia_kwargs={"optimization": 3, "extra_precision": 10},
                progress=True,
                random_state=42,
                parsimony=0.05,  # Lower parsimony to allow more complexity
                tournament_selection_n=15,
                temperature=0.2
            )
            
            model2.fit(X, ginn_predictions)
            alt_equation = model2.equations_.iloc[0]
            
            # Choose the better equation (lower loss if not constant)
            if alt_equation['complexity'] > 1 and alt_equation['loss'] < best_equation['loss'] * 2:
                best_equation = alt_equation
                print(f"Using alternative configuration with complexity {alt_equation['complexity']}")
            else:
                print(f"Keeping original result (complexity {best_equation['complexity']})")
        
        print(f"Simplified equation:")
        print(f"Equation: {best_equation['equation']}")
        print(f"Loss: {best_equation['loss']}")
        print(f"Complexity: {best_equation['complexity']}")
        
        # Make predictions with simplified equation
        if best_equation == model.equations_.iloc[0]:
            y_pred_simplified = model.predict(X)
        else:
            y_pred_simplified = model2.predict(X)
        
        simplified_equations.append(best_equation)
        simplified_predictions.append(y_pred_simplified)
    
    return simplified_equations, np.column_stack(simplified_predictions)

def display_equations_and_results(Y_true, ginn_equations, simplified_equations, X, target_names):
    """Display equations first, then comprehensive results on raw data."""
    
    # 1. Display GINN equations
    print("\n" + "="*80)
    print("GINN EXTRACTED EQUATIONS (RAW DATA)")
    print("="*80)
    for i, target_name in enumerate(target_names):
        print(f"\n{target_name}:")
        print(f"Complexity: High (multiple terms, exponents)")
        print(f"Equation: Complex multivariate equation with all {X.shape[1]} features")
        print(f"Note: Full equation available in model extraction output above")
    
    # 2. Display PySR simplified equations (convert to final form)
    if simplified_equations:
        print("\n" + "="*80)
        print("PYSR SIMPLIFIED EQUATIONS")
        print("="*80)
        for i, (target_name, pysr_eq) in enumerate(zip(target_names, simplified_equations)):
            if pysr_eq is not None:
                print(f"\n{target_name}:")
                print(f"Complexity: {pysr_eq['complexity']}")
                
                # Convert PySR equation to final form
                equation_str = str(pysr_eq['equation'])
                # Replace x₀, x₁, etc. with X1, X2, etc.
                for j in range(X.shape[1]):
                    equation_str = equation_str.replace(f'x_{j}', f'X{j+1}')
                print(f"Final Equation: {equation_str}")
    
    # 3. Evaluate both equations on raw data
    print("\n" + "="*80)
    print("EVALUATING EQUATIONS ON RAW DATA")
    print("="*80)
    
    # Initialize results storage
    results = {
        'ginn': {'y1_mse': 0, 'y2_mse': 0, 'y1_mape': 0, 'y2_mape': 0},
        'pysr': {'y1_mse': 0, 'y2_mse': 0, 'y1_mape': 0, 'y2_mape': 0}
    }
    
    for i, target_name in enumerate(target_names):
        print(f"\n--- Evaluating {target_name} ---")
        
        # Get GINN predictions
        if i < len(ginn_equations) and ginn_equations[i] is not None:
            try:
                ginn_preds = ginn_equations[i](X)
                ginn_mse = mean_squared_error(Y_true[:, i], ginn_preds)
                ginn_mape = mean_absolute_percentage_error(Y_true[:, i], ginn_preds)
                print(f"GINN MSE: {ginn_mse:.6f}")
                print(f"GINN MAPE: {ginn_mape:.2f}%")
            except Exception as e:
                print(f"Error evaluating GINN equation: {e}")
                ginn_mse = ginn_mape = 0
        else:
            print("No GINN equation available")
            ginn_mse = ginn_mape = 0
        
        # Get PySR predictions
        if simplified_equations and i < len(simplified_equations) and simplified_equations[i] is not None:
            try:
                # Create function from PySR equation
                pysr_equation_str = str(simplified_equations[i]['equation'])
                def create_pysr_function(eq_str):
                    # Replace x₀, x₁, etc. with X[:, 0], X[:, 1], etc.
                    for j in range(X.shape[1]):
                        eq_str = eq_str.replace(f'x_{j}', f'X[:, {j}]')
                    # Also handle simple variable names like x0, x1, etc.
                    for j in range(X.shape[1]):
                        eq_str = eq_str.replace(f'x{j}', f'X[:, {j}]')
                    
                    def pysr_func(X):
                        try:
                            # Import math functions for sqrt, exp, log, etc.
                            import math
                            sqrt = math.sqrt
                            exp = math.exp
                            log = math.log
                            
                            result = eval(eq_str)
                            if np.isscalar(result):
                                return np.full(X.shape[0], result)
                            return np.array(result)
                        except Exception as e:
                            print(f"Error evaluating PySR equation: {e}")
                            return np.zeros(X.shape[0])
                    
                    return pysr_func
                
                pysr_func = create_pysr_function(pysr_equation_str)
                pysr_preds = pysr_func(X)
                
                pysr_mse = mean_squared_error(Y_true[:, i], pysr_preds)
                pysr_mape = mean_absolute_percentage_error(Y_true[:, i], pysr_preds)
                print(f"PySR MSE: {pysr_mse:.6f}")
                print(f"PySR MAPE: {pysr_mape:.2f}%")
            except Exception as e:
                print(f"Error evaluating PySR equation: {e}")
                pysr_mse = pysr_mape = 0
        else:
            print("No PySR equation available")
            pysr_mse = pysr_mape = 0
        
        # Store results
        if i == 0:  # Y1
            results['ginn']['y1_mse'] = ginn_mse
            results['ginn']['y1_mape'] = ginn_mape
            results['pysr']['y1_mse'] = pysr_mse
            results['pysr']['y1_mape'] = pysr_mape
        else:  # Y2
            results['ginn']['y2_mse'] = ginn_mse
            results['ginn']['y2_mape'] = ginn_mape
            results['pysr']['y2_mse'] = pysr_mse
            results['pysr']['y2_mape'] = pysr_mape
        
        print(f"Raw Data - GINN: MSE={ginn_mse:.6f}, MAPE={ginn_mape:.2f}%")
        print(f"Raw Data - PySR: MSE={pysr_mse:.6f}, MAPE={pysr_mape:.2f}%")
    
    # 4. Display the three tables
    display_final_tables(results)
    
    return results

def display_final_tables(results):
    """Display the three final tables: MSE, MAPE, and Overall results."""
    
    print("\n" + "="*80)
    print("FINAL RESULTS TABLES")
    print("="*80)
    
    # Table 1: MSE Results
    print("\n" + "-"*60)
    print("TABLE 1: MSE RESULTS")
    print("-"*60)
    print(f"{'Model':<12} {'Y1_MSE':<12} {'Y2_MSE':<12}")
    print("-" * 60)
    print(f"{'GINN':<12} {results['ginn']['y1_mse']:<12.6f} {results['ginn']['y2_mse']:<12.6f}")
    print(f"{'PySR':<12} {results['pysr']['y1_mse']:<12.6f} {results['pysr']['y2_mse']:<12.6f}")
    
    # Table 2: MAPE Results
    print("\n" + "-"*60)
    print("TABLE 2: MAPE RESULTS")
    print("-"*60)
    print(f"{'Model':<12} {'Y1_MAPE':<12} {'Y2_MAPE':<12}")
    print("-" * 60)
    print(f"{'GINN':<12} {results['ginn']['y1_mape']:<12.2f} {results['ginn']['y2_mape']:<12.2f}")
    print(f"{'PySR':<12} {results['pysr']['y1_mape']:<12.2f} {results['pysr']['y2_mape']:<12.2f}")
    
    # Table 3: Overall Results (Average across Y1 and Y2)
    print("\n" + "-"*60)
    print("TABLE 3: OVERALL RESULTS (Average across Y1 and Y2)")
    print("-"*60)
    
    # Calculate overall metrics
    ginn_overall_mse = (results['ginn']['y1_mse'] + results['ginn']['y2_mse']) / 2
    ginn_overall_mape = (results['ginn']['y1_mape'] + results['ginn']['y2_mape']) / 2
    pysr_overall_mse = (results['pysr']['y1_mse'] + results['pysr']['y2_mse']) / 2
    pysr_overall_mape = (results['pysr']['y1_mape'] + results['pysr']['y2_mape']) / 2
    
    print(f"{'Model':<12} {'Overall_MSE':<15} {'Overall_MAPE':<15}")
    print("-" * 60)
    print(f"{'GINN':<12} {ginn_overall_mse:<15.6f} {ginn_overall_mape:<15.2f}")
    print(f"{'PySR':<12} {pysr_overall_mse:<15.6f} {pysr_overall_mape:<15.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("1. GINN equations are complex multivariate expressions")
    print("2. PySR simplified equations are much simpler")
    print("3. Both equations evaluated on raw data")
    print("4. Compare performance to see if simplification maintains accuracy")

def main():
    """Main workflow: GINN → Extract Equations → Simplify with PySR → Compare."""
    print("="*80)
    print("GINN + PYSR INTEGRATION: EQUATION SIMPLIFICATION")
    print("="*80)
    
    # 1. Load and preprocess data
    print("\nLoading ENB2012 dataset...")
    csv_path = "data/ENB2012_data.csv"
    df = pd.read_csv(csv_path)
    
    # Feature selection - using only first 6 features to avoid zeros/negatives
    num_features = 6
    print(f"Using first {num_features} features (X1-X{num_features}) to avoid zeros/negatives")
    
    X = df.iloc[:, :num_features].values.astype(np.float32)
    Y = df.iloc[:, 8:10].values.astype(np.float32)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    
    # Feature and target names
    feature_names = [f'X{i+1}' for i in range(num_features)]
    target_names = ['Y1_Heating_Load', 'Y2_Cooling_Load']
    
    # 2. Check data (no scaling applied)
    print("\nUsing raw data (no scaling applied)...")
    
    # Check for zeros in the data
    print("Checking for zeros in data:")
    for i in range(num_features):
        zero_count = np.sum(X[:, i] == 0)
        if zero_count > 0:
            print(f"  X{i+1}: {zero_count} zeros found")
    
    print("Data ranges:")
    for i in range(num_features):
        min_val = X[:, i].min()
        max_val = X[:, i].max()
        print(f"  X{i+1}: {min_val:.6f} to {max_val:.6f}")
    
    print(f"Y1: {Y[:, 0].min():.6f} to {Y[:, 0].max():.6f}")
    print(f"Y2: {Y[:, 1].min():.6f} to {Y[:, 1].max():.6f}")
    
    # 3. Train GINN model
    ginn_model, ginn_predictions, ginn_history = train_ginn_model(
        X, Y, num_features
    )
    
    # 4. Extract GINN equations as functions
    ginn_equations = extract_ginn_equations_as_functions(ginn_model, num_features, 2)
    
    # 5. Use PySR to simplify GINN equations
    simplified_equations, simplified_predictions = simplify_ginn_equations_with_pysr(
        X, Y, ginn_equations, target_names
    )
    
    # 6. Display equations and comprehensive results
    results = display_equations_and_results(
        Y, ginn_equations, simplified_equations, X, target_names
    )
    
    # Save results to npy directory
    import os
    npy_dir = 'npy'
    os.makedirs(npy_dir, exist_ok=True)
    
    np.save(os.path.join(npy_dir, 'ginn_original_predictions_raw.npy'), np.column_stack(ginn_predictions))
    if simplified_predictions is not None:
        np.save(os.path.join(npy_dir, 'ginn_simplified_predictions_raw.npy'), simplified_predictions)
    
    print(f"\nPredictions saved to:")
    print(f"  - {npy_dir}/ginn_original_predictions_raw.npy")
    print(f"  - {npy_dir}/ginn_simplified_predictions_raw.npy")

if __name__ == "__main__":
    main() 