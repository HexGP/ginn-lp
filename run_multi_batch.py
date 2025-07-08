import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from ginnlp.de_learn_network import eql_model_v3_multioutput, eql_opt, get_multioutput_sympy_expr
from tensorflow.keras.callbacks import EarlyStopping
import time

def weighted_multi_task_loss(task_weights):
    """
    Custom loss function with manual weights for each task.
    task_weights: list of weights that sum to 1.0
    """
    def loss(y_true, y_pred):
        # Convert task_weights to tensor
        weights = tf.constant(task_weights, dtype=tf.float32)
        
        # Calculate MSE for each task using TensorFlow operations
        mse_0 = tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])
        mse_1 = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
        
        # Apply weights
        weighted_mse_0 = weights[0] * mse_0
        weighted_mse_1 = weights[1] * mse_1
        
        # Sum the weighted losses
        total_loss = weighted_mse_0 + weighted_mse_1
        return total_loss
    return loss

def run_single_experiment(task_weights, run_number, verbose=False):
    """
    Run a single experiment with given task weights.
    Returns: dict with metrics
    """
    if verbose:
        print(f"\n=== Starting Run {run_number} with weights {task_weights} ===")
    
    # 1. Load data
    csv_path = "data/ENB2012_data.csv"
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :8].values.astype(np.float32)  # First 8 columns as input (X1-X8)
    Y = df.iloc[:, 8:10].values.astype(np.float32) # target_1 and target_2 columns

    # 2. Apply MinMaxScaler to ensure all values are strictly positive for log safety
    scaler_X = MinMaxScaler(feature_range=(0.1, 1.0))
    scaler_Y = MinMaxScaler(feature_range=(0.1, 1.0))

    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # 3. Model configuration
    input_size = 8
    num_outputs = 2
    ln_blocks = (4, 4)          # 2 shared layers: both with 4 PTA blocks
    lin_blocks = (1, 1)         # Must match ln_blocks
    output_ln_blocks = 4        # 4 PTA blocks per output

    decay_steps = 1000
    init_lr = 0.01
    opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)

    # 4. Create model
    model = eql_model_v3_multioutput(
        input_size=input_size,
        opt=opt,
        ln_blocks=ln_blocks,
        lin_blocks=lin_blocks,
        output_ln_blocks=output_ln_blocks,
        num_outputs=num_outputs,
        compile=False,
        l1_reg=1e-3,
        l2_reg=1e-3
    )

    # 5. Compile with custom weighted loss
    custom_loss = weighted_multi_task_loss(task_weights)
    model.compile(optimizer=opt, loss=custom_loss, metrics=['mean_squared_error', 'mean_absolute_percentage_error'])

    # 6. Train model (suppress output unless verbose)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=100,
        restore_best_weights=True,
        verbose=0
    )
    if verbose:
        history = model.fit(
            [X_scaled[:, i].reshape(-1, 1) for i in range(input_size)],
            [Y_scaled[:, i].reshape(-1, 1) for i in range(num_outputs)],
            epochs=10000,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stopping]
        )
    else:
        history = model.fit(
            [X_scaled[:, i].reshape(-1, 1) for i in range(input_size)],
            [Y_scaled[:, i].reshape(-1, 1) for i in range(num_outputs)],
            epochs=10000,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stopping]
        )

    # 7. Calculate predictions and convert back to original scale
    predictions_scaled = model.predict([X_scaled[:, i].reshape(-1, 1) for i in range(input_size)], verbose=0)
    predictions_original = scaler_Y.inverse_transform(np.column_stack(predictions_scaled))

    # 8. Calculate metrics on original scale
    results = {}
    for i in range(num_outputs):
        mse = mean_squared_error(Y[:, i], predictions_original[:, i])
        mape = mean_absolute_percentage_error(Y[:, i], predictions_original[:, i])
        results[f'target_{i+1}_mse'] = mse
        results[f'target_{i+1}_mape'] = mape

    # Calculate overall metrics
    overall_mse = np.mean([results[f'target_{i+1}_mse'] for i in range(num_outputs)])
    overall_mape = np.mean([results[f'target_{i+1}_mape'] for i in range(num_outputs)])
    results['overall_mse'] = overall_mse
    results['overall_mape'] = overall_mape

    return results

def run_batch_experiments(task_weights, num_runs=12, verbose=False):
    """
    Run multiple experiments with the same task weights.
    
    Args:
        task_weights: list of weights [w1, w2] that sum to 1.0
        num_runs: number of experiments to run
        verbose: whether to print detailed output
    
    Returns:
        list of results dictionaries
    """
    print(f"\n{'='*60}")
    print(f"BATCH EXPERIMENT: {num_runs} runs with task weights {task_weights}")
    print(f"{'='*60}")
    
    if sum(task_weights) != 1.0:
        print(f"WARNING: Task weights sum to {sum(task_weights)}, not 1.0!")
    
    start_time = time.time()
    results = []
    
    for run in range(1, num_runs + 1):
        print(f"\n--- RUN {run:2d}/{num_runs} ---")
        try:
            run_result = run_single_experiment(task_weights, run, verbose)
            run_result['run_number'] = run
            results.append(run_result)
            
            if verbose:
                print(f"✅ Run {run:2d} COMPLETED:")
                print(f"   Target 1: MSE = {run_result['target_1_mse']:.4f}, MAPE = {run_result['target_1_mape']:.2f}%")
                print(f"   Target 2: MSE = {run_result['target_2_mse']:.4f}, MAPE = {run_result['target_2_mape']:.2f}%")
                print(f"   Overall: MSE = {run_result['overall_mse']:.4f}, MAPE = {run_result['overall_mape']:.2f}%")
            else:
                print(f"✅ Run {run:2d} COMPLETED: Overall MSE = {run_result['overall_mse']:.4f}, MAPE = {run_result['overall_mape']:.2f}%")
                
        except Exception as e:
            print(f"❌ Run {run:2d} FAILED: {str(e)}")
            results.append({
                'run_number': run,
                'target_1_mse': float('inf'),
                'target_1_mape': float('inf'),
                'target_2_mse': float('inf'),
                'target_2_mape': float('inf'),
                'overall_mse': float('inf'),
                'overall_mape': float('inf'),
                'error': str(e)
            })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    successful_runs = [r for r in results if r['overall_mse'] != float('inf')]
    failed_runs = [r for r in results if r['overall_mse'] == float('inf')]
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Task weights: {task_weights}")
    print(f"Total runs: {num_runs}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Success rate: {len(successful_runs)/num_runs*100:.1f}%")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per run: {total_time/num_runs:.1f} seconds")
    
    if successful_runs:
        overall_mses = [r['overall_mse'] for r in successful_runs]
        overall_mapes = [r['overall_mape'] for r in successful_runs]
        
        # Individual target statistics
        target1_mses = [r['target_1_mse'] for r in successful_runs]
        target1_mapes = [r['target_1_mape'] for r in successful_runs]
        target2_mses = [r['target_2_mse'] for r in successful_runs]
        target2_mapes = [r['target_2_mape'] for r in successful_runs]
        
        print(f"\nPERFORMANCE STATISTICS (successful runs only):")
        print(f"{'='*50}")
        print(f"OVERALL PERFORMANCE:")
        print(f"  MSE - Mean: {np.mean(overall_mses):.4f}, Std: {np.std(overall_mses):.4f}")
        print(f"  MSE - Min: {np.min(overall_mses):.4f}, Max: {np.max(overall_mses):.4f}")
        print(f"  MAPE - Mean: {np.mean(overall_mapes):.4f}, Std: {np.std(overall_mapes):.4f}")
        print(f"  MAPE - Min: {np.min(overall_mapes):.4f}, Max: {np.max(overall_mapes):.4f}")
        
        print(f"\nTARGET 1 (Heating Load):")
        print(f"  MSE - Mean: {np.mean(target1_mses):.4f}, Std: {np.std(target1_mses):.4f}")
        print(f"  MSE - Min: {np.min(target1_mses):.4f}, Max: {np.max(target1_mses):.4f}")
        print(f"  MAPE - Mean: {np.mean(target1_mapes):.4f}, Std: {np.std(target1_mapes):.4f}")
        print(f"  MAPE - Min: {np.min(target1_mapes):.4f}, Max: {np.max(target1_mapes):.4f}")
        
        print(f"\nTARGET 2 (Cooling Load):")
        print(f"  MSE - Mean: {np.mean(target2_mses):.4f}, Std: {np.std(target2_mses):.4f}")
        print(f"  MSE - Min: {np.min(target2_mses):.4f}, Max: {np.max(target2_mses):.4f}")
        print(f"  MAPE - Mean: {np.mean(target2_mapes):.4f}, Std: {np.std(target2_mapes):.4f}")
        print(f"  MAPE - Min: {np.min(target2_mapes):.4f}, Max: {np.max(target2_mapes):.4f}")
        
        # Performance comparison between targets
        print(f"\nTARGET COMPARISON:")
        print(f"  Target 1 vs Target 2 MSE ratio: {np.mean(target1_mses)/np.mean(target2_mses):.3f}")
        print(f"  Target 1 vs Target 2 MAPE ratio: {np.mean(target1_mapes)/np.mean(target2_mapes):.3f}")
        if np.mean(target1_mses) < np.mean(target2_mses):
            print(f"  Target 1 performs better (lower MSE)")
        else:
            print(f"  Target 2 performs better (lower MSE)")
    
    return results

if __name__ == "__main__":
    # Configuration - CHANGE THESE VALUES
    TASK_WEIGHTS = [0.2, 0.8]  # Equal weighting
    NUM_RUNS = 12              # Number of experiments to run
    VERBOSE = False            # Set to True for detailed output
    
    # Run the batch experiment
    results = run_batch_experiments(TASK_WEIGHTS, NUM_RUNS, VERBOSE)
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"\n--- RUN {result['run_number']:2d} ---")
        if result['overall_mse'] != float('inf'):
            print(f"✅ SUCCESSFUL:")
            print(f"   Target 1: MSE = {result['target_1_mse']:.4f}, MAPE = {result['target_1_mape']:.2f}%")
            print(f"   Target 2: MSE = {result['target_2_mse']:.4f}, MAPE = {result['target_2_mape']:.2f}%")
            print(f"   Overall: MSE = {result['overall_mse']:.4f}, MAPE = {result['overall_mape']:.2f}%")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}") 