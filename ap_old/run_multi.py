import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from ginnlp.de_learn_network import eql_model_v3_multioutput, eql_opt, get_multioutput_sympy_expr
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Load data - automatically detects features and targets from ANY dataset
# Just change the csv_path to use any dataset - the script will automatically detect:
# - Feature columns (anything that's not a target)
# - Target columns (looks for target*, Y*, output*, label*, class*, result* patterns)
# - Or assumes last 1-3 columns are targets if no patterns found
csv_path = "data/ENB2012_data.csv"  # Change this to use any dataset
df = pd.read_csv(csv_path)

# Automatically detect features and targets from any dataset
def detect_features_and_targets(df):
    """Automatically detect feature and target columns from any dataset"""
    all_cols = df.columns.tolist()
    
    # Strategy 1: Look for common target patterns first
    target_patterns = [
        lambda col: col.startswith('target'),   # target_0, target_1, target
        lambda col: col.startswith('Y'),        # Y1, Y2, Y3
        lambda col: col.startswith('output'),   # output_0, output_1
        lambda col: col.startswith('label'),    # label_0, label_1
        lambda col: col.startswith('class'),    # class_0, class_1
        lambda col: col.startswith('result'),   # result_0, result_1
    ]
    
    # Try to detect targets first
    target_cols = []
    for pattern in target_patterns:
        target_cols = [col for col in all_cols if pattern(col)]
        if target_cols:
            break
    
    # Strategy 2: If no target patterns found, assume last columns are targets
    if not target_cols:
        # Try different numbers of target columns (1, 2, 3)
        for num_targets in [1, 2, 3]:
            if len(all_cols) >= num_targets:
                potential_targets = all_cols[-num_targets:]
                # Check if these look like targets (numeric data)
                try:
                    target_data = df[potential_targets].values
                    if target_data.shape[1] == num_targets:
                        target_cols = potential_targets
                        break
                except:
                    continue
    
    # Strategy 3: If still no targets, assume last 2 columns are targets
    if not target_cols and len(all_cols) >= 2:
        target_cols = all_cols[-2:]
    
    # Features are everything that's not a target
    feature_cols = [col for col in all_cols if col not in target_cols]
    
    return feature_cols, target_cols

# Detect features and targets
feature_cols, target_cols = detect_features_and_targets(df)

if not feature_cols or not target_cols:
    raise ValueError(f"Could not automatically detect features and targets in {csv_path}")

X = df[feature_cols].values.astype(np.float32)
Y = df[target_cols].values.astype(np.float32)

# Automatically detect number of features and outputs
num_features = len(feature_cols)
num_outputs = len(target_cols)

print(f"Dataset: {csv_path}")
print(f"Detected features: {feature_cols}")
print(f"Detected targets: {target_cols}")
print(f"Number of features: {num_features}")
print(f"Number of outputs: {num_outputs}")
print(f"Dataset shape: {X.shape}")

# 2. Apply scaling - choose one approach:

# Approach 1: MinMaxScaler (current)
scaler_X = MinMaxScaler(feature_range=(0.1, 10.0))  # Avoid exact zeros
scaler_Y = MinMaxScaler(feature_range=(0.1, 10.0))  # Avoid exact zeros

# Approach 2: StandardScaler (alternative - uncomment to use)
# from sklearn.preprocessing import StandardScaler
# scaler_X = StandardScaler()
# scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

print("Original input ranges:")
for i in range(num_features):
    print(f"{feature_cols[i]}: {X[:, i].min():.2f} to {X[:, i].max():.2f}")
print("\nScaled input ranges:")
for i in range(num_features):
    print(f"{feature_cols[i]}: {X_scaled[:, i].min():.2f} to {X_scaled[:, i].max():.2f}")

print("\nOriginal target ranges:")
for i in range(num_outputs):
    print(f"{target_cols[i]}: {Y[:, i].min():.2f} to {Y[:, i].max():.2f}")
print("\nScaled target ranges:")
for i in range(num_outputs):
    print(f"{target_cols[i]}: {Y_scaled[:, i].min():.2f} to {Y_scaled[:, i].max():.2f}")

# 3. Use simple architecture that was working well
input_size = num_features
ln_blocks = (4, 4, 4, 4,)          # 1 shared layer with 4 PTA blocks
lin_blocks = (1, 1, 1, 1,)         # Must match ln_blocks
output_ln_blocks = 2        # 2 PTA blocks per output for simpler equations

# Regularization strategy:
# - Standard L1/L2 on shared layers (1e-3)
# - Much stronger L1/L2 on output heads to encourage extremely simple equations
#   - L1 (0.2): encourages extreme sparsity - most coefficients will be zero
#   - L2 (0.1): encourages very small coefficients - remaining terms will be simple

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
    compile=False,  # We'll compile manually
    l1_reg=1e-3,   # Standard regularization for shared layers
    l2_reg=1e-3,   # Standard regularization for shared layers
    output_l1_reg=0.2,   # Much stronger L1 regularization for output heads to encourage extreme sparsity
    output_l2_reg=0.1    # Much stronger L2 regularization for output heads to encourage very small coefficients
)

# 4. Define custom loss with manual task weights
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

# Manual task weights (must sum to 1.0)
task_weights = [0.2, 0.8]  # 20% weight to target_1, 80% to target_2
print(f"Task weights: {task_weights} (sum = {sum(task_weights)})")

# Compile with custom weighted loss
custom_loss = weighted_multi_task_loss(task_weights)
model.compile(optimizer=opt, loss=custom_loss, metrics=['mean_squared_error', 'mean_absolute_percentage_error'])

# print("\nModel Summary:")
# model.summary()

# 5. Train model with scaled values
# Add EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,
    verbose=0
)

history = model.fit(
    [X_scaled[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y_scaled[:, i].reshape(-1, 1) for i in range(num_outputs)],
    epochs=10000,  # You can still set a high max, but training will stop early if no improvement
    batch_size=32,
    validation_split=0.2,
    verbose=2,
    callbacks=[early_stopping]
)

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Evaluate with scaled values
results = model.evaluate(
    [X_scaled[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y_scaled[:, i].reshape(-1, 1) for i in range(num_outputs)],
    verbose=0
)

# 7. Calculate predictions and convert back to original scale for metrics
predictions_scaled = model.predict([X_scaled[:, i].reshape(-1, 1) for i in range(input_size)])
# Save predictions as a single (n_samples, 2) array for both outputs
np.save('nn_preds_scaled.npy', np.column_stack(predictions_scaled))
print("Saved neural network predictions to nn_preds_scaled.npy")
predictions_original = scaler_Y.inverse_transform(np.column_stack(predictions_scaled))

# Calculate metrics on both scaled and original scale
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

print("\n=== EVALUATION ON SCALED DATA ===")
for i in range(num_outputs):
    mse_scaled = mean_squared_error(Y_scaled[:, i], predictions_scaled[i].flatten())
    mape_scaled = mean_absolute_percentage_error(Y_scaled[:, i], predictions_scaled[i].flatten())
    print(f"Target {i+1} (scaled): MSE = {mse_scaled:.6f}, MAPE = {mape_scaled:.2f}%")

print("\n=== EVALUATION ON ORIGINAL SCALE ===")
for i in range(num_outputs):
    mse = mean_squared_error(Y[:, i], predictions_original[:, i])
    mape = mean_absolute_percentage_error(Y[:, i], predictions_original[:, i])
    print(f"Target {i+1}: MSE = {mse:.4f}, MAPE = {mape:.2f}%")

# 8. Post-process model to zero out very small coefficients for simpler equations
print("\n=== POST-PROCESSING FOR SIMPLER EQUATIONS ===")
# Zero out very small coefficients in output layers to encourage sparsity
threshold = 0.01  # Coefficients smaller than this will be zeroed out
for layer in model.layers:
    if 'output_' in layer.name:
        weights = layer.get_weights()
        if len(weights) > 0:
            # Zero out small kernel weights
            kernel = weights[0]
            kernel_mask = np.abs(kernel) < threshold
            kernel[kernel_mask] = 0.0
            weights[0] = kernel
            
            # Zero out small bias weights if they exist
            if len(weights) > 1:
                bias = weights[1]
                bias_mask = np.abs(bias) < threshold
                bias[bias_mask] = 0.0
                weights[1] = bias
            
            layer.set_weights(weights)
            print(f"Post-processed {layer.name}: zeroed out {np.sum(kernel_mask)} small coefficients")

# 9. Extract and print symbolic equations
print("\n=== SYMBOLIC EQUATIONS ===")
get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3)

print(f"\nTraining completed successfully!")

# Calculate overall metrics for scaled data
overall_mse_scaled = np.mean([mean_squared_error(Y_scaled[:, i], predictions_scaled[i].flatten()) for i in range(num_outputs)])
overall_mape_scaled = np.mean([mean_absolute_percentage_error(Y_scaled[:, i], predictions_scaled[i].flatten()) for i in range(num_outputs)])

# Calculate overall metrics for original scale
overall_mse = np.mean([mean_squared_error(Y[:, i], predictions_original[:, i]) for i in range(num_outputs)])
overall_mape = np.mean([mean_absolute_percentage_error(Y[:, i], predictions_original[:, i]) for i in range(num_outputs)])


# 11. FINAL PERFORMANCE METRICS (at the end)
print(f"\n{'='*60}")
print(f"FINAL PERFORMANCE METRICS")
print(f"{'='*60}")

# Calculate individual metrics for each target (using scaled data)
mse_target1 = mean_squared_error(Y_scaled[:, 0], predictions_scaled[0].flatten())
mse_target2 = mean_squared_error(Y_scaled[:, 1], predictions_scaled[1].flatten())
mape_target1 = mean_absolute_percentage_error(Y_scaled[:, 0], predictions_scaled[0].flatten())
mape_target2 = mean_absolute_percentage_error(Y_scaled[:, 1], predictions_scaled[1].flatten())

# Calculate averages
avg_mse = (mse_target1 + mse_target2) / 2
avg_mape = (mape_target1 + mape_target2) / 2

# Print formatted table
print(f"\n{'='*80}")
print("Multi-GINN MODEL RESULTS")
print(f"{'='*80}")

print(f"{'Model':<25} | {'MSE':<15} | {'MAPE':<15}")
print(f"{'':<25} | {'Y1':<6} {'Y2':<6} {'Avg':<6} | {'Y1':<6} {'Y2':<6} {'Avg':<6}")
print("-" * 80)

print(f"{'Multi-GINN':<25} | {mse_target1:<6.4f} {mse_target2:<6.4f} {avg_mse:<6.4f} | "
      f"{mape_target1:<6.2f} {mape_target2:<6.2f} {avg_mape:<6.2f}")

print("="*80)

# 10. Print comprehensive final summary
print(f"\n" + "="*80)
print("=== FINAL SUMMARY ===")
print("="*80)
print(f"📊 DATASET:")
print(f"   • File: {csv_path}")
print(f"   • Features: {num_features} ({', '.join(feature_cols[:3])}{'...' if len(feature_cols) > 3 else ''})")
print(f"   • Targets: {num_outputs} ({', '.join(target_cols)})")
print(f"   • Data shape: {X.shape[0]} samples × {X.shape[1]} features")
print(f"   • Scaling: MinMaxScaler({scaler_X.feature_range[0]}, {scaler_X.feature_range[1]})")

print(f"\n🏗️  ARCHITECTURE:")
print(f"   • Shared layers: {len(ln_blocks)} layer(s) with {ln_blocks[0]} PTA blocks")
print(f"   • Output layers: {num_outputs} output heads with {output_ln_blocks} PTA blocks each")
print(f"   • Total PTA blocks: {ln_blocks[0] * len(ln_blocks) + num_outputs * output_ln_blocks}")
print(f"   • Input size: {input_size} features")
print(f"   • Output size: {num_outputs} targets")

print(f"\n⚙️  TRAINING PARAMETERS:")
print(f"   • Learning rate: {init_lr}")
print(f"   • Decay steps: {decay_steps}")
print(f"   • Task weights: {task_weights} (sum = {sum(task_weights):.2f})")
print(f"   • Shared layer L1/L2: 1e-3/1e-3")
print(f"   • Output layer L1/L2: 0.2/0.1")
print(f"   • Early stopping: patience = 50")

print(f"\n📈 FINAL METRICS (SCALED DATA):")
print(f"   • Target 1 ({target_cols[0]}): MSE = {mse_target1:.4f}, MAPE = {mape_target1:.2f}%")
if num_outputs > 1:
    print(f"   • Target 2 ({target_cols[1]}): MSE = {mse_target2:.4f}, MAPE = {mape_target2:.2f}%")
print(f"   • Average: MSE = {avg_mse:.4f}, MAPE = {avg_mape:.2f}%")

print(f"\n🎯 PERFORMANCE ASSESSMENT:")
if avg_mape < 5.0:
    performance = "EXCELLENT"
elif avg_mape < 10.0:
    performance = "GOOD"
elif avg_mape < 20.0:
    performance = "FAIR"
else:
    performance = "NEEDS IMPROVEMENT"
print(f"   • Overall MAPE: {avg_mape:.2f}% ({performance})")
print(f"   • Model complexity: {ln_blocks[0]} shared + {num_outputs * output_ln_blocks} output PTA blocks")
print("="*80)
print(f"Complexity reduction: Strong L1/L2 regularization + post-processing threshold {threshold}")

# Save results to CSV
results_data = {
    'Model': ['Multi-GINN'],
    'Y1_MSE': [mse_target1],
    'Y2_MSE': [mse_target2],
    'Avg_MSE': [avg_mse],
    'Y1_MAPE': [mape_target1],
    'Y2_MAPE': [mape_target2],
    'Avg_MAPE': [avg_mape]
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('multiginn_results.csv', index=False)
print(f"\nResults saved to 'multiginn_results.csv'")

print(f"\n=== DETAILED METRICS (SCALED DATA) ===")
print(f"Target 1: MSE = {mse_target1:.4f}, MAPE = {mape_target1:.2f}%")
print(f"Target 2: MSE = {mse_target2:.4f}, MAPE = {mape_target2:.2f}%")
print(f"Average: MSE = {avg_mse:.4f}, MAPE = {avg_mape:.2f}%")

# Also show original scale metrics for reference
mse_target1_orig = mean_squared_error(Y[:, 0], predictions_original[:, 0])
mse_target2_orig = mean_squared_error(Y[:, 1], predictions_original[:, 1])
mape_target1_orig = mean_absolute_percentage_error(Y[:, 0], predictions_original[:, 0])
mape_target2_orig = mean_absolute_percentage_error(Y[:, 1], predictions_original[:, 1])
avg_mape_orig = (mape_target1_orig + mape_target2_orig) / 2

print(f"\n=== ORIGINAL SCALE METRICS (FOR REFERENCE) ===")
print(f"Target 1: MSE = {mse_target1_orig:.4f}, MAPE = {mape_target1_orig:.2f}%")
print(f"Target 2: MSE = {mse_target2_orig:.4f}, MAPE = {mape_target2_orig:.2f}%")
print(f"Average MAPE: {avg_mape_orig:.2f}%")

print(f"\n=== PRIMARY METRIC: MAPE (Scale-Invariant) ===")
print(f"MAPE is the better metric for diverse data scales")
print(f"Overall MAPE (scaled): {avg_mape:.2f}% (excellent if < 5%)")