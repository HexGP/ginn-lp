import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from ginnlp.de_learn_network import eql_model_v3_multioutput, eql_opt, get_multioutput_sympy_expr

# 1. Load data
csv_path = "data/ENB2012_data.csv"  # Use original data file
df = pd.read_csv(csv_path)
X = df.iloc[:, :8].values.astype(np.float32)  # First 8 columns as input (X1-X8)
Y = df.iloc[:, 8:10].values.astype(np.float32) # target_1 and target_2 columns

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
for i in range(8):
    print(f"X{i+1}: {X[:, i].min():.2f} to {X[:, i].max():.2f}")
print("\nScaled input ranges:")
for i in range(8):
    print(f"X{i+1}: {X_scaled[:, i].min():.2f} to {X_scaled[:, i].max():.2f}")

print("\nOriginal target ranges:")
for i in range(2):
    print(f"Y{i+1}: {Y[:, i].min():.2f} to {Y[:, i].max():.2f}")
print("\nScaled target ranges:")
for i in range(2):
    print(f"Y{i+1}: {Y_scaled[:, i].min():.2f} to {Y_scaled[:, i].max():.2f}")

# 3. Use simple architecture that was working well
input_size = 8
num_outputs = 2
ln_blocks = (4, 4)          # 2 shared layers: both with 4 PTA blocks
lin_blocks = (1, 1)         # Must match ln_blocks
output_ln_blocks = 4        # 4 PTA blocks per output

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
    l1_reg=1e-3,   # Standard regularization
    l2_reg=1e-3
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
task_weights = [0.2, 0.8]  # 60% weight to target_1, 40% to target_2
print(f"Task weights: {task_weights} (sum = {sum(task_weights)})")

# Compile with custom weighted loss
custom_loss = weighted_multi_task_loss(task_weights)
model.compile(optimizer=opt, loss=custom_loss, metrics=['mean_squared_error', 'mean_absolute_percentage_error'])

print("\nModel Summary:")
model.summary()

# 5. Train model with scaled values
history = model.fit(
    [X_scaled[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y_scaled[:, i].reshape(-1, 1) for i in range(num_outputs)],
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# 6. Evaluate with scaled values
results = model.evaluate(
    [X_scaled[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y_scaled[:, i].reshape(-1, 1) for i in range(num_outputs)],
    verbose=0
)

# 7. Calculate predictions and convert back to original scale for metrics
predictions_scaled = model.predict([X_scaled[:, i].reshape(-1, 1) for i in range(input_size)])
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

# 8. Extract and print symbolic equations
print("\n=== SYMBOLIC EQUATIONS ===")
get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3)

# 9. Print final summary with actual metrics
print(f"\nTraining completed successfully!")
print("\n=== FINAL SUMMARY ===")
print("Architecture: 2 shared layers (4,4) + 4 output-specific PTA blocks")
print("Data: ENB2012_data.csv with MinMaxScaler normalization")
print(f"Target ranges: {scaler_Y.feature_range[0]} to {scaler_Y.feature_range[1]} (scaled)")
print(f"Task weights: {task_weights} (sum = {sum(task_weights)})")


# 10. FINAL PERFORMANCE METRICS (at the end)
print(f"\n{'='*60}")
print(f"FINAL PERFORMANCE METRICS")
print(f"{'='*60}")

print(f"\n=== SCALED DATA METRICS ===")
for i in range(num_outputs):
    mse_scaled = mean_squared_error(Y_scaled[:, i], predictions_scaled[i].flatten())
    mape_scaled = mean_absolute_percentage_error(Y_scaled[:, i], predictions_scaled[i].flatten())
    print(f"Target {i+1} (scaled): MSE = {mse_scaled:.6f}, MAPE = {mape_scaled:.2f}%")

print(f"\n=== ORIGINAL SCALE METRICS ===")
for i in range(num_outputs):
    mse = mean_squared_error(Y[:, i], predictions_original[:, i])
    mape = mean_absolute_percentage_error(Y[:, i], predictions_original[:, i])
    print(f"Target {i+1}: MSE = {mse:.4f}, MAPE = {mape:.2f}%")

# Calculate overall metrics
overall_mse = np.mean([mean_squared_error(Y[:, i], predictions_original[:, i]) for i in range(num_outputs)])
overall_mape = np.mean([mean_absolute_percentage_error(Y[:, i], predictions_original[:, i]) for i in range(num_outputs)])

print(f"\n=== OVERALL PERFORMANCE ===")
print(f"Overall MSE: {overall_mse:.4f}")
print(f"Overall MAPE: {overall_mape:.2f}%")

print(f"\n=== PRIMARY METRIC: MAPE (Scale-Invariant) ===")
print(f"MAPE is the better metric for diverse data scales")
print(f"Overall MAPE: {overall_mape:.2f}% (excellent if < 5%)")