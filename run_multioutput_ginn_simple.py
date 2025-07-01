import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from ginnlp.de_learn_network import eql_model_v3_multioutput, eql_opt, get_multioutput_sympy_expr

# 1. Load data
csv_path = "data/ENB2012_nozero.csv"
df = pd.read_csv(csv_path)
X = df.iloc[:, :6].values.astype(np.float32)  # First 6 columns as input
Y = df.iloc[:, 6:8].values.astype(np.float32) # Next 2 columns as targets

# 2. Use raw values directly (no scaling) - all values are already positive
print("Input ranges:")
for i in range(6):
    print(f"X{i+1}: {X[:, i].min():.2f} to {X[:, i].max():.2f}")
print("\nTarget ranges:")
for i in range(2):
    print(f"Y{i+1}: {Y[:, i].min():.2f} to {Y[:, i].max():.2f}")
    
# 3. Analyze target correlation
print("=== TARGET CORRELATION ANALYSIS ===")
target_corr = np.corrcoef(Y.T)[0, 1]
print(f"Correlation between targets: {target_corr:.4f}")

if abs(target_corr) > 0.8:
    print("WARNING: High correlation between targets (>0.8). Similar equations may be expected.")
elif abs(target_corr) > 0.5:
    print("Moderate correlation between targets. Some similarity in equations is expected.")
else:
    print("Low correlation between targets. Equations should be quite different.")

# Plot target correlation
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(Y[:, 0], Y[:, 1], alpha=0.6)
plt.xlabel('Target 1')
plt.ylabel('Target 2')
plt.title(f'Target Correlation: {target_corr:.4f}')

plt.subplot(1, 2, 2)
sns.heatmap(np.corrcoef(Y.T), annot=True, cmap='coolwarm', 
            xticklabels=['Target 1', 'Target 2'], 
            yticklabels=['Target 1', 'Target 2'])
plt.title('Target Correlation Matrix')
plt.tight_layout()
plt.savefig('target_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Data analysis
print("\n=== DATA ANALYSIS ===")
print("Input ranges:")
for i in range(6):
    print(f"X{i+1}: {X[:, i].min():.2f} to {X[:, i].max():.2f} (std: {X[:, i].std():.2f})")

print("\nTarget ranges:")
for i in range(2):
    print(f"Y{i+1}: {Y[:, i].min():.2f} to {Y[:, i].max():.2f} (std: {Y[:, i].std():.2f})")

# Check for any zero or negative values that could cause log issues
print(f"\nZero values in inputs: {np.sum(X == 0)}")
print(f"Negative values in inputs: {np.sum(X < 0)}")
print(f"Zero values in targets: {np.sum(Y == 0)}")
print(f"Negative values in targets: {np.sum(Y < 0)}")

# 4. Use original working architecture
print("\n=== MODEL ARCHITECTURE ===")
input_size = 6
num_outputs = 2

# Use 2 shared layers with more features
ln_blocks = (3, 3)          # 2 shared layers: both with 3 PTA blocks = 6 shared features total
lin_blocks = (1, 1)         # Must match ln_blocks
output_ln_blocks = 4        # Same as original

print(f"Shared layers: {len(ln_blocks)}")
print(f"Shared PTA blocks per layer: {ln_blocks}")
print(f"Output-specific PTA blocks: {output_ln_blocks}")

# 5. Use original training setup
decay_steps = 1000          # Same as original
init_lr = 0.01              # Same as original
opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)

# 6. Build model with reduced regularization
model = eql_model_v3_multioutput(
    input_size=input_size,
    opt=opt,
    ln_blocks=ln_blocks,
    lin_blocks=lin_blocks,
    output_ln_blocks=output_ln_blocks,
    num_outputs=num_outputs,
    compile=False,  # We'll compile manually
    l1_reg=1e-5,   # REDUCED L1 regularization
    l2_reg=1e-5    # REDUCED L2 regularization
)

# 7. Compile with custom metrics
model.compile(
    optimizer=opt, 
    loss='mean_squared_error', 
    metrics=['mean_squared_error', 'mean_absolute_percentage_error']
)

print("\nModel Summary:")
model.summary()

# 8. Train with simple setup
print("\n=== TRAINING ===")
history = model.fit(
    [X[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y[:, i].reshape(-1, 1) for i in range(num_outputs)],
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# 9. Comprehensive evaluation
print("\n=== EVALUATION ===")
results = model.evaluate(
    [X[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y[:, i].reshape(-1, 1) for i in range(num_outputs)],
    verbose=0
)

# Calculate predictions for detailed analysis
predictions = model.predict([X[:, i].reshape(-1, 1) for i in range(input_size)])

print("\nDetailed Metrics:")
for i in range(num_outputs):
    # Check for NaN in predictions
    if np.any(np.isnan(predictions[i])):
        print(f"Output {i}: Contains NaN values!")
        continue
    
    mse = mean_squared_error(Y[:, i], predictions[i].flatten())
    mape = mean_absolute_percentage_error(Y[:, i], predictions[i].flatten())
    print(f"Output {i}: MSE = {mse:.2f}, MAPE = {mape:.2f}")

# 10. Plot training history (only if we have valid data)
if 'loss' in history.history and not np.any(np.isnan(history.history['loss'])):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['output_0_mean_squared_error'], label='Output 0 MSE')
    plt.plot(history.history['output_1_mean_squared_error'], label='Output 1 MSE')
    plt.plot(history.history['val_output_0_mean_squared_error'], label='Val Output 0 MSE')
    plt.plot(history.history['val_output_1_mean_squared_error'], label='Val Output 1 MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['output_0_mean_absolute_percentage_error'], label='Output 0 MAPE')
    plt.plot(history.history['output_1_mean_absolute_percentage_error'], label='Output 1 MAPE')
    plt.plot(history.history['val_output_0_mean_absolute_percentage_error'], label='Val Output 0 MAPE')
    plt.plot(history.history['val_output_1_mean_absolute_percentage_error'], label='Val Output 1 MAPE')
    plt.title('Mean Absolute Percentage Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Cannot plot training history due to NaN values")

# 11. Extract and print symbolic equations
print("\n=== SYMBOLIC EQUATIONS ===")
get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3)

# 12. Final summary
print("\n=== FINAL SUMMARY ===")
print(f"Target correlation: {target_corr:.4f}")
if 'loss' in history.history and not np.any(np.isnan(history.history['loss'])):
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
else:
    print("Training failed due to NaN values")
print("Model architecture:")
print("- Using 2 shared layers (3, 3) = 6 shared features total")
print("- Non-trainable output-specific PTA blocks with diverse initialization")
print("- REDUCED L1/L2 regularization (1e-5) for feature diversity") 