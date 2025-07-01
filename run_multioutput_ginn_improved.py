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

# 2. Analyze target correlation
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

# 4. Improved model architecture
print("\n=== MODEL ARCHITECTURE ===")
input_size = 6
num_outputs = 2

# More diverse architecture
ln_blocks = (6, 4, 3)       # 3 shared layers with decreasing complexity
lin_blocks = (6, 4, 3)      # Must match ln_blocks
output_ln_blocks = 12       # More output-specific complexity

print(f"Shared layers: {len(ln_blocks)}")
print(f"Shared PTA blocks per layer: {ln_blocks}")
print(f"Output-specific PTA blocks: {output_ln_blocks}")

# 5. Improved training setup
decay_steps = 2000          # Slower decay
init_lr = 0.005             # Lower initial learning rate
opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)

# 6. Build model with stronger regularization for diversity
model = eql_model_v3_multioutput(
    input_size=input_size,
    opt=opt,
    ln_blocks=ln_blocks,
    lin_blocks=lin_blocks,
    output_ln_blocks=output_ln_blocks,
    num_outputs=num_outputs,
    compile=False,  # We'll compile manually
    l1_reg=1e-3,   # Stronger L1 regularization
    l2_reg=1e-3    # Stronger L2 regularization
)

# 7. Compile with custom metrics
model.compile(
    optimizer=opt, 
    loss='mean_squared_error', 
    metrics=['mean_squared_error', 'mean_absolute_percentage_error']
)

print("\nModel Summary:")
model.summary()

# 8. Train with early stopping and more epochs
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

print("\n=== TRAINING ===")
history = model.fit(
    [X[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y[:, i].reshape(-1, 1) for i in range(num_outputs)],
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
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
    mse = mean_squared_error(Y[:, i], predictions[i].flatten())
    mape = mean_absolute_percentage_error(Y[:, i], predictions[i].flatten())
    print(f"Output {i}: MSE = {mse:.2f}, MAPE = {mape:.2f}")

# 10. Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['mean_squared_error'], label='Training MSE')
plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
plt.title('Mean Squared Error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['mean_absolute_percentage_error'], label='Training MAPE')
plt.plot(history.history['val_mean_absolute_percentage_error'], label='Validation MAPE')
plt.title('Mean Absolute Percentage Error')
plt.xlabel('Epoch')
plt.ylabel('MAPE')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. Extract and print symbolic equations
print("\n=== SYMBOLIC EQUATIONS ===")
get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3)

# 12. Final summary
print("\n=== FINAL SUMMARY ===")
print(f"Target correlation: {target_corr:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print("Model architecture improvements:")
print("- More shared layers with decreasing complexity")
print("- More output-specific PTA blocks")
print("- Stronger regularization for diversity")
print("- Slower learning rate decay")
print("- Early stopping to prevent overfitting") 