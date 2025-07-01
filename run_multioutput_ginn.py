import pandas as pd
import numpy as np
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

# 3. Build model with REDUCED shared layers and INCREASED output-specific complexity
input_size = 6
num_outputs = 2
ln_blocks = (4, 4)          # REDUCED: 2 shared PTA layers instead of 3
lin_blocks = (4, 4)         # Must match ln_blocks
output_ln_blocks = 8        # INCREASED: 8 PTA blocks per output instead of 4

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
    compile=True,
    l1_reg=1e-4,
    l2_reg=1e-4
)

# 4. Add MSE and MAPE metrics
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_percentage_error'])

# 5. Train model with raw values
history = model.fit(
    [X[:, i].reshape(-1, 1) for i in range(input_size)],  # List of arrays, one per input
    [Y[:, i].reshape(-1, 1) for i in range(num_outputs)], # List of arrays, one per output
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# 6. Evaluate with raw values
results = model.evaluate(
    [X[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y[:, i].reshape(-1, 1) for i in range(num_outputs)],
    verbose=2
)
print("Evaluation (MSE, MAPE for each output):", results)

# 7. Extract and print symbolic equations for each output
get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3)

# 8. Print metrics again for clarity
print("\nFinal Evaluation Metrics (MSE, MAPE for each output):", results) 

# Print model summary
# model.summary()