import pandas as pd
import numpy as np
from ginnlp.de_learn_network import eql_model_v3_multioutput, eql_opt

# Load data
csv_path = "data/ENB2012_nozero.csv"
df = pd.read_csv(csv_path)
X = df.iloc[:, :6].values.astype(np.float32)
Y = df.iloc[:, 6:8].values.astype(np.float32)

# Build model with same architecture
input_size = 6
num_outputs = 2
ln_blocks = (4, 4)          # 2 shared PTA layers, 4 blocks each
lin_blocks = (4, 4)         # 4 output units per shared layer
output_ln_blocks = 8        # 8 PTA blocks per output

decay_steps = 1000
init_lr = 0.01
opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)

# Debug: Print what values we're passing
l1_reg = 1e-4
l2_reg = 1e-4
print(f"DEBUG: Passing l1_reg={l1_reg}, l2_reg={l2_reg}")
print(f"DEBUG: Type of l1_reg: {type(l1_reg)}, Type of l2_reg: {type(l2_reg)}")

model = eql_model_v3_multioutput(
    input_size=input_size,
    opt=opt,
    ln_blocks=ln_blocks,
    lin_blocks=lin_blocks,
    output_ln_blocks=output_ln_blocks,
    num_outputs=num_outputs,
    compile=True,
    l1_reg=l1_reg,
    l2_reg=l2_reg
)

# Train for a few epochs to see the weights
history = model.fit(
    [X[:, i].reshape(-1, 1) for i in range(input_size)],
    [Y[:, i].reshape(-1, 1) for i in range(num_outputs)],
    epochs=10,
    batch_size=32,
    verbose=1
)

print("\n" + "="*50)
print("DEBUGGING SHARED FEATURES")
print("="*50)

# Check the weights of the last shared layer's output_dense layers
print("\nLast shared layer output_dense weights:")
for i in range(4):  # 4 output units
    layer_name = f'output_dense_1_{i}'  # Layer 1 (second layer), unit i
    weights = model.get_layer(layer_name).get_weights()
    print(f"{layer_name}: {weights[0].flatten()}")

print("\nLast shared layer ln_dense weights:")
for i in range(4):  # 4 PTA blocks
    layer_name = f'ln_dense_1_{i}'  # Layer 1 (second layer), block i
    weights = model.get_layer(layer_name).get_weights()
    print(f"{layer_name}: {weights[0].flatten()}")

# Check if the output_dense layers have identical weights
print("\nChecking if output_dense layers are identical:")
base_weights = model.get_layer('output_dense_1_0').get_weights()[0].flatten()
for i in range(1, 4):
    layer_name = f'output_dense_1_{i}'
    weights = model.get_layer(layer_name).get_weights()[0].flatten()
    is_identical = np.allclose(base_weights, weights, atol=1e-6)
    print(f"{layer_name} identical to output_dense_1_0: {is_identical}")
    if not is_identical:
        print(f"  Difference: {np.max(np.abs(base_weights - weights))}")

print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
model.summary() 