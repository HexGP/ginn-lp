import tensorflow as tf
from tensorflow.keras import layers, activations, initializers
import numpy as np

def log_activation(in_x):
    return tf.math.log(in_x)

def eql_ln_block_simple(inputs_x, layer_num):
    """PTA block that shows exactly what it accepts and produces"""
    print(f"\n=== PTA Block {layer_num} ===")
    print(f"Input type: {type(inputs_x)}")
    print(f"Number of inputs: {len(inputs_x)}")
    
    # Show each input
    for i, input_tensor in enumerate(inputs_x):
        print(f"Input {i}: shape={input_tensor.shape}, value={input_tensor.numpy()[0][0]}")
    
    # Process each input through log transformation
    ln_layers = []
    for i, input_x in enumerate(inputs_x):
        ln_layer = layers.Dense(1, use_bias=False,
                              kernel_initializer=initializers.Identity(gain=1.0),
                              trainable=False,
                              activation=log_activation)(input_x)
        ln_layers.append(ln_layer)
        print(f"After log: ln(x_{i+1}) = {ln_layer.numpy()[0][0]:.4f}")
    
    # Concatenate and combine
    if len(ln_layers) == 1:
        ln_concat = ln_layers[0]
    else:
        ln_concat = layers.Concatenate()(ln_layers)
    
    # Final exponential transformation
    ln_dense = layers.Dense(1, use_bias=False, 
                          activation=activations.exponential)(ln_concat)
    
    output_value = ln_dense.numpy()[0][0]
    print(f"Output: {output_value:.4f}")
    print(f"Output type: {type(ln_dense)}")
    print(f"Output shape: {ln_dense.shape}")
    
    return ln_dense

# Example usage
if __name__ == "__main__":
    print("=== PTA BLOCK INPUT/OUTPUT EXPLANATION ===\n")
    
    # Example 1: 2 variables
    print("EXAMPLE 1: 2 variables (x₁=2.0, x₂=3.0)")
    x1 = tf.constant([[2.0]], dtype=tf.float32)
    x2 = tf.constant([[3.0]], dtype=tf.float32)
    inputs_x = [x1, x2]  # List of 2 separate tensors
    
    output = eql_ln_block_simple(inputs_x, layer_num=0)
    
    print(f"\nMathematical interpretation:")
    print(f"Input: x₁={x1.numpy()[0][0]}, x₂={x2.numpy()[0][0]}")
    print(f"PTA computes: exp(ln(x₁) + ln(x₂)) = exp(ln({x1.numpy()[0][0]}) + ln({x2.numpy()[0][0]}))")
    print(f"= exp({np.log(x1.numpy()[0][0]):.4f} + {np.log(x2.numpy()[0][0]):.4f})")
    print(f"= exp({np.log(x1.numpy()[0][0]) + np.log(x2.numpy()[0][0]):.4f})")
    print(f"= {output.numpy()[0][0]:.4f}")
    print(f"= x₁ * x₂ = {x1.numpy()[0][0]} * {x2.numpy()[0][0]} = {x1.numpy()[0][0] * x2.numpy()[0][0]}")
    
    print("\n" + "="*50)
    
    # Example 2: 3 variables
    print("\nEXAMPLE 2: 3 variables (x₁=2.0, x₂=3.0, x₃=4.0)")
    x1 = tf.constant([[2.0]], dtype=tf.float32)
    x2 = tf.constant([[3.0]], dtype=tf.float32)
    x3 = tf.constant([[4.0]], dtype=tf.float32)
    inputs_x = [x1, x2, x3]  # List of 3 separate tensors
    
    output = eql_ln_block_simple(inputs_x, layer_num=1)
    
    print(f"\nMathematical interpretation:")
    print(f"Input: x₁={x1.numpy()[0][0]}, x₂={x2.numpy()[0][0]}, x₃={x3.numpy()[0][0]}")
    print(f"PTA computes: exp(ln(x₁) + ln(x₂) + ln(x₃))")
    print(f"= exp(ln({x1.numpy()[0][0]}) + ln({x2.numpy()[0][0]}) + ln({x3.numpy()[0][0]}))")
    print(f"= exp({np.log(x1.numpy()[0][0]):.4f} + {np.log(x2.numpy()[0][0]):.4f} + {np.log(x3.numpy()[0][0]):.4f})")
    print(f"= exp({np.log(x1.numpy()[0][0]) + np.log(x2.numpy()[0][0]) + np.log(x3.numpy()[0][0]):.4f})")
    print(f"= {output.numpy()[0][0]:.4f}")
    print(f"= x₁ * x₂ * x₃ = {x1.numpy()[0][0]} * {x2.numpy()[0][0]} * {x3.numpy()[0][0]} = {x1.numpy()[0][0] * x2.numpy()[0][0] * x3.numpy()[0][0]}")
    
    print("\n" + "="*50)
    
    # Example 3: With different weights (showing polynomial terms)
    print("\nEXAMPLE 3: With learned weights (x₁=2.0, x₂=3.0, weights=[2.0, 1.5])")
    x1 = tf.constant([[2.0]], dtype=tf.float32)
    x2 = tf.constant([[3.0]], dtype=tf.float32)
    inputs_x = [x1, x2]
    
    # Create PTA with custom weights
    ln1 = layers.Dense(1, use_bias=False, 
                      kernel_initializer=initializers.Constant(2.0),
                      trainable=False, activation=log_activation)(x1)
    ln2 = layers.Dense(1, use_bias=False, 
                      kernel_initializer=initializers.Constant(1.5),
                      trainable=False, activation=log_activation)(x2)
    
    concat = layers.Concatenate()([ln1, ln2])
    output = layers.Dense(1, use_bias=False, 
                         kernel_initializer=initializers.Constant(1.0),
                         activation=activations.exponential)(concat)
    
    print(f"Input: x₁={x1.numpy()[0][0]}, x₂={x2.numpy()[0][0]}")
    print(f"Weights: w₁=2.0, w₂=1.5")
    print(f"PTA computes: exp(2.0*ln(x₁) + 1.5*ln(x₂))")
    print(f"= exp(2.0*ln({x1.numpy()[0][0]}) + 1.5*ln({x2.numpy()[0][0]}))")
    print(f"= exp({2.0*np.log(x1.numpy()[0][0]):.4f} + {1.5*np.log(x2.numpy()[0][0]):.4f})")
    print(f"= exp({2.0*np.log(x1.numpy()[0][0]) + 1.5*np.log(x2.numpy()[0][0]):.4f})")
    print(f"= {output.numpy()[0][0]:.4f}")
    print(f"= x₁^2.0 * x₂^1.5 = {x1.numpy()[0][0]}^2.0 * {x2.numpy()[0][0]}^1.5 = {x1.numpy()[0][0]**2.0 * x2.numpy()[0][0]**1.5:.4f}")
    
    print("\n" + "="*50)
    print("\nSUMMARY:")
    print("1. INPUT: List of separate scalar tensors (one per variable)")
    print("2. OUTPUT: Single scalar value representing a polynomial term")
    print("3. POLYNOMIAL TERM: Mathematical expression like x₁² * x₂³")
    print("4. PURPOSE: Converts variables into interpretable mathematical terms") 