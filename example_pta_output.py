import tensorflow as tf
from tensorflow.keras import layers, activations, initializers
import numpy as np

# Import the necessary functions from the GINN-LP codebase
def log_activation(in_x):
    return tf.math.log(in_x)

def eql_ln_block_simple(inputs_x, layer_num):
    ln_layers = [layers.Dense(1, use_bias=False,
                              kernel_initializer=initializers.Identity(gain=1.0),
                              trainable=False,
                              activation=log_activation,
                              name='ln_{}_{}'.format(layer_num, i))(input_x) for i, input_x in enumerate(inputs_x)]
    if len(ln_layers) == 1:
        ln_concat = ln_layers[0]
    else:
        ln_concat = layers.Concatenate()(ln_layers)
    ln_dense = layers.Dense(1,
                            use_bias=False, 
                            activation=activations.exponential,
                            name='ln_dense_{}'.format(layer_num))(ln_concat)
    return ln_dense

# Example usage
if __name__ == "__main__":
    # Create sample input data: [x1, x2] = [2.0, 3.0]
    x1 = tf.constant([[2.0]], dtype=tf.float32)
    x2 = tf.constant([[3.0]], dtype=tf.float32)
    inputs_x = [x1, x2]
    
    print("Input values:")
    print(f"x1 = {x1.numpy()[0][0]}")
    print(f"x2 = {x2.numpy()[0][0]}")
    print()
    
    # Create the PTA block
    pta_output = eql_ln_block_simple(inputs_x, layer_num=0)
    
    # Let's manually trace through the computation to show what happens:
    print("Step-by-step computation:")
    
    # Step 1: Log transformation
    log_x1 = tf.math.log(x1)
    log_x2 = tf.math.log(x2)
    print(f"ln(x1) = ln({x1.numpy()[0][0]}) = {log_x1.numpy()[0][0]:.4f}")
    print(f"ln(x2) = ln({x2.numpy()[0][0]}) = {log_x2.numpy()[0][0]:.4f}")
    
    # Step 2: Concatenate (with Identity weights = 1.0)
    # The Identity initializer sets weights to 1.0, so:
    # w1 * ln(x1) + w2 * ln(x2) = 1.0 * ln(x1) + 1.0 * ln(x2)
    combined_log = log_x1 + log_x2
    print(f"w1*ln(x1) + w2*ln(x2) = 1.0*{log_x1.numpy()[0][0]:.4f} + 1.0*{log_x2.numpy()[0][0]:.4f} = {combined_log.numpy()[0][0]:.4f}")
    
    # Step 3: Exponential activation
    final_output = tf.math.exp(combined_log)
    print(f"exp(w1*ln(x1) + w2*ln(x2)) = exp({combined_log.numpy()[0][0]:.4f}) = {final_output.numpy()[0][0]:.4f}")
    
    print()
    print("Mathematical interpretation:")
    print(f"exp(ln(x1) + ln(x2)) = exp(ln(x1 * x2)) = x1 * x2 = {x1.numpy()[0][0]} * {x2.numpy()[0][0]} = {final_output.numpy()[0][0]:.4f}")
    
    print()
    print("What the PTA block actually outputs:")
    print(f"PTA output = {pta_output.numpy()[0][0]:.4f}")
    
    print()
    print("With different weights (example):")
    # Let's show what happens with different weights
    # Create a model to demonstrate weight changes
    input1 = layers.Input(shape=(1,))
    input2 = layers.Input(shape=(1,))
    
    # Create the PTA block with custom weights
    ln1 = layers.Dense(1, use_bias=False, 
                      kernel_initializer=initializers.Constant(2.0),
                      trainable=False, activation=log_activation)(input1)
    ln2 = layers.Dense(1, use_bias=False, 
                      kernel_initializer=initializers.Constant(1.5),
                      trainable=False, activation=log_activation)(input2)
    
    concat = layers.Concatenate()([ln1, ln2])
    output = layers.Dense(1, use_bias=False, 
                         kernel_initializer=initializers.Constant(1.0),
                         activation=activations.exponential)(concat)
    
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    
    # Test with the same inputs
    result = model.predict([x1, x2])
    print(f"With weights w1=2.0, w2=1.5: exp(2.0*ln(x1) + 1.5*ln(x2)) = exp(2.0*ln({x1.numpy()[0][0]}) + 1.5*ln({x2.numpy()[0][0]}))")
    print(f"= exp({2.0*log_x1.numpy()[0][0]:.4f} + {1.5*log_x2.numpy()[0][0]:.4f}) = exp({2.0*log_x1.numpy()[0][0] + 1.5*log_x2.numpy()[0][0]:.4f})")
    print(f"= {result[0][0]:.4f}")
    print(f"= x1^2.0 * x2^1.5 = {x1.numpy()[0][0]}^2.0 * {x2.numpy()[0][0]}^1.5 = {x1.numpy()[0][0]**2.0 * x2.numpy()[0][0]**1.5:.4f}")
    
    print()
    print("Summary of PTA block output:")
    print("The PTA block takes input variables and outputs polynomial terms of the form:")
    print("exp(∑ᵢ wᵢ * ln(xᵢ)) = ∏ᵢ xᵢ^wᵢ")
    print("This allows the network to learn polynomial relationships between variables.") 