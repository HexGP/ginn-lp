from tensorflow.keras.models import Model
from tensorflow.keras import layers, activations, regularizers, optimizers, initializers, constraints
from tensorflow.keras.regularizers import Regularizer

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
from ginnlp.utils import get_sympy_expr_v3, get_multioutput_sympy_expr_v2


def my_tf_round(x, decimals=0):
    multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


class PosConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be positive (wider range)"""
    def __call__(self, w):
        return tf.clip_by_value(w, 0.0001, 100.0)


class IntConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be positive"""

    def __call__(self, w):
        return my_tf_round(w, 1)


class WeightMaskCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight_mask):
        super().__init__()
        self.weight_mask = weight_mask

    def on_train_batch_end(self, batch, logs=None):
        for i in range(len(self.model.layers)):
            new_weights = [self.model.layers[i].get_weights()[j] * self.weight_mask[i][j] for j in
                           range(len(self.model.layers[i].get_weights()))]
            self.model.layers[i].set_weights(new_weights)


class ClippedDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        super(ClippedDense, self).__init__(**kwargs)
        self.weight_thresh = None

    def set_weight_tresh(self, weight_tresh):
        self.weight_thresh = weight_tresh

    def call(self, inputs):
        if self.weight_thresh:
            kernel_mask = tf.cast(tf.abs(self.kernel) > self.weight_tresh, dtype=tf.float32)
            self.kernel = tf.multiply(self.kernel * kernel_mask)

            bias_mask = tf.cast(tf.abs(self.bias) > self.weight_tresh, dtype=tf.float32)
            self.bias = tf.multiply(self.bias * bias_mask)
        return tf.matmul(inputs, self.kernel) + self.bias


class L1L2_m(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.01, int_reg=0.0):
        with K.name_scope(self.__class__.__name__):
            self.l1 = K.variable(l1, name='l1')
            self.l2 = K.variable(l2, name='l2')
            self.val_l1 = l1
            self.val_l2 = l2
            self.val_int = int_reg

    def set_l1_l2(self, l1, l2):
        K.set_value(self.l1, l1)
        K.set_value(self.l2, l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def __call__(self, x):
        regularization = 0.
        if self.val_l1 > 0.:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.val_int > 0:
            regularization += K.sum(self.val_int*K.abs(x - K.round(x)))
        if self.val_l2 > 0.:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        config = {'l1': float(K.get_value(self.l1)),
                  'l2': float(K.get_value(self.l2))}
        return config


def set_model_l1_l2(model, l1, l2):
    for layer in model.layers:
        if 'kernel_regularizer' in dir(layer) and \
                isinstance(layer.kernel_regularizer, L1L2_m):
            layer.kernel_regularizer.set_l1_l2(l1, l2)


# Original Log Activation
def log_activation(in_x):
    return tf.math.log(in_x)

# def log_activation(in_x):
#     return tf.math.sign(in_x) * tf.math.log(1 + tf.math.abs(in_x))


def abs_activation(in_x):
    return tf.math.abs(in_x)


def sign_activation(in_x):
    return tf.math.sign(in_x)


def eql_model(input_size):
    inputs_x = [layers.Input(shape=(1,)) for i in range(input_size)]
    ln_layers = [layers.Dense(1, use_bias=False,
                              kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                              kernel_initializer=initializers.Identity(gain=1.0),
                              trainable=False,
                              activation=log_activation)(input_x) for input_x in inputs_x]
    ln_concat = layers.Concatenate()(ln_layers)
    ln_dense = layers.Dense(1,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                            use_bias=False, activation=activations.exponential,
                            kernel_initializer=initializers.Identity(gain=1.0))(ln_concat)

    input_x_concat = layers.Concatenate()(inputs_x)
    input_x_dense = layers.Dense(1,
                                 activation='linear',
                                 kernel_regularizer=regularizers.l1_l2(
                                     l1=1e-3,
                                     l2=1e-3),
                                 bias_regularizer=regularizers.l1_l2(
                                     l1=1e-3,
                                     l2=1e-3))(input_x_concat)

    output_concat = layers.Concatenate()([ln_dense, input_x_dense])
    output_dense = layers.Dense(1, use_bias=False, activation='linear',
                                kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))(output_concat)
    model = Model(inputs=inputs_x, outputs=output_dense, name='eql_model')

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.1,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)
    opt = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


def eql_ln_block(inputs_x, layer_num, l1_reg=0, l2_reg=0):
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
                            kernel_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg, int_reg=0.0),
                            use_bias=False, activation=activations.exponential,
                            # kernel_constraint=IntConstraint(),
                            name='ln_dense_{}'.format(layer_num))(ln_concat)
    return ln_dense


def eql_model_v2(input_size, ln_block_count=2, decay_steps=1000, compile=True):
    inputs_x = [layers.Input(shape=(1,)) for i in range(input_size)]
    ln_dense_units = [eql_ln_block(inputs_x, layer_num=i) for i in range(ln_block_count)]

    if ln_block_count == 1:
        ln_dense_concat = ln_dense_units[0]
    else:
        ln_dense_concat = layers.Concatenate()(ln_dense_units)
    output_dense = layers.Dense(1, activation='linear',
                                kernel_regularizer=L1L2_m(l1=1e-3, l2=1e-3),
                                bias_regularizer=L1L2_m(l1=1e-3, l2=1e-3),
                                name='output_dense')(ln_dense_concat)
    model = Model(inputs=inputs_x, outputs=output_dense, name='eql_model')

    if compile:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.01,
            decay_steps=decay_steps,
            decay_rate=0.96,
            staircase=True)
        opt = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
        print('get weights 2', model.get_weights())
    return model


def eql_opt(decay_steps=1000, init_lr=0.01):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        init_lr,
        decay_steps=decay_steps,
        decay_rate=0.96,
        staircase=True)
    opt = optimizers.Adam(learning_rate=lr_schedule)
    return opt


def eql_model_v3(input_size, opt, ln_blocks=(3,), lin_blocks=(1,),
                 compile=True, l1_reg=0, l2_reg=0):
    print(lin_blocks, ln_blocks)
    print(len(lin_blocks))
    print(len(ln_blocks))
    if len(ln_blocks) != len(lin_blocks):
        raise ValueError("length of ln_blocks should be equal to length of lin_blocks")

    inputs_x = [layers.Input(shape=(1,)) for i in range(input_size)]
    cur_input = inputs_x

    for depth_idx in range(len(ln_blocks)):
        cur_ln_blocks = ln_blocks[depth_idx]
        cur_output_units = lin_blocks[depth_idx]
        print('depth', depth_idx)
        cur_ln_dense_units = [
            eql_ln_block(cur_input,
                         layer_num=str(depth_idx) + '_' + str(i)) for i in range(cur_ln_blocks)]
        if cur_ln_blocks == 1:
            cur_ln_dense_concat = cur_ln_dense_units[0]
        else:
            cur_ln_dense_concat = layers.Concatenate()(cur_ln_dense_units)

        cur_output_dense = [layers.Dense(1, activation='linear',
                                         kernel_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
                                         bias_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
                                         kernel_constraint=PosConstraint(),
                                         kernel_initializer=initializers.RandomUniform(minval=0.5, maxval=1),
                                         use_bias=False,
                                         name='output_dense_{}_{}'.format(depth_idx, i))(cur_ln_dense_concat)
                            for i in range(cur_output_units)]
        cur_input = cur_input + cur_output_dense

    # if cur_ln_blocks == 1:
    #     ln_dense_concat = cur_ln_dense_units[0]
    # else:
    #     ln_dense_concat = layers.Concatenate()(cur_ln_dense_units)
    #
    # output_dense = layers.Dense(1, activation='linear',
    #                             kernel_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
    #                             bias_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
    #                             name='output_dense')(ln_dense_concat)
    model = Model(inputs=inputs_x, outputs=cur_output_dense[0], name='eql_model')
    if compile:
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def add_ln_block(input_size, trained_model, cur_ln_block_count, decay_steps=1000):
    model = eql_model_v2(input_size, ln_block_count=cur_ln_block_count + 1, decay_steps=decay_steps,
                         compile=False)
    for i in range(cur_ln_block_count):
        ln_block = 'ln_dense_{}'.format(i)
        model.get_layer(ln_block).set_weights(trained_model.get_layer(ln_block).get_weights())
    trained_output_kernel = trained_model.get_layer('output_dense').get_weights()[0]
    trained_output_bias = trained_model.get_layer('output_dense').get_weights()[1]
    init_output_kernel = model.get_layer('output_dense').get_weights()[0]
    init_output_bias = model.get_layer('output_dense').get_weights()[1]
    print(init_output_kernel)
    print(init_output_bias)
    model.get_layer('output_dense').set_weights([np.append(trained_output_kernel, [init_output_kernel[-1]], axis=0),
                                                 init_output_bias])
    return model


def add_width_ln_block(input_size, trained_model, cur_ln_blocks, cur_line_blocks, opt):
    ln_blocks = cur_ln_blocks.copy()
    ln_blocks[-1] += 1
    model = eql_model_v3(input_size, opt, ln_blocks=ln_blocks, lin_blocks=cur_line_blocks,
                         compile=False)
    for depth_idx in range(len(cur_ln_blocks)):
        cur_ln_block_count = cur_ln_blocks[depth_idx]
        cur_output_units = cur_line_blocks[depth_idx]
        for i in range(cur_ln_block_count):
            ln_block = 'ln_dense_{}_{}'.format(depth_idx, i)
            model.get_layer(ln_block).set_weights(trained_model.get_layer(ln_block).get_weights())
        if depth_idx != len(cur_ln_blocks) - 1:
            for i in range(cur_output_units - 1):
                output_block = 'output_dense_{}_{}'.format(depth_idx, i)
                model.get_layer(output_block).set_weights(trained_model.get_layer(output_block).get_weights())
        else:
            final_output_layer = 'output_dense_{}_{}'.format(depth_idx, 0)
            trained_output_kernel = trained_model.get_layer(final_output_layer).get_weights()[0]
            init_output_kernel = model.get_layer(final_output_layer).get_weights()[0]
            # init_output_bias = model.get_layer(final_output_layer).get_weights()[1]
            model.get_layer(final_output_layer).set_weights([np.append(trained_output_kernel,
                                                                       [init_output_kernel[-1]], axis=0)])
            # ,
            # init_output_bias])

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def add_depth_ln_layer(input_size, trained_model, cur_ln_blocks, new_lin_blocks, opt):
    ln_blocks = cur_ln_blocks.copy()
    ln_blocks.append(1)
    lin_blocks = new_lin_blocks.copy()
    model = eql_model_v3(input_size, opt, ln_blocks=ln_blocks, lin_blocks=lin_blocks,
                         compile=False)
    for depth_idx in range(len(cur_ln_blocks)):
        cur_ln_block_count = cur_ln_blocks[depth_idx]
        cur_output_units = lin_blocks[depth_idx]
        for i in range(cur_ln_block_count):
            ln_block = 'ln_dense_{}_{}'.format(depth_idx, i)
            model.get_layer(ln_block).set_weights(trained_model.get_layer(ln_block).get_weights())
        for i in range(cur_output_units - 1):
            output_block = 'output_dense_{}_{}'.format(depth_idx, i)
            model.get_layer(output_block).set_weights(trained_model.get_layer(output_block).get_weights())

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def copy_tf_model(cur_model):
    tf.keras.utils.get_custom_objects().update({'log_activation': log_activation,
                                                'L1L2_m': L1L2_m,
                                                'PosConstraint':PosConstraint})
    model_new = tf.keras.models.clone_model(cur_model)
    model_new.set_weights(cur_model.get_weights())
    return model_new


def eql_model_v3_multioutput(input_size, opt, ln_blocks=(3,), lin_blocks=(1,), output_ln_blocks=3, num_outputs=2, compile=True, l1_reg=0, l2_reg=0):
    """
    Build a multi-output GINN model with shared PTA layers, then output-specific PTA blocks and linear heads.
    - input_size: number of input variables
    - opt: optimizer
    - ln_blocks: tuple, number of PTA blocks per shared layer
    - lin_blocks: tuple, number of output units per shared layer (should match ln_blocks)
    - output_ln_blocks: number of PTA blocks per output head
    - num_outputs: number of outputs (tasks)
    - compile: whether to compile the model
    - l1_reg, l2_reg: regularization parameters
    """
    # Shared input (list of Input layers, as in original code)
    inputs_x = [layers.Input(shape=(1,)) for i in range(input_size)]
    cur_input = inputs_x

    # Shared PTA layers (as in eql_model_v3)
    for depth_idx in range(len(ln_blocks)):
        cur_ln_blocks = ln_blocks[depth_idx]
        cur_output_units = lin_blocks[depth_idx]
        cur_ln_dense_units = [
            eql_ln_block(cur_input, layer_num=str(depth_idx) + '_' + str(i)) for i in range(cur_ln_blocks)
        ]
        if cur_ln_blocks == 1:
            cur_ln_dense_concat = cur_ln_dense_units[0]
        else:
            cur_ln_dense_concat = layers.Concatenate()(cur_ln_dense_units)
        
        # Create output_dense layers with DIFFERENT weight initializations for each unit
        cur_output_dense = []
        for i in range(cur_output_units):
            # Use different random seeds for each output unit to ensure different initial weights
            different_init = initializers.RandomUniform(
                minval=0.5 + i*0.1,  # Different min values for each unit
                maxval=1.0 + i*0.1,  # Different max values for each unit
                seed=i+42  # Different seed for each unit
            )
            
            output_layer = layers.Dense(1, activation='linear',
                                       kernel_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
                                       bias_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
                                       kernel_constraint=PosConstraint(),
                                       kernel_initializer=different_init,
                                       use_bias=False,
                                       name='output_dense_{}_{}'.format(depth_idx, i))(cur_ln_dense_concat)
            cur_output_dense.append(output_layer)
        
        cur_input = cur_input + cur_output_dense

    # At this point, cur_input is a list: original inputs + all output_dense from last shared layer
    # We want to concatenate only the outputs of the last shared layer for the next step
    shared_features = layers.Concatenate()(cur_input[input_size:])

    # For each output, add output-specific PTA blocks and a linear head
    outputs = []
    for out_idx in range(num_outputs):
        # Output-specific PTA blocks - Different initialization but non-trainable for stability
        out_ln_dense_units = [
            layers.Dense(1, use_bias=False,
                         kernel_initializer=initializers.RandomUniform(minval=0.1, maxval=0.9, seed=out_idx*100+i),
                        #  kernel_initializer=initializers.Identity(gain=1.0), #Original initialization for each output unit (no seed)
                         trainable=False,  # Keep non-trainable for stability
                         activation=log_activation,
                         name=f'out{out_idx}_ln_{i}')(shared_features)
            for i in range(output_ln_blocks)
        ]
            
        if output_ln_blocks == 1:
            out_ln_concat = out_ln_dense_units[0]
        else:
            out_ln_concat = layers.Concatenate()(out_ln_dense_units)
            
        # Exponential activation to get polynomial terms
        out_ln_dense = layers.Dense(1,
                                   kernel_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg, int_reg=0.0),
                                   use_bias=False, activation=activations.exponential,
                                   name=f'out{out_idx}_ln_dense')(out_ln_concat)
        # Linear head with standard regularization
        out_linear = layers.Dense(1, activation='linear', 
                                 kernel_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
                                 bias_regularizer=L1L2_m(l1=l1_reg, l2=l2_reg),
                                 name=f'output_{out_idx}')(out_ln_dense)
        outputs.append(out_linear)

    model = Model(inputs=inputs_x, outputs=outputs, name='eql_model_multioutput')
    if compile:
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def get_multioutput_sympy_expr(model, input_size, output_ln_blocks, round_digits=3):
    """
    Extract and print symbolic equations for each output of a multi-output GINN model.
    """
    get_multioutput_sympy_expr_v2(model, input_size, output_ln_blocks, round_digits)


if __name__ == "__main__":
    # model = eql_model_v2(2, ln_block_count=1)
    # model_2 = add_ln_block(2, model, 1)
    # # print(model_2.summary())
    # # print(model_2.get_weights())
    # for layer in model_2.layers:
    #     print(layer.name, layer.get_weights())
    # print()
    # print(model.optimizer.get_gradients(model.total_loss, model.trainable_weights))
    print('v3')
    opt = eql_opt()
    model_3 = eql_model_v3(2, ln_blocks=(3, 1), lin_blocks=(1, 1), opt=opt)
    for layer in model_3.layers:
        print(layer.name, layer.get_weights())
    print()
    print(model_3.summary())

    # model_4 = add_width_ln_block(3, model_3, cur_ln_blocks=[2], opt=opt)
    # for layer in model_4.layers:
    #     print(layer.name, layer.get_weights())
    # print()
    # print(model_4.summary())
    #
    # model_5 = add_depth_ln_layer(2, model_4, cur_ln_blocks=[3], opt=opt)
    # for layer in model_5.layers:
    #     print(layer.name, layer.get_weights())
    # print()
    # print(model_5.summary())
    # get_sympy_expr_v3(model_3, 2, [2], 2)
