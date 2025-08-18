from sympy import symbols, powsimp, ln, log, preorder_traversal, Float, sqrt, count_ops, simplify


def eq_complexity(expr):
    c = 0
    for arg in preorder_traversal(expr):
        c += 1
    return c

def get_sympy_expr(model_weights, x_dim):
    sym = []
    for i in range(x_dim):
        sym.append(symbols('X_'+str(i+1), positive=True))
    sym_expr = (powsimp(sym[0]**model_weights[2][0][0])*powsimp(sym[1]**model_weights[2][1][0]))*model_weights[5][0][0] + \
               (sym[0]*model_weights[3][0][0] + sym[1]*model_weights[3][1][0])*model_weights[5][1][0]

    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            sym_expr = sym_expr.subs(a, round(a, 2))
    print(sym_expr)


def round_sympy_expr(sym_expr, round_digits=2):
    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            if int(round(a, round_digits)) == round(a, round_digits):
                sym_expr = sym_expr.subs(a, int(round(a, round_digits)))
            else:
                sym_expr = sym_expr.subs(a, round(a, round_digits))
    return sym_expr


def is_sqrt_term(expr):
    if count_ops(expr) != 1 or len(expr.free_symbols) != 1:
        return None
    ops = list(count_ops(expr, visual=True).free_symbols)
    op = ops[0].name
    op_sym = list(expr.free_symbols)[0]
    if op == 'POW' and op_sym**0.5 == expr:
        return sqrt(op_sym)
    elif op == 'POW' and op_sym**(-0.5) == expr:
        return 1/sqrt(op_sym)
    else:
        return None


def get_sympy_expr_v2(model, x_dim, ln_layer_count=2, round_digits=3):
    sym = []
    for i in range(x_dim):
        sym.append(symbols('X_'+str(i), positive=True))

    ln_dense_weights = [model.get_layer('ln_dense_{}'.format(i)).get_weights() for i in range(ln_layer_count)]
    output_dense_weights = model.get_layer('output_dense').get_weights()
    print(ln_dense_weights)
    print(output_dense_weights)
    sym_expr = 0
    for i in range(ln_layer_count):
        ln_block_expr = output_dense_weights[0][i][0]
        for j in range(x_dim):
            ln_block_expr *= sym[j]**ln_dense_weights[i][0][j][0]
        sym_expr += ln_block_expr
    sym_expr = round_sympy_expr(sym_expr, round_digits=round_digits)

    print(sym_expr)
    
    return sym_expr


def get_sympy_expr_v3(model, x_dim, ln_blocks, line_blocks, round_digits):
    sym = []
    for i in range(x_dim):
        sym.append(symbols('X_'+str(i+1)))

    ln_dense_weights = []
    output_dense_weights = []
    for depth_idx in range(len(ln_blocks)):
        cur_ln_block_count = ln_blocks[depth_idx]
        output_dense_count = line_blocks[depth_idx]
        ln_dense_weights.append(
            [model.get_layer('ln_dense_{}_{}'.format(depth_idx, i)).get_weights()
             for i in range(cur_ln_block_count)]
        )
        output_dense_weights.append(
            [model.get_layer('output_dense_{}_{}'.format(depth_idx, i)).get_weights()
             for i in range(output_dense_count)]
        )
    layer_inputs = sym
    for depth_idx in range(len(ln_blocks)):
        ln_block_count = ln_blocks[depth_idx]
        output_dense_count = line_blocks[depth_idx]
        ln_layer_outputs = [1 for _ in range(ln_block_count)]
        line_layer_outputs = [0 for _ in range(output_dense_count)]
        for i in range(ln_block_count):
            for j, layer_input in enumerate(layer_inputs):
                print('layer input', layer_input)
                ln_layer_outputs[i] *= layer_input ** ln_dense_weights[depth_idx][i][0][j][0]
                print('ln layer output', ln_layer_outputs[i])
        for o_num in range(output_dense_count):
            for i, ln_layer_out in enumerate(ln_layer_outputs):
                print('ln layer', ln_layer_out)
                line_layer_outputs[o_num] += output_dense_weights[depth_idx][o_num][0][i][0]*ln_layer_out
        print('line layer', line_layer_outputs)
        layer_inputs = layer_inputs.copy() + line_layer_outputs.copy()
        print('next layer in', layer_inputs)

    sym_expr = line_layer_outputs[0]
    # for i, final_layer_input in enumerate(layer_outputs):
    #     sym_expr += output_dense_weights[0][i][0]*final_layer_input
    # sym_expr += output_dense_weights[1][0]

    sym_expr = round_sympy_expr(sym_expr, round_digits=round_digits)
    # sym_expr = simplify(sym_expr)
    print(sym_expr)
    return sym_expr


def get_multioutput_sympy_expr_v2(model, input_size, output_ln_blocks, round_digits=3):
    """
    Extract symbolic equations for multi-output GINN models.
    Returns:
    - List of SymPy expressions for each output
    - List of input symbols
    """
    sym = [symbols(f'X_{i+1}', positive=True) for i in range(input_size)]

    # --- 1. Extract shared features (outputs of last shared PTA layer) ---
    # Find how many shared features there are (from model architecture)
    # We'll look for output_dense_{last_layer_idx}_* layers
    shared_feature_names = []
    shared_feature_exprs = []
    # Find last shared layer index
    last_shared_idx = 0
    while True:
        name = f'output_dense_{last_shared_idx}_0'
        try:
            model.get_layer(name)
            last_shared_idx += 1
        except Exception:
            break
    last_shared_idx -= 1
    # How many shared features?
    num_shared_features = 0
    while True:
        name = f'output_dense_{last_shared_idx}_{num_shared_features}'
        try:
            model.get_layer(name)
            num_shared_features += 1
        except Exception:
            break
    # For each shared feature, build its equation in terms of X1...Xn
    for feat_idx in range(num_shared_features):
        # Each shared feature is a linear combination of the outputs of the last shared PTA blocks
        # The last shared PTA blocks are ln_dense_{last_shared_idx}_*
        ln_block_exprs = []
        ln_block_count = 0
        while True:
            try:
                weights = model.get_layer(f'ln_dense_{last_shared_idx}_{ln_block_count}').get_weights()
                ln_block_count += 1
            except Exception:
                break
        for i in range(ln_block_count):
            # Each ln_dense is a Dense(1) with weights for each input
            ln_weights = model.get_layer(f'ln_dense_{last_shared_idx}_{i}').get_weights()[0].flatten()
            block_expr = 1
            for j in range(input_size):
                block_expr *= sym[j] ** ln_weights[j]
            ln_block_exprs.append(block_expr)
        # Now combine with output_dense weights
        out_weights = model.get_layer(f'output_dense_{last_shared_idx}_{feat_idx}').get_weights()[0].flatten()
        feat_expr = 0
        for i, block_expr in enumerate(ln_block_exprs):
            feat_expr += out_weights[i] * block_expr
        feat_expr = round_sympy_expr(feat_expr, round_digits=round_digits)
        shared_feature_names.append(f'F_{feat_idx+1}')
        shared_feature_exprs.append(feat_expr)

    # --- 2. Build both output equations in terms of F_i ---
    shared_syms = [symbols(f'F_{i+1}', positive=True) for i in range(num_shared_features)]
    output_expressions = []  # Store expressions to return
    
    for out_idx in range(2):  # Assuming 2 outputs
        out_ln_weights = []
        for i in range(output_ln_blocks):
            layer_name = f'out{out_idx}_ln_{i}'
            weights = model.get_layer(layer_name).get_weights()
            out_ln_weights.append(weights)
        out_dense_weights = model.get_layer(f'out{out_idx}_ln_dense').get_weights()
        output_weights = model.get_layer(f'output_{out_idx}').get_weights()
        ln_block_exprs = []
        for i in range(output_ln_blocks):
            block_expr = 1
            for j in range(num_shared_features):
                block_expr *= shared_syms[j] ** out_ln_weights[i][0][j][0]
            ln_block_exprs.append(block_expr)
        combined_expr = 0
        for i, block_expr in enumerate(ln_block_exprs):
            combined_expr += out_dense_weights[0][i][0] * block_expr
        final_expr = output_weights[0][0][0] * combined_expr
        if len(output_weights) > 1:
            final_expr += output_weights[1][0]
        final_expr = round_sympy_expr(final_expr, round_digits=round_digits)
        
        # --- 3. Substitute F_i in terms of X_j ---
        print(f"\nRecovered equation for output {out_idx} (in terms of X_j):")
        expr_sub = final_expr
        for F, F_expr in zip(shared_syms, shared_feature_exprs):
            expr_sub = expr_sub.subs(F, F_expr)
        expr_sub = simplify(expr_sub)
        print(expr_sub)
        
        # Store the final expression to return
        output_expressions.append(expr_sub)
    
    # Return the equations and symbols (ChatGPT's fix)
    return output_expressions, sym


def get_multioutput_log_symbolic_expr(model, input_size, output_ln_blocks, round_digits=3):
    """
    Extract LOGARITHMIC symbolic equations for multi-output GINN models.
    This transforms multiplicative relationships into additive ones in log-space.
    
    Returns equations in the form: y = log(c1) + a1*log(X1) + a2*log(X2) + ... + bias
    """
    from sympy import log, symbols, simplify
    
    # Create symbols for input features
    sym = [symbols(f'X_{i+1}', positive=True) for i in range(input_size)]
    
    # Create log symbols: log(X1), log(X2), etc.
    log_sym = [log(sym[i]) for i in range(input_size)]
    
    print(f"\n=== LOGARITHMIC EQUATION EXTRACTION ===")
    print(f"Transforming multiplicative relationships to additive log-space relationships")
    
    # Extract shared features in log-space
    shared_feature_names = []
    shared_feature_exprs = []
    
    # Find last shared layer index (same logic as before)
    last_shared_idx = 0
    while True:
        name = f'output_dense_{last_shared_idx}_0'
        try:
            model.get_layer(name)
            last_shared_idx += 1
        except Exception:
            break
    last_shared_idx -= 1
    
    # Count shared features
    num_shared_features = 0
    while True:
        name = f'output_dense_{last_shared_idx}_{num_shared_features}'
        try:
            model.get_layer(name)
            num_shared_features += 1
        except Exception:
            break
    
    # Build shared features in log-space
    for feat_idx in range(num_shared_features):
        ln_block_exprs = []
        ln_block_count = 0
        while True:
            try:
                weights = model.get_layer(f'ln_dense_{last_shared_idx}_{ln_block_count}').get_weights()
                ln_block_count += 1
            except Exception:
                break
        
        # Build log-space expression for this feature
        feat_expr = 0
        for i in range(ln_block_count):
            ln_weights = model.get_layer(f'ln_dense_{last_shared_idx}_{i}').get_weights()[0].flatten()
            # In log-space: log(x^a) = a*log(x)
            block_expr = 0
            for j in range(input_size):
                if ln_weights[j] != 0:  # Only add non-zero terms
                    block_expr += ln_weights[j] * log_sym[j]
            ln_block_exprs.append(block_expr)
        
        # Combine with output_dense weights
        out_weights = model.get_layer(f'output_dense_{last_shared_idx}_{feat_idx}').get_weights()[0].flatten()
        feat_expr = 0
        for i, block_expr in enumerate(ln_block_exprs):
            if out_weights[i] != 0:  # Only add non-zero terms
                feat_expr += out_weights[i] * block_expr
        
        feat_expr = round_sympy_expr(feat_expr, round_digits=round_digits)
        shared_feature_names.append(f'F_{feat_idx+1}')
        shared_feature_exprs.append(feat_expr)
    
    # Extract output equations in log-space
    shared_syms = [symbols(f'F_{i+1}', positive=True) for i in range(num_shared_features)]
    log_shared_syms = [log(sym) for sym in shared_syms]
    
    output_0_log_expr = None
    output_1_log_expr = None
    
    for out_idx in range(2):  # Assuming 2 outputs
        print(f"\n=== LOGARITHMIC EQUATION FOR OUTPUT {out_idx} ===")
        
        # Get output layer weights
        out_ln_weights = []
        for i in range(output_ln_blocks):
            layer_name = f'out{out_idx}_ln_{i}'
            weights = model.get_layer(layer_name).get_weights()
            out_ln_weights.append(weights)
        
        out_dense_weights = model.get_layer(f'out{out_idx}_ln_dense').get_weights()
        output_weights = model.get_layer(f'output_{out_idx}').get_weights()
        
        # Build log-space expression
        ln_block_exprs = []
        for i in range(output_ln_blocks):
            block_expr = 0
            for j in range(num_shared_features):
                if out_ln_weights[i][0][j][0] != 0:  # Only add non-zero terms
                    block_expr += out_ln_weights[i][0][j][0] * log_shared_syms[j]
            ln_block_exprs.append(block_expr)
        
        # Combine with dense weights
        combined_expr = 0
        for i, block_expr in enumerate(ln_block_exprs):
            if out_dense_weights[0][i][0] != 0:  # Only add non-zero terms
                combined_expr += out_dense_weights[0][i][0] * block_expr
        
        # Add bias
        final_expr = combined_expr
        if len(output_weights) > 1:
            final_expr += output_weights[1][0]
        
        final_expr = round_sympy_expr(final_expr, round_digits=round_digits)
        
        # Substitute shared features in terms of input features
        print(f"Log-space equation (in terms of F_i):")
        print(f"log(y_{out_idx}) = {final_expr}")
        
        # Substitute F_i in terms of X_j
        expr_sub = final_expr
        for F, F_expr in zip(shared_syms, shared_feature_exprs):
            expr_sub = expr_sub.subs(F, F_expr)
        
        expr_sub = simplify(expr_sub)
        print(f"\nLog-space equation (in terms of X_j):")
        print(f"log(y_{out_idx}) = {expr_sub}")
        
        # Convert back to original form for comparison
        print(f"\nOriginal form (y_{out_idx} = exp(...)):")
        print(f"y_{out_idx} = exp({expr_sub})")
        
        # Store expressions
        if out_idx == 0:
            output_0_log_expr = expr_sub
        else:
            output_1_log_expr = expr_sub
    
    print(f"\n=== LOGARITHMIC EXTRACTION COMPLETE ===")
    print(f"Both power-based and log-based equations extracted successfully!")
    
    return output_0_log_expr, output_1_log_expr


def evaluate_log_equations(X, log_expr_0, log_expr_1, round_digits=3):
    """
    Evaluate logarithmic equations on data.
    
    Args:
        X: Input features (numpy array)
        log_expr_0: Logarithmic expression for output 0
        log_expr_1: Logarithmic expression for output 1
        round_digits: Number of decimal places to round to
    
    Returns:
        y_pred_0, y_pred_1: Predictions from logarithmic equations
    """
    import numpy as np
    from sympy import lambdify, symbols
    
    print(f"\n=== EVALUATING LOGARITHMIC EQUATIONS ===")
    
    # Create symbols for input features
    sym = [symbols(f'X_{i+1}', positive=True) for i in range(X.shape[1])]
    
    # Convert symbolic expressions to callable functions
    try:
        f0 = lambdify(sym, log_expr_0, modules=['numpy'])
        f1 = lambdify(sym, log_expr_1, modules=['numpy'])
        print("Successfully created callable functions from symbolic expressions")
    except Exception as e:
        print(f"Error creating callable functions: {e}")
        return None, None
    
    # Evaluate on data
    y_pred_0 = np.zeros(X.shape[0])
    y_pred_1 = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        try:
            # Ensure all inputs are positive (log requires positive values)
            x_input = np.maximum(X[i], 1e-8)
            
            # Evaluate log expressions
            log_pred_0 = f0(*x_input)
            log_pred_1 = f1(*x_input)
            
            # Convert back from log-space to original space
            y_pred_0[i] = np.exp(log_pred_0)
            y_pred_1[i] = np.exp(log_pred_1)
            
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            y_pred_0[i] = np.nan
            y_pred_1[i] = np.nan
    
    # Handle any NaN or inf values
    y_pred_0 = np.nan_to_num(y_pred_0, nan=0.0, posinf=1e6, neginf=-1e6)
    y_pred_1 = np.nan_to_num(y_pred_1, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"Logarithmic equation evaluation complete!")
    print(f"Output 0 range: {y_pred_0.min():.6f} to {y_pred_0.max():.6f}")
    print(f"Output 1 range: {y_pred_1.min():.6f} to {y_pred_1.max():.6f}")
    
    return y_pred_0, y_pred_1

