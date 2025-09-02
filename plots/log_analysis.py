#!/usr/bin/env python3
"""
Professor's Request: Log-Space Expression Analysis for GINN Equations

This script implements the exact analysis your professor requested:
- Extract refitted equations from ginn_grad_ENB.json
- Convert to log-space expressions
- Plot sensitivity of each variable using the provided function

Based on the code section provided by your professor.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def plot_log_expression(log_expr, var_names, base_vals=None, x_range=(0.1, 10), num_points=100, title='Log-Space Expression Sensitivity'):
    """
    Plot sensitivity of a symbolic log-space expression to each variable.
    
    Parameters:
    - log_expr: sympy expression (e.g., 2.1*log(x1) - 1.2*log(x2) + ...)
    - var_names: list of variable names as strings (e.g., ['x1', 'x2', 'x3'])
    - base_vals: list of base values for all variables (defaults to 1)
    - x_range: tuple (min, max) for x-axis
    - num_points: number of points in plot
    - title: overall plot title
    """
    # Define variables with positive constraint
    variables = sp.symbols(' '.join(var_names), positive=True)
    assert len(variables) == len(var_names)

    # Lambdify the function
    f = sp.lambdify(variables, log_expr, modules='numpy')

    # Default base values
    if base_vals is None:
        base_vals = [1.0] * len(variables)

    # X-axis values
    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    # Prepare subplots
    fig, axs = plt.subplots(2, int(np.ceil(len(variables)/2)), figsize=(18, 8))
    axs = axs.ravel()

    # Loop over each variable
    for i, var in enumerate(variables):
        inputs = [np.full_like(x_vals, val) for val in base_vals]
        inputs[i] = x_vals  # vary i-th variable
        y_vals = f(*inputs)
        axs[i].plot(x_vals, y_vals, label=f'f vs {var_names[i]}', color='b')
        axs[i].set_xlabel(var_names[i])
        axs[i].set_ylabel('f')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle(title, fontsize=18, y=1.02)
    # Create a safe filename
    safe_title = title.replace(" ", "_").replace(":", "").replace("=", "").replace("*", "").replace("+", "").replace("-", "").replace("(", "").replace(")", "")
    safe_title = safe_title[:50]  # Limit length
    plt.savefig(f'{safe_title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def term_to_log_form(term):
    """Convert a term to log form if possible"""
    if term.is_number:
        return sp.log(term) if term > 0 else term
    elif term.is_Mul:
        coeff, var = term.as_coeff_Mul()
        if var.is_Symbol:
            return coeff * sp.log(var)
        else:
            return term
    else:
        return term

def load_equations():
    """Load the refitted equations from the JSON file"""
    with open('../outputs/ginn_multi_ENB_V1.json', 'r') as f:
        results = json.load(f)
    
    # Get the first fold results
    fold_data = results[0]
    
    equations = []
    target_names = []
    
    for target_data in fold_data['per_target']:
        target_name = target_data['target']
        refit_expr = target_data['expr_refit']
        
        equations.append(refit_expr)
        target_names.append(target_name)
    
    return equations, target_names

def parse_equation_to_sympy(expr_str):
    """Convert equation string to SymPy expression"""
    # Replace X_ with x_ for SymPy compatibility
    expr_str = expr_str.replace('X_', 'x')
    return parse_expr(expr_str)

def main():
    """Main function implementing professor's requested analysis"""
    print("Professor's Log-Space Expression Analysis")
    print("="*50)
    
    # Load equations
    equations, target_names = load_equations()
    
    print(f"Loaded {len(equations)} refitted equations:")
    for i, (eq, target) in enumerate(zip(equations, target_names)):
        print(f"  {target}: {eq}")
    
    # Define variables (8 features for ENB dataset)
    var_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    
    # Process each equation
    for idx, (eq_str, target_name) in enumerate(zip(equations, target_names)):
        print(f"\nProcessing {target_name.upper()}...")
        
        # Parse equation
        expr = parse_equation_to_sympy(eq_str)
        print(f"Original equation: {expr}")
        
        # Break expression into additive terms
        if expr.is_Add:
            terms = expr.args
        else:
            terms = [expr]
        
        print(f"Terms: {terms}")
        
        # Convert terms to log form
        log_terms = []
        for term in terms:
            if term.is_number:
                # Constant term
                if term > 0:
                    log_terms.append(sp.log(term))
                    print(f"  Constant {term} -> log({term})")
                else:
                    log_terms.append(term)  # Keep negative constants as is
                    print(f"  Constant {term} -> {term} (negative, kept as is)")
            elif term.is_Mul:
                coeff, var = term.as_coeff_Mul()
                if var.is_Symbol:
                    # coefficient * variable -> coefficient * log(variable)
                    log_term = coeff * sp.log(var)
                    log_terms.append(log_term)
                    print(f"  {term} -> {log_term}")
                else:
                    log_terms.append(term)
                    print(f"  {term} -> {term} (complex term, kept as is)")
            else:
                log_terms.append(term)
                print(f"  {term} -> {term} (other term, kept as is)")
        
        # Combine log-terms into symbolic sum
        log_expr_sum = sp.Add(*log_terms)
        print(f"Log-space expression: {log_expr_sum}")
        
        # Plot log expression using professor's function
        plot_log_expression(log_expr_sum, var_names, 
                           title=f'{target_name.upper()}: Log-Space Expression for f(x) = {log_expr_sum}')
    
    print("\nAnalysis complete!")
    print("Generated log-space sensitivity plots for each target.")

if __name__ == "__main__":
    main()
