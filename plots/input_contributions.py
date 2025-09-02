#!/usr/bin/env python3
"""
Input Contribution Analysis for GINN Extracted Equations

This script analyzes how each input feature contributes to the output based on 
the refitted equations extracted from the GINN model. It provides:

1. Feature importance analysis (coefficient magnitudes)
2. Sensitivity analysis (how output changes with input variations)
3. Contribution plots for each target
4. Log-space expression analysis (as requested by professor)

Based on the refitted equations from ginn_grad_ENB.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_equations():
    """Load the refitted equations from the JSON file"""
    with open('../outputs/ginn_grad_ENB.json', 'r') as f:
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
    
    return equations, target_names, fold_data

def parse_equation_to_sympy(expr_str):
    """Convert equation string to SymPy expression"""
    # Replace X_ with x_ for SymPy compatibility
    expr_str = expr_str.replace('X_', 'x')
    return parse_expr(expr_str)

def extract_coefficients(expr, num_features=8):
    """Extract coefficients for each feature from the equation"""
    coefficients = np.zeros(num_features)
    intercept = 0.0
    
    # Get the terms of the expression
    if expr.is_Add:
        terms = expr.args
    else:
        terms = [expr]
    
    for term in terms:
        if term.is_number:
            # This is the intercept
            intercept = float(term)
        else:
            # This is a coefficient * variable term
            coeff, var = term.as_coeff_Mul()
            if var.is_Symbol:
                # Extract variable index (x1, x2, etc.)
                var_name = str(var)
                if var_name.startswith('x'):
                    try:
                        var_idx = int(var_name[1:]) - 1  # Convert x1 -> 0, x2 -> 1, etc.
                        if 0 <= var_idx < num_features:
                            coefficients[var_idx] = float(coeff)
                    except ValueError:
                        pass
    
    return coefficients, intercept

def analyze_feature_importance(equations, target_names):
    """Analyze feature importance based on coefficient magnitudes"""
    num_features = 8
    feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Create a DataFrame to store coefficients
    coeff_data = []
    
    for i, (eq_str, target_name) in enumerate(zip(equations, target_names)):
        print(f"\n{target_name.upper()}:")
        print(f"Equation: {eq_str}")
        
        # Parse equation
        expr = parse_equation_to_sympy(eq_str)
        coefficients, intercept = extract_coefficients(expr, num_features)
        
        print(f"Intercept: {intercept:.6f}")
        print("Coefficients:")
        
        for j, (coeff, feature_name) in enumerate(zip(coefficients, feature_names)):
            print(f"  {feature_name}: {coeff:.6f}")
            coeff_data.append({
                'Target': target_name,
                'Feature': feature_name,
                'Feature_Index': j,
                'Coefficient': coeff,
                'Abs_Coefficient': abs(coeff)
            })
    
    # Create DataFrame
    df_coeff = pd.DataFrame(coeff_data)
    
    # Calculate relative importance (normalized by sum of absolute coefficients)
    for target in target_names:
        target_data = df_coeff[df_coeff['Target'] == target]
        total_abs_coeff = target_data['Abs_Coefficient'].sum()
        if total_abs_coeff > 0:
            df_coeff.loc[df_coeff['Target'] == target, 'Relative_Importance'] = \
                df_coeff.loc[df_coeff['Target'] == target, 'Abs_Coefficient'] / total_abs_coeff
    
    return df_coeff

def plot_feature_importance(df_coeff, target_names):
    """Plot feature importance for each target"""
    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 6))
    if len(target_names) == 1:
        axes = [axes]
    
    for i, target in enumerate(target_names):
        target_data = df_coeff[df_coeff['Target'] == target]
        
        # Sort by absolute coefficient value
        target_data = target_data.sort_values('Abs_Coefficient', ascending=True)
        
        # Create horizontal bar plot
        bars = axes[i].barh(target_data['Feature'], target_data['Coefficient'], 
                           color=['red' if x < 0 else 'blue' for x in target_data['Coefficient']])
        
        axes[i].set_xlabel('Coefficient Value')
        axes[i].set_title(f'{target.upper()}\nFeature Importance')
        axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, coeff) in enumerate(zip(bars, target_data['Coefficient'])):
            axes[i].text(coeff + (0.1 if coeff >= 0 else -0.1), j, f'{coeff:.3f}', 
                        va='center', ha='left' if coeff >= 0 else 'right')
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_relative_importance(df_coeff, target_names):
    """Plot relative importance (normalized) for each target"""
    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 6))
    if len(target_names) == 1:
        axes = [axes]
    
    for i, target in enumerate(target_names):
        target_data = df_coeff[df_coeff['Target'] == target]
        
        # Sort by relative importance
        target_data = target_data.sort_values('Relative_Importance', ascending=True)
        
        # Create horizontal bar plot
        bars = axes[i].barh(target_data['Feature'], target_data['Relative_Importance'], 
                           color='green', alpha=0.7)
        
        axes[i].set_xlabel('Relative Importance (0-1)')
        axes[i].set_title(f'{target.upper()}\nRelative Feature Importance')
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, importance) in enumerate(zip(bars, target_data['Relative_Importance'])):
            axes[i].text(importance + 0.01, j, f'{importance:.3f}', 
                        va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig('relative_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def sensitivity_analysis(equations, target_names, num_points=100):
    """Perform sensitivity analysis - how output changes with input variations"""
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS")
    print("="*80)
    
    num_features = 8
    feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    # Define base values (use mean values from test data)
    base_values = np.array([0.75, 650.0, 320.0, 175.0, 5.0, 3.0, 0.25, 3.0])
    
    # Define variation range (±20% of base values)
    variation_range = 0.2
    
    for i, (eq_str, target_name) in enumerate(zip(equations, target_names)):
        print(f"\n{target_name.upper()} Sensitivity Analysis:")
        
        # Parse equation
        expr = parse_equation_to_sympy(eq_str)
        
        # Create symbols
        symbols = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8')
        
        # Create lambda function for evaluation
        f = sp.lambdify(symbols, expr, modules='numpy')
        
        # Calculate base output
        base_output = f(*base_values)
        print(f"Base output: {base_output:.4f}")
        
        # Sensitivity for each feature
        sensitivities = []
        for j in range(num_features):
            # Vary feature j by ±20%
            var_values = base_values.copy()
            var_values[j] = base_values[j] * (1 + variation_range)
            output_plus = f(*var_values)
            
            var_values[j] = base_values[j] * (1 - variation_range)
            output_minus = f(*var_values)
            
            # Calculate sensitivity (change in output per unit change in input)
            input_change = base_values[j] * variation_range * 2
            output_change = output_plus - output_minus
            sensitivity = output_change / input_change if input_change != 0 else 0
            
            sensitivities.append(sensitivity)
            
            print(f"  {feature_names[j]}: Sensitivity = {sensitivity:.6f}")
        
        # Plot sensitivity
        plt.figure(figsize=(12, 6))
        bars = plt.bar(feature_names, sensitivities, 
                      color=['red' if x < 0 else 'blue' for x in sensitivities])
        plt.xlabel('Features')
        plt.ylabel('Sensitivity (ΔOutput/ΔInput)')
        plt.title(f'{target_name.upper()} - Feature Sensitivity Analysis')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, sens in zip(bars, sensitivities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if sens >= 0 else -0.01), 
                    f'{sens:.3f}', ha='center', va='bottom' if sens >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f'sensitivity_analysis_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_log_expression(log_expr, var_names, base_vals=None, x_range=(0.1, 10), num_points=100, title='Log-Space Expression Sensitivity'):
    """
    Plot sensitivity of a symbolic log-space expression to each variable.
    This is the function provided by your professor.
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

def log_space_analysis(equations, target_names):
    """Perform log-space analysis as requested by professor"""
    print("\n" + "="*80)
    print("LOG-SPACE EXPRESSION ANALYSIS")
    print("="*80)
    
    var_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    base_vals = [1.0] * 8  # Base values for log-space analysis
    
    for idx, (eq_str, target_name) in enumerate(zip(equations, target_names)):
        print(f"\n{target_name.upper()} Log-Space Analysis:")
        
        # Parse equation
        expr = parse_equation_to_sympy(eq_str)
        print(f"Original equation: {expr}")
        
        # Break expression into additive terms
        if expr.is_Add:
            terms = expr.args
        else:
            terms = [expr]
        
        # Convert terms to log form
        log_terms = []
        for term in terms:
            if term.is_number:
                # Constant term
                if term > 0:
                    log_terms.append(sp.log(term))
                else:
                    log_terms.append(term)  # Keep negative constants as is
            elif term.is_Mul:
                coeff, var = term.as_coeff_Mul()
                if var.is_Symbol and coeff > 0:
                    # Positive coefficient * variable -> coeff * log(var)
                    log_terms.append(coeff * sp.log(var))
                elif var.is_Symbol and coeff < 0:
                    # Negative coefficient * variable -> coeff * log(var) (coeff is negative)
                    log_terms.append(coeff * sp.log(var))
                else:
                    log_terms.append(term)
            else:
                log_terms.append(term)
        
        # Combine log-terms into symbolic sum
        log_expr_sum = sp.Add(*log_terms)
        print(f"Log-space expression: {log_expr_sum}")
        
        # Plot log expression using professor's function
        plot_log_expression(log_expr_sum, var_names, base_vals, 
                           title=f'{target_name.upper()}: Log-Space Expression for f(x) = {log_expr_sum}')

def contribution_analysis(equations, target_names, test_data):
    """Analyze actual contributions on test data"""
    print("\n" + "="*80)
    print("ACTUAL CONTRIBUTION ANALYSIS ON TEST DATA")
    print("="*80)
    
    X_test = np.array(test_data['X_test'])
    Y_test = np.array(test_data['Y_test'])
    
    num_features = 8
    feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    for i, (eq_str, target_name) in enumerate(zip(equations, target_names)):
        print(f"\n{target_name.upper()} Contribution Analysis:")
        
        # Parse equation
        expr = parse_equation_to_sympy(eq_str)
        coefficients, intercept = extract_coefficients(expr, num_features)
        
        # Calculate contributions for each sample
        contributions = np.zeros((len(X_test), num_features))
        predictions = np.zeros(len(X_test))
        
        for j, x_sample in enumerate(X_test):
            sample_contributions = coefficients * x_sample
            contributions[j] = sample_contributions
            predictions[j] = np.sum(sample_contributions) + intercept
        
        # Calculate average contributions
        avg_contributions = np.mean(contributions, axis=0)
        std_contributions = np.std(contributions, axis=0)
        
        print("Average contributions per feature:")
        for k, (avg, std, feature_name) in enumerate(zip(avg_contributions, std_contributions, feature_names)):
            print(f"  {feature_name}: {avg:.4f} ± {std:.4f}")
        
        # Plot contribution distribution
        plt.figure(figsize=(12, 8))
        
        # Box plot of contributions
        plt.subplot(2, 1, 1)
        plt.boxplot([contributions[:, k] for k in range(num_features)], 
                   labels=feature_names)
        plt.title(f'{target_name.upper()} - Feature Contribution Distribution')
        plt.ylabel('Contribution Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Average contributions bar plot
        plt.subplot(2, 1, 2)
        bars = plt.bar(feature_names, avg_contributions, 
                      color=['red' if x < 0 else 'blue' for x in avg_contributions])
        plt.title(f'{target_name.upper()} - Average Feature Contributions')
        plt.ylabel('Average Contribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add error bars
        plt.errorbar(range(len(feature_names)), avg_contributions, 
                    yerr=std_contributions, fmt='none', color='black', capsize=5)
        
        # Add value labels on bars
        for bar, avg in zip(bars, avg_contributions):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if avg >= 0 else -0.01), 
                    f'{avg:.3f}', ha='center', va='bottom' if avg >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f'contribution_analysis_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate R² to verify equation accuracy
        true_values = Y_test[:, i]
        r2 = r2_score(true_values, predictions)
        print(f"Equation R² on test data: {r2:.4f}")

def main():
    """Main analysis function"""
    print("GINN Input Contribution Analysis")
    print("="*50)
    
    # Load equations and data
    equations, target_names, fold_data = load_equations()
    
    print(f"Loaded {len(equations)} equations for targets: {target_names}")
    
    # 1. Feature Importance Analysis
    df_coeff = analyze_feature_importance(equations, target_names)
    
    # 2. Plot feature importance
    plot_feature_importance(df_coeff, target_names)
    plot_relative_importance(df_coeff, target_names)
    
    # 3. Sensitivity Analysis
    sensitivity_analysis(equations, target_names)
    
    # 4. Log-space Analysis (as requested by professor)
    log_space_analysis(equations, target_names)
    
    # 5. Actual Contribution Analysis on Test Data
    contribution_analysis(equations, target_names, fold_data['test_data'])
    
    # 6. Summary Report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    for i, (eq_str, target_name) in enumerate(zip(equations, target_names)):
        print(f"\n{target_name.upper()}:")
        print(f"  Equation: {eq_str}")
        
        # Parse equation
        expr = parse_equation_to_sympy(eq_str)
        coefficients, intercept = extract_coefficients(expr, 8)
        
        # Find most important features
        abs_coeffs = np.abs(coefficients)
        top_features = np.argsort(abs_coeffs)[-3:][::-1]  # Top 3 features
        
        print(f"  Top 3 Most Important Features:")
        for j, feat_idx in enumerate(top_features):
            print(f"    {j+1}. Feature {feat_idx+1}: coefficient = {coefficients[feat_idx]:.6f}")
        
        print(f"  Intercept: {intercept:.6f}")
    
    print(f"\nAnalysis complete! Generated plots:")
    print(f"  - feature_importance_analysis.png")
    print(f"  - relative_importance_analysis.png")
    print(f"  - sensitivity_analysis_target_1.png")
    print(f"  - sensitivity_analysis_target_2.png")
    print(f"  - Log-space plots for each target")
    print(f"  - contribution_analysis_target_1.png")
    print(f"  - contribution_analysis_target_2.png")

if __name__ == "__main__":
    main()
