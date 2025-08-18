import pandas as pd
import os

def print_formatted_table():
    """Print the results in the exact format specified by the user."""
    
    # Read the CSV file from the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, 'model_comparison_results.csv')
    df = pd.read_csv(csv_file)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Print the table in the exact format specified
    print(f"{'Model':<25} | {'MSE':<15} | {'MAPE':<15}")
    print(f"{'':<25} | {'Y1':<6} {'Y2':<6} {'Avg':<6} | {'Y1':<6} {'Y2':<6} {'Avg':<6}")
    print("-" * 80)
    
    # Print each model's results
    for _, row in df.iterrows():
        model_name = row['Model']
        y1_mse = row['Y1_MSE']
        y2_mse = row['Y2_MSE']
        avg_mse = row['Avg_MSE']
        y1_mape = row['Y1_MAPE']
        y2_mape = row['Y2_MAPE']
        avg_mape = row['Avg_MAPE']
        
        # Format the values appropriately
        if y1_mse > 1e10:  # For very large values like SGD
            y1_mse_str = f"{y1_mse:.2e}"
            y2_mse_str = f"{y2_mse:.2e}"
            avg_mse_str = f"{avg_mse:.2e}"
        else:
            y1_mse_str = f"{y1_mse:.4f}"
            y2_mse_str = f"{y2_mse:.4f}"
            avg_mse_str = f"{avg_mse:.4f}"
        
        if y1_mape > 1000:  # For very large MAPE values
            y1_mape_str = f"{y1_mape:.2e}"
            y2_mape_str = f"{y2_mape:.2e}"
            avg_mape_str = f"{avg_mape:.2e}"
        else:
            y1_mape_str = f"{y1_mape:.2f}"
            y2_mape_str = f"{y2_mape:.2f}"
            avg_mape_str = f"{avg_mape:.2f}"
        
        print(f"{model_name:<25} | {y1_mse_str:<6} {y2_mse_str:<6} {avg_mse_str:<6} | "
              f"{y1_mape_str:<6} {y2_mape_str:<6} {avg_mape_str:<6}")
    
    print("="*80)
    
    # Print summary statistics
    print("\nSUMMARY:")
    print("-" * 40)
    
    # Find best performing models
    best_mse_model = df.loc[df['Avg_MSE'].idxmin(), 'Model']
    best_mape_model = df.loc[df['Avg_MAPE'].idxmin(), 'Model']
    
    print(f"Best MSE (Average): {best_mse_model}")
    print(f"Best MAPE (Average): {best_mape_model}")
    
    # Show the actual best values
    best_mse_value = df['Avg_MSE'].min()
    best_mape_value = df['Avg_MAPE'].min()
    
    print(f"Best Average MSE: {best_mse_value:.4f}")
    print(f"Best Average MAPE: {best_mape_value:.2f}%")

if __name__ == "__main__":
    print_formatted_table() 