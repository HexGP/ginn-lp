import pandas as pd
import os

def split_enb2012_dataset():
    """
    Split the ENB2012 dataset into two separate datasets:
    1. ENB2012_Target1.csv - Contains X1-X8 features + target_1
    2. ENB2012_Target2.csv - Contains X1-X8 features + target_2
    """
    
    print("="*60)
    print("SPLITTING ENB2012 DATASET")
    print("="*60)
    
    # Define file paths
    input_file = "data/ENB2012_data.csv"
    output_dir = "data"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Loading dataset from: {input_file}")
    
    # Load the original dataset
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create dataset for Target 1
    print("\nCreating ENB2012_Target1 dataset...")
    target1_df = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'target_1']].copy()
    target1_df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'target']
    
    # Save Target 1 dataset
    target1_file = os.path.join(output_dir, "ENB2012_Target1.csv")
    target1_df.to_csv(target1_file, index=False)
    print(f"Target 1 dataset saved to: {target1_file}")
    print(f"Target 1 dataset shape: {target1_df.shape}")
    
    # Create dataset for Target 2
    print("\nCreating ENB2012_Target2 dataset...")
    target2_df = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'target_2']].copy()
    target2_df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'target']
    
    # Save Target 2 dataset
    target2_file = os.path.join(output_dir, "ENB2012_Target2.csv")
    target2_df.to_csv(target2_file, index=False)
    print(f"Target 2 dataset saved to: {target2_file}")
    print(f"Target 2 dataset shape: {target2_df.shape}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target 1 dataset: {target1_df.shape[0]} rows, {target1_df.shape[1]} columns")
    print(f"Target 2 dataset: {target2_df.shape[0]} rows, {target2_df.shape[1]} columns")
    
    print(f"\nTarget 1 statistics:")
    print(f"  Mean: {target1_df['target'].mean():.4f}")
    print(f"  Std:  {target1_df['target'].std():.4f}")
    print(f"  Min:  {target1_df['target'].min():.4f}")
    print(f"  Max:  {target1_df['target'].max():.4f}")
    
    print(f"\nTarget 2 statistics:")
    print(f"  Mean: {target2_df['target'].mean():.4f}")
    print(f"  Std:  {target2_df['target'].std():.4f}")
    print(f"  Min:  {target2_df['target'].min():.4f}")
    print(f"  Max:  {target2_df['target'].max():.4f}")
    
    print("\n" + "="*60)
    print("DATASET SPLITTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return target1_df, target2_df

if __name__ == "__main__":
    target1_df, target2_df = split_enb2012_dataset() 