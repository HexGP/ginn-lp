import pandas as pd
import os

def split_agricultural_dataset():
    """
    Split the merged agricultural dataset into two separate single-target datasets:
    1. Agricultural_Target1.csv - Contains features + target_SS (Sustainability Score)
    2. Agricultural_Target2.csv - Contains features + target_CTI (Consumer Trend Index)
    """
    
    print("="*60)
    print("SPLITTING AGRICULTURAL DATASET")
    print("="*60)
    
    # Define file paths
    input_file = "data/prepared/merged_stratified_1k.csv"
    output_dir = "data/prepared"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Loading dataset from: {input_file}")
    
    # Load the original dataset
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Define feature columns (exclude targets)
    target_cols = ['target_SS', 'target_CTI']  # Updated to new column names
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    print(f"Feature columns: {feature_cols}")
    print(f"Target columns: {target_cols}")
    
    # Create dataset for Target 1 (target_SS - Sustainability Score)
    print("\nCreating Agricultural_Target1 dataset (target_SS)...")
    target1_cols = feature_cols + ['target_SS']
    target1_df = df[target1_cols].copy()
    target1_df.columns = feature_cols + ['target']
    
    # Save Target 1 dataset
    target1_file = os.path.join(output_dir, "Agricultural_Target1.csv")
    target1_df.to_csv(target1_file, index=False)
    print(f"Target 1 dataset saved to: {target1_file}")
    print(f"Target 1 dataset shape: {target1_df.shape}")
    
    # Create dataset for Target 2 (target_CTI - Consumer Trend Index)
    print("\nCreating Agricultural_Target2 dataset (target_CTI)...")
    target2_cols = feature_cols + ['target_CTI']
    target2_df = df[target2_cols].copy()
    target2_df.columns = feature_cols + ['target']
    
    # Save Target 2 dataset
    target2_file = os.path.join(output_dir, "Agricultural_Target2.csv")
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
    
    print(f"\nSustainability Score (target_SS) statistics:")
    print(f"  Mean: {target1_df['target'].mean():.4f}")
    print(f"  Std:  {target1_df['target'].std():.4f}")
    print(f"  Min:  {target1_df['target'].min():.4f}")
    print(f"  Max:  {target1_df['target'].max():.4f}")
    
    print(f"\nConsumer Trend Index (target_CTI) statistics:")
    print(f"  Mean: {target2_df['target'].mean():.4f}")
    print(f"  Std:  {target2_df['target'].std():.4f}")
    print(f"  Min:  {target2_df['target'].min():.4f}")
    print(f"  Max:  {target2_df['target'].max():.4f}")
    
    print("\n" + "="*60)
    print("DATASET SPLITTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return target1_df, target2_df

if __name__ == "__main__":
    target1_df, target2_df = split_agricultural_dataset() 