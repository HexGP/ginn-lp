"""
Data preparation script for separate farmer and market datasets.
Processes farmer_advisor_dataset.csv and market_researcher_dataset.csv separately
without merging, following the same preprocessing approach but keeping datasets isolated.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def create_stratified_sample(df, sample_size, filename, dataset_name):
    """
    Create stratified sample from the dataset.
    Stratifies by categorical columns to maintain data distribution.
    """
    print(f"Creating stratified sample for {dataset_name}: {sample_size:,} samples -> {filename}")
    
    # Get categorical columns (these will be different for farmer vs market data)
    categorical_cols = []
    for col in df.columns:
        if col.startswith('X_') and col in ['X_CT', 'X_Season']:  # Only if these exist
            categorical_cols.append(col)
    
    if not categorical_cols:
        # If no categorical columns, use random sampling
        print(f"  No categorical columns found, using random sampling")
        df_stratified = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Created random sample: {len(df_stratified):,} samples")
        return df_stratified
    
    try:
        # Manual stratification to ensure proper distribution
        stratified_samples = []
        
        # Get unique combinations of categorical values
        df['strata'] = df[categorical_cols[0]].astype(str)
        if len(categorical_cols) > 1:
            for col in categorical_cols[1:]:
                df['strata'] = df['strata'] + '_' + df[col].astype(str)
        
        strata_counts = df['strata'].value_counts()
        
        # Calculate how many samples to take from each stratum
        total_samples = len(df)
        samples_per_stratum = {}
        
        for stratum, count in strata_counts.items():
            # Proportional sampling
            stratum_sample_size = max(1, int((count / total_samples) * sample_size))
            samples_per_stratum[stratum] = min(stratum_sample_size, count)
        
        # Sample from each stratum
        for stratum, stratum_size in samples_per_stratum.items():
            stratum_data = df[df['strata'] == stratum]
            if len(stratum_data) > 0:
                sampled_stratum = stratum_data.sample(n=stratum_size, random_state=42)
                stratified_samples.append(sampled_stratum)
        
        # Combine all stratified samples
        df_stratified = pd.concat(stratified_samples, ignore_index=True)
        
        # Remove the temporary strata column
        df_stratified = df_stratified.drop('strata', axis=1)
        
        # Shuffle the final sample
        df_stratified = df_stratified.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        print(f"  Created stratified sample: {len(df_stratified):,} samples")
        
        # Show distribution
        for col in categorical_cols:
            print(f"  {col} distribution:")
            dist = df_stratified[col].value_counts()
            for value, count in dist.items():
                pct = (count / len(df_stratified)) * 100
                print(f"    {col} {value}: {count:,} ({pct:.1f}%)")
        
        return df_stratified
        
    except Exception as e:
        print(f"  ⚠️ Stratified sampling failed: {e}")
        print(f"  🔄 Using random sampling as fallback...")
        
        # Fallback: random sampling
        df_random = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Created random sample: {len(df_random):,} samples")
        return df_random

def prepare_farmer_data():
    """
    Prepare farmer dataset separately with GINN-friendly column names.
    """
    print("="*60)
    print("PREPARING FARMER DATASET")
    print("="*60)
    
    # Load farmer dataset
    print("Loading farmer dataset...")
    farmer_data = pd.read_csv('data/original_data/farmer_advisor_dataset.csv')
    print(f"Farmer dataset shape: {farmer_data.shape}")
    
    # Apply 25% sampling with random_state=42
    print("Applying 25% sampling...")
    farmer_sampled = farmer_data.sample(frac=0.25, random_state=42)
    print(f"Sampled farmer data shape: {farmer_sampled.shape}")
    
    # Sort by categorical columns for consistency
    farmer_sampled = farmer_sampled.sort_values(['Crop_Type'])
    
    # Rename columns to GINN-friendly format
    column_mapping = {
        # Features
        'Crop_Type': 'X_CT',
        'Soil_pH': 'X_SpH', 
        'Soil_Moisture': 'X_SM',
        'Temperature_C': 'X_TempC',
        'Rainfall_mm': 'X_Rain',
        'Fertilizer_Usage_kg': 'X_Fert',
        'Pesticide_Usage_kg': 'X_Pest',
        'Crop_Yield_ton': 'X_Yield',
        'Sustainability_Score': 'target_SS'
    }
    
    # Apply column renaming
    farmer_sampled = farmer_sampled.rename(columns=column_mapping)
    
    # Reorder columns (features first, then target)
    feature_columns = ['X_CT', 'X_SpH', 'X_SM', 'X_TempC', 'X_Rain', 'X_Fert', 'X_Pest', 'X_Yield']
    target_columns = ['target_SS']
    
    farmer_sampled = farmer_sampled[feature_columns + target_columns]
    
    print(f"Farmer dataset after renaming: {farmer_sampled.shape}")
    print(f"Features: {feature_columns}")
    print(f"Target: {target_columns}")
    
    # Data cleaning - drop NULLs
    print("Cleaning data...")
    farmer_sampled.dropna(inplace=True)
    print(f"After cleaning: {farmer_sampled.shape}")
    
    # Convert categorical columns to category for memory efficiency
    categorical_columns = ['X_CT']
    for col in categorical_columns:
        farmer_sampled[col] = farmer_sampled[col].astype('category')
    
    # Label encoding for categorical variables
    print("Applying label encoding for categorical variables...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    for c in categorical_columns:
        farmer_sampled[c] = le.fit_transform(farmer_sampled[c])
    
    print(f"After encoding: {farmer_sampled.shape}")
    
    return farmer_sampled, feature_columns, target_columns

def prepare_market_data():
    """
    Prepare market dataset separately with GINN-friendly column names.
    """
    print("="*60)
    print("PREPARING MARKET DATASET")
    print("="*60)
    
    # Load market dataset
    print("Loading market dataset...")
    market_data = pd.read_csv('data/original_data/market_researcher_dataset.csv')
    print(f"Market dataset shape: {market_data.shape}")
    
    # Apply 25% sampling with random_state=42
    print("Applying 25% sampling...")
    market_sampled = market_data.sample(frac=0.25, random_state=42)
    print(f"Sampled market data shape: {market_sampled.shape}")
    
    # Sort by categorical columns for consistency
    market_sampled = market_sampled.sort_values(['Product'])
    
    # Rename columns to GINN-friendly format
    column_mapping = {
        # Features
        'Product': 'X_Product',  # Different categorical column for market data
        'Market_Price_per_ton': 'X_MPrice',
        'Demand_Index': 'X_Demand',
        'Supply_Index': 'X_Supply',
        'Competitor_Price_per_ton': 'X_CompPrice',
        'Economic_Indicator': 'X_Econ',
        'Weather_Impact_Score': 'X_Weather',
        'Seasonal_Factor': 'X_Season',
        'Consumer_Trend_Index': 'target_CTI'
    }
    
    # Apply column renaming
    market_sampled = market_sampled.rename(columns=column_mapping)
    
    # Reorder columns (features first, then target)
    feature_columns = ['X_Product', 'X_MPrice', 'X_Demand', 'X_Supply', 'X_CompPrice', 'X_Econ', 'X_Weather', 'X_Season']
    target_columns = ['target_CTI']
    
    market_sampled = market_sampled[feature_columns + target_columns]
    
    print(f"Market dataset after renaming: {market_sampled.shape}")
    print(f"Features: {feature_columns}")
    print(f"Target: {target_columns}")
    
    # Data cleaning - drop NULLs
    print("Cleaning data...")
    market_sampled.dropna(inplace=True)
    print(f"After cleaning: {market_sampled.shape}")
    
    # Convert categorical columns to category for memory efficiency
    categorical_columns = ['X_Product', 'X_Season']
    for col in categorical_columns:
        market_sampled[col] = market_sampled[col].astype('category')
    
    # Label encoding for categorical variables
    print("Applying label encoding for categorical variables...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    for c in categorical_columns:
        market_sampled[c] = le.fit_transform(market_sampled[c])
    
    print(f"After encoding: {market_sampled.shape}")
    
    return market_sampled, feature_columns, target_columns

def save_datasets(df, feature_cols, target_cols, dataset_name, output_dir):
    """
    Save datasets with stratified samples for a given dataset.
    """
    print(f"\nSaving {dataset_name} datasets...")
    
    # 1. Save full dataset
    print(f"1. Saving full {dataset_name} dataset...")
    df.to_csv(f'{output_dir}/{dataset_name}_full_dataset.csv', index=False)
    print(f"   ✅ Saved: {dataset_name}_full_dataset.csv ({len(df):,} samples)")
    
    # 2. Create and save stratified samples
    print(f"2. Creating stratified samples for {dataset_name}...")
    
    # 1K stratified sample
    df_1k = create_stratified_sample(df, 1000, f'{dataset_name}_stratified_1k.csv', dataset_name)
    df_1k.to_csv(f'{output_dir}/{dataset_name}_stratified_1k.csv', index=False)
    print(f"   ✅ Saved: {dataset_name}_stratified_1k.csv")
    
    # 10K stratified sample
    df_10k = create_stratified_sample(df, 10000, f'{dataset_name}_stratified_10k.csv', dataset_name)
    df_10k.to_csv(f'{output_dir}/{dataset_name}_stratified_10k.csv', index=False)
    print(f"   ✅ Saved: {dataset_name}_stratified_10k.csv")
    
    # 50K stratified sample
    df_50k = create_stratified_sample(df, 50000, f'{dataset_name}_stratified_50k.csv', dataset_name)
    df_50k.to_csv(f'{output_dir}/{dataset_name}_stratified_50k.csv', index=False)
    print(f"   ✅ Saved: {dataset_name}_stratified_50k.csv")
    
    # Save feature and target names
    print(f"3. Saving {dataset_name} metadata files...")
    pd.DataFrame({'feature_names': feature_cols}).to_csv(
        f'{output_dir}/{dataset_name}_feature_names.csv', index=False
    )
    pd.DataFrame({'target_names': target_cols}).to_csv(
        f'{output_dir}/{dataset_name}_target_names.csv', index=False
    )
    print(f"   ✅ Saved: {dataset_name}_feature_names.csv")
    print(f"   ✅ Saved: {dataset_name}_target_names.csv")
    
    return {
        'full': df,
        '1k': df_1k,
        '10k': df_10k,
        '50k': df_50k,
        'features': feature_cols,
        'targets': target_cols
    }

def prepare_separate_data():
    """
    Main function to prepare separate farmer and market datasets.
    """
    print("="*70)
    print("PREPARING SEPARATE FARMER AND MARKET DATASETS")
    print("="*70)
    
    # Create output directory
    os.makedirs('data/prepared/separate', exist_ok=True)
    
    # Prepare farmer data
    farmer_df, farmer_features, farmer_targets = prepare_farmer_data()
    
    # Prepare market data
    market_df, market_features, market_targets = prepare_market_data()
    
    # Save farmer datasets
    farmer_results = save_datasets(
        farmer_df, farmer_features, farmer_targets, 
        'farmer', 'data/prepared/separate'
    )
    
    # Save market datasets
    market_results = save_datasets(
        market_df, market_features, market_targets, 
        'market', 'data/prepared/separate'
    )
    
    # Create summary
    print("\n" + "="*70)
    print("SEPARATE DATASETS SUMMARY")
    print("="*70)
    
    print(f"\n🌾 FARMER DATASET:")
    print(f"   Shape: {farmer_df.shape}")
    print(f"   Features: {len(farmer_features)} ({farmer_features})")
    print(f"   Target: {farmer_targets[0]}")
    print(f"   Samples: 1K={len(farmer_results['1k']):,}, 10K={len(farmer_results['10k']):,}, 50K={len(farmer_results['50k']):,}")
    
    print(f"\n🏪 MARKET DATASET:")
    print(f"   Shape: {market_df.shape}")
    print(f"   Features: {len(market_features)} ({market_features})")
    print(f"   Target: {market_targets[0]}")
    print(f"   Samples: 1K={len(market_results['1k']):,}, 10K={len(market_results['10k']):,}, 50K={len(market_results['50k']):,}")
    
    print(f"\n📁 Files saved to data/prepared/separate/:")
    print(f"🌾 Farmer datasets:")
    print(f"   - farmer_full_dataset.csv ({len(farmer_df):,} samples)")
    print(f"   - farmer_stratified_1k.csv ({len(farmer_results['1k']):,} samples)")
    print(f"   - farmer_stratified_10k.csv ({len(farmer_results['10k']):,} samples)")
    print(f"   - farmer_stratified_50k.csv ({len(farmer_results['50k']):,} samples)")
    print(f"   - farmer_feature_names.csv")
    print(f"   - farmer_target_names.csv")
    
    print(f"🏪 Market datasets:")
    print(f"   - market_full_dataset.csv ({len(market_df):,} samples)")
    print(f"   - market_stratified_1k.csv ({len(market_results['1k']):,} samples)")
    print(f"   - market_stratified_10k.csv ({len(market_results['10k']):,} samples)")
    print(f"   - market_stratified_50k.csv ({len(market_results['50k']):,} samples)")
    print(f"   - market_feature_names.csv")
    print(f"   - market_target_names.csv")
    
    print(f"\n🎯 Usage for GINN training:")
    print(f"   Farmer dataset: 'data/prepared/separate/farmer_stratified_1k.csv'")
    print(f"   Market dataset: 'data/prepared/separate/market_stratified_1k.csv'")
    print(f"   Note: Each dataset has only one target (simpler for GINN)")
    
    # Create documentation
    doc_text = f"""
=== SEPARATE DATASETS DOCUMENTATION ===

FARMER DATASET (target_SS - Sustainability Score):
Features: {farmer_features}
Target: {farmer_targets[0]}
Samples: {len(farmer_df):,} total

MARKET DATASET (target_CTI - Consumer Trend Index):
Features: {market_features}
Target: {market_targets[0]}
Samples: {len(market_df):,} total

BENEFITS OF SEPARATE APPROACH:
- No merging complexity
- Cleaner mathematical relationships
- Better GINN performance
- Simpler equations to extract
- No spurious correlations

USAGE:
- Train GINN on farmer dataset for sustainability prediction
- Train GINN on market dataset for consumer trend prediction
- Compare results with merged approach

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('data/prepared/separate/separate_datasets_summary.txt', 'w', encoding='utf-8') as f:
        f.write(doc_text)
    
    print(f"\n✅ Saved documentation: separate_datasets_summary.txt")
    print("\n" + "="*70)
    print("SEPARATE DATA PREPARATION COMPLETED!")
    print("="*70)
    
    return farmer_results, market_results

if __name__ == "__main__":
    # Run the separate data preparation
    farmer_results, market_results = prepare_separate_data()
    print("\nSeparate data preparation completed successfully!")
