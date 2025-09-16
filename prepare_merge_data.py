"""
Data preparation script for merged farmer and market dataset.
Combines farmer_advisor_dataset.csv and market_researcher_dataset.csv
following the MTR-Agric-LR-MOR approach but adapted for GINN.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def create_stratified_sample(df, sample_size, filename):
    """
    Create stratified sample from the full dataset.
    Stratifies by X_CT (Crop_Type) and X_Season (Seasonal_Factor) to maintain data distribution.
    """
    print(f"Creating stratified sample: {sample_size:,} samples -> {filename}")
    
    # Get original categorical values (before label encoding)
    # We need to stratify by the original categories, not the encoded numbers
    categorical_cols = ['X_CT', 'X_Season']  # Updated to new column names
    
    # For stratified sampling, we'll use the encoded values as strata
    # This ensures we get proportional representation from each category
    try:
        # Use sklearn's train_test_split for stratified sampling
        # We'll stratify by the combination of categorical columns
        df_stratified, _ = train_test_split(
            df, 
            train_size=sample_size,
            random_state=42,
            stratify=None  # We'll do manual stratification below
        )
        
        # Manual stratification to ensure proper distribution
        stratified_samples = []
        
        # Get unique combinations of categorical values
        df['strata'] = df['X_CT'].astype(str) + '_' + df['X_Season'].astype(str)
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
        print(f"  X_CT (Crop_Type) distribution:")
        crop_dist = df_stratified['X_CT'].value_counts()
        for crop, count in crop_dist.items():
            pct = (count / len(df_stratified)) * 100
            print(f"    Crop {crop}: {count:,} ({pct:.1f}%)")
        
        print(f"  X_Season (Seasonal_Factor) distribution:")
        season_dist = df_stratified['X_Season'].value_counts()
        for season, count in season_dist.items():
            pct = (count / len(df_stratified)) * 100
            print(f"    Season {season}: {count:,} ({pct:.1f}%)")
        
        return df_stratified
        
    except Exception as e:
        print(f"  ⚠️ Stratified sampling failed: {e}")
        print(f"  🔄 Using random sampling as fallback...")
        
        # Fallback: random sampling
        df_random = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Created random sample: {len(df_random):,} samples")
        return df_random

def prepare_merged_data():
    """
    Prepare merged dataset combining farmer and market data.
    Adapted from MTR-Agric-LR-MOR approach but optimized for GINN.
    """
    
    # Load datasets
    print("Loading datasets...")
    farmer_data = pd.read_csv('data/original_data/farmer_advisor_dataset.csv')
    market_data = pd.read_csv('data/original_data/market_researcher_dataset.csv')
    
    print(f"Farmer dataset shape: {farmer_data.shape}")
    print(f"Market dataset shape: {market_data.shape}")
    
    # Apply 25% sampling with random_state=42 (following MTR approach)
    print("Applying 25% sampling...")
    farmer_sampled = farmer_data.sample(frac=0.25, random_state=42)
    market_sampled = market_data.sample(frac=0.25, random_state=42)
    
    print(f"Sampled farmer data shape: {farmer_sampled.shape}")
    print(f"Sampled market data shape: {market_sampled.shape}")
    
    # Sort by categorical columns for consistency
    farmer_sampled = farmer_sampled.sort_values(['Crop_Type'])
    market_sampled = market_sampled.sort_values(['Product'])
    
    # Merge datasets on Crop_Type (farmer) and Product (market)
    print("Merging datasets...")
    df_merged = pd.merge(
        farmer_sampled, 
        market_sampled, 
        left_on='Crop_Type', 
        right_on='Product', 
        how='inner'
    )
    
    print(f"Merged dataset shape: {df_merged.shape}")
    
    # Drop ID columns and Product (redundant with Crop_Type)
    df_merged = df_merged.drop(['Farm_ID', 'Market_ID', 'Product'], axis=1)
    
    # Re-arrange columns for consistency (features first, then targets)
    original_feature_columns = [
        'Crop_Type', 'Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
        'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Crop_Yield_ton',
        'Market_Price_per_ton', 'Demand_Index', 'Supply_Index', 
        'Competitor_Price_per_ton', 'Economic_Indicator', 
        'Weather_Impact_Score', 'Seasonal_Factor'
    ]
    
    original_target_columns = ['Sustainability_Score', 'Consumer_Trend_Index']
    
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
        'Market_Price_per_ton': 'X_MPrice',
        'Demand_Index': 'X_Demand',
        'Supply_Index': 'X_Supply',
        'Competitor_Price_per_ton': 'X_CompPrice',
        'Economic_Indicator': 'X_Econ',
        'Weather_Impact_Score': 'X_Weather',
        'Seasonal_Factor': 'X_Season',
        
        # Targets
        'Sustainability_Score': 'target_SS',
        'Consumer_Trend_Index': 'target_CTI'
    }
    
    # Apply column renaming
    df_merged = df_merged.rename(columns=column_mapping)
    
    # Define new column names after renaming
    feature_columns = [column_mapping[col] for col in original_feature_columns]
    target_columns = [column_mapping[col] for col in original_target_columns]
    
    # Reorder columns
    df_merged = df_merged[feature_columns + target_columns]
    
    print(f"Final dataset shape: {df_merged.shape}")
    print(f"Features: {len(feature_columns)}, Targets: {len(target_columns)}")
    
    # Data cleaning - drop NULLs
    print("Cleaning data...")
    df_merged.dropna(inplace=True)
    print(f"After cleaning: {df_merged.shape}")
    
    # Convert object columns to category for memory efficiency
    categorical_columns = ['X_CT', 'X_Season']  # Updated to new column names
    for col in categorical_columns:
        df_merged[col] = df_merged[col].astype('category')
    
    # Label encoding (exactly like MTR-Agric-LR-MOR.ipynb)
    print("Applying label encoding for categorical variables...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    for c in categorical_columns:
        df_merged[c] = le.fit_transform(df_merged[c])
    
    df_encoded = df_merged
    
    print(f"After encoding: {df_encoded.shape}")
    print(f"Encoded categorical columns: {categorical_columns}")
    
    # Separate features and targets for reference
    target_columns = ['target_SS', 'target_CTI']
    feature_columns = [col for col in df_encoded.columns if col not in target_columns]
    
    print(f"Features shape: {len(feature_columns)} columns")
    print(f"Targets shape: {len(target_columns)} columns")
    print(f"Full dataset shape: {df_encoded.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs('data/prepared/merged', exist_ok=True)
    
    # Save all datasets: full + 3 stratified samples
    print("Saving datasets: full + 3 stratified samples...")
    
    # 1. Save full dataset
    print("\n1. Saving full dataset...")
    df_encoded.to_csv('data/prepared/merged/merged_full_dataset.csv', index=False)
    print(f"   ✅ Saved: merged_full_dataset.csv ({len(df_encoded):,} samples)")
    
    # 2. Create and save stratified samples
    print("\n2. Creating stratified samples...")
    
    # 1K stratified sample (ultra-fast experiments)
    df_1k = create_stratified_sample(df_encoded, 1000, 'merged_stratified_1k.csv')
    df_1k.to_csv('data/prepared/merged/merged_stratified_1k.csv', index=False)
    print(f"   ✅ Saved: merged_stratified_1k.csv")
    
    # 10K stratified sample (very fast experiments)
    df_10k = create_stratified_sample(df_encoded, 10000, 'merged_stratified_10k.csv')
    df_10k.to_csv('data/prepared/merged/merged_stratified_10k.csv', index=False)
    print(f"   ✅ Saved: merged_stratified_10k.csv")
    
    # 50K stratified sample (fast experiments)
    df_50k = create_stratified_sample(df_encoded, 50000, 'merged_stratified_50k.csv')
    df_50k.to_csv('data/prepared/merged/merged_stratified_50k.csv', index=False)
    print(f"   ✅ Saved: merged_stratified_50k.csv")
    
    # 100K stratified sample (standard training)
    df_100k = create_stratified_sample(df_encoded, 100000, 'merged_stratified_100k.csv')
    df_100k.to_csv('data/prepared/merged/merged_stratified_100k.csv', index=False)
    print(f"   ✅ Saved: merged_stratified_100k.csv")
    
    # 200K stratified sample (best performance)
    df_200k = create_stratified_sample(df_encoded, 200000, 'merged_stratified_200k.csv')
    df_200k.to_csv('data/prepared/merged/merged_stratified_200k.csv', index=False)
    print(f"   ✅ Saved: merged_stratified_200k.csv")
    
    # Save feature and target names (same for all datasets)
    print("\n3. Saving metadata files...")
    pd.DataFrame({'feature_names': feature_columns}).to_csv(
        'data/prepared/merged/merged_feature_names.csv', index=False
    )
    pd.DataFrame({'target_names': target_columns}).to_csv(
        'data/prepared/merged/merged_target_names.csv', index=False
    )
    print("   ✅ Saved: merged_feature_names.csv")
    print("   ✅ Saved: merged_target_names.csv")
    
    # Create formatted distribution strings for each dataset
    def format_distribution(df, column_name, dataset_name):
        dist = df[column_name].value_counts()
        total = len(df)
        formatted_lines = []
        for value, count in dist.items():
            pct = (count / total) * 100
            formatted_lines.append(f"    {column_name} {value}: {count:,} ({pct:.1f}%)")
        return f"{dataset_name} {column_name} Distribution:\n" + "\n".join(formatted_lines)
    
    # Create column mapping documentation
    def create_column_mapping_doc():
        doc = """
=== COLUMN MAPPING DOCUMENTATION ===

Original Column Names -> GINN-Friendly Names:

FEATURES:
- Crop_Type -> X_CT (Crop Type)
- Soil_pH -> X_SpH (Soil pH)
- Soil_Moisture -> X_SM (Soil Moisture)
- Temperature_C -> X_TempC (Temperature in Celsius)
- Rainfall_mm -> X_Rain (Rainfall in mm)
- Fertilizer_Usage_kg -> X_Fert (Fertilizer Usage in kg)
- Pesticide_Usage_kg -> X_Pest (Pesticide Usage in kg)
- Crop_Yield_ton -> X_Yield (Crop Yield in tons)
- Market_Price_per_ton -> X_MPrice (Market Price per ton)
- Demand_Index -> X_Demand (Demand Index)
- Supply_Index -> X_Supply (Supply Index)
- Competitor_Price_per_ton -> X_CompPrice (Competitor Price per ton)
- Economic_Indicator -> X_Econ (Economic Indicator)
- Weather_Impact_Score -> X_Weather (Weather Impact Score)
- Seasonal_Factor -> X_Season (Seasonal Factor)

TARGETS:
- Sustainability_Score -> target_SS (Sustainability Score)
- Consumer_Trend_Index -> target_CTI (Consumer Trend Index)

CATEGORICAL FEATURES (Label Encoded):
- X_CT (Crop_Type): 0, 1, 2, 3 (4 crop types)
- X_Season (Seasonal_Factor): 0, 1, 2 (3 seasons)

CONTINUOUS FEATURES:
- X_SpH, X_SM, X_TempC, X_Rain, X_Fert, X_Pest, X_Yield
- X_MPrice, X_Demand, X_Supply, X_CompPrice, X_Econ, X_Weather

USAGE IN GINN:
- Features: X_CT, X_SpH, X_SM, ..., X_Season (15 total)
- Targets: target_SS, target_CTI (2 total)
- Remove categorical features with zeros: X_CT, X_Season
- Use continuous features only: 13 features for GINN training
"""
        return doc
    
    # Create summary text for saving
    summary_text = f"""
=== Dataset Summary ===
Original farmer data: {farmer_data.shape}
Original market data: {market_data.shape}
After 25% sampling: {farmer_sampled.shape}, {market_sampled.shape}
After merging: {df_merged.shape}
After encoding: {df_encoded.shape}
Features: {len(feature_columns)}
Targets: {len(target_columns)}

📁 Files saved to data/prepared/merged/:
📊 Datasets:
   - merged_full_dataset.csv ({len(df_encoded):,} samples)
   - merged_stratified_1k.csv ({len(df_1k):,} samples)
   - merged_stratified_10k.csv ({len(df_10k):,} samples)
   - merged_stratified_50k.csv ({len(df_50k):,} samples)
   - merged_stratified_100k.csv ({len(df_100k):,} samples)
   - merged_stratified_200k.csv ({len(df_200k):,} samples)
📋 Metadata:
   - merged_feature_names.csv (feature column names)
   - merged_target_names.csv (target column names)
   - column_mapping.txt (column renaming documentation)

🎯 Usage in run_shift_farm.py:
   Change DATA_CSV to:
   - 'data/prepared/merged/merged_stratified_1k.csv'   (ultra-fast experiments)
   - 'data/prepared/merged/merged_stratified_10k.csv'  (very fast experiments)
   - 'data/prepared/merged/merged_stratified_50k.csv'  (fast experiments)
   - 'data/prepared/merged/merged_stratified_100k.csv' (standard training)
   - 'data/prepared/merged/merged_stratified_200k.csv' (best performance)
   - 'data/prepared/merged/merged_full_dataset.csv'    (full dataset)

📊 Stratified Sample Details:

{format_distribution(df_1k, 'X_CT', '1K Sample')}

{format_distribution(df_1k, 'X_Season', '1K Sample')}

{format_distribution(df_10k, 'X_CT', '10K Sample')}

{format_distribution(df_10k, 'X_Season', '10K Sample')}

{format_distribution(df_50k, 'X_CT', '50K Sample')}

{format_distribution(df_50k, 'X_Season', '50K Sample')}

{format_distribution(df_100k, 'X_CT', '100K Sample')}

{format_distribution(df_100k, 'X_Season', '100K Sample')}

{format_distribution(df_200k, 'X_CT', '200K Sample')}

{format_distribution(df_200k, 'X_Season', '200K Sample')}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    print("\n=== Dataset Summary ===")
    print(f"Original farmer data: {farmer_data.shape}")
    print(f"Original market data: {market_data.shape}")
    print(f"After 25% sampling: {farmer_sampled.shape}, {market_sampled.shape}")
    print(f"After merging: {df_merged.shape}")
    print(f"After encoding: {df_encoded.shape}")
    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {len(target_columns)}")
    
    print(f"\n📁 Files saved to data/prepared/merged/:")
    print(f"📊 Datasets:")
    print(f"   - merged_full_dataset.csv ({len(df_encoded):,} samples)")
    print(f"   - merged_stratified_1k.csv ({len(df_1k):,} samples)")
    print(f"   - merged_stratified_10k.csv ({len(df_10k):,} samples)")
    print(f"   - merged_stratified_50k.csv ({len(df_50k):,} samples)")
    print(f"   - merged_stratified_100k.csv ({len(df_100k):,} samples)")
    print(f"   - merged_stratified_200k.csv ({len(df_200k):,} samples)")
    print(f"📋 Metadata:")
    print(f"   - merged_feature_names.csv (feature column names)")
    print(f"   - merged_target_names.csv (target column names)")
    
    print(f"\n🎯 Usage in run_shift_farm.py:")
    print(f"   Change DATA_CSV to:")
    print(f"   - 'data/prepared/merged/merged_stratified_1k.csv'   (ultra-fast experiments)")
    print(f"   - 'data/prepared/merged/merged_stratified_10k.csv'  (very fast experiments)")
    print(f"   - 'data/prepared/merged/merged_stratified_50k.csv'  (fast experiments)")
    print(f"   - 'data/prepared/merged/merged_stratified_100k.csv' (standard training)")
    print(f"   - 'data/prepared/merged/merged_stratified_200k.csv' (best performance)")
    print(f"   - 'data/prepared/merged/merged_full_dataset.csv'    (full dataset)")
    
    # Save summary to text file
    print(f"\n4. Saving dataset summary...")
    with open('data/prepared/merged/dataset_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"   ✅ Saved: dataset_summary.txt")
    
    # Save column mapping documentation
    print(f"\n5. Saving column mapping documentation...")
    column_mapping_doc = create_column_mapping_doc()
    with open('data/prepared/merged/column_mapping.txt', 'w', encoding='utf-8') as f:
        f.write(column_mapping_doc)
    print(f"   ✅ Saved: column_mapping.txt")
    
    return {
        'feature_names': feature_columns,
        'target_names': target_columns,
        'full_dataset': df_encoded
    }

if __name__ == "__main__":
    # Run the data preparation
    data = prepare_merged_data()
    print("\nData preparation completed successfully!")
