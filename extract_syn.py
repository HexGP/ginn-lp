#!/usr/bin/env python3
"""
Simple script to extract metrics from all JSON files in outputs/JSON_SYN folder
"""

import json
import os
import pandas as pd

def main():
    folder_path = "outputs/JSON_SYN"
    
    # Get all JSON files
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Found {len(json_files)} JSON files")
    
    all_results = []
    
    for filename in json_files:
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Get architecture info
        arch = data[0]['architecture']
        shared_layers = arch['shared_layers']
        pta_blocks = arch['pta_blocks_per_layer']
        
        # Get model type from filename
        if 'grad_' in filename:
            model_type = 'Gradient'
        elif 'surr_' in filename:
            model_type = 'Surrogate'
        elif 'trad_' in filename:
            model_type = 'Traditional'
        
        # Process each target
        for target_data in data[0]['per_target']:
            target_name = target_data['target']
            
            # Map targets: target_0 -> Y1, target_1 -> Y2
            if target_name == 'target_0':
                target_label = 'Y1'
            elif target_name == 'target_1':
                target_label = 'Y2'
            
            # Get metrics
            model_mse = target_data['model']['MSE']
            model_mape = target_data['model']['MAPE']
            eq_refit_mse = target_data['eq_refit']['MSE']
            eq_refit_mape = target_data['eq_refit']['MAPE']
            
            # Create model names
            layer_text = f"{shared_layers} Layer" if shared_layers == 1 else f"{shared_layers} Layers"
            
            # Multi-GINN entry
            all_results.append({
                'Model': f"Multi-GINN {layer_text} ({pta_blocks} PTA) ({model_type})",
                'Target': target_label,
                'MSE': model_mse,
                'MAPE': model_mape,
                'Type': 'Model'
            })
            
            # Refit Eq entry
            all_results.append({
                'Model': f"Refit Eq {layer_text} ({pta_blocks} PTA) ({model_type})",
                'Target': target_label,
                'MSE': eq_refit_mse,
                'MAPE': eq_refit_mape,
                'Type': 'Refit Eq'
            })
    
    # Create summary table
    y1_data = [r for r in all_results if r['Target'] == 'Y1']
    y2_data = [r for r in all_results if r['Target'] == 'Y2']
    
    summary_rows = []
    
    # Match Y1 and Y2 entries by model name
    for y1_entry in y1_data:
        model_name = y1_entry['Model']
        y1_mse = y1_entry['MSE']
        y1_mape = y1_entry['MAPE']
        
        # Find matching Y2 entry
        y2_entry = next((r for r in y2_data if r['Model'] == model_name), None)
        if y2_entry:
            y2_mse = y2_entry['MSE']
            y2_mape = y2_entry['MAPE']
            
            avg_mse = (y1_mse + y2_mse) / 2
            avg_mape = (y1_mape + y2_mape) / 2
            
            summary_rows.append({
                'Model': model_name,
                'MSE (Y1)': round(y1_mse, 3),
                'MSE (Y2)': round(y2_mse, 3),
                'Avg MSE': round(avg_mse, 3),
                'MAPE (Y1)': round(y1_mape, 3),
                'MAPE (Y2)': round(y2_mape, 3),
                'Avg MAPE': round(avg_mape, 3)
            })
    
    # Print table
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n" + "="*120)
        print("SYN METRICS TABLE")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
    else:
        print("No data to display")

if __name__ == "__main__":
    main()

