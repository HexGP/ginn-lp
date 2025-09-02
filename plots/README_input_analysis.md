# Input Contribution Analysis for GINN Equations

This repository contains scripts to analyze how each input feature contributes to the output based on the extracted equations from your GINN model.

## Files

1. **`professor_log_analysis.py`** - Simple script implementing exactly what your professor requested
2. **`analyze_input_contributions.py`** - Comprehensive analysis with multiple visualization types

## Quick Start (Professor's Request)

To run the exact analysis your professor requested:

```bash
python professor_log_analysis.py
```

This will:
- Load the refitted equations from `outputs/ginn_grad_ENB.json`
- Convert them to log-space expressions
- Plot sensitivity analysis for each variable using the provided function
- Generate plots showing how each input contributes to the output

## Comprehensive Analysis

For a more detailed analysis:

```bash
python analyze_input_contributions.py
```

This will generate:
- Feature importance analysis (coefficient magnitudes)
- Relative importance plots
- Sensitivity analysis (how output changes with input variations)
- Log-space expression analysis
- Actual contribution analysis on test data
- Multiple visualization plots

## What the Analysis Shows

### For Target 1 (Heating Load):
- **Most Important Features**: Feature 1 (X_1) and Feature 7 (X_7) have the largest coefficients
- **Equation**: `-57.82*X_1 - 0.050*X_2 + 0.036*X_3 - 0.046*X_4 + 4.58*X_5 + 0.275*X_6 + 18.02*X_7 + 0.415*X_8 + 66.20`

### For Target 2 (Cooling Load):
- **Most Important Features**: Feature 1 (X_1) and Feature 7 (X_7) are also most important
- **Equation**: `-74.20*X_1 - 0.064*X_2 + 0.015*X_3 - 0.045*X_4 + 4.60*X_5 + 0.328*X_6 + 11.68*X_7 + 0.307*X_8 + 98.50`

## Key Insights

1. **Feature 1 (X_1)** has the strongest negative influence on both targets
2. **Feature 7 (X_7)** has the strongest positive influence on both targets
3. **Features 2, 3, 4** have relatively small contributions
4. **Features 5, 6, 8** have moderate positive contributions

## Generated Outputs

The scripts will generate several PNG files:
- `feature_importance_analysis.png` - Bar charts showing coefficient magnitudes
- `relative_importance_analysis.png` - Normalized importance scores
- `sensitivity_analysis_target_1.png` - Sensitivity analysis for heating load
- `sensitivity_analysis_target_2.png` - Sensitivity analysis for cooling load
- Log-space plots for each target
- Contribution analysis plots

## Requirements

- Python 3.7+
- numpy
- matplotlib
- sympy
- pandas
- scikit-learn

## Data Source

The analysis uses the refitted equations from `outputs/ginn_grad_ENB.json`, which contains the results from your GINN model training on the ENB2012 dataset.
