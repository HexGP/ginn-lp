import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load data
DATA_PATH = 'data/ENB2012_data.csv'
df = pd.read_csv(DATA_PATH)

# Assume the first 8 columns are X1-X8, and columns 8 and 9 are Y1, Y2
def get_XY(df):
    X = df.iloc[:, :8].values
    Y = df.iloc[:, 8:10].values  # shape (n_samples, 2)
    return X, Y

X_raw, Y_raw = get_XY(df)

# --- MinMax Scaling (range 0.1 to 10.0, as in other scripts) ---
scaler_X = MinMaxScaler(feature_range=(0.1, 10.0))
scaler_Y = MinMaxScaler(feature_range=(0.1, 10.0))
X = scaler_X.fit_transform(X_raw)
Y = scaler_Y.fit_transform(Y_raw)
Y1_true = Y[:, 0]
Y2_true = Y[:, 1]

epsilon = 1e-8  # To avoid divide by zero

# --- Simplified Equations ---
def expr_A_simplified(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    term1 = X2**1.0 * X4**1.5 * X5**1.3
    term2 = X1**0.2 * X3**1.1 * X4**1.7 * X5**1.2
    return term1 + term2

def denom1_simplified(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X2**0.8 * X4**1.6 * X8**0.35 + epsilon

def denom2_simplified(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X2**0.28 * X4**0.53 * X8**0.12 + epsilon

def denom3_simplified(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X2**0.29 * X4**0.56 * X8**0.12 + epsilon

def denom4_simplified(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X2**0.18 * X4**0.34 * X8**0.07 + epsilon

def Y1_simplified(X):
    A = expr_A_simplified(X)
    d1 = denom1_simplified(X)
    d2 = denom2_simplified(X)
    return (0.145 * A**0.8 / d1) - (0.075 * A**0.27 / d2) + 5.43

def Y2_simplified(X):
    A = expr_A_simplified(X)
    d3 = denom3_simplified(X)
    d4 = denom4_simplified(X)
    return (-0.007 * A**0.28 / d3) + (0.16 * A**0.17 / d4) + 4.78

# --- Original (Recovered) Equations ---
def expr_A_orig(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    t1 = 0.722*X1**0.059*X2**0.855*X3**0.104*X4**1.431*X5**1.326*X6**0.041*X7**0.093*X8**0.373
    t2 = X1**0.192*X2**0.449*X3**1.14*X4**1.778*X5**1.209*X6**0.011*X7**0.128*X8**0.485
    t3 = 0.656*X1**0.193*X2**1.34*X3**0.876*X4**0.98*X5**1.58*X6**0.041*X7**0.091*X8**0.552
    t4 = 0.707*X2**0.772*X3**0.057*X4**1.787*X5**0.769*X6**0.05*X7**0.451*X8**0.06
    return t1 + t2 + t3 + t4

def denom1_orig(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X1**0.037*X2**0.833*X3**0.129*X4**1.6*X8**0.348 + epsilon

def denom2_orig(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X1**0.0124*X2**0.279*X3**0.0433*X4**0.5358*X8**0.116 + epsilon

def denom3_orig(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X1**0.0131*X2**0.295*X3**0.0457*X4**0.5657*X8**0.123 + epsilon

def denom4_orig(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    return X1**0.00796*X2**0.18*X3**0.0279*X4**0.3446*X8**0.0749 + epsilon

def Y1_orig(X):
    A = expr_A_orig(X)
    d1 = denom1_orig(X)
    d2 = denom2_orig(X)
    return 0.156*A**0.803/d1 - 0.0794*A**0.269/d2 + 5.432

def Y2_orig(X):
    A = expr_A_orig(X)
    d3 = denom3_orig(X)
    d4 = denom4_orig(X)
    return -0.00723*A**0.284/d3 + 0.166*A**0.173/d4 + 4.777

# --- Compute predictions (on scaled X) ---
Y1_pred_simp = Y1_simplified(X)
Y2_pred_simp = Y2_simplified(X)
Y1_pred_orig = Y1_orig(X)
Y2_pred_orig = Y2_orig(X)

# --- Compute metrics (on scaled Y) ---
def metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100  # percent
    }

results = {
    'Y1_Original': metrics(Y1_true, Y1_pred_orig),
    'Y1_Simplified': metrics(Y1_true, Y1_pred_simp),
    'Y2_Original': metrics(Y2_true, Y2_pred_orig),
    'Y2_Simplified': metrics(Y2_true, Y2_pred_simp),
}

# --- Print results as a table ---
print("\nPerformance Comparison Table (Scaled Data):\n")
print(f"{'Output':<15}{'Equation':<15}{'MSE':<12}{'MAPE (%)':<12}")
print('-'*54)
for key, val in results.items():
    output, eqn = key.split('_')
    print(f"{output:<15}{eqn:<15}{val['MSE']:<12.6f}{val['MAPE']:<12.2f}")

# --- Debug: Compare symbolic implementation to printed equation for first 5 rows ---
def nn_printed_equation_output(X):
    X1, X2, X3, X4, X5, X6, X7, X8 = X.T
    expr = (
        0.722*X1**0.059*X2**0.855*X3**0.104*X4**1.431*X5**1.326*X6**0.041*X7**0.093*X8**0.373
        + X1**0.192*X2**0.449*X3**1.14*X4**1.778*X5**1.209*X6**0.011*X7**0.128*X8**0.485
        + 0.656*X1**0.193*X2**1.34*X3**0.876*X4**0.98*X5**1.58*X6**0.041*X7**0.091*X8**0.552
        + 0.707*X2**0.772*X3**0.057*X4**1.787*X5**0.769*X6**0.05*X7**0.451*X8**0.06
    )
    d1 = X1**0.037*X2**0.833*X3**0.129*X4**1.6*X8**0.348 + epsilon
    d2 = X1**0.0124*X2**0.279*X3**0.0433*X4**0.5358*X8**0.116 + epsilon
    return 0.156*expr**0.803/d1 - 0.0794*expr**0.269/d2 + 5.432

print("\n--- DEBUG: Compare symbolic Y1_orig to printed equation for first 5 rows (scaled) ---")
for i in range(5):
    print(f"Row {i}")
    print(f"X (scaled): {X[i]}")
    y1_our_impl = Y1_orig(X[i:i+1])[0]
    y1_printed = nn_printed_equation_output(X[i:i+1])[0]
    print(f"Y1_orig (our impl): {y1_our_impl}")
    print(f"Y1_orig (printed eq): {y1_printed}")
    print(f"Difference: {abs(y1_our_impl - y1_printed)}\n") 