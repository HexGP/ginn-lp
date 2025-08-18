#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GINN multi-output training + equation extraction + periodic “equation sync”
(one-file runner; no scaling; Savitzky–Golay smoothing + positivity clamp)

What this does:
  • Builds/uses your GINN multi-output model (shared PTA layer + 2 heads)
  • Trains with multitask loss (two targets)
  • Every N epochs:
      - Extracts SymPy equations for y1,y2 (flattens weird nested returns)
      - Evaluates them safely (Laurent terms, no zeros/negatives explode)
      - Optionally refits ONLY numeric constants in the printed equations
        to better match your (smoothed) data (structure/exponents fixed)
      - Reports R²/MAE/RMSE and faithfulness R²(model ↔ equation)
  • No scaling; uses Savitzky–Golay smoothing + min-positive clamp.

Requirements (pip):
  numpy pandas sympy scikit-learn scipy tensorflow
  ginnlp  (your fork / repo providing eql_model_v3_multioutput, eql_opt, get_multioutput_sympy_expr*)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# --- ML / math ---
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter

# --- TF / GINN ---
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping

# IMPORTANT: adjust these imports to your local GINN package path
from ginnlp.de_learn_network import (
    eql_model_v3_multioutput,
    eql_opt,
    get_multioutput_sympy_expr,     # wrapper that forwards to v2 in your code
    # get_multioutput_sympy_expr_v2  # if you prefer to call v2 explicitly
)

# --- BEGIN: Fix the GINN regularizer to always return tensors ---
import tensorflow as tf
from ginnlp import de_learn_network as dln

class _FixedL1L2(dln.L1L2_m):
    """Fixed version of L1L2_m that always returns tensors"""
    def __call__(self, x):
        regularization = tf.constant(0.0, dtype=tf.float32)  # Always a tensor
        if self.val_l1 > 0.:
            regularization += tf.reduce_sum(self.l1 * tf.abs(x))
        if self.val_int > 0:
            regularization += tf.reduce_sum(self.val_int * tf.abs(x - tf.round(x)))
        if self.val_l2 > 0.:
            regularization += tf.reduce_sum(self.l2 * tf.square(x))
        return regularization

# Replace the problematic regularizer
dln.L1L2_m = _FixedL1L2
# --- END: Fix ---

# =============== USER CONFIG ===============
DATA_CSV = "data/ENB2012_data.csv"   # <--- change to your dataset file
# If you know exact target col names, set them here (otherwise auto-detect below).
TARGET_COLS = None                    # e.g. ["Y1","Y2"] or leave None to auto-detect
K_FOLDS = 5
VALIDATE_EVERY = 100                  # equation sync every N epochs
ROUND_DIGITS = 3
RIDGE_LAMBDA = 1e-6                   # small ridge in coefficient refit
MIN_POSITIVE = 1e-2                   # clamp after smoothing to avoid zeros/negatives
EPS_LAURENT = 1e-12

# GINN architecture (your description)
LN_BLOCKS_SHARED = (4, 4)             # 1 shared layer with 4 PTA blocks
LIN_BLOCKS_SHARED = (1, 1)            # must match per GINN builder
OUTPUT_LN_BLOCKS = 4                  # you said “their own four” per head; set 4
L1 = 1e-3; L2 = 1e-3
OUT_L1 = 0.2; OUT_L2 = 0.1
INIT_LR = 1e-2; DECAY_STEPS = 1000
BATCH_SIZE = 32
EPOCHS = 10000
VAL_SPLIT = 0.2
PATIENCE = 100
TASK_WEIGHTS = [0.5, 0.5]             # adjust weighting if one target is more important
# ==========================================


# ---------- Data utilities ----------
def detect_features_and_targets(df, override=None):
    if override:
        tcols = override
    else:
        # try patterns, else last two columns
        patterns = [
            lambda c: c.lower().startswith("target"),
            lambda c: c.upper().startswith("Y"),
            lambda c: c.lower().startswith("output"),
            lambda c: c.lower().startswith("label"),
            lambda c: c.lower().startswith("class"),
            lambda c: c.lower().startswith("result"),
        ]
        tcols = []
        for p in patterns:
            tcols = [c for c in df.columns if p(c)]
            if tcols:
                break
        if not tcols and len(df.columns) >= 2:
            tcols = df.columns[-2:].tolist()
    fcols = [c for c in df.columns if c not in tcols]
    return fcols, tcols


def savgol_positive(X, window_length=15, polyorder=3, min_positive=MIN_POSITIVE):
    """
    Savitzky–Golay smoothing; then clamp to strictly positive floor.
    Works for both features (X) and targets (Y).
    """
    arr = np.asarray(X, dtype=float).copy()
    n, d = arr.shape
    wl = max(3, min(window_length, (n // 2) * 2 + 1))  # must be odd, <= n and >=3
    for j in range(d):
        if n >= wl:
            arr[:, j] = savgol_filter(arr[:, j], wl, polyorder)
    # clamp: (i) avoid true zeros, (ii) avoid negatives for Laurent stability
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.sign(arr) * np.maximum(np.abs(arr), EPS_LAURENT)   # avoid exact 0
    arr = np.maximum(arr, min_positive)                         # enforce positive domain
    return arr


# ---------- SymPy helpers (robust evaluation + coeff refit) ----------
def _to_expr(x):
    # peel nested containers like [(expr, meta)], [[expr]], (expr, meta), etc.
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    # sympy Matrix → scalar expr if 1x1
    if isinstance(x, sp.Matrix):
        if x.shape == (1, 1):
            x = x[0]
        else:
            x = sp.simplify(sp.Matrix(x))  # leave as Matrix if truly vector/matrix
    if isinstance(x, sp.Expr):
        return x
    if isinstance(x, str):
        return parse_expr(x)
    return sp.sympify(x)

def normalize_expr_list(power_equations):
    """
    Accepts whatever get_multioutput_sympy_expr returns, produces a flat [Expr,...].
    """
    if isinstance(power_equations, tuple) and len(power_equations) >= 1:
        exprs = power_equations[0]
        syms = power_equations[1] if len(power_equations) > 1 else None
    else:
        exprs, syms = power_equations, None

    # force list
    exprs = exprs if isinstance(exprs, (list, tuple)) else [exprs]
    expr_list = [_to_expr(e) for e in exprs]
    # assert we ended with sympy Expr objects
    for k, e in enumerate(expr_list):
        if not isinstance(e, sp.Expr):
            raise TypeError(f"Output {k} is not a SymPy Expr: {type(e)}")
    return expr_list, syms

# ---------- Robust evaluation helpers (drop-in) ----------
def _coerce_sympy_scalar(expr):
    """
    Ensure a SymPy object is a scalar Expr (not Matrix). If it's a 1x1 Matrix, unwrap.
    If it's a vector/matrix with >1 element, raise to force us to fix the extractor.
    """
    if isinstance(expr, sp.MatrixBase):
        if expr.shape == (1, 1):
            return sp.simplify(expr[0, 0])
        raise ValueError(f"Expression evaluated to non-scalar Matrix with shape {expr.shape}: {expr}")
    return expr

def _to_float_vector_anyshape(val, n_rows):
    """
    Accept scalar, 0-D, (n,), (n,1), (1,n) or 1x1 Matrix results
    and return a contiguous float vector of length n_rows.
    """
    # Unwrap SymPy Matrix(1,1) -> scalar if it's still sympy-ish:
    if isinstance(val, sp.MatrixBase):
        if val.shape == (1, 1):
            val = float(val[0, 0])
        else:
            raise ValueError(f"Non-scalar Matrix result: shape={val.shape}")

    arr = np.asarray(val)

    # Handle scalar / 0-D
    if arr.shape == () or arr.ndim == 0:
        return np.full((n_rows,), float(arr))

    # Squeeze simple column/row vectors
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)

    # At this point we expect a 1-D vector. Broadcast if length == 1
    if arr.ndim == 1:
        if arr.size == 1:
            return np.full((n_rows,), float(arr[0]))
        if arr.size == n_rows:
            return arr.astype(float, copy=False)
        # If length mismatches, try last resort broadcast if divisible (shouldn't happen)
        raise ValueError(f"Vector length {arr.size} != n_rows {n_rows}; cannot coerce")
    else:
        raise ValueError(f"Unexpected result ndim={arr.ndim}, shape={arr.shape}")

def make_vector_fn_debug(exprs, n_features, symbols=None, eps=1e-12, log_every_expr=True):
    """
    Robust evaluator:
      * NO SymPy Max/Abs/sign inside expressions (keep tree pure)
      * Clamp inputs in NumPy just before evaluation
      * Normalize shapes for stacking
    """
    if symbols is None:
        symbols = sp.symbols([f"X_{i+1}" for i in range(n_features)])

    clean_exprs = []
    for idx, e in enumerate(exprs):
        e = _coerce_sympy_scalar(e)
        e = sp.simplify(e)    # keep pure
        if log_every_expr:
            print(f"[EvalPrep] expr[{idx}] -> {type(e).__name__} | {str(e)[:120]}")
        clean_exprs.append(e)

    # Plain NumPy backend; no custom mappings needed
    fns = [sp.lambdify(symbols, e, modules="numpy") for e in clean_exprs]

    def f(X):
        n = X.shape[0]

        # ---- Clamp inputs here (NumPy) ----
        cols = []
        for i in range(n_features):
            v = X[:, i]
            v = np.where(np.isfinite(v), v, 0.0)
            v = np.sign(v) * np.maximum(np.abs(v), eps)  # avoid exact zeros for Laurent terms
            v = np.maximum(v, MIN_POSITIVE)               # your MIN_POSITIVE
            cols.append(v)
        # -----------------------------------

        outs = []
        for idx, fn in enumerate(fns):
            raw = fn(*cols)
            vec = _to_float_vector_anyshape(raw, n)
            if log_every_expr and idx < 3:
                print(f"[EvalRun] expr[{idx}] type={type(raw).__name__}, coerced={vec.shape}, preview={vec[:3]}")
            outs.append(vec)
        return np.column_stack(outs)

    return f

# Keep the old function for backward compatibility
def make_vector_fn(exprs, n_features, symbols=None):
    """
    Returns f(X[n,nf]) -> Y[n,no] for list of expressions.
    Guarantees each output is 1-D float array of length n.
    """
    if symbols is None:
        symbols = sp.symbols([f"X_{i+1}" for i in range(n_features)])

    # Prepare lambdas once
    exprs = [sp.simplify(e) for e in exprs]
    fns = [sp.lambdify(symbols, e, "numpy") for e in exprs]

    def _to_1d_float(y, n):
        """
        Coerce any scalar / (n,) / (n,1) / list / Matrix into a 1-D float ndarray of length n.
        """
        # SymPy Matrix?
        if isinstance(y, sp.Matrix):
            y = np.asarray(y.tolist(), dtype=float)
        else:
            y = np.asarray(y, dtype=float)

        if y.ndim == 0:
            # broadcast scalar to length n
            y = np.full(n, float(y))
        elif y.ndim == 1:
            if y.shape[0] != n:
                # rare case: wrong length term; try to broadcast if size 1
                if y.shape[0] == 1:
                    y = np.full(n, float(y[0]))
                else:
                    raise ValueError(f"Expected length {n}, got {y.shape[0]}")
        else:
            # squeeze (n,1) → (n,), (1,n) → (n,)
            y = np.squeeze(y)
            if y.ndim == 0:
                y = np.full(n, float(y))
            elif y.ndim == 1 and y.shape[0] != n:
                # if became (1,), broadcast
                if y.shape[0] == 1:
                    y = np.full(n, float(y[0]))
                else:
                    raise ValueError(f"Squeezed shape mismatch: {y.shape}")
            elif y.ndim > 1:
                # last resort: flatten and check
                y = y.reshape(-1)
                if y.shape[0] != n:
                    raise ValueError(f"Flatten shape mismatch: {y.shape[0]} != {n}")

        # ensure finite
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return y.astype(float, copy=False)

    def f(X):
        # X: (n, n_features)
        n = X.shape[0]
        cols = [X[:, i] for i in range(n_features)]
        outs = []
        for fn in fns:
            yi = fn(*cols)           # could be scalar, list, (n,), (n,1), Matrix, etc.
            yi = _to_1d_float(yi, n) # force to (n,)
            outs.append(yi)
        return np.column_stack(outs)  # (n, no)

    return f

def decompose_monomials(expr):
    """
    Expand into additive terms; for each term t = coeff * monomial, return (coeff_number, monomial_expr).
    """
    terms = sp.Add.make_args(sp.expand(expr))
    pairs = []
    for t in terms:
        c, m = t.as_coeff_Mul()
        if not c.free_symbols and m != 1:
            c2, m2 = sp.factor(m).as_coeff_Mul()
            c = sp.simplify(c * c2)
            m = m2
        pairs.append((sp.N(c), sp.simplify(m)))
    return pairs

def _coerce_len_n(val, n):
    """Coerce lambdify output to shape (n,)."""
    arr = np.asarray(val)
    if arr.shape == () or arr.ndim == 0:
        return np.full((n,), float(arr))
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1).astype(float, copy=False)
    if arr.ndim == 1 and arr.size == n:
        return arr.astype(float, copy=False)
    raise ValueError(f"Unexpected basis shape {arr.shape}")

def _numeric_clamp_columns(X, eps=1e-12, min_pos=1e-2):
    """Same clamp you used in the evaluator: avoid exact zero & enforce positivity."""
    cols = []
    for i in range(X.shape[1]):
        v = X[:, i]
        v = np.where(np.isfinite(v), v, 0.0)
        v = np.sign(v) * np.maximum(np.abs(v), eps)  # avoid zeros in Laurent terms
        v = np.maximum(v, min_pos)                  # global floor for positivity
        cols.append(v)
    return cols

def refit_coeffs_single(expr, symbols, X, y, ridge=RIDGE_LAMBDA):
    """
    Keep structure fixed; refit only numeric coefficients via ridge LS.
    NO symbolic clamp inside expressions; inputs are clamped numerically.
    """
    # 1) Decompose expression into additive terms
    pairs = decompose_monomials(expr)  # [(coeff, monomial_expr), ...]
    if len(pairs) == 0:
        return sp.simplify(expr)

    # 2) Prepare numeric-safe input columns
    n = X.shape[0]
    Xcols = _numeric_clamp_columns(X, eps=EPS_LAURENT, min_pos=MIN_POSITIVE)

    # 3) Build design matrix from monomials
    monoms = [m for _, m in pairs]
    basis_cols = []
    for m in monoms:
        if isinstance(m, (int, float)) or (isinstance(m, sp.Expr) and m == 1):
            col = np.ones(n, dtype=float)
        else:
            m_simpl = sp.simplify(m)
            fn = sp.lambdify(symbols, m_simpl, modules="numpy")
            val = fn(*Xcols)  # evaluate with numeric-clamped inputs
            col = _coerce_len_n(val, n)
        basis_cols.append(col)

    M = np.column_stack(basis_cols)   # shape (n, k)
    b = y.reshape(-1)                 # shape (n,)

    # 4) Ridge solve
    MTM = M.T @ M
    A = MTM + ridge * np.eye(MTM.shape[0])
    rhs = M.T @ b
    try:
        coeffs = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    # 5) Reassemble expression with refit coefficients
    new_expr = sum(sp.Float(coeffs[k]) * monoms[k] for k in range(len(monoms)))
    return sp.simplify(new_expr)

def refit_coeffs_multi(exprs, n_features, X, Y, symbols=None, ridge=RIDGE_LAMBDA):
    if symbols is None:
        symbols = sp.symbols([f"X_{i+1}" for i in range(n_features)])
    out = []
    for j, e in enumerate(exprs):
        out.append(refit_coeffs_single(e, symbols, X, Y[:, j], ridge=ridge))
    return out

# ---------- Extraction faithfulness check ----------
def assert_head_layers_exist(model, head_idx, out_ln_blocks):
    expected = [f"out{head_idx}_ln_{i}" for i in range(out_ln_blocks)]
    expected += [f"out{head_idx}_ln_dense", f"output_{head_idx}"]
    missing = []
    for name in expected:
        try:
            model.get_layer(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(f"[Extractor Mismatch] Missing layers for head {head_idx}: {missing}")

def check_extraction_faithfulness(model, X_train_s, Y_train_s, num_features, output_ln_blocks, get_expr_fn):
    # 1) ensure both heads' layer names match what the extractor expects
    for h in range(Y_train_s.shape[1]):
        assert_head_layers_exist(model, h, output_ln_blocks)

    # 2) model predictions (ground truth for the extractor test)
    y_pred_list = model.predict([X_train_s[:, i].reshape(-1, 1) for i in range(num_features)], verbose=0)
    Yhat_model = np.column_stack(y_pred_list)

    # 3) extract + evaluate
    exprs, maybe_syms = normalize_expr_list(get_expr_fn(model, num_features, output_ln_blocks, round_digits=3))
    symbols = maybe_syms if isinstance(maybe_syms, (list, tuple)) and len(maybe_syms) else sp.symbols([f"X_{i+1}" for i in range(num_features)])
    f_vec = make_vector_fn_debug(exprs, num_features, symbols=symbols, eps=EPS_LAURENT, log_every_expr=False)
    Yhat_eq = f_vec(X_train_s)

    # 4) compare model vs equation (the key check)
    for j in range(Y_train_s.shape[1]):
        r2_m_eq = r2_score(Yhat_model[:, j], Yhat_eq[:, j])
        r2_t_eq = r2_score(Y_train_s[:, j], Yhat_eq[:, j])
        print(f"[Sanity] Target {j}: R²(model↔eq)={r2_m_eq:.6f}  R²(truth↔eq)={r2_t_eq:.6f}")
    return exprs


# ---------- Callback: periodic equation sync ----------
class EquationSyncCallback(Callback):
    def __init__(self, X_train, Y_train, num_features, output_ln_blocks,
                 validate_every=VALIDATE_EVERY, round_digits=ROUND_DIGITS,
                 ridge_lambda=RIDGE_LAMBDA, min_log=True):
        super().__init__()
        self.Xt = X_train
        self.Yt = Y_train
        self.nf = num_features
        self.out_ln = output_ln_blocks
        self.validate_every = validate_every
        self.round_digits = round_digits
        self.ridge = ridge_lambda
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch % self.validate_every != 0):
            return

        try:
            # 1) Extract equations (power form)
            power_equations = get_multioutput_sympy_expr(self.model, self.nf, self.out_ln, round_digits=self.round_digits)
            exprs, maybe_syms = normalize_expr_list(power_equations)
            symbols = maybe_syms if isinstance(maybe_syms, (list, tuple)) and len(maybe_syms) else sp.symbols([f"X_{i+1}" for i in range(self.nf)])

            # 2) Evaluate equations on TRAIN
            f_vec = make_vector_fn_debug(exprs, self.nf, symbols=symbols, eps=EPS_LAURENT, log_every_expr=True)
            try:
                y_eq = f_vec(self.Xt)  # [n,2]
                print(f"[EqSync] Equation evaluation successful: y_eq shape = {y_eq.shape}")
            except Exception as e:
                print(f"[EqSync] Equation evaluation failed: {e}")
                print(f"[EqSync] Expression types: {[type(e).__name__ for e in exprs]}")
                print(f"[EqSync] Expression shapes: {[str(e)[:100] for e in exprs]}")
                raise

            # 3) Model predictions on TRAIN (same split the model sees)
            y_pred_list = self.model.predict([self.Xt[:, i].reshape(-1, 1) for i in range(self.nf)], verbose=0)
            y_pred = np.column_stack(y_pred_list)

            # 4) Metrics (pre-refit)
            r2_eq_truth = [r2_score(self.Yt[:, j], y_eq[:, j]) for j in range(self.Yt.shape[1])]
            r2_model_truth = [r2_score(self.Yt[:, j], y_pred[:, j]) for j in range(self.Yt.shape[1])]
            r2_model_eq = [r2_score(y_pred[:, j], y_eq[:, j]) for j in range(self.Yt.shape[1])]

            # 5) Refit constants (OPTIONAL but recommended)
            exprs_refit = refit_coeffs_multi(exprs, self.nf, self.Xt, self.Yt, symbols=symbols, ridge=self.ridge)
            f_vec_refit = make_vector_fn_debug(exprs_refit, self.nf, symbols=symbols, eps=EPS_LAURENT, log_every_expr=True)
            try:
                y_eq_refit = f_vec_refit(self.Xt)
                print(f"[EqSync] Refit evaluation successful: y_eq_refit shape = {y_eq_refit.shape}")
            except Exception as e:
                print(f"[EqSync] Refit evaluation failed: {e}")
                print(f"[EqSync] Refit expression types: {[type(e).__name__ for e in exprs_refit]}")
                raise

            r2_eq_truth_refit = [r2_score(self.Yt[:, j], y_eq_refit[:, j]) for j in range(self.Yt.shape[1])]
            r2_model_eq_refit = [r2_score(y_pred[:, j], y_eq_refit[:, j]) for j in range(self.Yt.shape[1])]

            msg = (
                f"[EqSync @ epoch {epoch}] "
                f"R2(eq→truth): {r2_eq_truth}  "
                f"R2(eq_refit→truth): {r2_eq_truth_refit}  "
                f"R2(model↔eq): {r2_model_eq}  "
                f"R2(model↔eq_refit): {r2_model_eq_refit}"
            )
            print(msg)

            self.history.append({
                "epoch": epoch,
                "r2_eq_truth": r2_eq_truth,
                "r2_model_truth": r2_model_truth,
                "r2_model_eq": r2_model_eq,
                "r2_eq_truth_refit": r2_eq_truth_refit,
                "r2_model_eq_refit": r2_model_eq_refit,
                "exprs": [str(e) for e in exprs],
                "exprs_refit": [str(e) for e in exprs_refit],
            })

            # NOTE: Writing refit constants back into Keras layers is model-specific.
            # If your top head is strictly linear over PTA block outputs, you can
            # optionally add a ridge-refit of head weights and set_weights here.
            # (Left out by default to avoid mismapping; equations are still logged.)
        except Exception as e:
            print(f"[EqSync @ epoch {epoch}] Equation sync failed: {e}")


# ---------- Loss (weighted multitask MSE with mild regularization on outputs) ----------
def faithfulness_aware_loss(task_weights, faithfulness_weight=0.0):
    def loss(y_true, y_pred):
        # Simple approach: just use standard MSE loss for now
        # This will help us isolate if the issue is in our custom loss function
        if isinstance(y_true, (list, tuple)) and isinstance(y_pred, (list, tuple)):
            # Handle list of tensors case
            total_loss = tf.constant(0.0, dtype=tf.float32)
            for i in range(len(task_weights)):
                mse_loss = tf.keras.losses.mse(y_true[i], y_pred[i])
                total_loss += task_weights[i] * mse_loss
            return total_loss
        else:
            # Fallback to standard MSE
            return tf.keras.losses.mse(y_true, y_pred)
    return loss


# ================== MAIN ==================
def main():
    # 1) Load data
    df = pd.read_csv(DATA_CSV)
    feature_cols, target_cols = detect_features_and_targets(df, override=TARGET_COLS)
    X_raw = df[feature_cols].values.astype(np.float32)
    Y_raw = df[target_cols].values.astype(np.float32)
    num_features = len(feature_cols)
    num_outputs = Y_raw.shape[1]
    print(f"Features: {feature_cols}")
    print(f"Targets:  {target_cols}  (num_outputs={num_outputs})")

    # 2) K-fold CV
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    all_results = []

    for fold, (tr, te) in enumerate(kf.split(X_raw), 1):
        print("\n" + "="*70)
        print(f"FOLD {fold}/{K_FOLDS}   Train={len(tr)}  Test={len(te)}")
        print("="*70)

        X_train, X_test = X_raw[tr], X_raw[te]
        Y_train, Y_test = Y_raw[tr], Y_raw[te]

        # 3) Smoothing (Savitzky–Golay) + positivity clamp; NO SCALING
        X_train_s = savgol_positive(X_train)
        Y_train_s = savgol_positive(Y_train)
        X_test_s  = savgol_positive(X_test)
        Y_test_s  = savgol_positive(Y_test)

        # 4) Build GINN model (use it directly; no wrapper)
        opt = eql_opt(decay_steps=DECAY_STEPS, init_lr=INIT_LR)
        model = eql_model_v3_multioutput(
            input_size=num_features,
            opt=opt,
            ln_blocks=LN_BLOCKS_SHARED,
            lin_blocks=LIN_BLOCKS_SHARED,
            output_ln_blocks=OUTPUT_LN_BLOCKS,
            num_outputs=num_outputs,
            compile=False,
            # use zeros, not None (your L1L2_m wraps these in tf.Variable)
            l1_reg=0.0, l2_reg=0.0,
            output_l1_reg=0.0, output_l2_reg=0.0,
        )

        # 5) Callbacks
        eqsync = EquationSyncCallback(
            X_train=X_train_s,
            Y_train=Y_train_s,
            num_features=num_features,
            output_ln_blocks=OUTPUT_LN_BLOCKS,
            validate_every=VALIDATE_EVERY,
            round_digits=ROUND_DIGITS,
            ridge_lambda=RIDGE_LAMBDA
        )
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)

        # 6) Compile (multi-output: one loss per head + weights)
        model.compile(
            optimizer=opt,
            loss=['mse'] * num_outputs,          # ['mse','mse']
            loss_weights=TASK_WEIGHTS,           # e.g. [0.5, 0.5]
            # metrics=['mse']  # Simplified: single metric for all outputs
            # run_eagerly=False  # leave default; flip to True only if you need step-by-step debug
        )

        # Quick dry run to confirm forward works
        _ = model.predict([X_train_s[:5, i].reshape(-1, 1) for i in range(num_features)], verbose=0)

        # Extra guard: check for additional losses (should now be tensors)
        if model.losses:
            print(f"DEBUG: Found {len(model.losses)} additional losses")
            print(f"DEBUG: First few loss types: {[type(x).__name__ for x in model.losses[:5]]}")
            if len(model.losses) > 5:
                print(f"DEBUG: ... and {len(model.losses) - 5} more")
        else:
            print("DEBUG: No additional losses found (good!)")

        # Model configuration check
        print(f"DEBUG: Model inputs: {len(model.inputs)} inputs with shapes: {[inp.shape for inp in model.inputs]}")
        print(f"DEBUG: Model outputs: {len(model.outputs)} outputs with shapes: {[out.shape for out in model.outputs]}")
        print(f"DEBUG: Model loss function: {model.loss}")
        print(f"DEBUG: Model optimizer: {model.optimizer}")
        print(f"DEBUG: Model is compiled: {model.compiled}")

        # 7) Train
        print(f"DEBUG: X_train_s shape: {X_train_s.shape}")
        print(f"DEBUG: Y_train_s shape: {Y_train_s.shape}")
        print(f"DEBUG: X_train_s[:, 0].reshape(-1, 1) shape: {X_train_s[:, 0].reshape(-1, 1).shape}")
        print(f"DEBUG: Y_train_s[:, 0].reshape(-1, 1) shape: {Y_train_s[:, 0].reshape(-1, 1).shape}")
        
        print("DEBUG: Starting training...")
        try:
            history = model.fit(
                [X_train_s[:, i].reshape(-1, 1) for i in range(num_features)],
                [Y_train_s[:, i].reshape(-1, 1) for i in range(num_outputs)],
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VAL_SPLIT,
                verbose=1,
                callbacks=[es, eqsync]
            )
            print("DEBUG: Training completed successfully!")
            
            # Check extraction faithfulness on training data
            print("\n" + "="*50)
            print("CHECKING EXTRACTION FAITHFULNESS")
            print("="*50)
            try:
                exprs_train = check_extraction_faithfulness(
                    model=model,
                    X_train_s=X_train_s,
                    Y_train_s=Y_train_s,
                    num_features=num_features,
                    output_ln_blocks=OUTPUT_LN_BLOCKS,
                    get_expr_fn=get_multioutput_sympy_expr
                )
                print("✅ Extraction faithfulness check completed")
            except Exception as e:
                print(f"❌ Extraction faithfulness check failed: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"DEBUG: Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 8) Inference
        nn_pred_list = model.predict([X_test_s[:, i].reshape(-1, 1) for i in range(num_features)], verbose=0)
        Yhat_nn = np.column_stack(nn_pred_list)

        # 9) Extract equations (final), evaluate (raw + refit)
        try:
            power_equations = get_multioutput_sympy_expr(model, num_features, OUTPUT_LN_BLOCKS, round_digits=ROUND_DIGITS)
            exprs, maybe_syms = normalize_expr_list(power_equations)
            symbols = (maybe_syms if isinstance(maybe_syms, (list, tuple)) and len(maybe_syms)
                       else sp.symbols([f"X_{i+1}" for i in range(num_features)]))
            f_vec = make_vector_fn_debug(exprs, num_features, symbols=symbols, eps=EPS_LAURENT, log_every_expr=True)
            Yhat_eq = f_vec(X_test_s)

            exprs_refit = refit_coeffs_multi(exprs, num_features, X_train_s, Y_train_s, symbols=symbols, ridge=RIDGE_LAMBDA)
            f_vec_refit = make_vector_fn_debug(exprs_refit, num_features, symbols=symbols, eps=EPS_LAURENT, log_every_expr=True)
            Yhat_eq_refit = f_vec_refit(X_test_s)
        except Exception as e:
            print(f"[Fold {fold}] Equation evaluation failed: {e}")
            Yhat_eq = Yhat_nn.copy()
            Yhat_eq_refit = Yhat_nn.copy()
            exprs, exprs_refit = ["<n/a>","<n/a>"], ["<n/a>","<n/a>"]

        # 10) Metrics
        def metrics(y, yhat):
            return dict(
                R2=float(r2_score(y, yhat)),
                MAE=float(mean_absolute_error(y, yhat)),
                RMSE=float(np.sqrt(mean_squared_error(y, yhat)))
            )
        per_target = []
        for j in range(num_outputs):
            m_nn = metrics(Y_test_s[:, j], Yhat_nn[:, j])
            m_eq = metrics(Y_test_s[:, j], Yhat_eq[:, j])
            m_eqr= metrics(Y_test_s[:, j], Yhat_eq_refit[:, j])
            r2_nn_eq  = float(r2_score(Yhat_nn[:, j], Yhat_eq[:, j]))
            r2_nn_eqr = float(r2_score(Yhat_nn[:, j], Yhat_eq_refit[:, j]))
            per_target.append(dict(
                target=target_cols[j],
                model=m_nn, eq=m_eq, eq_refit=m_eqr,
                R2_model_eq=r2_nn_eq, R2_model_eq_refit=r2_nn_eqr,
                expr=str(exprs[j]) if j < len(exprs) else "<n/a>",
                expr_refit=str(exprs_refit[j]) if j < len(exprs_refit) else "<n/a>",
            ))

        all_results.append(dict(fold=fold, per_target=per_target))
        # Pretty print
        print("\nResults (TEST):")
        for r in per_target:
            print(f" - {r['target']}: "
                  f"R2(model→truth)={r['model']['R2']:.4f} | "
                  f"R2(eq→truth)={r['eq']['R2']:.4f} | "
                  f"R2(eq_refit→truth)={r['eq_refit']['R2']:.4f} | "
                  f"R2(model↔eq)={r['R2_model_eq']:.4f} | "
                  f"R2(model↔eq_refit)={r['R2_model_eq_refit']:.4f}")

    # 11) (Optional) Save results
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/ginn_multitask_eqsync_results.json"
    import json
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
