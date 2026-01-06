#!/usr/bin/env python3
"""
Video Games JP_Sales: CatBoost + Ridge + Optuna Hyperparameter Tuning
Скрипт для запуска на мощном сервере

Использование:
    python train_optuna.py                    # Полный цикл с Optuna
    python train_optuna.py --skip-optuna      # Только базовая модель (без Optuna)
    python train_optuna.py --trials 500       # Указать число trials
    python train_optuna.py --timeout 3600     # Таймаут в секундах
"""

import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================
RANDOM_STATE = 42
TRAIN_PATH = "Video_Games.csv"
TEST_PATH = "Video_Games_Test.csv"
TARGET = "JP_Sales"

# Optuna settings
BEST_PARAMS_FILE = "optuna_best_params.json"
STUDY_NAME = "catboost_jp_sales"
N_TRIALS_DEFAULT = 2000
SAVE_INTERVAL_SEC = 600  # 10 минут
TIMEOUT_DEFAULT = 60 * 60 * 8  # 8 часов

# Output directories
OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "optuna_plots")


# =========================
# UTILITY FUNCTIONS
# =========================
def rmse(y_true, y_pred):
    """Calculate RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def find_id_col(df):
    """Find ID column in DataFrame."""
    for c in df.columns:
        if str(c).strip().lower() == "id":
            return c
    return None


def make_ohe(min_freq: int):
    """Create OneHotEncoder with compatibility for different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_freq, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_freq, sparse=True)


def make_catboost_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for CatBoost (string categories)."""
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) in ("object", "category", "string", "bool"):
            out[c] = out[c].astype("string").fillna("__MISSING__")
    return out


# =========================
# NAME FEATURE ENGINEERING
# =========================
EDITION_RE = re.compile(
    r"\b(remaster(ed)?|hd|definitive|ultimate|complete|collector'?s|"
    r"game of the year|goty|gold|deluxe|premium|special|limited|edition|"
    r"director'?s cut|anniversary|bundle|collection)\b",
    flags=re.IGNORECASE
)

ROMAN_RE = re.compile(
    r"\b(i{1,3}|iv|v|vi{0,3}|ix|x|xi|xii|xiii|xiv|xv)\b",
    flags=re.IGNORECASE
)


def normalize_name(s: pd.Series) -> pd.Series:
    """Normalize game names."""
    s = s.astype("string").fillna("__MISSING__").str.lower()
    s = s.str.replace(r"[™®©]", "", regex=True)
    s = s.str.replace(r"[\(\)\[\]\{\}]", " ", regex=True)
    s = s.str.replace(r"[/:;,\.!\?\|\\]", " ", regex=True)
    s = s.str.replace(r"[-_]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.replace("", "__MISSING__")


def split_base(s: pd.Series) -> pd.Series:
    """Extract base name before subtitle."""
    s2 = s.str.replace(r"\s*:\s*", " : ", regex=True)
    base = s2.str.split(r"\s:\s|\s-\s|\s—\s", n=1, expand=True)[0]
    base = base.str.strip()
    return base.replace("", "__MISSING__")


def franchise_key(s: pd.Series) -> pd.Series:
    """Create franchise key by removing edition words and numbers."""
    s = s.copy()
    s = s.str.replace(EDITION_RE, " ", regex=True)
    s = s.str.replace(ROMAN_RE, " ", regex=True)
    s = s.str.replace(r"\b\d+\b", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.replace("", "__MISSING__")


def add_name_flags(df):
    """Add name-based features."""
    s = df["Name_norm"].astype("string")
    df["name_len"] = s.str.len().fillna(0).astype(int)
    df["name_words"] = s.str.split().str.len().fillna(0).astype(int)
    df["has_colon_or_dash"] = s.str.contains(r"\s:\s|\s-\s|\s—\s", regex=True).astype(int)
    df["has_digit"] = s.str.contains(r"\d").astype(int)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["has_roman"] = s.str.contains(ROMAN_RE, regex=True).astype(int)
        df["has_edition_word"] = s.str.contains(EDITION_RE, regex=True).astype(int)
    
    return df


def add_name_features(X_all):
    """Add all name-based features to DataFrame."""
    if "Name" in X_all.columns:
        X_all["Name_norm"] = normalize_name(X_all["Name"])
        X_all["Name_base"] = split_base(X_all["Name_norm"])
        X_all["Franchise_key"] = franchise_key(X_all["Name_base"])
    
    if "Name_norm" in X_all.columns:
        X_all = add_name_flags(X_all)
    
    return X_all


# =========================
# DATA PREPROCESSING
# =========================
def preprocess_all_data(
    X_all_df: pd.DataFrame,
    text_cols,
    cat_cols,
    num_cols,
    min_freq: int = 50,
    tfidf_max_features: int = 80000,
):
    """Preprocess all data (train + test) together."""
    mats = []

    # 1) NUM
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        num_scaler = StandardScaler(with_mean=False)
        num_data = num_scaler.fit_transform(num_imputer.fit_transform(X_all_df[num_cols]))
        mats.append(sparse.csr_matrix(num_data))
        print(f"  NUM: {len(num_cols)} cols -> {num_data.shape}")

    # 2) CAT
    if len(cat_cols) > 0:
        cat_df = X_all_df[cat_cols].fillna("__MISSING__").astype(str)
        ohe = make_ohe(min_freq)
        cat_ohe = ohe.fit_transform(cat_df)
        mats.append(cat_ohe.tocsr())
        print(f"  CAT: {len(cat_cols)} cols -> OHE shape {cat_ohe.shape}")

    # 3) TEXT (TF-IDF)
    def _flatten_1d(x):
        arr = np.asarray(x).ravel().astype(str)
        return np.where((arr == 'nan') | (arr == 'None') | (arr == '<NA>'), '', arr)

    for c in text_cols:
        text_data = X_all_df[c].fillna("").astype(str).values
        text_data = _flatten_1d(text_data.reshape(-1, 1))
        
        tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=tfidf_max_features,
        )
        tfidf_mat = tfidf.fit_transform(text_data)
        mats.append(tfidf_mat.tocsr())
        print(f"  TEXT '{c}': TF-IDF shape {tfidf_mat.shape}, vocab={len(tfidf.vocabulary_)}")

    # Stack all
    X_proc = sparse.hstack(mats, format="csr")
    print(f"  TOTAL features: {X_proc.shape[1]}")
    return X_proc


# =========================
# CV FUNCTIONS
# =========================
def cv_oof_ridge(X_proc, y_series: pd.Series, cv, alpha: float = 2.0):
    """Cross-validation with Ridge regression."""
    oof = np.zeros(len(y_series), dtype=float)
    fold_scores = []
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_proc, y_series), 1):
        m = Ridge(alpha=alpha)
        m.fit(X_proc[tr_idx], y_series.iloc[tr_idx])
        pred = np.clip(m.predict(X_proc[va_idx]), 0, None)
        oof[va_idx] = pred
        mae = float(mean_absolute_error(y_series.iloc[va_idx], pred))
        r = rmse(y_series.iloc[va_idx], pred)
        fold_scores.append((mae, r))
        print(f"  [Ridge][fold {fold}] MAE={mae:.6f} RMSE={r:.6f}")
    
    maes = np.array([s[0] for s in fold_scores])
    rmses = np.array([s[1] for s in fold_scores])
    print(f"  Ridge: MAE mean={maes.mean():.6f} std={maes.std():.6f} | RMSE mean={rmses.mean():.6f} std={rmses.std():.6f}")
    
    return oof, fold_scores


def cv_oof_catboost(X_df: pd.DataFrame, y_series: pd.Series, cv):
    """Cross-validation with CatBoost."""
    drop_cols = [c for c in ["Name", "Name_root"] if c in X_df.columns]
    X_cb = X_df.drop(columns=drop_cols).reset_index(drop=True)
    X_cb = make_catboost_frame(X_cb)

    cat_cols = X_cb.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    cat_idx = [X_cb.columns.get_loc(c) for c in cat_cols]

    oof = np.zeros(len(y_series), dtype=float)
    fold_scores = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_cb, y_series), 1):
        X_tr, X_va = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
        y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]

        model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=20000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=6.0,
            random_seed=RANDOM_STATE,
            task_type="CPU",
            subsample=0.8,
            rsm=0.8,
            bootstrap_type="Bernoulli",
            verbose=200,
        )

        model.fit(
            X_tr, y_tr,
            cat_features=cat_idx,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=500,
        )

        pred = np.clip(model.predict(X_va), 0, None)
        oof[va_idx] = pred

        mae = float(mean_absolute_error(y_va, pred))
        r = rmse(y_va, pred)
        fold_scores.append((mae, r))
        best_iters.append(int(model.get_best_iteration()))
        print(f"  [CatBoost][fold {fold}] MAE={mae:.6f} RMSE={r:.6f} best_iter={best_iters[-1]}")

    maes = np.array([s[0] for s in fold_scores])
    rmses = np.array([s[1] for s in fold_scores])
    print(f"  CatBoost: MAE mean={maes.mean():.6f} std={maes.std():.6f} | RMSE mean={rmses.mean():.6f} std={rmses.std():.6f}")

    return oof, fold_scores, best_iters


# =========================
# LOAD AND PREPARE DATA
# =========================
def load_data():
    """Load and prepare train/test data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    assert TARGET in train_df.columns, f"'{TARGET}' not found in train"

    y = train_df[TARGET].astype(float)
    X_train_raw = train_df.drop(columns=[TARGET]).copy()
    X_test_raw = test_df.copy()

    # Handle ID columns
    id_col_train = find_id_col(X_train_raw)
    id_col_test = find_id_col(X_test_raw)

    if id_col_test is not None:
        test_ids = X_test_raw[id_col_test].values
    else:
        test_ids = np.arange(1, len(X_test_raw) + 1)

    if id_col_train is not None:
        X_train_raw.drop(columns=[id_col_train], inplace=True)
    if id_col_test is not None:
        X_test_raw.drop(columns=[id_col_test], inplace=True)

    # Combine for joint preprocessing
    X_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

    print(f"  Train rows: {len(X_train_raw)}, Test rows: {len(X_test_raw)}, All rows: {len(X_all)}")
    print(f"  Columns: {list(X_all.columns)}")

    return X_train_raw, X_test_raw, X_all, y, test_ids


def prepare_features(X_all, X_train_raw, X_test_raw, y):
    """Add features and preprocess data."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Add name features
    X_all = add_name_features(X_all)
    
    # Select column types
    text_cols = [c for c in ["Name", "Name_root"] if c in X_all.columns]
    cat_cols_all = X_all.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols_all if c not in set(text_cols)]
    num_cols = [c for c in X_all.columns if c not in set(cat_cols) and c not in set(text_cols)]

    print(f"  text_cols: {text_cols}")
    print(f"  cat_cols: {cat_cols}")
    print(f"  num_cols: {num_cols}")

    # Preprocess
    print("\nPreprocessing...")
    X_all_proc = preprocess_all_data(
        X_all,
        text_cols=text_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        min_freq=50,
        tfidf_max_features=80000,
    )

    n_train = len(X_train_raw)
    X_train_proc = X_all_proc[:n_train]
    X_test_proc = X_all_proc[n_train:]

    print(f"\n  X_train_proc: {X_train_proc.shape}, X_test_proc: {X_test_proc.shape}")

    # Update raw dataframes with name features
    n_train = len(X_train_raw)
    for col in ["Name_norm", "Name_base", "Franchise_key", "name_len", "name_words", 
                "has_colon_or_dash", "has_digit", "has_roman", "has_edition_word"]:
        if col in X_all.columns:
            X_train_raw[col] = X_all[col].iloc[:n_train].values
            X_test_raw[col] = X_all[col].iloc[n_train:].values

    return X_train_proc, X_test_proc, X_train_raw, X_test_raw


# =========================
# BASELINE TRAINING (WITHOUT OPTUNA)
# =========================
def train_baseline(X_train_raw, X_test_raw, X_train_proc, X_test_proc, y, test_ids, best_w=None):
    """Train baseline models and create submission."""
    print("\n" + "=" * 60)
    print("FINAL FIT + PREDICT TEST (BASELINE)")
    print("=" * 60)
    
    # Ridge final on all processed train
    print("\nTraining Ridge...")
    ridge_final = Ridge(alpha=2.0)
    ridge_final.fit(X_train_proc, y)
    pred_ridge = np.clip(ridge_final.predict(X_test_proc), 0, None)

    # CatBoost final
    print("\nTraining CatBoost...")
    drop_cols = [c for c in ["Name", "Name_root"] if c in X_train_raw.columns]
    X_cb_full = X_train_raw.drop(columns=drop_cols).reset_index(drop=True)
    X_cb_test = X_test_raw.drop(columns=drop_cols).reset_index(drop=True)

    X_cb_full = make_catboost_frame(X_cb_full)
    X_cb_test = make_catboost_frame(X_cb_test)

    cat_cols = X_cb_full.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    cat_idx = [X_cb_full.columns.get_loc(c) for c in cat_cols]

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_cb_full, y, test_size=0.15, random_state=RANDOM_STATE
    )

    cb_final = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=30000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=RANDOM_STATE,
        task_type="CPU",
        subsample=0.8,
        rsm=0.8,
        bootstrap_type="Bernoulli",
        verbose=200,
    )

    cb_final.fit(
        X_tr, y_tr,
        cat_features=cat_idx,
        eval_set=(X_va, y_va),
        use_best_model=True,
        early_stopping_rounds=500,
    )

    pred_cb = np.clip(cb_final.predict(X_cb_test), 0, None)

    # Ensemble
    w = best_w if best_w is not None else 0.5
    pred_ens = np.clip(w * pred_cb + (1 - w) * pred_ridge, 0, None)

    print(f"\nPred stats: ridge_mean={pred_ridge.mean():.4f}, cb_mean={pred_cb.mean():.4f}, ens_mean={pred_ens.mean():.4f}, w={w}")

    # Save submission
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub = pd.DataFrame({"Id": test_ids, "JP_Sales": pred_ens})
    sub_path = os.path.join(OUTPUT_DIR, "sub_baseline.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved baseline submission to {sub_path}")

    return pred_ridge, pred_cb, pred_ens


# =========================
# OPTUNA HYPERPARAMETER TUNING
# =========================
def run_optuna_optimization(X_train_raw, y, n_trials, timeout):
    """Run Optuna hyperparameter optimization."""
    import optuna
    from optuna.samplers import TPESampler
    
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Global variables for callback
    global last_save_time
    last_save_time = time.time()
    
    def save_best_params(study, trial=None):
        """Save best params to JSON periodically."""
        global last_save_time
        current_time = time.time()
        
        if current_time - last_save_time >= SAVE_INTERVAL_SEC or trial is None:
            if study.best_trial is not None:
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "n_trials_completed": len(study.trials),
                    "best_trial_number": study.best_trial.number,
                }
                
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                params_path = os.path.join(OUTPUT_DIR, BEST_PARAMS_FILE)
                with open(params_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saved best params: MAE={study.best_value:.6f}")
                last_save_time = current_time
    
    def objective(trial):
        """Optuna objective function."""
        # Bootstrap type selection
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
        
        # Base hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
            'rsm': trial.suggest_float('rsm', 0.5, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'bootstrap_type': bootstrap_type,
        }
        
        # Bootstrap-dependent parameters
        if bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 5.0)
        elif bootstrap_type in ['Bernoulli', 'MVS']:
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        
        # Prepare data
        drop_cols = [c for c in ["Name", "Name_root"] if c in X_train_raw.columns]
        X_cb = X_train_raw.drop(columns=drop_cols).reset_index(drop=True)
        X_cb = make_catboost_frame(X_cb)
        
        cat_cols = X_cb.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
        cat_idx = [X_cb.columns.get_loc(c) for c in cat_cols]
        
        # CV
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fold_maes = []
        
        for fold, (tr_idx, va_idx) in enumerate(cv_inner.split(X_cb), 1):
            X_tr, X_va = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            
            model = CatBoostRegressor(
                loss_function="MAE",
                eval_metric="MAE",
                iterations=10000,
                **params,
                random_seed=RANDOM_STATE,
                task_type="CPU",
                verbose=0,
            )
            
            model.fit(
                X_tr, y_tr,
                cat_features=cat_idx,
                eval_set=(X_va, y_va),
                use_best_model=True,
                early_stopping_rounds=200,
            )
            
            pred = np.clip(model.predict(X_va), 0, None)
            mae = float(mean_absolute_error(y_va, pred))
            fold_maes.append(mae)
            
            trial.report(np.mean(fold_maes), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(fold_maes)
    
    # Create study
    storage = f"sqlite:///{os.path.join(OUTPUT_DIR, STUDY_NAME)}.db"
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"\nStarting Optuna optimization: {n_trials} trials")
    print(f"Best params will be saved to '{BEST_PARAMS_FILE}' every {SAVE_INTERVAL_SEC//60} minutes")
    print(f"Study stored in '{STUDY_NAME}.db' (can resume if interrupted)")
    print("=" * 60)

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[save_best_params],
        show_progress_bar=True,
        timeout=timeout,
    )

    # Final save
    save_best_params(study)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print(f"Best MAE: {study.best_value:.6f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    
    return study


# =========================
# OPTUNA VISUALIZATION
# =========================
def save_optuna_visualizations(study):
    """Save Optuna visualization plots to files."""
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ OPTUNA")
    print("=" * 60)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    try:
        import optuna.visualization as vis
        import plotly.io as pio
        
        plots_saved = []
        
        # 1. История оптимизации
        try:
            fig1 = vis.plot_optimization_history(study)
            path1 = os.path.join(PLOTS_DIR, "optimization_history.html")
            pio.write_html(fig1, path1)
            pio.write_image(fig1, os.path.join(PLOTS_DIR, "optimization_history.png"), width=1200, height=600)
            plots_saved.append("optimization_history")
            print(f"  ✓ Saved optimization_history.html/png")
        except Exception as e:
            print(f"  ✗ Failed to save optimization_history: {e}")
        
        # 2. Важность параметров
        try:
            fig2 = vis.plot_param_importances(study)
            path2 = os.path.join(PLOTS_DIR, "param_importances.html")
            pio.write_html(fig2, path2)
            pio.write_image(fig2, os.path.join(PLOTS_DIR, "param_importances.png"), width=1200, height=600)
            plots_saved.append("param_importances")
            print(f"  ✓ Saved param_importances.html/png")
        except Exception as e:
            print(f"  ✗ Failed to save param_importances: {e}")
        
        # 3. Slice plot
        try:
            fig3 = vis.plot_slice(study)
            path3 = os.path.join(PLOTS_DIR, "slice_plot.html")
            pio.write_html(fig3, path3)
            pio.write_image(fig3, os.path.join(PLOTS_DIR, "slice_plot.png"), width=1600, height=800)
            plots_saved.append("slice_plot")
            print(f"  ✓ Saved slice_plot.html/png")
        except Exception as e:
            print(f"  ✗ Failed to save slice_plot: {e}")
        
        # 4. Parallel coordinate plot
        try:
            fig4 = vis.plot_parallel_coordinate(study)
            path4 = os.path.join(PLOTS_DIR, "parallel_coordinate.html")
            pio.write_html(fig4, path4)
            pio.write_image(fig4, os.path.join(PLOTS_DIR, "parallel_coordinate.png"), width=1600, height=600)
            plots_saved.append("parallel_coordinate")
            print(f"  ✓ Saved parallel_coordinate.html/png")
        except Exception as e:
            print(f"  ✗ Failed to save parallel_coordinate: {e}")
        
        # 5. Contour plot (для пар параметров)
        try:
            fig5 = vis.plot_contour(study, params=["learning_rate", "depth"])
            path5 = os.path.join(PLOTS_DIR, "contour_lr_depth.html")
            pio.write_html(fig5, path5)
            pio.write_image(fig5, os.path.join(PLOTS_DIR, "contour_lr_depth.png"), width=800, height=600)
            plots_saved.append("contour_lr_depth")
            print(f"  ✓ Saved contour_lr_depth.html/png")
        except Exception as e:
            print(f"  ✗ Failed to save contour_lr_depth: {e}")
        
        # 6. Timeline plot
        try:
            fig6 = vis.plot_timeline(study)
            path6 = os.path.join(PLOTS_DIR, "timeline.html")
            pio.write_html(fig6, path6)
            pio.write_image(fig6, os.path.join(PLOTS_DIR, "timeline.png"), width=1200, height=600)
            plots_saved.append("timeline")
            print(f"  ✓ Saved timeline.html/png")
        except Exception as e:
            print(f"  ✗ Failed to save timeline: {e}")
        
        print(f"\n  Total plots saved: {len(plots_saved)}")
        print(f"  Location: {PLOTS_DIR}/")
        
    except ImportError as e:
        print(f"  Plotly not available: {e}")
        print("  Install with: pip install plotly kaleido")
        print("\n  Top 10 trials (fallback):")
        trials_df = study.trials_dataframe()
        print(trials_df.nsmallest(10, 'value')[['number', 'value', 'params_learning_rate', 'params_depth', 'params_l2_leaf_reg']])
        
        # Save trials dataframe as CSV
        trials_path = os.path.join(PLOTS_DIR, "trials_results.csv")
        trials_df.to_csv(trials_path, index=False)
        print(f"\n  Saved trials results to {trials_path}")


# =========================
# FINAL TRAINING WITH OPTUNA PARAMS
# =========================
def train_with_optuna_params(X_train_raw, X_test_raw, X_train_proc, X_test_proc, y, test_ids, pred_ridge, best_w, study=None):
    """Train final model with best Optuna parameters."""
    print("\n" + "=" * 60)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ С ЛУЧШИМИ ПАРАМЕТРАМИ OPTUNA")
    print("=" * 60)
    
    # Load best params
    params_path = os.path.join(OUTPUT_DIR, BEST_PARAMS_FILE)
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            saved_result = json.load(f)
        best_params = saved_result['best_params']
        print(f"  Loaded best params from {params_path}")
    elif study is not None:
        best_params = study.best_params
        print("  Using params from study object")
    else:
        print("  ERROR: No best params found!")
        return None

    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    # Prepare data
    drop_cols = [c for c in ["Name", "Name_root"] if c in X_train_raw.columns]
    X_cb_full = X_train_raw.drop(columns=drop_cols).reset_index(drop=True)
    X_cb_test = X_test_raw.drop(columns=drop_cols).reset_index(drop=True)

    X_cb_full = make_catboost_frame(X_cb_full)
    X_cb_test = make_catboost_frame(X_cb_test)

    cat_cols = X_cb_full.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    cat_idx = [X_cb_full.columns.get_loc(c) for c in cat_cols]

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_cb_full, y, test_size=0.15, random_state=RANDOM_STATE
    )

    # Determine bootstrap type
    bootstrap_type = best_params.get('bootstrap_type', 'Bernoulli')
    
    # Build model params
    model_params = {
        'loss_function': "MAE",
        'eval_metric': "MAE",
        'iterations': 30000,
        'random_seed': RANDOM_STATE,
        'task_type': "CPU",
        'verbose': 200,
    }
    
    # Add optimized params (excluding bootstrap-specific ones initially)
    for key, value in best_params.items():
        if key not in ['bagging_temperature', 'subsample']:
            model_params[key] = value
    
    # Add bootstrap-specific param
    if bootstrap_type == 'Bayesian' and 'bagging_temperature' in best_params:
        model_params['bagging_temperature'] = best_params['bagging_temperature']
    elif bootstrap_type in ['Bernoulli', 'MVS'] and 'subsample' in best_params:
        model_params['subsample'] = best_params['subsample']

    print("\nTraining CatBoost with Optuna params...")
    cb_optuna_final = CatBoostRegressor(**model_params)

    cb_optuna_final.fit(
        X_tr, y_tr,
        cat_features=cat_idx,
        eval_set=(X_va, y_va),
        use_best_model=True,
        early_stopping_rounds=500,
    )

    pred_cb_optuna = np.clip(cb_optuna_final.predict(X_cb_test), 0, None)

    # Ensemble with Ridge
    w = best_w if best_w is not None else 0.5
    pred_ens_optuna = np.clip(w * pred_cb_optuna + (1 - w) * pred_ridge, 0, None)

    print(f"\nPred stats: cb_mean={pred_cb_optuna.mean():.4f}, ens_mean={pred_ens_optuna.mean():.4f}, w={w}")

    # Save submission
    sub_optuna = pd.DataFrame({"Id": test_ids, "JP_Sales": pred_ens_optuna})
    sub_path = os.path.join(OUTPUT_DIR, "sub_optuna.csv")
    sub_optuna.to_csv(sub_path, index=False)
    print(f"\nSaved Optuna submission to {sub_path}")

    return pred_ens_optuna


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Train CatBoost + Ridge with Optuna")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Optuna optimization")
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT, help="Optuna timeout in seconds")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV (use for quick testing)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    X_train_raw, X_test_raw, X_all, y, test_ids = load_data()
    
    # Feature engineering and preprocessing
    X_train_proc, X_test_proc, X_train_raw, X_test_raw = prepare_features(
        X_all, X_train_raw, X_test_raw, y
    )
    
    # CV
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    if not args.skip_cv:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION: RIDGE")
        print("=" * 60)
        oof_ridge, ridge_scores = cv_oof_ridge(X_train_proc, y, cv, alpha=2.0)
        
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION: CATBOOST")
        print("=" * 60)
        oof_cb, cb_scores, cb_best_iters = cv_oof_catboost(X_train_raw, y, cv)
        
        # Find best ensemble weight
        print("\n" + "=" * 60)
        print("ENSEMBLE WEIGHT OPTIMIZATION")
        print("=" * 60)
        weights = np.linspace(0, 1, 201)
        best = {"w": None, "mae": np.inf, "rmse": np.inf}

        for w in weights:
            ens = w * oof_cb + (1 - w) * oof_ridge
            mae = float(mean_absolute_error(y, ens))
            r = rmse(y, ens)
            if mae < best["mae"]:
                best = {"w": float(w), "mae": mae, "rmse": r}

        print(f"  Best ensemble: {best}")
        best_w = best["w"]
    else:
        best_w = 1.0  # Default to CatBoost only
        print("\n  [Skipping CV, using default w=1.0]")
    
    # Train baseline and get predictions
    pred_ridge, pred_cb, pred_ens = train_baseline(
        X_train_raw, X_test_raw, X_train_proc, X_test_proc, y, test_ids, best_w
    )
    
    # Optuna optimization
    study = None
    if not args.skip_optuna:
        study = run_optuna_optimization(X_train_raw, y, args.trials, args.timeout)
        
        # Save visualizations
        save_optuna_visualizations(study)
        
        # Train with best params
        train_with_optuna_params(
            X_train_raw, X_test_raw, X_train_proc, X_test_proc, y, test_ids, pred_ridge, best_w, study
        )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    if study:
        print(f"  - Optuna DB: {STUDY_NAME}.db")
        print(f"  - Best params: {BEST_PARAMS_FILE}")
        print(f"  - Plots: {PLOTS_DIR}/")
    print(f"  - Submissions: sub_baseline.csv, sub_optuna.csv")


if __name__ == "__main__":
    main()
