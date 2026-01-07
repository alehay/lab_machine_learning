#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from catboost import CatBoostClassifier


# =========================
# Hardcoded I/O
# =========================
TRAIN_PATH = "insclass_train.csv"
TEST_PATH  = "insclass_test.csv"
BASE_OUTDIR = "runs/insclass_cb"

SEED = 42
FINAL_SPLITS = 5          # финальная CV
TUNE_SPLITS = 3           # tuning CV внутри Optuna
N_STARTUP_TRIALS = 10     # для pruner
DEFAULT_TRIALS = 40


# =========================
# Utils
# =========================
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("insclass_optuna")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_learning_curve(evals_result: dict, out_png: str, title: str) -> None:
    if not evals_result:
        return

    datasets = list(evals_result.keys())
    if not datasets:
        return

    metric_name = None
    for ds in datasets:
        for m in evals_result.get(ds, {}).keys():
            if m.upper() == "AUC":
                metric_name = m
                break
        if metric_name:
            break
    if metric_name is None:
        metric_name = next(iter(evals_result[datasets[0]].keys()), None)
    if metric_name is None:
        return

    plt.figure()
    for ds in datasets:
        series = evals_result.get(ds, {}).get(metric_name, None)
        if series:
            plt.plot(series, label=f"{ds}:{metric_name}")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_hist(values: np.ndarray, out_png: str, title: str) -> None:
    plt.figure()
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_optuna_history(trial_values: list, out_png: str, title: str) -> None:
    # trial_values: list of (trial_number, value)
    if not trial_values:
        return
    xs = [t for t, _ in trial_values]
    ys = [v for _, v in trial_values]
    best_so_far = np.maximum.accumulate(np.array(ys, dtype=float))

    plt.figure()
    plt.plot(xs, ys, marker="o", linewidth=1, markersize=3, label="trial value")
    plt.plot(xs, best_so_far, linewidth=2, label="best so far")
    plt.title(title)
    plt.xlabel("Trial")
    plt.ylabel("AUC (CV mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# =========================
# Preprocess
# =========================
def preprocess(train: pd.DataFrame, test: pd.DataFrame, logger: logging.Logger):
    if "target" not in train.columns:
        raise ValueError("train: column 'target' not found")
    if "id" not in test.columns:
        raise ValueError("test: column 'id' not found")

    y = train["target"].astype(int).values
    X = train.drop(columns=["target"]).copy()

    test_id = test["id"].values
    X_test = test.drop(columns=["id"]).copy()

    # drop constants in train
    drop_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]

    # drop variable_15 if almost empty in test (guard)
    if "variable_15" in X_test.columns:
        miss_rate = float(X_test["variable_15"].isna().mean())
        if miss_rate > 0.999:
            drop_cols.append("variable_15")
            logger.info("Drop variable_15 due to miss_rate_in_test=%.6f", miss_rate)

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        logger.info("Dropping %d columns (constants/empty): %s",
                    len(drop_cols),
                    drop_cols[:25] + (["..."] if len(drop_cols) > 25 else []))

    X.drop(columns=drop_cols, inplace=True, errors="ignore")
    X_test.drop(columns=drop_cols, inplace=True, errors="ignore")

    # object -> string and fill missing
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        logger.info("Categorical(object) cols: %s", cat_cols)

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("string").fillna("__MISSING__")
        else:
            X_test[c] = "__MISSING__"

    # missing indicators for any columns with NA in train
    na_cols = [c for c in X.columns if X[c].isna().any()]
    if na_cols:
        logger.info("Adding %d missing indicator columns", len(na_cols))
    for c in na_cols:
        X[c + "__isna"] = X[c].isna().astype(np.uint8)
        X_test[c + "__isna"] = X_test[c].isna().astype(np.uint8)

    # align columns
    missing_in_test = [c for c in X.columns if c not in X_test.columns]
    if missing_in_test:
        logger.warning("Test missing %d columns -> fill NaN", len(missing_in_test))
        for c in missing_in_test:
            X_test[c] = np.nan

    extra_in_test = [c for c in X_test.columns if c not in X.columns]
    if extra_in_test:
        logger.warning("Test has %d extra columns -> drop", len(extra_in_test))
        X_test.drop(columns=extra_in_test, inplace=True)

    X_test = X_test[X.columns]

    cat_cols2 = [c for c in X.columns if (X[c].dtype == "string" or X[c].dtype == "object")]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols2]
    logger.info("cat_features count=%d", len(cat_idx))

    return X, y, X_test, test_id, cat_idx


# =========================
# Optuna tuning (all params)
# =========================
def tune_with_optuna(X: pd.DataFrame, y: np.ndarray, cat_idx, out_dir: str, trials: int, timeout: int | None, logger):
    import optuna

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=1)

    storage_path = os.path.join(out_dir, "optuna_study.db")
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name="insclass_cb_auc",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    skf = StratifiedKFold(n_splits=TUNE_SPLITS, shuffle=True, random_state=SEED)

    def objective(trial: "optuna.Trial") -> float:
        # Conditionally sampled parameters
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        grow_policy = trial.suggest_categorical("grow_policy", ["Depthwise", "Lossguide"])

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": SEED,
            "allow_writing_files": False,
            "verbose": False,

            # tuned knobs
            "iterations": trial.suggest_int("iterations", 3000, 15000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 80.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 64),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),

            "bootstrap_type": bootstrap_type,
            "grow_policy": grow_policy,

            # categorical/CTR related
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 6),

            # class imbalance handling (also tuned)
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced", None]),
        }

        # grow_policy specific
        if grow_policy == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 16, 128)

        # bootstrap specific
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        # early stopping also tuned
        early_stop = trial.suggest_int("early_stopping_rounds", 100, 600)

        fold_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                cat_features=cat_idx,
                use_best_model=True,
                early_stopping_rounds=early_stop,
            )
            pred = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, pred)
            fold_aucs.append(float(auc))

            trial.report(float(np.mean(fold_aucs)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_aucs))

    logger.info("Optuna: trials=%d timeout=%s (sec)", trials, str(timeout))
    study.optimize(objective, n_trials=trials, timeout=timeout, n_jobs=1, gc_after_trial=True)

    best_params = dict(study.best_params)
    best_value = float(study.best_value)

    # Persist tuning result
    save_json(os.path.join(out_dir, "optuna_best.json"), {"best_value": best_value, "best_params": best_params})
    logger.info("Optuna best: AUC=%.6f params=%s", best_value, best_params)

    # Plot optuna history
    tv = [(t.number, t.value) for t in study.trials if t.value is not None]
    plot_optuna_history(tv, os.path.join(out_dir, "optuna_history.png"), "Optuna CV mean AUC")

    return best_params, best_value, storage_path


# =========================
# Final training and artifacts
# =========================
def final_train_and_predict(X, y, X_test, cat_idx, best_params: dict, out_dir: str, logger):
    skf = StratifiedKFold(n_splits=FINAL_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    fold_scores = []
    best_iters = []

    # Extract tuned early stopping
    early_stop = int(best_params.pop("early_stopping_rounds"))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        ensure_dir(fold_dir)

        params = dict(best_params)
        # enforce stable non-tuned runtime params
        params.update({
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": SEED,
            "verbose": 200,
            "allow_writing_files": True,
            "train_dir": os.path.join(fold_dir, "catboost_info"),
            "thread_count": -1,
        })

        logger.info("Fold %d/%d training...", fold, FINAL_SPLITS)

        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_idx,
            use_best_model=True,
            early_stopping_rounds=early_stop,
        )

        oof_va = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = oof_va
        test_pred += model.predict_proba(X_test)[:, 1] / FINAL_SPLITS

        fold_auc = roc_auc_score(y_va, oof_va)
        fold_scores.append(float(fold_auc))

        best_it = model.get_best_iteration()
        best_iters.append(int(best_it) if best_it is not None else None)

        logger.info("Fold %d AUC(prob)=%.6f best_iter=%s", fold, fold_auc, str(best_it))

        # plots
        plot_learning_curve(
            model.get_evals_result(),
            out_png=os.path.join(fold_dir, "learning_curve.png"),
            title=f"Fold {fold} learning curve",
        )

    oof_auc = roc_auc_score(y, oof)
    logger.info("OOF AUC(prob)=%.6f | fold mean=%.6f std=%.6f",
                oof_auc, float(np.mean(fold_scores)), float(np.std(fold_scores)))

    # Save preds and hist
    np.save(os.path.join(out_dir, "oof_pred.npy"), oof)
    np.save(os.path.join(out_dir, "test_pred.npy"), test_pred)
    plot_hist(oof, os.path.join(out_dir, "hist_oof.png"), "OOF predicted probabilities")
    plot_hist(test_pred, os.path.join(out_dir, "hist_test.png"), "TEST predicted probabilities")

    # Also make a binary submission with threshold tuned by F1 (optional)
    thr_grid = np.linspace(0.01, 0.99, 199)
    best_thr, best_f1 = 0.5, -1.0
    for t in thr_grid:
        f1 = f1_score(y, (oof >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)

    logger.info("Best threshold by F1: thr=%.4f f1=%.6f", best_thr, best_f1)

    return oof_auc, fold_scores, best_iters, test_pred, best_thr, best_f1


def maybe_update_best_run(current_run_dir: str, current_oof_auc: float, logger: logging.Logger):
    best_dir = os.path.join(BASE_OUTDIR, "best")
    best_summary_path = os.path.join(best_dir, "cv_summary.json")

    prev = safe_read_json(best_summary_path)
    prev_auc = float(prev.get("metrics", {}).get("oof_auc_prob", -1e9)) if isinstance(prev, dict) else -1e9

    if current_oof_auc > prev_auc:
        logger.info("New BEST run: %.6f > %.6f. Updating %s", current_oof_auc, prev_auc, best_dir)
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(current_run_dir, best_dir)
    else:
        logger.info("Best not improved: current=%.6f best=%.6f. Keep existing best.", current_oof_auc, prev_auc)


def main():
    parser = argparse.ArgumentParser(description="InsClass CatBoost + Optuna (only trials & timeout)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Optuna trials count")
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds (optional)")
    args = parser.parse_args()

    logger = setup_logging()
    ensure_dir(BASE_OUTDIR)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_OUTDIR, run_ts)
    ensure_dir(out_dir)
    logger.info("Run dir: %s", out_dir)

    # Load
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    logger.info("Loaded train=%s test=%s", train.shape, test.shape)

    # Preprocess
    X, y, X_test, test_id, cat_idx = preprocess(train, test, logger)
    logger.info("After preprocess: X=%s X_test=%s", X.shape, X_test.shape)

    # Tune (all params)
    best_params, best_cv_auc, study_db = tune_with_optuna(
        X=X, y=y, cat_idx=cat_idx, out_dir=out_dir,
        trials=args.trials, timeout=args.timeout, logger=logger
    )

    # Final train
    # IMPORTANT: we mutate best_params inside final_train (pop early_stopping_rounds),
    # so pass a copy to preserve for summary.
    tuned_params_for_summary = dict(best_params)
    oof_auc, fold_aucs, best_iters, test_pred, best_thr, best_f1 = final_train_and_predict(
        X=X, y=y, X_test=X_test, cat_idx=cat_idx, best_params=dict(best_params), out_dir=out_dir, logger=logger
    )

    # Submissions
    sub_proba = pd.DataFrame({"id": test_id, "target_proba": test_pred})
    sub_proba_path = os.path.join(out_dir, "submission_proba.csv")
    sub_proba.to_csv(sub_proba_path, index=False)

    sub_bin = pd.DataFrame({"id": test_id, "target": (test_pred >= best_thr).astype(int)})
    sub_bin_path = os.path.join(out_dir, "submission.csv")
    sub_bin.to_csv(sub_bin_path, index=False)

    logger.info("Saved submissions: %s and %s", sub_bin_path, sub_proba_path)
    logger.info("Submission head:\n%s", sub_bin.head(10).to_string(index=False))

    # Summary
    summary = {
        "files": {
            "train_path": TRAIN_PATH,
            "test_path": TEST_PATH,
        },
        "output_dir": out_dir,
        "seed": SEED,
        "tuning": {
            "trials": int(args.trials),
            "timeout_sec": args.timeout,
            "tune_splits": TUNE_SPLITS,
            "best_cv_auc_mean": float(best_cv_auc),
            "study_db": study_db,
            "best_params": tuned_params_for_summary,
        },
        "metrics": {
            "oof_auc_prob": float(oof_auc),
            "fold_auc_prob": fold_aucs,
            "fold_auc_mean": float(np.mean(fold_aucs)),
            "fold_auc_std": float(np.std(fold_aucs)),
        },
        "best_iteration_per_fold": best_iters,
        "threshold_by_f1": {
            "thr": float(best_thr),
            "f1": float(best_f1),
        },
        "submission": {
            "binary_path": sub_bin_path,
            "proba_path": sub_proba_path,
        },
    }
    save_json(os.path.join(out_dir, "cv_summary.json"), summary)

    # Update "best" run
    maybe_update_best_run(out_dir, oof_auc, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
