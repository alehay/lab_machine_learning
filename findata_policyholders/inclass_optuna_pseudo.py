#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InsClass CatBoost + Optuna + Pseudo-Labeling
Оптимизировано для GPU V100 16GB

Использование:
  python inclass_optuna_pseudo.py --trials 100 --timeout 7200 --device gpu
  python inclass_optuna_pseudo.py --trials 50 --device cpu  # для CPU
"""

import os
import json
import shutil
import argparse
import logging
import gc
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from catboost import CatBoostClassifier


# =========================
# Configuration
# =========================
TRAIN_PATH = "insclass_train.csv"
TEST_PATH = "insclass_test.csv"
PSEUDO_PATH = "score_068192.csv"  # лучший submission для pseudo-labels

BASE_OUTDIR = "runs/insclass_cb_pseudo"

SEED = 42
FINAL_SPLITS = 5
TUNE_SPLITS = 3
N_STARTUP_TRIALS = 10
DEFAULT_TRIALS = 60

# Multi-seed для финального обучения
FINAL_SEEDS = [42, 7896, 99999999, 568, 789]


# =========================
# Hardware Configuration
# =========================
def get_catboost_device_params(device: str) -> Dict[str, Any]:
    """Возвращает параметры CatBoost для выбранного устройства."""
    if device == "gpu":
        return {
            "task_type": "GPU",
            "devices": "0",  # первый GPU
            "gpu_ram_part": 0.9,  # используем 90% GPU RAM
        }
    else:
        return {
            "task_type": "CPU",
            "thread_count": -1,  # использовать все ядра
        }


def is_gpu_mode(device: str) -> bool:
    """Проверяет, используется ли GPU."""
    return device == "gpu"


# =========================
# Logging & Utils
# =========================
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("insclass_optuna_pseudo")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_json(path: str) -> Optional[dict]:
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
# Preprocessing
# =========================
def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[int]]:
    """Препроцессинг данных."""
    if "target" not in train.columns:
        raise ValueError("train: column 'target' not found")
    if "id" not in test.columns:
        raise ValueError("test: column 'id' not found")

    y = train["target"].astype(int).values
    X = train.drop(columns=["target"]).copy()

    test_id = test["id"].values
    X_test = test.drop(columns=["id"]).copy()

    # Удаляем константные столбцы
    drop_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]

    # Удаляем variable_15 если почти пустой в test
    if "variable_15" in X_test.columns:
        miss_rate = float(X_test["variable_15"].isna().mean())
        if miss_rate > 0.999:
            drop_cols.append("variable_15")
            logger.info("Drop variable_15 due to miss_rate_in_test=%.6f", miss_rate)

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        logger.info(
            "Dropping %d columns (constants/empty): %s",
            len(drop_cols),
            drop_cols[:25] + (["..."] if len(drop_cols) > 25 else [])
        )

    X.drop(columns=drop_cols, inplace=True, errors="ignore")
    X_test.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Обрабатываем категориальные признаки
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        logger.info("Categorical(object) cols: %s", cat_cols)

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("string").fillna("__MISSING__")
        else:
            X_test[c] = "__MISSING__"

    # Индикаторы пропусков
    na_cols = [c for c in X.columns if X[c].isna().any()]
    if na_cols:
        logger.info("Adding %d missing indicator columns", len(na_cols))
    for c in na_cols:
        X[c + "__isna"] = X[c].isna().astype(np.uint8)
        X_test[c + "__isna"] = X_test[c].isna().astype(np.uint8)

    # Выравниваем столбцы
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

    # Индексы категориальных признаков
    cat_cols2 = [c for c in X.columns if (X[c].dtype == "string" or X[c].dtype == "object")]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols2]
    logger.info("cat_features count=%d", len(cat_idx))

    return X, y, X_test, test_id, cat_idx


def load_pseudo_labels(
    pseudo_path: str,
    X_test: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Загружает pseudo-labels из submission файла."""
    if not os.path.exists(pseudo_path):
        logger.warning("Pseudo-labels file not found: %s", pseudo_path)
        return None, None

    pseudo_sub = pd.read_csv(pseudo_path)
    pseudo_labels = pseudo_sub["target"].values

    # Проверяем размеры
    if len(pseudo_labels) != len(X_test):
        logger.error(
            "Pseudo labels size mismatch: %d vs %d",
            len(pseudo_labels), len(X_test)
        )
        return None, None

    X_pseudo = X_test.copy()
    y_pseudo = pseudo_labels

    logger.info("Loaded pseudo-labels: %d samples", len(y_pseudo))
    logger.info("Pseudo labels distribution: %s", pd.Series(y_pseudo).value_counts().to_dict())

    return X_pseudo, y_pseudo


# =========================
# Optuna Tuning with Pseudo-Labeling
# =========================
def tune_with_optuna(
    X: pd.DataFrame,
    y: np.ndarray,
    X_pseudo: Optional[pd.DataFrame],
    y_pseudo: Optional[np.ndarray],
    cat_idx: List[int],
    device: str,
    out_dir: str,
    trials: int,
    timeout: Optional[int],
    logger: logging.Logger
) -> Tuple[dict, float, str]:
    """Optuna оптимизация с учётом pseudo-labeling."""
    import optuna

    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=1,
        interval_steps=1
    )

    storage_path = os.path.join(out_dir, "optuna_study.db")
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name="insclass_cb_pseudo_auc",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    device_params = get_catboost_device_params(device)
    has_pseudo = X_pseudo is not None and y_pseudo is not None

    skf = StratifiedKFold(n_splits=TUNE_SPLITS, shuffle=True, random_state=SEED)

    def objective(trial: "optuna.Trial") -> float:
        # Параметры pseudo-labeling (если доступны)
        if has_pseudo:
            pseudo_weight = trial.suggest_float("pseudo_weight", 0.0, 1.0)
            pseudo_confidence_threshold = trial.suggest_float(
                "pseudo_confidence_threshold", 0.0, 0.4
            )
        else:
            pseudo_weight = 0.0
            pseudo_confidence_threshold = 0.0

        # Основные гиперпараметры
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        grow_policy = trial.suggest_categorical("grow_policy", ["Depthwise", "Lossguide"])

        # GPU-специфичные ограничения
        if device == "gpu":
            max_ctr = 4  # GPU ограничение
            border_cnt = trial.suggest_categorical("border_count", [32, 64, 128, 254])
        else:
            max_ctr = trial.suggest_int("max_ctr_complexity", 1, 6)
            border_cnt = trial.suggest_int("border_count", 32, 255)

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": SEED,
            "allow_writing_files": False,
            "verbose": False,

            # Основные параметры
            "iterations": trial.suggest_int("iterations", 2000, 12000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "border_count": border_cnt,
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),

            "bootstrap_type": bootstrap_type,
            "grow_policy": grow_policy,

            # Категориальные параметры
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 25),
            "max_ctr_complexity": max_ctr,

            # Балансировка классов
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "SqrtBalanced", None]
            ),
        }

        # rsm НЕ поддерживается на GPU для classification (только pairwise)
        if device != "gpu":
            params["rsm"] = trial.suggest_float("rsm", 0.5, 1.0)

        # Добавляем device параметры
        params.update(device_params)

        # Условные параметры
        if grow_policy == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 16, 128)

        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        early_stop = trial.suggest_int("early_stopping_rounds", 100, 500)

        fold_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx].copy()
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            # Добавляем pseudo-labels если включено
            if has_pseudo and pseudo_weight > 0:
                # Фильтруем по confidence (только уверенные предсказания)
                if pseudo_confidence_threshold > 0:
                    confident_mask = (
                        (y_pseudo <= pseudo_confidence_threshold) |
                        (y_pseudo >= 1 - pseudo_confidence_threshold)
                    )
                    X_pseudo_filtered = X_pseudo[confident_mask]
                    y_pseudo_filtered = y_pseudo[confident_mask]
                else:
                    X_pseudo_filtered = X_pseudo
                    y_pseudo_filtered = y_pseudo

                # Сэмплируем pseudo-labels
                n_pseudo_sample = int(len(X_tr) * pseudo_weight)
                if n_pseudo_sample > 0 and len(X_pseudo_filtered) > 0:
                    n_pseudo_sample = min(n_pseudo_sample, len(X_pseudo_filtered))
                    rng = np.random.RandomState(SEED + fold)
                    pseudo_idx = rng.choice(len(X_pseudo_filtered), n_pseudo_sample, replace=False)

                    X_tr = pd.concat([X_tr, X_pseudo_filtered.iloc[pseudo_idx]], ignore_index=True)
                    y_tr = np.concatenate([y_tr, y_pseudo_filtered[pseudo_idx]])

            # Пересчитываем cat_idx для новых данных
            cat_cols2 = [c for c in X_tr.columns if (X_tr[c].dtype == "string" or X_tr[c].dtype == "object")]
            cat_idx_fold = [X_tr.columns.get_loc(c) for c in cat_cols2]

            try:
                model = CatBoostClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_va, y_va),
                    cat_features=cat_idx_fold,
                    use_best_model=True,
                    early_stopping_rounds=early_stop,
                )
                pred = model.predict_proba(X_va)[:, 1]
                auc = roc_auc_score(y_va, pred)
                fold_aucs.append(float(auc))

                # Освобождаем память
                del model
                gc.collect()

            except Exception as e:
                logger.warning(f"Trial {trial.number} fold {fold} failed: {e}")
                raise optuna.TrialPruned()

            trial.report(float(np.mean(fold_aucs)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_aucs))

    logger.info("Optuna: trials=%d timeout=%s device=%s", trials, str(timeout), device)
    logger.info("Pseudo-labeling enabled: %s", has_pseudo)

    study.optimize(
        objective,
        n_trials=trials,
        timeout=timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True
    )

    # Проверяем, есть ли успешные trials
    completed_trials = [t for t in study.trials if t.value is not None]
    
    if not completed_trials:
        logger.error("All trials failed! Using default parameters.")
        # Fallback параметры
        best_params = {
            "iterations": 5000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 128,
            "min_data_in_leaf": 30,
            "bootstrap_type": "Bernoulli",
            "grow_policy": "Depthwise",
            "one_hot_max_size": 10,
            "max_ctr_complexity": 4 if device == "gpu" else 4,
            "auto_class_weights": "SqrtBalanced",
            "subsample": 0.8,
            "early_stopping_rounds": 300,
        }
        if has_pseudo:
            best_params["pseudo_weight"] = 0.3
            best_params["pseudo_confidence_threshold"] = 0.1
        if device != "gpu":
            best_params["rsm"] = 0.9
        best_value = 0.0
    else:
        best_params = dict(study.best_params)
        best_value = float(study.best_value)
        logger.info("Optuna best: AUC=%.6f", best_value)
        logger.info("Best params: %s", best_params)

    # Сохраняем результаты
    save_json(
        os.path.join(out_dir, "optuna_best.json"),
        {"best_value": best_value, "best_params": best_params, "completed_trials": len(completed_trials)}
    )

    # График истории
    tv = [(t.number, t.value) for t in study.trials if t.value is not None]
    plot_optuna_history(tv, os.path.join(out_dir, "optuna_history.png"), "Optuna CV mean AUC")

    return best_params, best_value, storage_path


# =========================
# Final Training with Multi-Seed
# =========================
def final_train_and_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    X_pseudo: Optional[pd.DataFrame],
    y_pseudo: Optional[np.ndarray],
    cat_idx: List[int],
    best_params: dict,
    device: str,
    out_dir: str,
    logger: logging.Logger
) -> Tuple[float, List[float], np.ndarray, float, float]:
    """Финальное обучение с multi-seed averaging."""

    device_params = get_catboost_device_params(device)
    has_pseudo = X_pseudo is not None and y_pseudo is not None

    # Извлекаем pseudo параметры
    pseudo_weight = best_params.pop("pseudo_weight", 0.0) if has_pseudo else 0.0
    pseudo_confidence_threshold = best_params.pop("pseudo_confidence_threshold", 0.0) if has_pseudo else 0.0
    early_stop = int(best_params.pop("early_stopping_rounds", 300))
    
    # Удаляем rsm на GPU (не поддерживается для classification)
    if device == "gpu" and "rsm" in best_params:
        logger.warning("Removing 'rsm' parameter (not supported on GPU for classification)")
        best_params.pop("rsm")
    
    # GPU ограничения
    if device == "gpu":
        # max_ctr_complexity должен быть <= 4 на GPU
        if best_params.get("max_ctr_complexity", 0) > 4:
            logger.warning("Limiting max_ctr_complexity to 4 for GPU")
            best_params["max_ctr_complexity"] = 4
        # border_count должен быть 32, 64, 128, или 254 на GPU
        bc = best_params.get("border_count", 128)
        valid_bc = [32, 64, 128, 254]
        if bc not in valid_bc:
            new_bc = min(valid_bc, key=lambda x: abs(x - bc))
            logger.warning(f"Adjusting border_count from {bc} to {new_bc} for GPU")
            best_params["border_count"] = new_bc

    all_oof = []
    all_test_preds = []
    all_fold_scores = []

    for seed_idx, seed in enumerate(FINAL_SEEDS):
        logger.info("\n" + "=" * 60)
        logger.info(f"Training with seed={seed} ({seed_idx + 1}/{len(FINAL_SEEDS)})")
        logger.info("=" * 60)

        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        ensure_dir(seed_dir)

        skf = StratifiedKFold(n_splits=FINAL_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X), dtype=float)
        test_pred = np.zeros(len(X_test), dtype=float)
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            ensure_dir(fold_dir)

            params = dict(best_params)
            params.update({
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": seed,
                "verbose": 200,
                "allow_writing_files": True,
                "train_dir": os.path.join(fold_dir, "catboost_info"),
            })
            params.update(device_params)

            logger.info(f"Seed {seed}, Fold {fold}/{FINAL_SPLITS}")

            X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx].copy()
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            # Добавляем pseudo-labels
            if has_pseudo and pseudo_weight > 0:
                if pseudo_confidence_threshold > 0:
                    confident_mask = (
                        (y_pseudo <= pseudo_confidence_threshold) |
                        (y_pseudo >= 1 - pseudo_confidence_threshold)
                    )
                    X_pseudo_filtered = X_pseudo[confident_mask]
                    y_pseudo_filtered = y_pseudo[confident_mask]
                else:
                    X_pseudo_filtered = X_pseudo
                    y_pseudo_filtered = y_pseudo

                n_pseudo_sample = int(len(X_tr) * pseudo_weight)
                if n_pseudo_sample > 0 and len(X_pseudo_filtered) > 0:
                    n_pseudo_sample = min(n_pseudo_sample, len(X_pseudo_filtered))
                    rng = np.random.RandomState(seed + fold)
                    pseudo_idx = rng.choice(len(X_pseudo_filtered), n_pseudo_sample, replace=False)

                    X_tr = pd.concat([X_tr, X_pseudo_filtered.iloc[pseudo_idx]], ignore_index=True)
                    y_tr = np.concatenate([y_tr, y_pseudo_filtered[pseudo_idx]])

                    logger.info(f"  Added {n_pseudo_sample} pseudo samples (total train: {len(X_tr)})")

            cat_cols2 = [c for c in X_tr.columns if (X_tr[c].dtype == "string" or X_tr[c].dtype == "object")]
            cat_idx_fold = [X_tr.columns.get_loc(c) for c in cat_cols2]

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                cat_features=cat_idx_fold,
                use_best_model=True,
                early_stopping_rounds=early_stop,
            )

            oof_va = model.predict_proba(X_va)[:, 1]
            oof[va_idx] = oof_va
            test_pred += model.predict_proba(X_test)[:, 1] / FINAL_SPLITS

            fold_auc = roc_auc_score(y_va, oof_va)
            fold_scores.append(float(fold_auc))

            best_it = model.get_best_iteration()
            logger.info(f"  Fold {fold} AUC={fold_auc:.6f} best_iter={best_it}")

            # График обучения
            plot_learning_curve(
                model.get_evals_result(),
                out_png=os.path.join(fold_dir, "learning_curve.png"),
                title=f"Seed {seed} Fold {fold}",
            )

            del model
            gc.collect()

        seed_oof_auc = roc_auc_score(y, oof)
        logger.info(f"Seed {seed} OOF AUC={seed_oof_auc:.6f}")

        all_oof.append(oof)
        all_test_preds.append(test_pred)
        all_fold_scores.extend(fold_scores)

        # Сохраняем промежуточные результаты
        np.save(os.path.join(seed_dir, "oof_pred.npy"), oof)
        np.save(os.path.join(seed_dir, "test_pred.npy"), test_pred)

    # Усреднение по всем seeds
    avg_oof = np.mean(all_oof, axis=0)
    avg_test_pred = np.mean(all_test_preds, axis=0)

    oof_auc = roc_auc_score(y, avg_oof)
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (averaged over %d seeds)", len(FINAL_SEEDS))
    logger.info("=" * 60)
    logger.info(f"Average OOF AUC: {oof_auc:.6f}")
    logger.info(f"Fold mean: {np.mean(all_fold_scores):.6f} std: {np.std(all_fold_scores):.6f}")

    # Сохраняем усредненные предсказания
    np.save(os.path.join(out_dir, "oof_pred_avg.npy"), avg_oof)
    np.save(os.path.join(out_dir, "test_pred_avg.npy"), avg_test_pred)

    # Графики
    plot_hist(avg_oof, os.path.join(out_dir, "hist_oof_avg.png"), "OOF predictions (averaged)")
    plot_hist(avg_test_pred, os.path.join(out_dir, "hist_test_avg.png"), "Test predictions (averaged)")

    # Оптимальный порог по F1
    thr_grid = np.linspace(0.01, 0.99, 199)
    best_thr, best_f1 = 0.5, -1.0
    for t in thr_grid:
        f1 = f1_score(y, (avg_oof >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)

    logger.info(f"Best threshold by F1: thr={best_thr:.4f} f1={best_f1:.6f}")

    return oof_auc, all_fold_scores, avg_test_pred, best_thr, best_f1


def maybe_update_best_run(
    current_run_dir: str,
    current_oof_auc: float,
    logger: logging.Logger
) -> None:
    """Обновляет лучший run если текущий лучше."""
    best_dir = os.path.join(BASE_OUTDIR, "best")
    best_summary_path = os.path.join(best_dir, "cv_summary.json")

    prev = safe_read_json(best_summary_path)
    prev_auc = float(prev.get("metrics", {}).get("oof_auc_avg", -1e9)) if isinstance(prev, dict) else -1e9

    if current_oof_auc > prev_auc:
        logger.info("New BEST run: %.6f > %.6f", current_oof_auc, prev_auc)
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(current_run_dir, best_dir)
    else:
        logger.info("Best not improved: current=%.6f best=%.6f", current_oof_auc, prev_auc)


def main():
    parser = argparse.ArgumentParser(
        description="InsClass CatBoost + Optuna + Pseudo-Labeling"
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help="Optuna trials count (default: 60)"
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Optuna timeout in seconds"
    )
    parser.add_argument(
        "--device", type=str, default="gpu", choices=["gpu", "cpu"],
        help="Device to use: gpu or cpu (default: gpu)"
    )
    parser.add_argument(
        "--no-pseudo", action="store_true",
        help="Disable pseudo-labeling"
    )
    parser.add_argument(
        "--pseudo-path", type=str, default=PSEUDO_PATH,
        help=f"Path to pseudo-labels file (default: {PSEUDO_PATH})"
    )
    args = parser.parse_args()

    logger = setup_logging()
    ensure_dir(BASE_OUTDIR)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_OUTDIR, run_ts)
    ensure_dir(out_dir)
    logger.info("Run dir: %s", out_dir)
    logger.info("Device: %s", args.device)

    # Загрузка данных
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    logger.info("Loaded train=%s test=%s", train.shape, test.shape)

    # Препроцессинг
    X, y, X_test, test_id, cat_idx = preprocess(train, test, logger)
    logger.info("After preprocess: X=%s X_test=%s", X.shape, X_test.shape)

    # Загрузка pseudo-labels
    X_pseudo, y_pseudo = None, None
    if not args.no_pseudo:
        X_pseudo, y_pseudo = load_pseudo_labels(args.pseudo_path, X_test, logger)

    # Optuna tuning
    best_params, best_cv_auc, study_db = tune_with_optuna(
        X=X, y=y,
        X_pseudo=X_pseudo, y_pseudo=y_pseudo,
        cat_idx=cat_idx,
        device=args.device,
        out_dir=out_dir,
        trials=args.trials,
        timeout=args.timeout,
        logger=logger
    )

    # Финальное обучение
    tuned_params_for_summary = dict(best_params)
    oof_auc, fold_aucs, test_pred, best_thr, best_f1 = final_train_and_predict(
        X=X, y=y, X_test=X_test,
        X_pseudo=X_pseudo, y_pseudo=y_pseudo,
        cat_idx=cat_idx,
        best_params=dict(best_params),
        device=args.device,
        out_dir=out_dir,
        logger=logger
    )

    # Сохранение submissions
    # Probability submission
    sub_proba = pd.DataFrame({"id": test_id, "target": test_pred})
    sub_proba_path = os.path.join(out_dir, "submission_proba.csv")
    sub_proba.to_csv(sub_proba_path, index=False)

    # Binary submission
    sub_bin = pd.DataFrame({"id": test_id, "target": (test_pred >= best_thr).astype(int)})
    sub_bin_path = os.path.join(out_dir, "submission.csv")
    sub_bin.to_csv(sub_bin_path, index=False)

    logger.info("Saved submissions: %s and %s", sub_bin_path, sub_proba_path)
    logger.info("Target distribution: %s", sub_bin["target"].value_counts().to_dict())

    # Summary
    summary = {
        "files": {
            "train_path": TRAIN_PATH,
            "test_path": TEST_PATH,
            "pseudo_path": args.pseudo_path if not args.no_pseudo else None,
        },
        "output_dir": out_dir,
        "device": args.device,
        "seeds": FINAL_SEEDS,
        "tuning": {
            "trials": int(args.trials),
            "timeout_sec": args.timeout,
            "tune_splits": TUNE_SPLITS,
            "best_cv_auc_mean": float(best_cv_auc),
            "study_db": study_db,
            "best_params": tuned_params_for_summary,
        },
        "metrics": {
            "oof_auc_avg": float(oof_auc),
            "fold_auc_all": fold_aucs,
            "fold_auc_mean": float(np.mean(fold_aucs)),
            "fold_auc_std": float(np.std(fold_aucs)),
        },
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

    # Обновляем лучший run
    maybe_update_best_run(out_dir, oof_auc, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
