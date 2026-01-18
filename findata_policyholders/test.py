#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier


# =========================
# Paths
# =========================
TRAIN_PATH  = "insclass_train.csv"
TEST_PATH   = "insclass_test.csv"
PSEUDO_PATH = "score_068192.csv"   # лучший submission / предсказания для pseudo


# =========================
# Optuna best
# =========================
OPTUNA_BEST = {
    "best_value": 0.7460452268088806,
    "best_params": {
        "pseudo_weight": 0.32320293202075523,
        "pseudo_confidence_threshold": 0.20751624869734644,
        "bootstrap_type": "MVS",
        "grow_policy": "Depthwise",
        "max_ctr_complexity": 3,
        "border_count": 99,
        "iterations": 4848,
        "learning_rate": 0.011050512402941122,
        "depth": 8,
        "l2_leaf_reg": 7.145565133513967,
        "random_strength": 0.25739375624994676,
        "min_data_in_leaf": 28,
        "one_hot_max_size": 23,
        "auto_class_weights": None,
        "rsm": 0.9928252270553004,
        "subsample": 0.6210276357557503,
        "early_stopping_rounds": 369
    },
    "completed_trials": 15
}


def align_pseudo_to_test(pseudo_sub: pd.DataFrame, test_ids: np.ndarray) -> np.ndarray:
    """
    Возвращает pseudo_pred_proba в порядке test_ids.
    Поддерживает варианты:
      - pseudo_sub содержит колонку id и target
      - pseudo_sub содержит только target в правильном порядке
    """
    pseudo_sub = pseudo_sub.copy()
    pseudo_sub.columns = pseudo_sub.columns.str.strip()

    if "target" not in pseudo_sub.columns:
        raise ValueError("В PSEUDO_PATH нет колонки 'target'.")

    if "id" in pseudo_sub.columns:
        s = pseudo_sub.set_index("id")["target"]
        # reindex гарантирует порядок как в тесте
        s = s.reindex(test_ids)
        if s.isna().any():
            missing = int(s.isna().sum())
            raise ValueError(f"В pseudo_sub отсутствуют {missing} id из теста (или есть лишние/дубли).")
        return s.astype(float).values

    # иначе считаем, что порядок уже совпадает с test_ids
    if len(pseudo_sub) != len(test_ids):
        raise ValueError(f"Длина pseudo_sub ({len(pseudo_sub)}) != длины test ({len(test_ids)}), "
                         f"и при этом нет колонки 'id'.")
    return pseudo_sub["target"].astype(float).values


def main():
    # =========================
    # Load
    # =========================
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    pseudo_sub = pd.read_csv(PSEUDO_PATH)

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    y = train["target"].astype(int).values
    X = train.drop(columns=["target"]).copy()

    test_id = test["id"].values
    X_test = test.drop(columns=["id"]).copy()

    # =========================
    # Preprocess (как у тебя, но аккуратнее)
    # =========================
    drop_cols = []
    for c in X.columns:
        if X[c].nunique(dropna=False) <= 1:
            drop_cols.append(c)

    if "variable_15" in X_test.columns and X_test["variable_15"].isna().mean() > 0.999:
        drop_cols.append("variable_15")

    drop_cols = sorted(set(drop_cols))
    X.drop(columns=drop_cols, inplace=True, errors="ignore")
    X_test.drop(columns=drop_cols, inplace=True, errors="ignore")

    # категориальные в string
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        X_test[c] = X_test[c].astype("string").fillna("__MISSING__")

    # missing indicators для числовых/любых NA-колонок
    na_cols = [c for c in X.columns if X[c].isna().any()]
    for c in na_cols:
        X[c + "__isna"] = X[c].isna().astype(np.uint8)
        X_test[c + "__isna"] = X_test[c].isna().astype(np.uint8)

    # cat feature list для CatBoost
    cat_cols2 = [c for c in X.columns if (X[c].dtype == "string" or X[c].dtype == "object")]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols2]

    # =========================
    # Pseudo-labeling (confidence filtering)
    # =========================
    best_params = dict(OPTUNA_BEST["best_params"])

    PSEUDO_WEIGHT = float(best_params.pop("pseudo_weight"))
    PSEUDO_CONF_THR = float(best_params.pop("pseudo_confidence_threshold"))
    EARLY_STOP = int(best_params.pop("early_stopping_rounds"))

    # если None — лучше убрать ключ совсем
    if best_params.get("auto_class_weights", None) is None:
        best_params.pop("auto_class_weights", None)

    # pseudo predictions (желательно probabilities)
    pseudo_pred = align_pseudo_to_test(pseudo_sub, test_id)

    # confidence: |p-0.5|
    # если pseudo_pred бинарный 0/1 — confidence всегда 0.5, значит фильтр оставит всё (это нормально)
    conf = np.abs(pseudo_pred - 0.5)
    mask_conf = conf >= PSEUDO_CONF_THR

    X_pseudo = X_test.loc[mask_conf].reset_index(drop=True)
    pseudo_pred_f = pseudo_pred[mask_conf]
    y_pseudo = (pseudo_pred_f >= 0.5).astype(int)

    print(f"Original train size: {len(X)}")
    print(f"Test size: {len(X_test)}")
    print(f"Pseudo kept: {len(X_pseudo)} / {len(X_test)} ({len(X_pseudo)/max(1,len(X_test)):.1%}) "
          f"with conf_thr={PSEUDO_CONF_THR:.6f}")
    print(f"Pseudo labels distribution (kept): {pd.Series(y_pseudo).value_counts().to_dict()}")

    # =========================
    # CatBoost params (из Optuna)
    # =========================
    cb_params_base = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=200,
        thread_count=-1,
        **best_params
    )

    # =========================
    # Training with pseudo-labels
    # =========================
    seeds = [7896, 99999999, 568, 789]
    all_oof = []
    all_test_preds = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training seed={seed} | pseudo_weight={PSEUDO_WEIGHT:.6f} | early_stop={EARLY_STOP}")
        print(f"{'='*60}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        oof = np.zeros(len(X), dtype=float)
        test_pred = np.zeros(len(X_test), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n--- Seed {seed}, Fold {fold}/5 ---")

            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            # сколько pseudo добавляем
            n_pseudo_sample = int(len(X_tr) * PSEUDO_WEIGHT)
            n_pseudo_sample = min(n_pseudo_sample, len(X_pseudo))

            if n_pseudo_sample > 0:
                rng = np.random.RandomState(seed + fold)
                pseudo_idx = rng.choice(len(X_pseudo), n_pseudo_sample, replace=False)

                X_tr_aug = pd.concat([X_tr, X_pseudo.iloc[pseudo_idx]], ignore_index=True)
                y_tr_aug = np.concatenate([y_tr, y_pseudo[pseudo_idx]])
            else:
                X_tr_aug = X_tr.copy()
                y_tr_aug = y_tr.copy()

            print(f"  Train aug size: {len(X_tr)} + {n_pseudo_sample} pseudo = {len(X_tr_aug)}")

            cat_idx_aug = [X_tr_aug.columns.get_loc(c) for c in cat_cols2]

            cb_params = dict(cb_params_base)
            cb_params["random_seed"] = seed

            model = CatBoostClassifier(**cb_params)
            model.fit(
                X_tr_aug, y_tr_aug,
                eval_set=(X_va, y_va),
                cat_features=cat_idx_aug,
                use_best_model=True,
                early_stopping_rounds=EARLY_STOP,
            )

            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits

        oof_auc = roc_auc_score(y, oof)
        print(f"\nSeed {seed} OOF AUC: {oof_auc:.6f}")

        all_oof.append(oof)
        all_test_preds.append(test_pred)

    # =========================
    # Ensembling + Submission
    # =========================
    print(f"\n{'='*60}")
    print("RESULTS WITH PSEUDO-LABELING (PROBABILITIES SUBMISSION)")
    print(f"{'='*60}")

    avg_oof = np.mean(all_oof, axis=0)
    avg_oof_auc = roc_auc_score(y, avg_oof)
    print(f"Average OOF AUC: {avg_oof_auc:.6f}")

    avg_test_pred = np.mean(all_test_preds, axis=0)

    # ВАЖНО: для AUC в submission почти всегда нужны вероятности, НЕ 0/1.
    sub = pd.DataFrame({"id": test_id, "target": avg_test_pred})
    out_path = "submission_pseudo_probs.csv"
    sub.to_csv(out_path, index=False)

    print(f"\nSubmission saved: {out_path}")
    print(f"Pred stats: min={avg_test_pred.min():.6f}, max={avg_test_pred.max():.6f}, mean={avg_test_pred.mean():.6f}")


if __name__ == "__main__":
    main()
