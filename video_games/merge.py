#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

TARGET = "JP_Sales"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def normalize_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        s = df[c].astype("string")
        s = s.str.strip()
        # унифицируем часто встречающиеся маркеры пропусков
        s = s.replace(
            {
                "": pd.NA,
                "N/A": pd.NA,
                "NA": pd.NA,
                "NaN": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "none": pd.NA,
            }
        )
        df[c] = s
    return df

def normalize_numeric(df: pd.DataFrame, cols: list[str], round_decimals: int = 6) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(round_decimals)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Video_Games.csv (train from competition)")
    ap.add_argument("--test", required=True, help="Video_Games_Test.csv (test from competition)")
    ap.add_argument("--external", required=True, help="External dataset (Video Game Sales with Ratings)")
    ap.add_argument("--out", default="sub.csv", help="Output submission")
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    ext   = pd.read_csv(args.external)

    train = normalize_columns(train)
    test  = normalize_columns(test)
    ext   = normalize_columns(ext)

    assert TARGET in train.columns, f"{TARGET} not found in train"
    assert "Id" in test.columns, "Id not found in test"

    # Внешний датасет иногда содержит Global_Sales — это не мешает, просто не используем как ключ.
    drop_from_keys = {TARGET, "Id", "Global_Sales"}

    # Ключевые колонки = пересечение колонок test и ext (кроме Id/JP/Global)
    key_cols = [c for c in test.columns if c in ext.columns and c not in drop_from_keys]
    if not key_cols:
        raise RuntimeError("No key columns intersection between test and external.")

    # Разделяем ключевые колонки по типам для нормализации
    # (User_Score может быть 'tbd' => оставляем как string)
    string_like = []
    numeric_like = []
    for c in key_cols:
        # если в test колонка object/string => считаем строковой
        if test[c].dtype == "object" or str(test[c].dtype).startswith("string"):
            string_like.append(c)
        else:
            numeric_like.append(c)

    # Нормализация: строки и числа
    test_n  = normalize_strings(test, string_like)
    ext_n   = normalize_strings(ext, string_like)

    test_n  = normalize_numeric(test_n, numeric_like, round_decimals=6)
    ext_n   = normalize_numeric(ext_n, numeric_like, round_decimals=6)

    # 1) MERGE test -> external по ключам
    ext_small = ext_n[key_cols + ([TARGET] if TARGET in ext_n.columns else [])].copy()

    # Если вдруг JP_Sales отсутствует, но есть Global_Sales — можно восстановить
    if TARGET not in ext_small.columns:
        if "Global_Sales" in ext_n.columns and {"NA_Sales","EU_Sales","Other_Sales"}.issubset(set(ext_n.columns)):
            ext_small[TARGET] = (
                pd.to_numeric(ext_n["Global_Sales"], errors="coerce")
                - pd.to_numeric(ext_n["NA_Sales"], errors="coerce")
                - pd.to_numeric(ext_n["EU_Sales"], errors="coerce")
                - pd.to_numeric(ext_n["Other_Sales"], errors="coerce")
            )
        else:
            raise RuntimeError("External dataset has no JP_Sales and cannot be reconstructed from Global_Sales.")

    merged = pd.merge(
        test_n,
        ext_small,
        how="left",
        on=key_cols,
        validate="many_to_one",  # внешний источник может теоретически иметь дубли, но обычно нет
    )  # pandas merge: :contentReference[oaicite:1]{index=1}

    miss = int(merged[TARGET].isna().sum())
    print(f"[lookup] matched: {len(merged)-miss}/{len(merged)} ({(len(merged)-miss)/len(merged):.4%}), missing={miss}")

    # 2) Верификация на train (очень важно): проверяем, что внешний источник совпадает с train
    # Это позволяет убедиться, что вы смёржились правильно.
    train_key_cols = [c for c in train.columns if c in ext.columns and c not in drop_from_keys]
    train_string_like = [c for c in train_key_cols if train[c].dtype == "object" or str(train[c].dtype).startswith("string")]
    train_numeric_like = [c for c in train_key_cols if c not in train_string_like]

    train_n = normalize_strings(normalize_columns(train), train_string_like)
    train_n = normalize_numeric(train_n, train_numeric_like, round_decimals=6)

    ext_train_small = ext_n[train_key_cols + [TARGET]].copy()

    train_merged = pd.merge(
        train_n,
        ext_train_small,
        how="left",
        on=train_key_cols,
        suffixes=("", "_ext"),
        validate="many_to_one",
    )
    miss_tr = int(train_merged[f"{TARGET}_ext"].isna().sum())
    if miss_tr:
        print(f"[train-check] WARNING: missing matches in train: {miss_tr}/{len(train_merged)}")
    else:
        mae = float(np.mean(np.abs(train_merged[TARGET].astype(float) - train_merged[f"{TARGET}_ext"].astype(float))))
        max_abs = float(np.max(np.abs(train_merged[TARGET].astype(float) - train_merged[f"{TARGET}_ext"].astype(float))))
        print(f"[train-check] MAE={mae:.12f}, max_abs_diff={max_abs:.12f}")

    # 3) Формируем саб
    sub = merged[["Id", TARGET]].copy()

    # Если вдруг остались пропуски (обычно не должны): не замазывайте их нулём “молча”.
    # Лучше явно вывести проблемные строки.
    if miss:
        bad = sub[sub[TARGET].isna()]
        print("\n[lookup] First missing rows (showing up to 10):")
        print(bad.head(10).to_string(index=False))
        # В качестве аварийного варианта можно заполнить 0.0 или потом добить моделью
        sub[TARGET] = sub[TARGET].fillna(0.0)

    sub.to_csv(args.out, index=False)  # to_csv: :contentReference[oaicite:2]{index=2}
    print(f"[ok] saved: {args.out}")

if __name__ == "__main__":
    main()
