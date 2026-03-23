"""
ベイズ最適化による実験候補点提案システム
アミン化合物の修飾実験（水溶性最適化）

実験概要:
  - 化合物A〜Fから2種類を選び、アミン化合物を修飾する
  - 各化合物の修飾率の合計は100（整数）
  - 非ゼロになるのはちょうど2変数のみ
  - 目的: 水溶性生成物（1=溶ける）の探索

使い方:
  python bayesian_optimization.py                          # デフォルトファイルで実行
  python bayesian_optimization.py -i results.csv           # 入力ファイルを指定
  python bayesian_optimization.py -i results.csv -o next.csv  # 出力ファイルも指定
  python bayesian_optimization.py -n 10                    # 提案数を指定
  python bayesian_optimization.py --init-csv               # サンプル入力CSVを生成

入力CSVフォーマット（必須列）:
  A, B, C, D, E, F  ... 各化合物の修飾率（整数、合計=100、非ゼロは2列のみ）
  soluble            ... 水溶性結果（1=溶ける, 0=溶けない）

出力CSVフォーマット:
  化合物A修飾率(%), ..., 化合物F修飾率(%), 組み合わせ, 獲得スコア
"""

import argparse
import numpy as np
import pandas as pd
import csv
import os
import sys
import warnings
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# 定数
# ============================================================
COMPOUNDS = ["A", "B", "C", "D", "E", "F"]
DEFAULT_INPUT  = "observations.csv"
DEFAULT_OUTPUT = "suggestions.csv"


# ============================================================
# 探索空間の生成
# ============================================================
def generate_all_candidates(step: int = 1) -> np.ndarray:
    """
    制約を満たす全候補点を生成する。

    制約:
      - 非ゼロ変数はちょうど2つ
      - 非ゼロ変数はそれぞれ [1, 99] の整数
      - 全変数の合計 = 100
    """
    candidates = []
    for i, j in combinations(range(6), 2):
        for r in range(1, 100, step):
            row = [0] * 6
            row[i] = r
            row[j] = 100 - r
            candidates.append(row)
    return np.array(candidates, dtype=float)


# ============================================================
# データ入出力
# ============================================================
def load_observations(filepath: str):
    """
    入力CSVから観測データを読み込む。

    必須列: A, B, C, D, E, F, soluble
    """
    if not os.path.exists(filepath):
        print(f"[エラー] ファイルが見つかりません: {filepath}")
        print("  --init-csv オプションでサンプル入力CSVを生成できます。")
        sys.exit(1)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[エラー] CSV読み込み失敗: {e}")
        sys.exit(1)

    required = ["A", "B", "C", "D", "E", "F", "soluble"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[エラー] 必須列が不足しています: {missing}")
        print(f"  必須列: {required}")
        sys.exit(1)

    df = df.dropna(subset=required)
    if len(df) == 0:
        print("[警告] 有効なデータ行がありません。ランダム提案を行います。")
        return np.empty((0, 6)), np.empty(0, dtype=int)

    X = df[["A", "B", "C", "D", "E", "F"]].values.astype(float)
    y = df["soluble"].values.astype(int)
    return X, y


def save_suggestions(df: pd.DataFrame, filepath: str):
    """次回候補を出力CSVに書き出す"""
    df.to_csv(filepath, encoding="utf-8-sig")  # utf-8-sig で Excel でも文字化けしない
    print(f"次回候補を保存しました: {filepath}")


def create_sample_csv(filepath: str):
    """サンプル入力CSVファイルを生成する"""
    sample = [
        ["A",  "B",  "C",  "D",  "E",  "F",  "soluble"],
        [ 50,   50,    0,    0,    0,    0,    1],
        [ 30,    0,   70,    0,    0,    0,    0],
        [  0,    0,    0,   60,   40,    0,    1],
        [  0,   20,    0,    0,   80,    0,    0],
        [  0,    0,    0,    0,   45,   55,    1],
        [ 10,    0,    0,    0,    0,   90,    0],
        [  0,   60,    0,   40,    0,    0,    1],
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(sample)
    print(f"サンプル入力CSVを生成しました: {filepath}")
    print(f"  列: A〜F（修飾率）, soluble（1=溶ける, 0=溶けない）")


# ============================================================
# ユーティリティ
# ============================================================
def format_candidate(x) -> str:
    parts = [f"{COMPOUNDS[i]}:{int(x[i])}%" for i in range(6) if x[i] > 0]
    return " + ".join(parts)


def show_summary(X_obs: np.ndarray, y_obs: np.ndarray):
    """観測データのサマリをターミナルに表示"""
    if len(X_obs) == 0:
        print("  観測データなし")
        return
    n_pos = int(y_obs.sum())
    print(f"  件数: {len(y_obs)} 件（溶ける: {n_pos}, 溶けない: {len(y_obs) - n_pos}）")
    print()
    for x, y in zip(X_obs, y_obs):
        label = "溶ける  " if y == 1 else "溶けない"
        print(f"    {format_candidate(x):<25} → {label}")


# ============================================================
# ベイズ最適化コア
# ============================================================
def fit_gpc(X: np.ndarray, y: np.ndarray) -> GaussianProcessClassifier:
    """ガウス過程分類器を学習する"""
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=15.0,
        length_scale_bounds=(1.0, 200.0),
        nu=2.5,
    )
    gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpc.fit(X, y)
    return gpc


def acquisition_ucb(proba: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """UCB 獲得関数（logit スコア + 不確実性ボーナス）"""
    eps = 1e-6
    p = np.clip(proba, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    uncertainty = -np.abs(p - 0.5)   # 0.5 に近いほど不確実性が高い
    return logit + beta * uncertainty


def suggest_next(
    candidates: np.ndarray,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    n: int = 5,
    beta: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    次回実験候補点を提案する。

    観測データが不足（3件未満 or クラスが1種類）の場合はランダム提案。
    それ以外は GPC + UCB で上位 n 件を返す。
    """
    rng = np.random.default_rng(random_state)

    # 観測済みを除外
    if len(X_obs) > 0:
        obs_set = set(map(tuple, X_obs.astype(int).tolist()))
        mask = np.array([tuple(c.astype(int).tolist()) not in obs_set for c in candidates])
    else:
        mask = np.ones(len(candidates), dtype=bool)

    remaining = candidates[mask]

    if len(remaining) == 0:
        print("全候補点の観測が完了しています。")
        return pd.DataFrame()

    if len(X_obs) < 3 or len(np.unique(y_obs)) < 2:
        print("  （観測データ不足のためランダム提案）")
        idx = rng.choice(len(remaining), size=min(n, len(remaining)), replace=False)
        suggestions = remaining[idx]
        scores = ["-"] * len(suggestions)
    else:
        gpc = fit_gpc(X_obs, y_obs)
        proba = gpc.predict_proba(remaining)[:, 1]
        scores_all = acquisition_ucb(proba, beta=beta)
        top_idx = np.argsort(scores_all)[::-1][:n]
        suggestions = remaining[top_idx]
        scores = [round(float(scores_all[i]), 4) for i in top_idx]

    rows = []
    for s, sc in zip(suggestions, scores):
        row = {f"化合物{COMPOUNDS[j]}修飾率(%)": int(s[j]) for j in range(6)}
        row["組み合わせ"] = format_candidate(s)
        row["獲得スコア"] = sc
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1)
    df.index.name = "候補No."
    return df


# ============================================================
# エントリーポイント
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="ベイズ最適化 - アミン化合物修飾実験（水溶性最適化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python bayesian_optimization.py
  python bayesian_optimization.py -i results.csv -o next.csv
  python bayesian_optimization.py -n 10 --beta 3.0
  python bayesian_optimization.py --init-csv
        """,
    )
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT,
        metavar="FILE",
        help=f"実験結果の入力CSVファイル（デフォルト: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"次回候補の出力CSVファイル（デフォルト: {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "-n", "--num-suggestions",
        type=int,
        default=5,
        metavar="N",
        help="提案する候補点の数（デフォルト: 5）",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="UCB探索強度。大きいほど未探索領域を優先（デフォルト: 2.0）",
    )
    parser.add_argument(
        "--init-csv",
        action="store_true",
        help="サンプル入力CSVを生成して終了",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.init_csv:
        create_sample_csv(args.input)
        sys.exit(0)

    print("=" * 60)
    print("  ベイズ最適化 - アミン化合物修飾実験")
    print("=" * 60)

    # 探索空間
    print(f"\n探索空間を生成中... ", end="", flush=True)
    candidates = generate_all_candidates(step=1)
    print(f"{len(candidates):,} 候補点")

    # 入力CSV 読み込み
    print(f"\n入力ファイル: {args.input}")
    X_obs, y_obs = load_observations(args.input)
    print("\n--- 観測済みデータ ---")
    show_summary(X_obs, y_obs)

    # 次回候補を提案
    print(f"\n--- 次回実験候補（{args.num_suggestions}件）---")
    df = suggest_next(candidates, X_obs, y_obs, n=args.num_suggestions, beta=args.beta)
    print(df.to_string())

    # 出力CSV に書き出し
    print()
    save_suggestions(df, args.output)


if __name__ == "__main__":
    main()
