"""
ベイズ最適化による実験候補点提案システム
アミン化合物の修飾実験（水溶性最適化）

実験概要:
  - 化合物A〜Fから2種類を選び、アミン化合物を修飾する
  - 各化合物の修飾率の合計は100（整数）
  - 非ゼロになるのはちょうど2変数のみ
  - 目的: 水溶性生成物（1=溶ける）の探索

使い方:
  python bayesian_optimization.py                        # デフォルト observations.csv を読み込む
  python bayesian_optimization.py -i results.csv         # ファイルを指定
  python bayesian_optimization.py -i results.csv -n 10   # 提案数を指定
  python bayesian_optimization.py --init-csv             # サンプルCSVを生成して終了

CSVフォーマット（必須列）:
  A, B, C, D, E, F  ... 各化合物の修飾率（整数、合計=100、非ゼロは2列のみ）
  soluble            ... 水溶性結果（1=溶ける, 0=溶けない）
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
DATA_FILE = "observations.csv"


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

    Parameters
    ----------
    step : int
        修飾率の刻み幅（デフォルト1）

    Returns
    -------
    np.ndarray, shape (N, 6)
    """
    candidates = []
    rates = list(range(1, 100, step))  # 1〜99（合計が100になるよう）
    for i, j in combinations(range(6), 2):
        for r in rates:
            row = [0] * 6
            row[i] = r
            row[j] = 100 - r
            candidates.append(row)
    return np.array(candidates, dtype=float)


# ============================================================
# データ管理
# ============================================================
def load_observations(filepath: str):
    """
    CSVから観測データを読み込む。

    必須列: A, B, C, D, E, F, soluble
    存在しない場合や列が足りない場合はエラーメッセージを出して終了する。
    """
    if not os.path.exists(filepath):
        print(f"[エラー] ファイルが見つかりません: {filepath}")
        print("  --init-csv オプションでサンプルCSVを生成できます。")
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


def save_observation(x: list, y: int, filepath: str):
    """観測結果をCSVに追記する"""
    header = ["A", "B", "C", "D", "E", "F", "soluble"]
    row = list(x) + [y]
    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def create_sample_csv(filepath: str):
    """サンプルCSVファイルを生成する"""
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
    print(f"サンプルCSVを生成しました: {filepath}")
    print(f"  列: A, B, C, D, E, F（修飾率）, soluble（1=溶ける, 0=溶けない）")


def format_candidate(x) -> str:
    """候補点を読みやすい文字列に変換"""
    parts = [f"{COMPOUNDS[i]}:{int(x[i])}%" for i in range(6) if x[i] > 0]
    return " + ".join(parts)


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
    gpc = GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42,
    )
    gpc.fit(X, y)
    return gpc


def acquisition_ucb(proba: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) 獲得関数。

    GPC はシグモイド変換後の確率を返すため、
    ここでは単純に P(y=1) を最大化する greedy + ε探索で代用する。
    beta が大きいほど探索（exploration）を重視。
    """
    # GPC は確率の不確実性を直接持たないため、
    # proba から逆シグモイドで潜在値の近似スコアを作成
    eps = 1e-6
    p = np.clip(proba, eps, 1 - eps)
    logit = np.log(p / (1 - p))   # 潜在スコアの近似
    # 探索ボーナス: 確率が 0.5 に近い点（不確実性が高い）を優遇
    uncertainty = -np.abs(p - 0.5)
    score = logit + beta * uncertainty
    return score


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

    Parameters
    ----------
    candidates : np.ndarray
        全候補点
    X_obs : np.ndarray
        観測済み説明変数
    y_obs : np.ndarray
        観測済み目的変数
    n : int
        提案数
    beta : float
        UCB の探索重み（大きいほど探索優先）

    Returns
    -------
    pd.DataFrame
        提案候補の一覧
    """
    rng = np.random.default_rng(random_state)

    # 観測済み候補を除外
    if len(X_obs) > 0:
        obs_set = set(map(tuple, X_obs.astype(int).tolist()))
        mask = np.array([tuple(c.astype(int).tolist()) not in obs_set for c in candidates])
    else:
        mask = np.ones(len(candidates), dtype=bool)

    remaining = candidates[mask]

    if len(remaining) == 0:
        print("全候補点の観測が完了しています。")
        return pd.DataFrame()

    # 観測データが不足している場合はランダム提案
    if len(X_obs) < 3 or len(np.unique(y_obs)) < 2:
        idx = rng.choice(len(remaining), size=min(n, len(remaining)), replace=False)
        suggestions = remaining[idx]
        scores = np.full(len(suggestions), np.nan)
        note = "（初期ランダム提案）"
    else:
        gpc = fit_gpc(X_obs, y_obs)
        proba = gpc.predict_proba(remaining)[:, 1]
        scores_all = acquisition_ucb(proba, beta=beta)
        top_idx = np.argsort(scores_all)[::-1][:n]
        suggestions = remaining[top_idx]
        scores = scores_all[top_idx]
        note = ""

    rows = []
    for i, (s, sc) in enumerate(zip(suggestions, scores), 1):
        row = {f"化合物{COMPOUNDS[j]}修飾率(%)": int(s[j]) for j in range(6)}
        row["組み合わせ"] = format_candidate(s)
        row["獲得スコア"] = round(float(sc), 4) if not np.isnan(sc) else "-"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
    df.index.name = "候補No."
    if note:
        print(note)
    return df


# ============================================================
# 結果サマリ
# ============================================================
def show_summary(X_obs: np.ndarray, y_obs: np.ndarray):
    """観測データのサマリを表示"""
    if len(X_obs) == 0:
        print("観測データなし")
        return

    df = pd.DataFrame(X_obs, columns=[f"化合物{c}" for c in COMPOUNDS])
    df["水溶性"] = ["溶ける" if y == 1 else "溶けない" for y in y_obs]
    df["組み合わせ"] = [format_candidate(x) for x in X_obs]
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1)
    df.index.name = "No."

    print(f"\n観測済みデータ（計 {len(df)} 件）:")
    print(df[["組み合わせ", "水溶性"]].to_string())
    n_pos = int(y_obs.sum())
    print(f"\n  溶ける: {n_pos} 件 / 溶けない: {len(y_obs) - n_pos} 件")


# ============================================================
# インタラクティブ実行
# ============================================================
def interactive_mode(candidates: np.ndarray, data_file: str):
    """対話形式で実験結果の追加・提案を繰り返す"""
    print("\n" + "=" * 60)
    print(f"対話モード  (データファイル: {data_file})")
    print("  [s] 次回候補を提案  [a] 観測データを追加")
    print("  [v] 観測データ一覧  [q] 終了")
    print("=" * 60)

    while True:
        cmd = input("\nコマンド > ").strip().lower()

        if cmd == "q":
            print("終了します。")
            break

        elif cmd == "v":
            X_obs, y_obs = load_observations(data_file)
            show_summary(X_obs, y_obs)

        elif cmd == "s":
            X_obs, y_obs = load_observations(data_file)
            show_summary(X_obs, y_obs)

            try:
                n = int(input("提案数（デフォルト5）> ").strip() or "5")
                beta = float(input("探索強度 beta（デフォルト2.0, 大きいほど探索優先）> ").strip() or "2.0")
            except ValueError:
                n, beta = 5, 2.0

            print("\n次回実験候補点:")
            df = suggest_next(candidates, X_obs, y_obs, n=n, beta=beta)
            print(df.to_string())

        elif cmd == "a":
            print("使用する2化合物とその修飾率を入力してください。")
            try:
                x = _input_condition()
                y_str = input("結果（1=溶ける / 0=溶けない）> ").strip()
                y = int(y_str)
                if y not in (0, 1):
                    raise ValueError
            except (ValueError, KeyboardInterrupt):
                print("入力を中断しました。")
                continue

            save_observation(x, y, data_file)
            print(f"保存しました: {format_candidate(x)} -> {'溶ける' if y == 1 else '溶けない'}")

        else:
            print("不明なコマンドです。")


def _input_condition() -> list:
    """実験条件を対話入力する"""
    while True:
        comp_input = input(f"化合物を2つ選択（例: A B）> ").strip().upper().split()
        if len(comp_input) == 2 and all(c in COMPOUNDS for c in comp_input) and len(set(comp_input)) == 2:
            break
        print(f"  → {COMPOUNDS} から異なる2つを入力してください。")

    c1, c2 = comp_input
    while True:
        try:
            r1 = int(input(f"化合物{c1}の修飾率 [1-99]> ").strip())
            r2 = 100 - r1
            if 1 <= r1 <= 99:
                print(f"  → 化合物{c1}: {r1}%, 化合物{c2}: {r2}%")
                break
        except ValueError:
            pass
        print("  → 1〜99 の整数を入力してください。")

    x = [0] * 6
    x[COMPOUNDS.index(c1)] = r1
    x[COMPOUNDS.index(c2)] = r2
    return x


def _print_proba_top(
    candidates: np.ndarray,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    top: int = 10,
):
    """全候補の予測確率上位を表示"""
    if len(np.unique(y_obs)) < 2:
        print("（クラスが1種類のため予測不可）")
        return
    gpc = fit_gpc(X_obs, y_obs)
    proba = gpc.predict_proba(candidates)[:, 1]
    top_idx = np.argsort(proba)[::-1][:top]
    print(f"{'組み合わせ':<28} 溶解確率")
    print("-" * 44)
    for i in top_idx:
        print(f"  {format_candidate(candidates[i]):<26} {proba[i]:.3f}")


# ============================================================
# エントリーポイント
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="ベイズ最適化 - アミン化合物修飾実験（水溶性最適化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSVフォーマット（必須列）:
  A, B, C, D, E, F  ... 各化合物の修飾率（整数、合計=100、非ゼロは2列のみ）
  soluble            ... 水溶性結果（1=溶ける, 0=溶けない）

使用例:
  python bayesian_optimization.py                        # デフォルトCSVを読み込む
  python bayesian_optimization.py -i results.csv         # ファイルを指定
  python bayesian_optimization.py -i results.csv -n 10   # 提案数10件
  python bayesian_optimization.py --init-csv             # サンプルCSVを生成
  python bayesian_optimization.py -i results.csv --interactive  # 対話モード
        """,
    )
    parser.add_argument(
        "-i", "--input",
        default=DATA_FILE,
        metavar="CSV_FILE",
        help=f"実験結果CSVファイルのパス（デフォルト: {DATA_FILE}）",
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
        "--top-proba",
        type=int,
        default=10,
        metavar="K",
        help="水溶性確率マップの表示件数（デフォルト: 10, 0で非表示）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="対話モードを起動（結果追加と提案を繰り返す）",
    )
    parser.add_argument(
        "--init-csv",
        action="store_true",
        help="サンプルCSVファイルを生成して終了",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # サンプルCSV生成モード
    if args.init_csv:
        create_sample_csv(args.input)
        sys.exit(0)

    print("=" * 60)
    print("  ベイズ最適化 - アミン化合物修飾実験")
    print("  目標: 水溶性生成物の最適組成探索")
    print("=" * 60)

    print(f"\n探索空間を生成中... ", end="", flush=True)
    candidates = generate_all_candidates(step=1)
    print(f"{len(candidates):,} 候補点")

    # 対話モード
    if args.interactive:
        interactive_mode(candidates, args.input)
        return

    # ---- 通常モード: CSVを読み込んで候補提案 ----
    print(f"\nデータファイル: {args.input}")
    X_obs, y_obs = load_observations(args.input)
    show_summary(X_obs, y_obs)

    print(f"\n--- ベイズ最適化による次回候補提案（{args.num_suggestions}件）---")
    df = suggest_next(candidates, X_obs, y_obs, n=args.num_suggestions, beta=args.beta)
    print(df.to_string())

    if args.top_proba > 0 and len(X_obs) >= 3 and len(np.unique(y_obs)) >= 2:
        print(f"\n--- 水溶性確率マップ（上位{args.top_proba}件）---")
        _print_proba_top(candidates, X_obs, y_obs, top=args.top_proba)


if __name__ == "__main__":
    main()
