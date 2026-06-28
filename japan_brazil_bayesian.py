"""
日本対ブラジル戦のベイズ推定（ディリクレ-多項分布モデル）

事前分布: WINNERオッズ → ディリクレ分布のαパラメータに変換
尤度    : 日本対ブラジル戦の過去の通算戦績 → 多項分布
事後分布: ディリクレ分布（共役事前分布のため解析的に計算可能）

ディリクレ-多項分布モデルの更新式:
  α_posterior = α_prior + counts

使い方:
  python japan_brazil_bayesian.py
  python japan_brazil_bayesian.py --concentration 10.0
  python japan_brazil_bayesian.py --japan-wins 2 --draws 3 --brazil-wins 11
"""

import argparse
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import japanize_matplotlib  # noqa: F401
    HAS_JP_FONT = True
except ImportError:
    HAS_JP_FONT = False


# ============================================================
# WINNERオッズ（例値）
# ※ 実際の試合前にWINNER公式サイトの最新オッズへ更新してください
# ============================================================
WINNER_ODDS = {
    "Japan Win":   12.0,
    "Draw":         5.5,
    "Brazil Win":   1.35,
}

# ラベル（日本語フォントがある場合のみ使用）
JP_LABELS = ["日本勝利", "引き分け", "ブラジル勝利"]

# ============================================================
# 日本対ブラジル 過去の通算戦績（概算）
# ※ FIFA/JFA公式記録で最新の数字に更新してください
#   参考: 1970年代以降の全対戦（親善試合・国際大会を含む）
#   W=1 / D=3 / L=12 は2023年末時点の概算値
# ============================================================
DEFAULT_HISTORICAL = {
    "japan_wins":   1,
    "draws":        3,
    "brazil_wins": 12,
}


# ============================================================
# オッズ → ディリクレ事前分布
# ============================================================
def odds_to_prior(odds_dict: dict, concentration: float) -> tuple[list, np.ndarray, np.ndarray]:
    """
    implied_prob_i = 1 / odds_i  （オーバーラウンドを含む）
    正規化後: α_i = p_i × concentration

    concentration（κ）はオッズ情報の「等価試合数」を意味する。
    例) κ=5 → オッズ情報は5試合分の根拠に相当
    """
    names = list(odds_dict.keys())
    raw = np.array([1.0 / v for v in odds_dict.values()])
    overround = raw.sum()
    probs = raw / overround
    alphas = probs * concentration

    print(f"\n--- オッズ → 事前ディリクレ分布 (κ={concentration}) ---")
    print(f"{'結果':<15} {'オッズ':>8} {'含意確率':>10} {'α_prior':>10}")
    print("-" * 46)
    for n, o, p, a in zip(names, odds_dict.values(), probs, alphas):
        print(f"{n:<15} {o:>8.2f} {p:>10.4f} {a:>10.4f}")
    print(f"  オーバーラウンド: {overround:.4f}  (ブックメーカーの利幅込み)")
    return names, probs, alphas


# ============================================================
# 過去の戦績（尤度）
# ============================================================
def get_historical(wins: int, draws: int, losses: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([wins, draws, losses], dtype=float)
    total = counts.sum()
    mle = counts / total

    print(f"\n--- 過去の通算戦績（尤度） ---")
    labels = ["Japan Win", "Draw", "Brazil Win"]
    print(f"{'結果':<15} {'試合数':>8} {'MLE確率':>10}")
    print("-" * 36)
    for n, c, p in zip(labels, counts, mle):
        print(f"{n:<15} {c:>8.0f} {p:>10.4f}")
    print(f"{'合計':<15} {total:>8.0f}")
    return counts, mle


# ============================================================
# 事後分布（ディリクレ更新）
# ============================================================
def compute_posterior(alpha_prior: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha_post = alpha_prior + counts
    mean_post = alpha_post / alpha_post.sum()
    return alpha_post, mean_post


# ============================================================
# 結果サマリ
# ============================================================
def print_summary(names, prior_probs, mle_probs, alpha_prior, alpha_post, post_probs):
    alpha0 = alpha_post.sum()
    print(f"\n{'='*62}")
    print(f"  事後分布サマリ")
    print(f"{'='*62}")
    print(f"{'結果':<15} {'事前(オッズ)':>12} {'尤度(戦績)':>12} {'事後(更新後)':>12}")
    print("-" * 54)
    for n, pr, ml, po in zip(names, prior_probs, mle_probs, post_probs):
        print(f"{n:<15} {pr:>12.4f} {ml:>12.4f} {po:>12.4f}")

    print(f"\n  α_prior  = {np.round(alpha_prior, 3)}")
    print(f"  counts   = {[int(x) for x in (alpha_post - alpha_prior)]}")
    print(f"  α_post   = {np.round(alpha_post, 3)}")

    print(f"\n  95%信頼区間（周辺ベータ分布から）:")
    for n, a in zip(names, alpha_post):
        b = alpha0 - a
        mu = a / alpha0
        lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
        print(f"    {n:<15}: {mu:.4f}  [95% CI: {lo:.4f} – {hi:.4f}]")


# ============================================================
# 可視化
# ============================================================
def visualize(names, prior_probs, mle_probs, alpha_post, post_probs, output="japan_brazil_posterior.png"):
    if not HAS_MATPLOTLIB:
        print("\n[info] matplotlib が見つかりません。pip install matplotlib で可視化できます。")
        return None

    labels = JP_LABELS if HAS_JP_FONT else names
    title_main = "日本対ブラジル ベイズ推定" if HAS_JP_FONT else "Japan vs Brazil Bayesian Estimation"
    title_sub = "(事前: WINNERオッズ / 尤度: 過去の通算戦績)" if HAS_JP_FONT else "(Prior: WINNER odds / Likelihood: historical record)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{title_main}\n{title_sub}", fontsize=12)

    palette = ["#5B9BD5", "#ED7D31", "#70AD47"]
    x = np.arange(len(names))
    w = 0.26

    # 左: 棒グラフ
    ax = axes[0]
    bl1 = "Prior (WINNER odds)" if not HAS_JP_FONT else "事前分布（WINNERオッズ）"
    bl2 = "Likelihood MLE (history)" if not HAS_JP_FONT else "尤度MLE（過去戦績）"
    bl3 = "Posterior" if not HAS_JP_FONT else "事後分布"
    ax.bar(x - w, prior_probs, w, label=bl1, color=palette[0], alpha=0.85)
    ax.bar(x,     mle_probs,   w, label=bl2, color=palette[1], alpha=0.85)
    ax.bar(x + w, post_probs,  w, label=bl3, color=palette[2], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Probability Comparison" if not HAS_JP_FONT else "各結果の確率比較")
    ax.grid(axis="y", alpha=0.3)

    # 右: 事後周辺ベータ分布
    ax2 = axes[1]
    t = np.linspace(0, 1, 400)
    alpha0 = alpha_post.sum()
    for i, (lbl, a) in enumerate(zip(labels, alpha_post)):
        b = alpha0 - a
        pdf = stats.beta.pdf(t, a, b)
        ax2.plot(t, pdf, color=palette[i], lw=2.2, label=lbl)
        mu = a / alpha0
        ax2.axvline(mu, color=palette[i], ls="--", lw=1.2, alpha=0.6)
    ax2.set_xlabel("Probability p")
    ax2.set_ylabel("Density")
    ax2.set_title(
        "Posterior Marginal Beta Distributions\n(dashed = posterior mean)"
        if not HAS_JP_FONT
        else "事後周辺ベータ分布\n（破線 = 事後平均）"
    )
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\n図を保存しました: {output}")
    return output


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="日本対ブラジル戦のベイズ推定（ディリクレ-多項分布モデル）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python japan_brazil_bayesian.py
  python japan_brazil_bayesian.py --concentration 10
  python japan_brazil_bayesian.py --japan-wins 2 --draws 3 --brazil-wins 11
  python japan_brazil_bayesian.py --japan-odds 15.0 --draw-odds 5.0 --brazil-odds 1.25
        """,
    )
    p.add_argument("--japan-odds",   type=float, default=WINNER_ODDS["Japan Win"],
                   help=f"WINNERの日本勝利オッズ（デフォルト: {WINNER_ODDS['Japan Win']}）")
    p.add_argument("--draw-odds",    type=float, default=WINNER_ODDS["Draw"],
                   help=f"WINNERの引き分けオッズ（デフォルト: {WINNER_ODDS['Draw']}）")
    p.add_argument("--brazil-odds",  type=float, default=WINNER_ODDS["Brazil Win"],
                   help=f"WINNERのブラジル勝利オッズ（デフォルト: {WINNER_ODDS['Brazil Win']}）")
    p.add_argument("--japan-wins",   type=int, default=DEFAULT_HISTORICAL["japan_wins"],
                   help=f"日本の勝ち数（デフォルト: {DEFAULT_HISTORICAL['japan_wins']}）")
    p.add_argument("--draws",        type=int, default=DEFAULT_HISTORICAL["draws"],
                   help=f"引き分け数（デフォルト: {DEFAULT_HISTORICAL['draws']}）")
    p.add_argument("--brazil-wins",  type=int, default=DEFAULT_HISTORICAL["brazil_wins"],
                   help=f"ブラジルの勝ち数（デフォルト: {DEFAULT_HISTORICAL['brazil_wins']}）")
    p.add_argument("--concentration", type=float, default=5.0,
                   help="事前分布の強さ κ（等価試合数、デフォルト: 5.0）")
    p.add_argument("--output", default="japan_brazil_posterior.png",
                   help="出力画像ファイル名")
    return p.parse_args()


def main():
    args = parse_args()

    odds = {
        "Japan Win":  args.japan_odds,
        "Draw":       args.draw_odds,
        "Brazil Win": args.brazil_odds,
    }

    print("=" * 62)
    print("  日本対ブラジル戦 ベイズ推定")
    print("  モデル: ディリクレ-多項分布（共役事前分布）")
    print("=" * 62)

    names, prior_probs, alpha_prior = odds_to_prior(odds, args.concentration)
    counts, mle_probs = get_historical(args.japan_wins, args.draws, args.brazil_wins)
    alpha_post, post_probs = compute_posterior(alpha_prior, counts)
    print_summary(names, prior_probs, mle_probs, alpha_prior, alpha_post, post_probs)
    visualize(names, prior_probs, mle_probs, alpha_post, post_probs, output=args.output)


if __name__ == "__main__":
    main()
