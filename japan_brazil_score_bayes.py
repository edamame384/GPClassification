"""
日本対ブラジル スコア予測 - 二変量正規-正規ベイズ更新

未知量  : μ = (μ_J, μ_B) ... 各チームの期待得点（試合ごとの真の平均）
事前分布 : μ ~ BivariateNormal(μ_0, Σ_0)  [国際サッカー一般統計から設定]
尤度    : x_i ~ BivariateNormal(μ, Σ_obs)  [過去の対戦スコアを観測値として使用]
事後分布 : μ | data ~ BivariateNormal(μ_n, Σ_n)

正規-正規共役更新式（精度行列表現）:
  Λ_0  = Σ_0^{-1}
  Λ    = Σ_obs^{-1}
  Σ_n  = (Λ_0 + n Λ)^{-1}
  μ_n  = Σ_n (Λ_0 μ_0 + n Λ x̄)

事後予測分布（次の試合のスコア X_new）:
  X_new | data ~ BivariateNormal(μ_n, Σ_obs + Σ_n)

使い方:
  python japan_brazil_score_bayes.py
  python japan_brazil_score_bayes.py --mu0-japan 0.8 --mu0-brazil 2.5 --sigma0 1.5
"""

import argparse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    import japanize_matplotlib  # noqa: F401
    HAS_JP = True
except ImportError:
    HAS_JP = False

# ====================================================================
# 過去の対戦スコアデータ（概算 - FIFA/JFA公式記録での確認を推奨）
# 形式: (日本得点, ブラジル得点, 年, 大会名)
# ====================================================================
HISTORICAL_SCORES = [
    (0, 4, 1981, "Friendly"),
    (0, 3, 1990, "Friendly"),
    (0, 3, 1995, "Friendly"),
    (0, 2, 1999, "Confederations Cup"),
    (0, 0, 2001, "Confederations Cup"),
    (2, 1, 2003, "Friendly"),            # 日本の歴史的勝利
    (0, 1, 2004, "Friendly"),
    (1, 4, 2006, "World Cup"),
    (0, 3, 2008, "Friendly"),
    (0, 1, 2010, "Friendly"),
    (0, 4, 2013, "Confederations Cup"),
    (1, 1, 2014, "Friendly"),
    (0, 2, 2016, "Olympics"),
    (1, 3, 2019, "Friendly"),
    (0, 0, 2021, "Friendly"),
    (0, 1, 2022, "Friendly"),
]

# ====================================================================
# 事前分布のデフォルト値
# μ_0 : 事前の期待得点 (日本, ブラジル)
#        国際Aマッチ平均(~1.3点/試合)を参考に、
#        対強豪戦の傾向（日本少なめ, ブラジル多め）を反映
# Σ_0 : 事前の不確実性 (σ=1.5 → 弱い事前知識)
# ====================================================================
DEFAULT_MU_0    = np.array([1.0, 2.0])
DEFAULT_SIGMA_0 = np.diag([1.5**2, 1.5**2])


# ====================================================================
# データ読み込み
# ====================================================================
def load_data() -> np.ndarray:
    return np.array([(j, b) for j, b, *_ in HISTORICAL_SCORES], dtype=float)


def print_data_summary(data: np.ndarray):
    x_bar = data.mean(axis=0)
    print(f"\n--- 過去の対戦スコア ({len(data)} 試合) ---")
    print(f"{'年':<6} {'大会':<25} {'スコア':<14} {'結果'}")
    print("-" * 56)
    for (j, b), (_, _, yr, comp) in zip(data, HISTORICAL_SCORES):
        result = "日本WIN" if j > b else ("DRAW" if j == b else "ブラジルWIN")
        print(f"{yr:<6} {comp:<25} Japan {int(j)}-{int(b)} Brazil   [{result}]")
    print(f"\n  標本平均 : 日本 {x_bar[0]:.3f} 点 / ブラジル {x_bar[1]:.3f} 点")
    n_jwin = sum(1 for j, b in data if j > b)
    n_draw = sum(1 for j, b in data if j == b)
    n_bwin = sum(1 for j, b in data if j < b)
    print(f"  通算戦績 : 日本 {n_jwin}勝 {n_draw}分 {n_bwin}敗")


# ====================================================================
# ベイズ更新（正規-正規共役）
# ====================================================================
def bayesian_update(
    data: np.ndarray,
    mu_0: np.ndarray,
    Sigma_0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    mu_n     : 事後平均
    Sigma_n  : 事後共分散
    x_bar    : 標本平均
    Sigma_obs: 観測共分散（標本共分散 + 正則化）
    """
    n = len(data)
    x_bar = data.mean(axis=0)

    S = np.cov(data.T)
    # 正則化: 小さな対角成分を加えて正定値を保証
    Sigma_obs = S + np.eye(2) * 0.05

    Lambda_0   = np.linalg.inv(Sigma_0)
    Lambda_obs = np.linalg.inv(Sigma_obs)
    Lambda_n   = Lambda_0 + n * Lambda_obs
    Sigma_n    = np.linalg.inv(Lambda_n)
    mu_n       = Sigma_n @ (Lambda_0 @ mu_0 + n * Lambda_obs @ x_bar)

    return mu_n, Sigma_n, x_bar, Sigma_obs


def print_posterior_summary(mu_0, mu_n, Sigma_n, x_bar, Sigma_obs):
    print("\n--- ベイズ更新結果 ---")
    print(f"  観測共分散 Σ_obs =")
    print(f"    [[{Sigma_obs[0,0]:.3f}, {Sigma_obs[0,1]:.3f}],")
    print(f"     [{Sigma_obs[1,0]:.3f}, {Sigma_obs[1,1]:.3f}]]")
    print(f"  共分散の相関 ρ = {Sigma_obs[0,1] / np.sqrt(Sigma_obs[0,0]*Sigma_obs[1,1]):.3f}")
    print()
    print(f"  {'':12} {'日本 μ_J':>12} {'ブラジル μ_B':>12}")
    print(f"  {'事前 μ_0':12} {mu_0[0]:>12.3f} {mu_0[1]:>12.3f}")
    print(f"  {'標本平均 x̄':12} {x_bar[0]:>12.3f} {x_bar[1]:>12.3f}")
    print(f"  {'事後 μ_n':12} {mu_n[0]:>12.3f} {mu_n[1]:>12.3f}")
    print()
    print(f"  事後標準偏差 σ(μ_J) = {np.sqrt(Sigma_n[0,0]):.4f}  "
          f"σ(μ_B) = {np.sqrt(Sigma_n[1,1]):.4f}")
    rho_post = Sigma_n[0,1] / np.sqrt(Sigma_n[0,0] * Sigma_n[1,1])
    print(f"  事後共分散の相関 ρ_n = {rho_post:.4f}")


# ====================================================================
# 事後予測分布 → スコア確率ヒートマップ
# ====================================================================
def score_probabilities(
    mu_pred: np.ndarray,
    Sigma_pred: np.ndarray,
    max_goals: int = 6,
) -> np.ndarray:
    """
    P(Japan=j, Brazil=b) を矩形積分で計算する。
    probs[j, b] = P(j-0.5 ≤ J ≤ j+0.5, b-0.5 ≤ B ≤ b+0.5)
    """
    rv = stats.multivariate_normal(mean=mu_pred, cov=Sigma_pred)
    probs = np.zeros((max_goals + 1, max_goals + 1))

    for j in range(max_goals + 1):
        for b in range(max_goals + 1):
            lo = [max(j - 0.5, -0.5), max(b - 0.5, -0.5)]
            hi = [j + 0.5, b + 0.5]
            # 包除原理: P(lo ≤ X ≤ hi) = CDF(hi) - CDF(lo[0],hi[1]) - CDF(hi[0],lo[1]) + CDF(lo)
            p = (rv.cdf(hi)
                 - rv.cdf([lo[0], hi[1]])
                 - rv.cdf([hi[0], lo[1]])
                 + rv.cdf(lo))
            probs[j, b] = max(0.0, p)

    probs /= probs.sum()
    return probs


def print_score_summary(probs: np.ndarray):
    mg = probs.shape[0] - 1
    p_japan  = sum(probs[j, b] for j in range(mg+1) for b in range(mg+1) if j > b)
    p_draw   = sum(probs[j, b] for j in range(mg+1) for b in range(mg+1) if j == b)
    p_brazil = sum(probs[j, b] for j in range(mg+1) for b in range(mg+1) if j < b)

    print("\n--- 事後予測確率（次の試合） ---")
    print(f"  日本勝利    : {p_japan*100:6.2f}%")
    print(f"  引き分け    : {p_draw*100:6.2f}%")
    print(f"  ブラジル勝利: {p_brazil*100:6.2f}%")

    print("\n  最も確率の高いスコア（上位10件）:")
    scores = sorted(
        [(probs[j, b], j, b) for j in range(mg+1) for b in range(mg+1)],
        reverse=True,
    )
    for rank, (p, j, b) in enumerate(scores[:10], 1):
        mark = " ← 最高確率" if rank == 1 else ""
        print(f"  {rank:2d}. Japan {j}-{b} Brazil  {p*100:6.2f}%{mark}")

    return p_japan, p_draw, p_brazil


# ====================================================================
# 可視化
# ====================================================================
def _contour_grid(mu, cov, x_range, y_range):
    XX, YY = np.meshgrid(x_range, y_range)
    pos = np.dstack([XX, YY])
    rv = stats.multivariate_normal(mean=mu, cov=cov)
    return XX, YY, rv.pdf(pos)


def visualize(
    data: np.ndarray,
    mu_0: np.ndarray, Sigma_0: np.ndarray,
    mu_n: np.ndarray, Sigma_n: np.ndarray,
    Sigma_obs: np.ndarray,
    probs: np.ndarray,
    output: str = "japan_brazil_score_posterior.png",
):
    lj = "日本の期待得点 μ_J" if HAS_JP else "Japan expected goals μ_J"
    lb = "ブラジルの期待得点 μ_B" if HAS_JP else "Brazil expected goals μ_B"

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "日本対ブラジル スコア予測 - 二変量正規-正規ベイズ更新" if HAS_JP
        else "Japan vs Brazil Score Prediction - Bivariate Normal Bayesian Update",
        fontsize=13, y=0.99
    )
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])   # 事前分布 2D
    ax2 = fig.add_subplot(gs[0, 1])   # 事後分布 2D
    ax3 = fig.add_subplot(gs[0, 2])   # スコアヒートマップ
    ax4 = fig.add_subplot(gs[1, 0])   # 日本の得点分布（周辺）
    ax5 = fig.add_subplot(gs[1, 1])   # ブラジルの得点分布（周辺）
    ax6 = fig.add_subplot(gs[1, 2])   # 勝敗確率

    j_vals = np.linspace(-1.5, 5.5, 200)
    b_vals = np.linspace(-1.5, 7.5, 200)

    Sigma_pred = Sigma_obs + Sigma_n

    # ---- Panel 1: 事前分布 ----
    XX, YY, Z0 = _contour_grid(mu_0, Sigma_0, j_vals, b_vals)
    ax1.contourf(XX, YY, Z0, levels=8, cmap="Blues", alpha=0.7)
    ax1.contour(XX, YY, Z0, levels=5, colors="steelblue", linewidths=0.8)
    ax1.scatter(data[:, 0], data[:, 1], color="tomato", s=45, zorder=5, alpha=0.7,
                label="観測スコア" if HAS_JP else "Observed scores")
    ax1.axvline(mu_0[0], color="navy", ls="--", lw=1.2, alpha=0.6)
    ax1.axhline(mu_0[1], color="navy", ls="--", lw=1.2, alpha=0.6)
    ax1.set_xlabel(lj); ax1.set_ylabel(lb)
    ax1.set_title("事前分布 p(μ)" if HAS_JP else "Prior p(μ)")
    ax1.set_xlim(-1.5, 5.5); ax1.set_ylim(-1.5, 7.5)
    ax1.legend(fontsize=8)
    ax1.text(mu_0[0]+0.1, mu_0[1]+0.15,
             f"μ_0=({mu_0[0]:.1f},{mu_0[1]:.1f})", fontsize=8, color="navy")

    # ---- Panel 2: 事後分布 ----
    XX, YY, Zn = _contour_grid(mu_n, Sigma_n, j_vals, b_vals)
    ax2.contourf(XX, YY, Zn, levels=8, cmap="Greens", alpha=0.7)
    ax2.contour(XX, YY, Zn, levels=5, colors="darkgreen", linewidths=0.8)
    ax2.scatter(data[:, 0], data[:, 1], color="tomato", s=45, zorder=5, alpha=0.7)
    ax2.axvline(mu_n[0], color="darkgreen", ls="--", lw=1.5, alpha=0.8)
    ax2.axhline(mu_n[1], color="darkgreen", ls="--", lw=1.5, alpha=0.8)
    ax2.set_xlabel(lj); ax2.set_ylabel(lb)
    ax2.set_title("事後分布 p(μ|data)" if HAS_JP else "Posterior p(μ|data)")
    ax2.set_xlim(-1.5, 5.5); ax2.set_ylim(-1.5, 7.5)
    ax2.text(mu_n[0]+0.1, mu_n[1]+0.15,
             f"μ_n=({mu_n[0]:.2f},{mu_n[1]:.2f})", fontsize=8, color="darkgreen")

    # ---- Panel 3: スコアヒートマップ ----
    mg = probs.shape[0] - 1
    im = ax3.imshow(
        probs.T * 100,          # transpose: rows=Brazil, cols=Japan
        origin="lower", aspect="auto", cmap="YlOrRd",
        extent=[-0.5, mg + 0.5, -0.5, mg + 0.5],
        vmin=0,
    )
    plt.colorbar(im, ax=ax3, shrink=0.82, label="%")
    max_p = probs.max()
    for j in range(mg + 1):
        for b in range(mg + 1):
            v = probs[j, b] * 100
            color = "white" if probs[j, b] > max_p * 0.55 else "black"
            ax3.text(j, b, f"{v:.1f}", ha="center", va="center",
                     fontsize=7, color=color, fontweight="bold")
    # 対角線 (引き分けライン)
    ax3.plot([-0.5, mg + 0.5], [-0.5, mg + 0.5], "w--", lw=1.2, alpha=0.6)
    ax3.set_xlabel("Japan goals" if not HAS_JP else "日本の得点")
    ax3.set_ylabel("Brazil goals" if not HAS_JP else "ブラジルの得点")
    ax3.set_title("スコア確率 [%]\n（破線=引き分けライン）" if HAS_JP
                  else "Score Probability [%]\n(dashed=draw line)")
    ax3.set_xticks(range(mg + 1)); ax3.set_yticks(range(mg + 1))

    # ---- Panel 4: 日本の得点分布（周辺予測） ----
    jp_x = np.linspace(-1, 6.5, 400)
    jp_post_pdf  = stats.norm.pdf(jp_x, mu_n[0],  np.sqrt(Sigma_pred[0, 0]))
    jp_prior_pdf = stats.norm.pdf(jp_x, mu_0[0],  np.sqrt(Sigma_0[0, 0] + Sigma_obs[0, 0]))
    ax4.plot(jp_x, jp_post_pdf,  "b-",  lw=2.2, label="事後予測" if HAS_JP else "Posterior pred.")
    ax4.plot(jp_x, jp_prior_pdf, "b--", lw=1.5, alpha=0.5, label="事前予測" if HAS_JP else "Prior pred.")
    ax4.fill_between(jp_x, jp_post_pdf, alpha=0.15, color="b")
    ax4.axvline(mu_n[0], color="b", ls=":", lw=1.5,
                label=f"μ_J={mu_n[0]:.2f}")
    # 整数点での確率棒
    jp_bar_x = np.arange(0, 7)
    jp_bar_p = [probs[j, :].sum() for j in range(mg + 1)]
    ax4_twin = ax4.twinx()
    ax4_twin.bar(jp_bar_x, jp_bar_p, alpha=0.25, color="steelblue", width=0.5)
    ax4_twin.set_ylabel("P(Japan goals = k)", fontsize=8)
    ax4_twin.set_ylim(0, max(jp_bar_p) * 4)
    ax4.set_xlabel("Japan goals" if not HAS_JP else "日本の得点")
    ax4.set_ylabel("密度" if HAS_JP else "Density")
    ax4.set_title("日本の得点分布（事後予測）" if HAS_JP else "Japan Goals Predictive")
    ax4.legend(fontsize=8); ax4.set_xlim(-1, 6.5); ax4.grid(alpha=0.3)

    # ---- Panel 5: ブラジルの得点分布（周辺予測） ----
    br_x = np.linspace(-1, 9, 400)
    br_post_pdf  = stats.norm.pdf(br_x, mu_n[1],  np.sqrt(Sigma_pred[1, 1]))
    br_prior_pdf = stats.norm.pdf(br_x, mu_0[1],  np.sqrt(Sigma_0[1, 1] + Sigma_obs[1, 1]))
    ax5.plot(br_x, br_post_pdf,  "r-",  lw=2.2, label="事後予測" if HAS_JP else "Posterior pred.")
    ax5.plot(br_x, br_prior_pdf, "r--", lw=1.5, alpha=0.5, label="事前予測" if HAS_JP else "Prior pred.")
    ax5.fill_between(br_x, br_post_pdf, alpha=0.15, color="r")
    ax5.axvline(mu_n[1], color="r", ls=":", lw=1.5,
                label=f"μ_B={mu_n[1]:.2f}")
    br_bar_x = np.arange(0, mg + 1)
    br_bar_p = [probs[:, b].sum() for b in range(mg + 1)]
    ax5_twin = ax5.twinx()
    ax5_twin.bar(br_bar_x, br_bar_p, alpha=0.25, color="tomato", width=0.5)
    ax5_twin.set_ylabel("P(Brazil goals = k)", fontsize=8)
    ax5_twin.set_ylim(0, max(br_bar_p) * 4)
    ax5.set_xlabel("Brazil goals" if not HAS_JP else "ブラジルの得点")
    ax5.set_ylabel("密度" if HAS_JP else "Density")
    ax5.set_title("ブラジルの得点分布（事後予測）" if HAS_JP else "Brazil Goals Predictive")
    ax5.legend(fontsize=8); ax5.set_xlim(-1, 9); ax5.grid(alpha=0.3)

    # ---- Panel 6: 勝敗確率棒グラフ ----
    p_japan  = sum(probs[j, b] for j in range(mg+1) for b in range(mg+1) if j > b)
    p_draw   = sum(probs[j, b] for j in range(mg+1) for b in range(mg+1) if j == b)
    p_brazil = sum(probs[j, b] for j in range(mg+1) for b in range(mg+1) if j < b)
    labels_wdl = (["日本勝利", "引き分け", "ブラジル勝利"] if HAS_JP
                  else ["Japan Win", "Draw", "Brazil Win"])
    vals_wdl   = [p_japan, p_draw, p_brazil]
    colors_wdl = ["#2E75B6", "#999999", "#C00000"]
    bars = ax6.bar(labels_wdl, vals_wdl, color=colors_wdl, alpha=0.88)
    for bar, v in zip(bars, vals_wdl):
        ax6.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                 f"{v*100:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax6.set_ylim(0, 1.05)
    ax6.set_ylabel("確率" if HAS_JP else "Probability")
    ax6.set_title("勝敗確率（事後予測）" if HAS_JP else "Win/Draw/Loss Probability")
    ax6.grid(axis="y", alpha=0.3)

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\n図を保存しました: {output}")
    return output


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="日本対ブラジル スコア予測（二変量正規-正規ベイズ更新）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python japan_brazil_score_bayes.py
  python japan_brazil_score_bayes.py --mu0-japan 0.5 --mu0-brazil 3.0
  python japan_brazil_score_bayes.py --sigma0 2.0   # より弱い事前知識
        """,
    )
    p.add_argument("--mu0-japan",  type=float, default=DEFAULT_MU_0[0],
                   help=f"事前の期待得点（日本）（デフォルト: {DEFAULT_MU_0[0]}）")
    p.add_argument("--mu0-brazil", type=float, default=DEFAULT_MU_0[1],
                   help=f"事前の期待得点（ブラジル）（デフォルト: {DEFAULT_MU_0[1]}）")
    p.add_argument("--sigma0",     type=float, default=1.5,
                   help="事前分布の標準偏差 σ（デフォルト: 1.5）")
    p.add_argument("--output",     default="japan_brazil_score_posterior.png")
    return p.parse_args()


def main():
    args = parse_args()
    mu_0    = np.array([args.mu0_japan, args.mu0_brazil])
    Sigma_0 = np.diag([args.sigma0**2, args.sigma0**2])

    print("=" * 62)
    print("  日本対ブラジル スコア予測")
    print("  モデル: 二変量正規-正規共役ベイズ更新")
    print("=" * 62)

    data = load_data()
    print_data_summary(data)

    mu_n, Sigma_n, x_bar, Sigma_obs = bayesian_update(data, mu_0, Sigma_0)
    print_posterior_summary(mu_0, mu_n, Sigma_n, x_bar, Sigma_obs)

    Sigma_pred = Sigma_obs + Sigma_n
    probs = score_probabilities(mu_n, Sigma_pred)

    print("\n" + "=" * 62)
    print("  スコア予測サマリ")
    print("=" * 62)
    print_score_summary(probs)

    visualize(data, mu_0, Sigma_0, mu_n, Sigma_n, Sigma_obs, probs, output=args.output)


if __name__ == "__main__":
    main()
