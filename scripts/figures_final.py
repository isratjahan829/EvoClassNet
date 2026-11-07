"""The manuscript's main figures, drawn to a single publication style.

Each figure is a self-contained argument rather than a dump of one table. Panel letters, colour
and typography come from evoclassnet.figstyle, so the set reads as one piece of work. Every value
is read from results/; nothing here is hand-entered.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import figstyle as F

F.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures" / "final"
J = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
C = lambda n: pd.read_csv(R / n) if (R / n).exists() else pd.DataFrame()


def num(t):
    import re
    m = re.match(r"\s*([+-]?[0-9.]+)", str(t))
    return float(m.group(1)) if m else np.nan


def sd(t):
    import re
    m = re.search(r"±\s*([0-9.]+)", str(t))
    return float(m.group(1)) if m else 0.0


# ================================================================ Figure 1
def fig1():
    """The whole argument: what a random split rewards, and what survives a position-based one."""
    D = C("proteingym_per_assay.csv")
    if not len(D):
        return None
    fig = plt.figure(figsize=(F.DOUBLE, 4.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.3], hspace=0.52, wspace=0.40,
                          left=0.075, right=0.985, top=0.91, bottom=0.11)

    # --- a: the two protocols, drawn as a variant x position matrix
    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off()
    F.panel(ax, "a", dx=-0.06, dy=1.20)
    ax.set_title("Two ways to hold out data", loc="left", pad=14)
    rng = np.random.default_rng(3)
    n_pos, n_var = 14, 8
    test_block = list(range(n_pos - 4, n_pos))
    grid = np.zeros((n_var, n_pos), dtype=int)
    for i in range(n_var):
        if i in (1, 5):          # variants whose every mutation lies in the withheld block
            grid[i, rng.choice(test_block, 2, replace=False)] = 1
        elif i in (3,):          # a variant that spans the boundary, and is therefore discarded
            grid[i, rng.choice(range(n_pos - 4), 1)] = 1
            grid[i, rng.choice(test_block, 1)] = 1
        else:
            grid[i, rng.choice(range(n_pos - 4), 2, replace=False)] = 1
    cw, ch = 0.058, 0.030
    for k, (title, y0) in enumerate((("random", 0.60), ("held-out position", 0.08))):
        test_rows = set(rng.choice(n_var, 3, replace=False)) if k == 0 else set()
        test_cols = set(range(n_pos - 4, n_pos)) if k == 1 else set()
        for i in range(n_var):
            cells = np.flatnonzero(grid[i])
            if k == 0:
                state = "test" if i in test_rows else "train"
            elif all(c in test_cols for c in cells):
                state = "test"
            elif any(c in test_cols for c in cells):
                state = "dropped"
            else:
                state = "train"
            for j in cells:
                kw = dict(lw=0, color={"test": F.OI["vermillion"], "train": F.OI["sky"]}.get(state))
                if state == "dropped":
                    kw = dict(lw=0.7, facecolor="white", edgecolor=F.LIGHT)
                ax.add_patch(Rectangle((0.04 + j * cw, y0 + ch * (n_var - 1 - i)),
                                       cw * 0.85, ch * 0.78, transform=ax.transAxes, **kw))
        if k == 1:
            ax.add_patch(Rectangle((0.04 + (n_pos - 4) * cw - 0.008, y0 - 0.012),
                                   4 * cw, ch * n_var + 0.012, fill=False,
                                   edgecolor=F.OI["vermillion"], lw=0.9, ls=(0, (2, 1.4)),
                                   transform=ax.transAxes))
        ax.text(0.04, y0 + ch * n_var + 0.030, title, transform=ax.transAxes,
                fontsize=7.5, fontweight="bold")
    ax.text(0.04, 0.015, "rows, variants   columns, positions", transform=ax.transAxes,
            fontsize=6.6, color=F.GREY)
    ax.text(0.04, 1.03, "train", transform=ax.transAxes, fontsize=6.8, color=F.OI["sky"],
            fontweight="bold")
    ax.text(0.175, 1.03, "test", transform=ax.transAxes, fontsize=6.8,
            color=F.OI["vermillion"], fontweight="bold")
    ax.text(0.300, 1.03, "spans the split, discarded", transform=ax.transAxes, fontsize=6.8,
            color=F.GREY)

    # --- b, c: per-assay scatter under each protocol
    for col, (split, lab, letter, colr) in enumerate(
            (("random", "Random split", "b", F.GREY),
             ("pg_modulo_5", "ProteinGym fold_modulo_5", "c", F.OI["blue"]))):
        ax = fig.add_subplot(gs[0, col + 1])
        d = D[D.split == split].dropna(subset=["position_mean", "physchem"])
        ax.scatter(d.position_mean, d.physchem, s=9, alpha=0.55, linewidths=0, color=colr)
        lim = (-0.06, 1.02)
        ax.plot(lim, lim, color=F.INK, lw=0.6, ls=(0, (2.5, 1.6)), zorder=0)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("position-only model  ρ")
        ax.set_ylabel("chemistry model  ρ" if col == 0 else "")
        ax.set_title(f"{lab}\n{len(d)} assays", loc="left", pad=14)
        F.panel(ax, letter, dx=-0.30 if col == 0 else -0.22, dy=1.26)
        above = (d.physchem > d.position_mean).mean()
        ax.text(0.97, 0.05, f"chemistry ahead in\n{100*above:.0f}% of assays",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8)

    # --- d: median and interquartile range per model and protocol
    ax = fig.add_subplot(gs[1, :])
    F.panel(ax, "d", dx=-0.075, dy=1.16)
    order = [("mutation_count", "Mutation count", "untrained"),
             ("blosum_zeroshot", "BLOSUM62", "untrained"),
             ("ridge_additive", "Additive ridge", "position-indexed"),
             ("position_mean", "Position mean", "position-indexed"),
             ("residue_only", "Residue identity", "transferable"),
             ("physchem", "Chemistry ridge", "transferable")]
    splits = [("random", "random split", F.GREY),
              ("pg_modulo_5", "fold_modulo_5", F.OI["blue"]),
              ("pg_contiguous_5", "fold_contiguous_5", F.OI["green"])]
    h = 0.23
    for i, (key, lab, fam) in enumerate(order):
        for j, (sp, splab, col) in enumerate(splits):
            v = D[D.split == sp][key].dropna().to_numpy()
            if not len(v):
                continue
            y = i + (j - 1) * h
            p5, q1, med, q3, p95 = np.percentile(v, [5, 25, 50, 75, 95])
            ax.plot([p5, p95], [y, y], color=col, lw=0.8, alpha=0.75, solid_capstyle="butt")
            ax.plot([q1, q3], [y, y], color=col, lw=3.4, solid_capstyle="butt", alpha=0.95)
            ax.plot([med], [y], marker="|", ms=8.5, mew=1.5, color="white", zorder=4)
            if j == 0:
                ax.text(1.035, y, f"{med:.3f}", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=6.6, color=col)
            else:
                ax.text(1.035 + 0.047 * j, y, f"{med:.3f}",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=6.6, color=col)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([l for _, l, _ in order])
    for tick, (_, _, fam) in zip(ax.get_yticklabels(), order):
        tick.set_color(F.FAMILY[fam]); tick.set_fontweight("bold")
    ax.invert_yaxis()
    F.zeroline(ax)
    ax.set_xlim(-0.12, 1.0)
    ax.set_xlabel("Spearman ρ per assay   (thick bar, interquartile range; thin, 5th–95th centile;"
                  " white tick, median)")
    ax.set_title("Every ProteinGym substitution assay under three evaluation protocols",
                 loc="left", pad=8)
    ax.legend(handles=[Line2D([], [], color=c, lw=3.0, label=l) for _, l, c in splits],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=7)
    ax.text(1.035, -0.95, "median ρ", transform=ax.get_yaxis_transform(), fontsize=6.4,
            color=F.GREY, va="center")
    return F.save(fig, FIG / "Figure1_protocol.png")




# ================================================================ Figure 2
def fig2():
    """The published leaderboard, audited against the position-only control."""
    A = C("table_position_only_audit.csv")
    L = C("table_leaderboard_pivot.csv")
    if not len(A):
        return None
    fig = plt.figure(figsize=(F.DOUBLE, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 0.92], wspace=0.46,
                          left=0.205, right=0.775, top=0.86, bottom=0.17)

    schemes = [("Random split", "random split", F.GREY),
               ("fold_modulo_5", "fold_modulo_5", F.OI["blue"]),
               ("fold_contiguous_5", "fold_contiguous_5", F.OI["green"])]

    # --- a: fraction of assays won, per model and scheme
    ax = fig.add_subplot(gs[0, 0])
    piv = A.pivot(index="model", columns="scheme",
                  values="frac_assays_model_beats_position_only")
    cols = [c for c, _, _ in schemes if c in piv.columns]
    piv = piv[cols].sort_values(cols[0])
    y = np.arange(len(piv))
    for i, (c, lab, colr) in enumerate([s for s in schemes if s[0] in cols]):
        ax.barh(y + (i - 1) * 0.26, piv[c], 0.24, color=colr, lw=0)
    ax.axvline(0.5, color=F.INK, lw=0.7, ls=(0, (2.5, 1.6)), zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=6.8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of assays the published model wins")
    ax.set_title("Published model vs a position-only model", loc="left", pad=8)
    F.panel(ax, "a", dx=-0.80, dy=1.10)
    ax.legend(handles=[Line2D([], [], color=c, lw=3.2, label=l)
                       for cc, l, c in schemes if cc in cols],
              loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=3, fontsize=6.9)

    # --- b: median rho, random -> position-based, as a slope chart
    ax = fig.add_subplot(gs[0, 1])
    if len(L):
        d = L.dropna(subset=["Random split", "fold_modulo_5"]).copy()
        d = d.sort_values("fold_modulo_5", ascending=False).reset_index(drop=True)
        # Spread the right-hand labels so they do not overlap, then lead each back to its
        # endpoint with a hairline. Without this the eleven published models collide.
        ys = d["fold_modulo_5"].to_numpy(float)
        span = float(ys.max() - ys.min()) or 1.0
        gap = span * 0.052
        lab_y = ys.copy()
        for i in range(1, len(lab_y)):
            if lab_y[i - 1] - lab_y[i] < gap:
                lab_y[i] = lab_y[i - 1] - gap
        shift = max(0.0, lab_y.min() * -1 + ys.min() - lab_y.min()) if lab_y.min() < ys.min() else 0.0
        lab_y = lab_y + shift * 0.0
        for i, r in d.iterrows():
            mine = r["source"] == "This work"
            col = F.FAMILY["this work"] if mine else F.FAMILY["published"]
            ax.plot([0, 1], [r["Random split"], r["fold_modulo_5"]], marker="o",
                    color=col, lw=1.0, ms=2.8, alpha=0.9,
                    ls=(0, (2, 1.3)) if mine else "-")
            ax.plot([1.0, 1.045], [r["fold_modulo_5"], lab_y[i]], lw=0.4, color=col,
                    alpha=0.7, clip_on=False)
            lab = str(r["model"]).replace(" (this work)", "*")
            ax.annotate(lab, (1.055, lab_y[i]), fontsize=5.9, va="center",
                        color=col, annotation_clip=False)
        ax.set_xlim(-0.06, 1.0); ax.set_xticks([0, 1])
        ax.set_xticklabels(["random\nsplit", "fold_\nmodulo_5"])
        ax.set_ylabel("mean Spearman ρ across assays")
        ax.set_title("How far each model falls", loc="left", pad=8)
        F.panel(ax, "b", dx=-0.22, dy=1.10)
        ax.text(0.0, -0.165, "* baselines added here; dashed", transform=ax.transAxes,
                fontsize=6.3, color=F.GREY)
    return F.save(fig, FIG / "Figure2_leaderboard.png")


# ================================================================ Figure 3
def fig3():
    """The receptor-binding-domain panel: what survives once the confound is removed."""
    P = C("table_panel.csv")
    Z = C("table_partial_vs_zero.csv")
    if not len(P):
        return None
    order = ["Mutation count", "BLOSUM62", "ESM-2 zero-shot", "Additive ridge",
             "Global epistasis", "EvoClassNet", "Structure GNN", "EvoClassNet + ESM/str",
             "k-NN (features)", "Gradient boosting", "Physicochemical ridge"]
    fam = {r.model: r.family for _, r in P.iterrows()}
    fig = plt.figure(figsize=(F.DOUBLE, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.42,
                          left=0.215, right=0.985, top=0.86, bottom=0.16)

    # --- a: raw Spearman, both splits
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(order))
    for i, (sp, lab, colr) in enumerate((("random", "random split", F.GREY),
                                         ("heldout_sites", "held-out positions", F.OI["blue"]))):
        m, e = [], []
        for mod in order:
            r = P[(P.split == sp) & (P.task == "ACE2 binding") & (P.model == mod)]
            m.append(num(r.iloc[0].spearman) if len(r) else np.nan)
            e.append(sd(r.iloc[0].spearman) if len(r) else 0.0)
        ax.barh(y + (i - 0.5) * 0.36, m, 0.33, xerr=e, color=colr, lw=0,
                error_kw=dict(lw=0.6, capsize=1.6, ecolor=F.INK))
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6.9)
    for t, mod in zip(ax.get_yticklabels(), order):
        t.set_color(F.FAMILY.get(fam.get(mod, ""), F.INK)); t.set_fontweight("bold")
    ax.invert_yaxis(); F.zeroline(ax)
    ax.set_xlabel("Spearman ρ, ACE2 binding")
    ax.set_title("Conventional metric", loc="left", pad=8)
    F.panel(ax, "a", dx=-0.80, dy=1.10)
    ax.legend(handles=[Line2D([], [], color=c, lw=3.2, label=l)
                       for c, l in ((F.GREY, "random split"), (F.OI["blue"], "held-out positions"))],
              loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=2, fontsize=6.9)

    # --- b: partial correlation at held-out positions, with intervals
    ax = fig.add_subplot(gs[0, 1])
    zh = Z[Z.split == "heldout_sites"] if len(Z) else pd.DataFrame()
    for i, mod in enumerate(order):
        r = zh[(zh.model == mod) & (zh.task == "ACE2 binding")] if len(zh) else pd.DataFrame()
        if not len(r):
            ax.plot([0], [i], marker="|", ms=7, color=F.INK)
            continue
        r = r.iloc[0]
        colr = F.FAMILY.get(fam.get(mod, ""), F.INK)
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=colr, lw=1.5, solid_capstyle="round")
        ax.plot([r.mean_partial], [i], "o", color=colr, ms=4.0,
                mfc=colr if r.nonzero_fdr else "white", mew=1.0)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([])
    ax.invert_yaxis(); F.zeroline(ax)
    ax.set_xlabel("partial Spearman ρ   (substitution count removed)")
    ax.set_title("After removing the confound", loc="left", pad=8)
    F.panel(ax, "b", dx=-0.09, dy=1.10)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=F.INK, ms=4, label="above zero"),
                       Line2D([], [], marker="o", ls="", color=F.INK, ms=4, mfc="white",
                              label="not above zero")],
              loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=2, fontsize=6.9)
    return F.save(fig, FIG / "Figure3_rbd_panel.png")


# ================================================================ Figure 4
def fig4():
    """Where the transferable signal comes from."""
    FA = J("feature_ablation.json").get("summary", {})
    TB = J("transfer_by_position.json")
    if not FA:
        return None
    fig = plt.figure(figsize=(F.DOUBLE, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.46,
                          left=0.255, right=0.98, top=0.86, bottom=0.18)

    arms = [("nsub_only", "substitution count only"), ("esm_only", "ESM-2 score only"),
            ("chem_only", "residue chemistry"), ("struct_nointerface", "structure, no interface"),
            ("struct_only", "structure"), ("chem+struct", "chemistry + structure"),
            ("chem+struct+esm", "chemistry + structure + ESM-2")]
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(arms))
    mu = [FA.get(a, {}).get("bind", {}).get("partial_mean", np.nan) for a, _ in arms]
    er = [FA.get(a, {}).get("bind", {}).get("partial_sd", 0.0) for a, _ in arms]
    cols = [F.FAMILY["untrained"] if a == "nsub_only" else
            (F.OI["purple"] if "esm" in a and a != "chem+struct+esm" else F.OI["blue"])
            for a, _ in arms]
    ax.barh(y, mu, 0.62, xerr=er, color=cols, lw=0,
            error_kw=dict(lw=0.6, capsize=1.8, ecolor=F.INK))
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in arms], fontsize=7)
    ax.invert_yaxis(); F.zeroline(ax)
    ax.set_xlabel("partial Spearman ρ, ACE2 binding, held-out positions")
    ax.set_title("Which features carry the transfer", loc="left", pad=8)
    F.panel(ax, "a", dx=-0.62, dy=1.11)

    ax = fig.add_subplot(gs[0, 1])
    if TB:
        tasks = [("bind", "ACE2 binding"), ("expr", "Expression"), ("escape", "Antibody escape")]
        x = np.arange(len(tasks))
        mm = [TB.get(t, {}).get("median_rho", np.nan) for t, _ in tasks]
        bb = [TB.get(t, {}).get("vs_blosum", {}).get("blosum_median", np.nan) for t, _ in tasks]
        ax.bar(x - 0.19, mm, 0.36, color=F.OI["blue"], lw=0, label="transferable model")
        ax.bar(x + 0.19, bb, 0.36, color=F.FAMILY["untrained"], lw=0, label="BLOSUM62")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in tasks], fontsize=7)
        F.zeroline(ax, x=False)
        ax.set_ylabel("median per-position ρ")
        ax.set_title("Within one withheld position", loc="left", pad=8)
        F.panel(ax, "b", dx=-0.24, dy=1.11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.9)
        for xi, (t, _) in enumerate(tasks):
            v = TB.get(t, {}).get("vs_blosum", {})
            if v and v.get("p") == v.get("p"):
                star = "n.s." if v["p"] > 0.05 else ("***" if v["p"] < 1e-3 else "*")
                top = max(mm[xi], bb[xi])
                ax.text(xi, top + 0.045, star, ha="center", fontsize=6.6)
    return F.save(fig, FIG / "Figure4_decomposition.png")


# ================================================================ Figure 5
def fig5():
    """Controls: the language model, and the genome task."""
    E = J("esm_control.json")
    G = C("table_genomes.csv")
    if not (E or len(G)):
        return None
    fig = plt.figure(figsize=(F.DOUBLE, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.3], wspace=0.60,
                          left=0.085, right=0.985, top=0.82, bottom=0.21)

    ax = fig.add_subplot(gs[0, 0])
    if E:
        labs = [("ubiquitin", "ubiquitin\n(positive control)"),
                ("rbd_fragment", "RBD alone"), ("rbd_in_spike", "RBD inside\nfull spike")]
        x = np.arange(len(labs))
        rec = [E.get(k, {}).get("recovery", np.nan) for k, _ in labs]
        prob = [E.get(k, {}).get("median_top_prob", np.nan) for k, _ in labs]
        ax.bar(x - 0.19, rec, 0.36, color=F.OI["purple"], lw=0, label="wild-type recovery")
        ax.bar(x + 0.19, prob, 0.36, color=F.LIGHT, lw=0, label="median top-token probability")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in labs], fontsize=6.8)
        ax.set_ylim(0, 1.0); ax.set_ylabel("fraction")
        ax.set_title("ESM-2 works, but not on this domain", loc="left", pad=8)
        F.panel(ax, "a", dx=-0.19, dy=1.12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1, fontsize=6.8)
        for xi, v in enumerate(rec):
            ax.text(xi - 0.19, v + 0.02, f"{v:.1%}", ha="center", fontsize=6.4)

    ax = fig.add_subplot(gs[0, 1])
    if len(G):
        mods = ["Majority class", "Genome length + base composition",
                "5-mer + logistic regression", "5-mer + gradient boosting",
                "5-mer + random forest", "1D CNN on nucleotides"]
        y = np.arange(len(mods))
        for i, (sp, lab, colr) in enumerate((("random", "random split", F.GREY),
                                             ("heldout_species", "held-out species",
                                              F.OI["blue"]))):
            m, e = [], []
            for mod in mods:
                r = G[(G.split == sp) & (G.model == mod)]
                m.append(num(r.iloc[0].balanced_accuracy) if len(r) else np.nan)
                e.append(sd(r.iloc[0].balanced_accuracy) if len(r) else 0.0)
            ax.barh(y + (i - 0.5) * 0.36, m, 0.33, xerr=e, color=colr, lw=0,
                    error_kw=dict(lw=0.6, capsize=1.6, ecolor=F.INK))
        ax.axvline(0.25, color=F.INK, lw=0.7, ls=(0, (2.5, 1.6)), zorder=0)
        ax.text(0.26, len(mods) - 0.45, "chance", fontsize=6.3, color=F.GREY, va="bottom")
        ax.set_yticks(y); ax.set_yticklabels(mods, fontsize=6.8)
        ax.invert_yaxis(); ax.set_xlim(0, 1.05)
        ax.set_xlabel("balanced accuracy")
        ax.set_title("Coronavirus genus classification", loc="left", pad=8)
        F.panel(ax, "b", dx=-0.52, dy=1.16)
        ax.legend(handles=[Line2D([], [], color=c, lw=3.2, label=l)
                           for c, l in ((F.GREY, "random split"),
                                        (F.OI["blue"], "held-out species"))],
                  loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=6.8)
    return F.save(fig, FIG / "Figure5_controls.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig1, fig2, fig3, fig4, fig5):
        p = fn()
        print("wrote", p.name if p else f"{fn.__name__}: inputs missing")
