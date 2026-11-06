"""Architecture and design schematics, drawn from the real configuration.

The uploaded draft illustrated the method with schematic panels. Those panels described an
architecture that was never built and their labels were largely unreadable, so they are redrawn
here for the system that this repository actually contains. Every dimension printed on a box is
read from the trained configuration or from the modules themselves, so a label cannot disagree
with the code.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C = dict(seq="#4c78a8", plm="#8c62aa", struct="#2e8b57", head="#d9541e",
         data="#51789e", grey="#8a8a8a", bg="#f4f6f8", edge="#2f3640")
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, fc, fs=8.0, tc="white", lw=0.0, style="round,pad=0.012"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                                edgecolor=C["edge"], facecolor=fc, mutation_aspect=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.35)


def arrow(ax, p, q, color=None, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9,
                                 linewidth=lw, color=color or C["edge"],
                                 shrinkA=1.5, shrinkB=1.5))


def canvas(w, h, title, sub=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=11.5, fontweight="bold", va="top")
    if sub:
        ax.text(0.0, 0.952, sub, fontsize=8.5, color="#44506a", va="top")
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------- 1. evaluation design
def fig_design(g_stats):
    fig, ax = canvas(7.4, 3.5, "Two evaluation designs on the same library",
                     "A variant enters the held-out-position test set only if every position it "
                     "mutates is withheld from training.")
    n, nb, ne, nx = g_stats["n"], g_stats["bind"], g_stats["escape"], g_stats["expr"]
    box(ax, 0.02, 0.60, 0.26, 0.20,
        f"SARS-CoV-2 RBD\ndeep mutational scan\n{n:,} unique genotypes\n"
        f"binding {nb:,} · expression {nx:,}\nescape {ne:,}", C["data"], 7.6)
    arrow(ax, (0.28, 0.70), (0.36, 0.78)); arrow(ax, (0.28, 0.70), (0.36, 0.40))

    box(ax, 0.36, 0.68, 0.28, 0.20,
        "Random genotype split\n80 / 10 / 10 %\n\nTest positions are also\nmutated in training",
        C["grey"], 7.6)
    box(ax, 0.36, 0.26, 0.28, 0.24,
        "Held-out-position split\npositions 75 / 10 / 15 %\n\nNo training variant mutates\n"
        "any test position;\nspanning variants discarded", C["seq"], 7.6)

    arrow(ax, (0.64, 0.78), (0.72, 0.78)); arrow(ax, (0.64, 0.38), (0.72, 0.38))
    box(ax, 0.72, 0.68, 0.26, 0.20, "Measures\ninterpolation", C["grey"], 8.2)
    box(ax, 0.72, 0.28, 0.26, 0.20, "Measures\nposition-level\nextrapolation", C["seq"], 8.2)

    box(ax, 0.36, 0.04, 0.62, 0.14,
        "Untrained controls evaluated on the identical partitions:  −n_substitutions  ·  "
        "BLOSUM62 exchangeability  ·  ESM-2 masked marginals",
        C["head"], 7.8)
    save(fig, "schema1_design.png")


# ----------------------------------------------------------------- 2. EvoClassNet backbone
def fig_backbone(hp, n_params):
    emb, ch, ks, lstm = hp["emb"], tuple(hp["ch"]), tuple(hp["ks"]), hp["lstm"]
    out_len = [100, 50, 50]          # after block 1 (pool), block 2 (pool), block 3 (no pool)
    fig, ax = canvas(9.2, 4.2, "EvoClassNet backbone: convolution, recurrence, attention",
                     f"As implemented in evoclassnet/model.py ({n_params:,} parameters, trained on CPU).")
    ROW, H = 0.60, 0.22
    xs, ws = [], []
    x = 0.012
    plan = [("Variant sequence\n201 residues\n(spike 331–531)\ninteger coded", 0.132, C["data"]),
            (f"Residue embedding\n{emb}-dim  +  learned\npositional embedding", 0.124, C["seq"])]
    for i, (c, k) in enumerate(zip(ch, ks)):
        pool = "max-pool /2\n" if i < 2 else ""
        plan.append((f"Conv1d block {i+1}\n{c} channels · k={k}\nBatchNorm · ReLU\n{pool}→ {out_len[i]} positions",
                     0.118, C["seq"]))
    plan.append((f"BiLSTM\n{lstm} per\ndirection\n→ {2*lstm}-dim", 0.098, C["seq"]))
    plan.append(("4-head\nself-attention\nresidual\n+ LayerNorm", 0.112, C["seq"]))
    for i, (txt, w, col) in enumerate(plan):
        box(ax, x, ROW, w, H, txt, col, 7.0, lw=0.9)
        xs.append(x); ws.append(w)
        if i + 1 < len(plan):
            arrow(ax, (x + w + 0.001, ROW + H / 2), (x + w + 0.010, ROW + H / 2))
        x += w + 0.012

    # down to pooling, then out to the heads
    last_mid = xs[-1] + ws[-1] / 2
    arrow(ax, (last_mid, ROW), (0.40, 0.425))
    box(ax, 0.17, 0.215, 0.23, 0.20,
        f"Pooling\nlearned attention pooling\n‖ max pooling\n→ {4*lstm}-dim", C["grey"], 7.3)
    for i, lab in enumerate(("ACE2 binding", "Expression", "Antibody escape")):
        yy = 0.370 - i * 0.070
        box(ax, 0.455, yy, 0.205, 0.054, f"{lab} head · MLP 128", C["head"], 7.0, lw=0.9)
        arrow(ax, (0.402, 0.315), (0.452, yy + 0.027))
    ax.text(0.695, 0.445,
            "Masked Huber loss: a genotype\nmissing one phenotype still trains\nthe others. Task weights are learned\n"
            "homoscedastic uncertainties, so the\nscarce escape labels are not drowned\nby the two large ones.",
            fontsize=7.2, va="top", color="#2f3640", linespacing=1.5)
    box(ax, 0.012, 0.015, 0.975, 0.125,
        "Optional per-position side information (evoclassnet_plus): the 8 structural descriptors from PDB 6M0J and 16 PCA components of "
        "ESM-2\nper-residue embeddings are concatenated to the residue embedding before the first convolution. The same training code "
        "is used, so the\naugmented and plain models differ only in that input.", C["bg"], 7.2, tc="#2f3640")
    save(fig, "schema2_backbone.png")


# ----------------------------------------------------------------- 3. ESM-2 integration
def fig_plm(model_name, mm_shape, emb_dim, pca_dim):
    fig, ax = canvas(7.4, 3.2, "ESM-2 integration: a zero-shot scorer and a feature source",
                     f"{model_name} run once on the wild-type sequence; neither product sees any assay measurement.")
    box(ax, 0.02, 0.60, 0.20, 0.22, "Wild-type RBD\n201 residues\n(context: full spike\nP0DTC2)", C["data"], 7.5)
    arrow(ax, (0.22, 0.71), (0.28, 0.71))
    box(ax, 0.28, 0.58, 0.20, 0.26, f"ESM-2\n650M parameters\n33 layers\nmasked language model", C["plm"], 7.6)
    arrow(ax, (0.48, 0.76), (0.55, 0.82)); arrow(ax, (0.48, 0.64), (0.55, 0.52))

    box(ax, 0.55, 0.70, 0.43, 0.22,
        f"Masked marginals  {mm_shape[0]}×{mm_shape[1]}\n"
        "mask position i → log p(mut | context) − log p(wt | context)\n"
        "genotype score = sum over its substitutions", C["plm"], 7.4)
    box(ax, 0.55, 0.40, 0.43, 0.22,
        f"Per-residue embeddings  {mm_shape[0]}×{emb_dim}\n"
        f"PCA → {pca_dim} dims per position\n"
        "concatenated to the residue embedding (extra_dim hook)", C["plm"], 7.4)
    box(ax, 0.02, 0.08, 0.96, 0.22,
        "Why both: the masked-marginal scorer is untrained on the assay, so it is leakage-free at held-out positions "
        "by construction and tests whether\ninformation from outside the training library supplies the transfer the "
        "supervised models lack. The embeddings test the same question in the\nsupervised setting, by giving the "
        "network per-position context it cannot infer from the library.", C["bg"], 7.5, tc="#2f3640")
    save(fig, "schema3_plm.png")


# ----------------------------------------------------------------- 4. structure + GNN
def fig_gnn(n_resolved, n_edges, mean_deg, feat_names, iface):
    fig, ax = canvas(8.6, 4.0, "Structure-aware model: contact graph from PDB 6M0J",
                     "Crystal structure of the RBD (chain E) bound to human ACE2 (chain A). "
                     "Every descriptor is known independently of the assay.")
    ROW, H = 0.52, 0.34
    box(ax, 0.012, ROW + 0.04, 0.185, H - 0.08,
        f"PDB 6M0J\n\nRBD = chain E\n{n_resolved} of 201 positions\nresolved\n\nACE2 = chain A", C["data"], 7.3, lw=0.9)
    arrow(ax, (0.199, ROW + H / 2), (0.213, ROW + H / 2))
    box(ax, 0.215, ROW, 0.225, H,
        "Per-position features\n" + "\n".join("· " + n for n in feat_names), C["struct"], 6.8, lw=0.9)
    arrow(ax, (0.442, ROW + H / 2), (0.456, ROW + H / 2))
    box(ax, 0.458, ROW + 0.055, 0.235, H - 0.11,
        f"Contact graph\n\nCB–CB < 8 Å, plus\nbackbone neighbours\n\n{n_edges} edges\nmean degree {mean_deg:.1f}",
        C["struct"], 7.3, lw=0.9)
    arrow(ax, (0.695, ROW + H / 2), (0.709, ROW + H / 2))
    box(ax, 0.711, ROW + 0.04, 0.215, H - 0.08,
        "GNN\n\n3 × ( Â H W )\nLayerNorm · ReLU\n\nattention pooling\n→ 3 task heads", C["struct"], 7.3, lw=0.9)

    box(ax, 0.012, 0.235, 0.914, 0.235,
        "Node = one RBD position, carrying an embedding of the residue this variant places there plus the fixed structural\n"
        "descriptors of that position. Message passing lets a substitution influence the residues it physically touches — the\n"
        "mechanism a sequence model has no access to — and it remains defined at positions never seen mutated, because\n"
        "the structure does not come from the assay.", C["bg"], 7.4, tc="#2f3640")
    ax.text(0.012, 0.165, "ACE2 interface positions recovered by ΔRSA > 0.05 (spike numbering), matching the known contact set:",
            fontsize=7.5, fontweight="bold", color="#2f3640")
    ax.text(0.012, 0.085, "   " + "   ".join(str(s) for s in iface), fontsize=8.6, color=C["struct"],
            fontweight="bold")
    save(fig, "schema4_structure_gnn.png")


# ----------------------------------------------------------------- 5. genome task
def fig_genome(meta):
    cc = meta["class_counts"]
    fig, ax = canvas(7.2, 3.0, "Second task: coronavirus genus classification",
                     "The same evaluation question on an independent data type.")
    box(ax, 0.02, 0.62, 0.26, 0.24,
        f"NCBI nucleotide\n{meta['n_genomes']} complete genomes\n{meta['n_species']} species, 4 genera\n"
        "SARS-CoV-2 excluded\nfrom Betacoronavirus", C["data"], 7.4)
    arrow(ax, (0.28, 0.74), (0.34, 0.80)); arrow(ax, (0.28, 0.70), (0.34, 0.42))
    box(ax, 0.34, 0.70, 0.28, 0.18, "Random split\nnear-identical relatives\non both sides", C["grey"], 7.5)
    box(ax, 0.34, 0.32, 0.28, 0.18, "Held-out-species split\nno species in both\ntrain and test", C["seq"], 7.5)
    arrow(ax, (0.62, 0.79), (0.68, 0.70)); arrow(ax, (0.62, 0.41), (0.68, 0.56))
    box(ax, 0.68, 0.44, 0.30, 0.32,
        f"Models\n· majority class\n· genome length + base\n  composition (untrained)\n"
        f"· {meta['k']}-mer + logistic / GBM / RF\n· 1D CNN on nucleotides", C["head"], 7.3)
    ax.text(0.02, 0.52, "Class sizes", fontsize=7.8, fontweight="bold")
    for i, (g, n) in enumerate(cc.items()):
        ax.text(0.02, 0.44 - i * 0.075, f"{g}: {n}", fontsize=7.4, color="#2f3640")
    box(ax, 0.34, 0.04, 0.64, 0.18,
        "Genome length and base composition need no training and no sequence model, yet separate coronavirus\n"
        "genera. They play the role the mutation-count baseline plays in the assay study: the score an\n"
        "uninformative model already achieves, which any reported accuracy must be read against.",
        C["bg"], 7.4, tc="#2f3640")
    save(fig, "schema5_genome_task.png")


if __name__ == "__main__":
    from evoclassnet import data, structure as S, plm
    hp = json.load(open(OUT / "ga_hyperparams.json"))["best"]
    g = data.load_genotypes()
    fig_design(dict(n=len(g), bind=int(g.bind.notna().sum()), expr=int(g.expr.notna().sum()),
                    escape=int(g.escape.notna().sum())))

    from evoclassnet import model as M
    net = M.build(M.EvoClassNetLite, 0, n_tasks=3,
                  **{k: (tuple(v) if isinstance(v, list) else v)
                     for k, v in hp.items() if k in ("emb", "ch", "ks", "lstm", "dropout")})
    fig_backbone(hp, sum(p.numel() for p in net.parameters()))

    mm, hid = plm.load()
    if mm is not None:
        fig_plm(plm.DEFAULT_MODEL.split("/")[-1], mm.shape, hid.shape[1], 16)

    f, mask, adj, names = S.compute()
    iface = [i + 1 + S.OFFSET for i in range(201) if mask[i] and f[i, names.index("is_interface")] > 0.5]
    fig_gnn(int(mask.sum()), int(adj.sum() // 2), float(adj.sum(1).mean()), names, iface)

    gj = OUT / "genome_study.json"
    if gj.exists():
        fig_genome(json.load(open(gj))["meta"])
    print("schematics done")
