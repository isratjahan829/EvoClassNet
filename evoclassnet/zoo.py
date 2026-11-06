"""Additional model families, chosen so that the comparison contains models that *could*
extrapolate to positions never mutated in training.

The original study compared an additive one-hot model, a global-epistasis fit and a sequence
neural network. All three represent a substitution by *which position and which residue* it is,
so none of them has any parameter that applies to a position absent from training; their failure
at held-out positions is therefore partly a property of the representation, not a finding about
proteins. The families here close that gap:

  physchem      a substitution is described only by how the residue changes (hydrophobicity,
                volume, charge, ...), never by where. Transfers to unseen positions by
                construction.
  blosum        the substitution's BLOSUM62 exchangeability, summed over a genotype. Untrained.
  esm           ESM-2 masked-marginal log-likelihood ratio, summed over a genotype. Untrained on
                the assay, so leakage-free at held-out positions.
  structure     per-position features taken from the crystal structure 6M0J (solvent
                accessibility, distance to ACE2, burial). Known for every position regardless of
                whether it was ever assayed, so again available at held-out positions.
  gbm / knn     gradient boosting and nearest neighbours over the combined feature block.
  struct_gnn    a graph neural network propagating over the residue contact graph.

Every feature block is position-transferable except where explicitly noted, which is the point.
"""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy, side-chain volume (A^3), charge at pH 7, polarity, flexibility,
# Chou-Fasman helix and sheet propensity, and molecular weight. Standard published scales.
_PROPS = {
    #      hyd    vol  chg  pol   flex  helix sheet   mw
    "A": (1.80,  88.6, 0.0, 0.0, 0.360, 1.42, 0.83,  89.1),
    "C": (2.50, 108.5, 0.0, 0.0, 0.350, 0.70, 1.19, 121.2),
    "D": (-3.50, 111.1, -1.0, 1.0, 0.510, 1.01, 0.54, 133.1),
    "E": (-3.50, 138.4, -1.0, 1.0, 0.500, 1.51, 0.37, 147.1),
    "F": (2.80, 189.9, 0.0, 0.0, 0.310, 1.13, 1.38, 165.2),
    "G": (-0.40,  60.1, 0.0, 0.0, 0.540, 0.57, 0.75,  75.1),
    "H": (-3.20, 153.2, 0.1, 1.0, 0.320, 1.00, 0.87, 155.2),
    "I": (4.50, 166.7, 0.0, 0.0, 0.460, 1.08, 1.60, 131.2),
    "K": (-3.90, 168.6, 1.0, 1.0, 0.470, 1.16, 0.74, 146.2),
    "L": (3.80, 166.7, 0.0, 0.0, 0.370, 1.21, 1.30, 131.2),
    "M": (1.90, 162.9, 0.0, 0.0, 0.300, 1.45, 1.05, 149.2),
    "N": (-3.50, 114.1, 0.0, 1.0, 0.460, 0.67, 0.89, 132.1),
    "P": (-1.60, 112.7, 0.0, 0.0, 0.510, 0.57, 0.55, 115.1),
    "Q": (-3.50, 143.8, 0.0, 1.0, 0.490, 1.11, 1.10, 146.2),
    "R": (-4.50, 173.4, 1.0, 1.0, 0.530, 0.98, 0.93, 174.2),
    "S": (-0.80,  89.0, 0.0, 1.0, 0.510, 0.77, 0.75, 105.1),
    "T": (-0.70, 116.1, 0.0, 1.0, 0.440, 0.83, 1.19, 119.1),
    "V": (4.20, 140.0, 0.0, 0.0, 0.390, 1.06, 1.70, 117.1),
    "W": (-0.90, 227.8, 0.0, 0.0, 0.310, 1.08, 1.37, 204.2),
    "Y": (-1.30, 193.6, 0.0, 1.0, 0.420, 0.69, 1.47, 181.2),
}
PROP_NAMES = ["hydropathy", "volume", "charge", "polarity", "flexibility", "helix", "sheet", "mw"]
_P = np.array([_PROPS[a] for a in AA], dtype=np.float32)
_P = (_P - _P.mean(0)) / (_P.std(0) + 1e-8)


def _blosum():
    from Bio.Align import substitution_matrices
    B = substitution_matrices.load("BLOSUM62")
    M = np.zeros((20, 20), dtype=np.float32)
    for i, a in enumerate(AA):
        for j, b in enumerate(AA):
            M[i, j] = float(B[a, b]) if (a in B.alphabet and b in B.alphabet) else 0.0
    return M


_B = None


def blosum_matrix():
    global _B
    if _B is None:
        _B = _blosum()
    return _B


def substitution_feature_names(use_esm, use_struct, struct_names=()):
    names = [f"d_{n}" for n in PROP_NAMES] + [f"wt_{n}" for n in PROP_NAMES] + ["blosum"]
    names += [f"wt_is_{a}" for a in AA] + [f"mut_is_{a}" for a in AA]
    if use_esm:
        names += ["esm_llr"]
    if use_struct:
        names += [f"site_{n}" for n in struct_names]
    return names + ["n_sub"]


def genotype_features(subs_list, esm_mm=None, struct=None, struct_names=()):
    """Additive feature block per genotype: the sum over its substitutions, plus substitution count.

    Nothing here indexes a position by identity, so a feature vector is defined for a genotype
    that mutates positions the training set never touched. Structural columns are per-position but
    come from the crystal structure, not from the assay, so they too are available at held-out
    positions.
    """
    B = blosum_matrix()
    use_esm, use_struct = esm_mm is not None, struct is not None
    d = 2 * len(PROP_NAMES) + 1 + 40 + (1 if use_esm else 0) + (struct.shape[1] if use_struct else 0)
    out = np.zeros((len(subs_list), d + 1), dtype=np.float32)
    for i, subs in enumerate(subs_list):
        for w, p, m in subs:
            if w not in AA2I or m not in AA2I:
                continue
            iw, im = AA2I[w], AA2I[m]
            v, k = out[i], 0
            v[k:k + 8] += _P[im] - _P[iw]; k += 8
            v[k:k + 8] += _P[iw]; k += 8
            v[k] += B[iw, im]; k += 1
            v[k + iw] += 1.0; k += 20
            v[k + im] += 1.0; k += 20
            if use_esm:
                v[k] += float(esm_mm[p - 1, im]); k += 1
            if use_struct:
                n = struct.shape[1]
                v[k:k + n] += struct[p - 1]; k += n
        out[i, -1] = len(subs)
    return out


# ------------------------------------------------------------------ untrained scorers
def esm_scores(subs_list, esm_mm):
    """Sum of masked-marginal log-likelihood ratios. Untrained; higher = more plausible."""
    return np.array([sum(float(esm_mm[p - 1, AA2I[m]]) for w, p, m in s if m in AA2I)
                     for s in subs_list], dtype=float)


def blosum_scores(subs_list):
    """Sum of BLOSUM62 exchangeabilities. Untrained; higher = more conservative substitution."""
    B = blosum_matrix()
    return np.array([sum(float(B[AA2I[w], AA2I[m]]) for w, p, m in s if w in AA2I and m in AA2I)
                     for s in subs_list], dtype=float)


# ------------------------------------------------------------------ trained feature models
def fit_predict_sklearn(kind, F, Y, tr, te, seed=0, n_tasks=3):
    """Fit one regressor per task on the feature block and predict the test rows."""
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    P = np.full((len(te), n_tasks), np.nan)
    sc = StandardScaler().fit(F[tr])
    Ftr, Fte = sc.transform(F[tr]), sc.transform(F[te])
    for t in range(n_tasks):
        m = ~np.isnan(Y[tr, t])
        if m.sum() < 20:
            continue
        if kind == "physchem":
            mdl = RidgeCV(alphas=np.logspace(-2, 3, 12))
        elif kind == "gbm":
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                early_stopping=True, random_state=seed)
        elif kind == "rf":
            mdl = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=seed,
                                        min_samples_leaf=3)
        elif kind == "knn":
            mdl = KNeighborsRegressor(n_neighbors=min(15, int(m.sum())), weights="distance")
        else:
            raise ValueError(kind)
        mdl.fit(Ftr[m], Y[tr, t][m])
        P[:, t] = mdl.predict(Fte)
    return P


# ------------------------------------------------------------------ structure-aware GNN
def build_gnn(seed, adj, struct, n_tasks=3, emb=24, hidden=64, layers=3, dropout=0.1):
    """Graph neural network over the residue contact graph of the folded RBD.

    Each of the 201 positions is a node carrying (a) an embedding of the amino acid currently
    occupying it in this variant and (b) the fixed structural descriptors of that position. Edges
    are CB-CB contacts under 8 A plus the backbone. Message passing therefore lets a substitution
    influence the representation of the residues it physically touches, which is the mechanism the
    sequence models have no access to, and it is defined at positions never seen mutated.
    """
    import torch
    import torch.nn as nn

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            A = torch.tensor(adj, dtype=torch.float32)
            A = A + torch.eye(A.shape[0])
            d = A.sum(1).clamp(min=1).pow(-0.5)
            self.register_buffer("A", d[:, None] * A * d[None, :])     # symmetric normalisation
            self.register_buffer("S", torch.tensor(struct, dtype=torch.float32))
            self.emb = nn.Embedding(20, emb)
            din = emb + struct.shape[1]
            self.gc = nn.ModuleList([nn.Linear(din if i == 0 else hidden, hidden) for i in range(layers)])
            self.ln = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.drop = nn.Dropout(dropout)
            self.att = nn.Linear(hidden, 1)
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden, 128), nn.ReLU(),
                                                      nn.Linear(128, 1)) for _ in range(n_tasks)])
            self.log_var = nn.Parameter(torch.zeros(n_tasks))

        def forward(self, x):
            h = torch.cat([self.emb(x), self.S.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=-1)
            for lin, ln in zip(self.gc, self.ln):
                h = self.drop(torch.relu(ln(lin(self.A @ h))))
            w = torch.softmax(self.att(h), dim=1)
            pooled = torch.cat([(w * h).sum(1), h.max(1).values], dim=-1)
            return torch.cat([head(pooled) for head in self.heads], dim=-1)

        def loss(self, pred, y):
            total = 0.0
            for t in range(pred.shape[1]):
                m = ~torch.isnan(y[:, t])
                if m.sum() == 0:
                    continue
                l = torch.nn.functional.smooth_l1_loss(pred[m, t], y[m, t])
                total = total + torch.exp(-self.log_var[t]) * l + self.log_var[t]
            return total

    torch.manual_seed(seed)
    np.random.seed(seed)
    return GNN()
