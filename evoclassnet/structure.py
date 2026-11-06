"""Per-residue structural features and a residue contact graph for the RBD, from PDB 6M0J.

6M0J is the crystal structure of the SARS-CoV-2 receptor-binding domain (chain E, spike residues
333-526) bound to human ACE2 (chain A). Using the complex rather than an isolated RBD model gives
two things the sequence-only study cannot have: a measured solvent accessibility for every
residue, and the actual distance from each residue to the receptor it has to bind.

Features per RBD position (1-based within the 201-residue RBD, spike = position + 330):

    rsa_unbound     relative solvent accessibility of the RBD alone
    rsa_bound       relative solvent accessibility in the ACE2 complex
    d_rsa           rsa_unbound - rsa_bound; non-zero only at the receptor interface
    min_dist_ace2   shortest heavy-atom distance to ACE2
    contact_number  number of other RBD residues within 8 A (CB-CB, CA for glycine)
    depth           count of heavy atoms of other residues within 10 A of the centroid (burial)
    bfactor         mean crystallographic B-factor, a proxy for local disorder
    is_interface    d_rsa > 0.05, i.e. the residue is buried by receptor binding

`contact_graph` returns the CB-CB adjacency within a cutoff, which the graph neural network in
models.py propagates over. Positions absent from the crystal structure (the disordered termini)
are marked in the returned mask and given feature values imputed from the column median, so that
every model still sees all 201 positions.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "data" / "raw" / "structure" / "6m0j.pdb"
RBD_CHAIN, ACE2_CHAIN = "E", "A"
OFFSET = 330          # spike number = RBD position + OFFSET
L = 201

# Tien et al. 2013 theoretical maxima, used to turn absolute SASA into a relative value.
MAXASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225, "G": 104,
          "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155,
          "T": 172, "W": 285, "Y": 263, "V": 174}
THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
             "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
             "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
FEATURES = ["rsa_unbound", "rsa_bound", "d_rsa", "min_dist_ace2", "contact_number", "depth",
            "bfactor", "is_interface"]


def _load():
    from Bio.PDB import PDBParser
    if not PDB.exists():
        raise FileNotFoundError(f"{PDB} missing; run scripts/download_data.py")
    st = PDBParser(QUIET=True).get_structure("6m0j", str(PDB))[0]
    rbd = [r for r in st[RBD_CHAIN] if r.id[0] == " " and r.get_resname() in THREE2ONE]
    ace2 = [r for r in st[ACE2_CHAIN] if r.id[0] == " " and r.get_resname() in THREE2ONE]
    return st, rbd, ace2


def _cb(res):
    """CB coordinate, or CA for glycine / residues with a missing side chain."""
    for name in ("CB", "CA"):
        if name in res:
            return res[name].get_coord()
    return np.mean([a.get_coord() for a in res], axis=0)


def compute(cutoff=8.0, interface_cut=0.05):
    """Return (features [L, n_feat], mask [L] bool, adjacency [L, L] float, names)."""
    from Bio.PDB import ShrakeRupley
    st, rbd, ace2 = _load()
    sr = ShrakeRupley()

    # SASA of the isolated RBD chain, then of the complex
    sr.compute(st[RBD_CHAIN], level="R")
    unbound = {r.id[1]: r.sasa for r in rbd}
    sr.compute(st, level="R")
    bound = {r.id[1]: r.sasa for r in rbd}

    ace2_atoms = np.array([a.get_coord() for r in ace2 for a in r])
    all_atoms = np.array([a.get_coord() for r in rbd for a in r])

    feats = np.full((L, len(FEATURES)), np.nan)
    mask = np.zeros(L, dtype=bool)
    cb = np.full((L, 3), np.nan)

    for r in rbd:
        spike = r.id[1]
        p = spike - OFFSET                       # 1-based RBD position
        if not (1 <= p <= L):
            continue
        aa = THREE2ONE[r.get_resname()]
        mx = MAXASA[aa]
        ru, rb = unbound.get(spike, np.nan) / mx, bound.get(spike, np.nan) / mx
        atoms = np.array([a.get_coord() for a in r])
        d_ace2 = float(np.min(np.linalg.norm(atoms[:, None, :] - ace2_atoms[None, :, :], axis=-1)))
        centroid = atoms.mean(0)
        # Burial: how many heavy atoms of *other* residues lie within 10 A of this residue's
        # centroid. A buried core residue is surrounded; a surface loop residue is not. Atoms of
        # the residue itself are excluded so the count does not simply track side-chain size.
        other = np.array([a.get_coord() for rr in rbd if rr is not r for a in rr])
        depth = float((np.linalg.norm(other - centroid, axis=-1) < 10.0).sum())
        bf = float(np.mean([a.get_bfactor() for a in r]))
        drsa = ru - rb
        feats[p - 1] = [ru, rb, drsa, d_ace2, np.nan, depth, bf, float(drsa > interface_cut)]
        mask[p - 1] = True
        cb[p - 1] = _cb(r)

    # contact graph and contact number from CB positions
    adj = np.zeros((L, L), dtype=np.float32)
    ok = ~np.isnan(cb[:, 0])
    idx = np.flatnonzero(ok)
    d = np.linalg.norm(cb[idx][:, None, :] - cb[idx][None, :, :], axis=-1)
    close = (d < cutoff).astype(np.float32)
    np.fill_diagonal(close, 0.0)
    adj[np.ix_(idx, idx)] = close
    feats[idx, FEATURES.index("contact_number")] = close.sum(1)

    # impute the disordered termini from the column median so every position is usable
    for j in range(feats.shape[1]):
        col = feats[:, j]
        col[np.isnan(col)] = np.nanmedian(col)
    # backbone-adjacent edges keep the graph connected across unresolved stretches
    for i in range(L - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return feats.astype(np.float32), mask, adj, list(FEATURES)


_CACHE = {}


def features(standardise=True):
    """Cached per-position structural features, optionally z-scored column-wise."""
    key = ("f", standardise)
    if key not in _CACHE:
        f, m, a, names = compute()
        if standardise:
            f = (f - f.mean(0)) / (f.std(0) + 1e-8)
        _CACHE[key] = (f.astype(np.float32), m, a, names)
    return _CACHE[key]
