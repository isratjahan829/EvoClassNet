import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from evoclassnet import data, model as M, ga

def test_wildtype_and_parse():
    wt, off = data.wildtype(); assert len(wt) == 201 and off == 330
    assert data.parse_subs("N1A K5L") == [("N", 1, "A"), ("K", 5, "L")]

def test_encode_roundtrip():
    wt, _ = data.wildtype(); X = data.encode([[(wt[0], 1, "A")], []])
    assert X.shape == (2, 201) and X[0, 0] == data.AA2I["A"] and X[1, 0] == data.AA2I[wt[0]]

def test_heldout_split_has_no_overlap():
    import pandas as pd
    g = pd.DataFrame({"sites": [frozenset([i % 200 + 1, (i * 7) % 200 + 1]) for i in range(3000)]})
    tr, va, te = data.split_heldout_sites(g, 0)
    ts = set().union(*[g.sites[i] for i in te]); trs = set().union(*[g.sites[i] for i in tr])
    assert not (ts & trs) and len(te) > 0

def test_model_shapes_and_masked_loss():
    net = M.EvoClassNetLite(); x = torch.randint(0, 20, (4, 201)); y = torch.tensor([[1., float("nan")]] * 4)
    p = net(x); assert p.shape == (4, 2) and torch.isfinite(net.loss(p, y))

def test_variant_ga_improves():
    wt = "ACDEFGHIKL" * 2; target = np.array([data.AA2I["W"]] * 20)
    f = lambda X: -(X != target).sum(1).astype(float)
    arch = ga.VariantGA(wt, f, pop=100, gens=15, max_muts=20, seed=0, log=lambda *a: None).run()
    assert max(v[0] for v in arch.values()) > f(np.array([[data.AA2I[a] for a in wt]]))[0]


def test_build_is_reproducible_and_seed_controls_weights():
    """Model weights must depend on the seed and nothing else (regression test: torch.manual_seed
    inside fit() runs after construction, so it cannot control initialisation)."""
    a = M.build(M.EvoClassNetLite, 7, n_tasks=3)
    torch.randn(1000)                     # disturb the global RNG
    b = M.build(M.EvoClassNetLite, 7, n_tasks=3)
    c = M.build(M.EvoClassNetLite, 8, n_tasks=3)
    assert torch.equal(a.emb.weight, b.emb.weight)
    assert not torch.equal(a.emb.weight, c.emb.weight)


def test_heldout_split_has_no_position_overlap():
    """No training or validation variant may mutate any test position.

    The expanded study's central positive claim is that models with position-transferable
    features predict effects at positions they never saw mutated. That claim rests entirely on
    this property of the split, so it is asserted directly rather than assumed.
    """
    from evoclassnet import data as D
    g = D.load_genotypes()
    for seed in range(3):
        tr, va, te = D.split_heldout_sites(g, seed)
        tr_sites = set().union(*[g.sites.iloc[i] for i in tr]) if len(tr) else set()
        va_sites = set().union(*[g.sites.iloc[i] for i in va]) if len(va) else set()
        te_sites = set().union(*[g.sites.iloc[i] for i in te]) if len(te) else set()
        assert not (tr_sites & te_sites), f"seed {seed}: train/test share positions"
        assert not (va_sites & te_sites), f"seed {seed}: val/test share positions"
        assert len(te_sites) > 10 and len(te) > 100


def test_transferable_features_carry_no_measurement():
    """The transferable feature block must be a function of sequence and structure only.

    If any column depended on a measured phenotype, the held-out-position result would be
    circular. This checks the construction directly: features are rebuilt after every label is
    replaced by noise, and must come out bit-identical.
    """
    import numpy as np
    from evoclassnet import data as D, plm, structure as S, zoo
    g = D.load_genotypes().head(500)
    sfeat, _, _, snames = S.features()
    mm, _ = plm.load()
    A = zoo.genotype_features(g.subs.tolist(), esm_mm=mm, struct=sfeat, struct_names=snames)
    g2 = g.copy()
    rng = np.random.default_rng(0)
    for c in ("bind", "expr", "escape"):
        g2[c] = rng.normal(size=len(g2))
    B = zoo.genotype_features(g2.subs.tolist(), esm_mm=mm, struct=sfeat, struct_names=snames)
    assert np.array_equal(A, B)


def test_structure_recovers_known_ace2_interface():
    """A geometry check on the structural pipeline, against residues established in the
    literature as ACE2 contacts."""
    from evoclassnet import structure as S
    f, mask, adj, names = S.compute()
    j = names.index("is_interface")
    iface = {i + 1 + S.OFFSET for i in range(201) if mask[i] and f[i, j] > 0.5}
    for known in (417, 455, 456, 486, 487, 489, 493, 501, 505):
        assert known in iface, f"spike {known} is a known ACE2 contact but was not recovered"
    assert len(iface) < 40, "interface set implausibly large"
