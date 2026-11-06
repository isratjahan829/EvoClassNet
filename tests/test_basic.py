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
