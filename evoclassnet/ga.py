"""Genetic algorithms used in EvoClassNet.

1. `GAHyperSearch`  - evolves model hyper-parameters (fitness = validation Spearman).
2. `VariantGA`      - evolves RBD sequences to maximise model-predicted phenotype.
Both are standard (mu+lambda) GAs with tournament selection, crossover and mutation.
"""
import numpy as np
from .data import AA, AA2I


class GAHyperSearch:
    SPACE = {"lr": [5e-4, 1e-3, 2e-3, 4e-3], "emb": [16, 32, 48], "lstm": [32, 64, 96],
             "dropout": [0.0, 0.1, 0.2, 0.3], "ch": [(32, 48, 32), (64, 96, 64), (96, 128, 96)],
             "ks": [(3, 3, 3), (3, 5, 3), (5, 7, 5)], "wd": [1e-5, 1e-4, 1e-3]}

    def __init__(self, fitness, pop=6, gens=3, seed=0, pc=0.7, pm=0.25, tourn=2, log=print):
        self.f, self.pop, self.gens, self.pc, self.pm, self.t = fitness, pop, gens, pc, pm, tourn
        self.rng = np.random.default_rng(seed); self.log = log; self.cache = {}

    def _rand(self): return {k: v[self.rng.integers(len(v))] for k, v in self.SPACE.items()}
    def _key(self, g): return tuple((k, str(g[k])) for k in sorted(g))

    def _fit(self, g):
        k = self._key(g)
        if k not in self.cache: self.cache[k] = self.f(g)
        return self.cache[k]

    def _select(self, P, F):
        i = self.rng.choice(len(P), self.t, replace=False); return P[i[np.argmax(F[i])]]

    def run(self):
        P = [self._rand() for _ in range(self.pop)]; hist = []
        for gen in range(self.gens):
            F = np.array([self._fit(g) for g in P]); hist.append((gen, F.max(), F.mean()))
            self.log(f"[GA-hp] gen{gen} best={F.max():.4f} mean={F.mean():.4f} best={P[int(F.argmax())]}")
            elite = P[int(F.argmax())]; N = [elite]
            while len(N) < self.pop:
                a, b = self._select(P, F), self._select(P, F)
                c = {k: (b[k] if (self.rng.random() < self.pc and self.rng.random() < .5) else a[k]) for k in a}
                for k in c:
                    if self.rng.random() < self.pm: c[k] = self.SPACE[k][self.rng.integers(len(self.SPACE[k]))]
                N.append(c)
            P = N
        F = np.array([self._fit(g) for g in P]); best = P[int(F.argmax())]
        return best, float(F.max()), hist


class VariantGA:
    """Evolve variants of a seed sequence. `score(X)->(n,)` is any vectorised fitness
    (e.g. predicted binding + w * predicted expression)."""

    def __init__(self, wt, score, site_weights=None, pop=400, gens=20, pm_sites=2.0, pc=0.5,
                 max_muts=8, seed=0, tourn=3, log=print):
        self.wt = np.array([AA2I[a] for a in wt]); self.L = len(wt); self.score = score
        w = np.ones(self.L) if site_weights is None else np.asarray(site_weights, float)
        self.sw = w / w.sum(); self.pop, self.gens, self.pm, self.pc = pop, gens, pm_sites, pc
        self.max_muts, self.t, self.log = max_muts, tourn, log
        self.rng = np.random.default_rng(seed)

    def _mutate(self, x):
        x = x.copy()
        for _ in range(max(1, self.rng.poisson(self.pm))):
            p = self.rng.choice(self.L, p=self.sw); x[p] = self.rng.integers(20)
        d = np.flatnonzero(x != self.wt)
        if len(d) > self.max_muts:  # revert extras
            for p in self.rng.choice(d, len(d) - self.max_muts, replace=False): x[p] = self.wt[p]
        return x

    def run(self):
        P = np.stack([self._mutate(self.wt) for _ in range(self.pop)]); archive = {}
        for g in range(self.gens):
            F = self.score(P)
            for x, f in zip(P, F): archive[x.tobytes()] = (f, x)
            self.log(f"[GA-var] gen{g} best={F.max():.3f} mean={F.mean():.3f}")
            new = [P[int(F.argmax())]]
            while len(new) < self.pop:
                i = self.rng.choice(self.pop, self.t, replace=False); a = P[i[np.argmax(F[i])]]
                j = self.rng.choice(self.pop, self.t, replace=False); b = P[j[np.argmax(F[j])]]
                if self.rng.random() < self.pc:  # recombination: single crossover point
                    c = self.rng.integers(1, self.L); child = np.concatenate([a[:c], b[c:]])
                else: child = a
                new.append(self._mutate(child))
            P = np.stack(new)
        return archive
