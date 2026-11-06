"""EvoClassNet-lite: CNN -> Bi-LSTM -> multi-head self-attention pooling -> multi-task heads
with learned uncertainty weighting (Kendall et al. 2018).

Not included (cannot be run in this environment, see README): frozen ESM-1b embeddings
(weights unreachable) and the structure GNN (no PDB access). `embed_extra` is the hook where
precomputed per-residue ESM features can be concatenated on a GPU machine.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvoClassNetLite(nn.Module):
    def __init__(self, n_tasks=2, emb=32, ch=(64, 96, 64), ks=(3, 5, 3), lstm=64, heads=4,
                 dropout=0.2, extra_dim=0, L=201):
        super().__init__()
        self.emb = nn.Embedding(20, emb)
        self.pos = nn.Parameter(torch.zeros(L, emb))
        layers, c_in = [], emb + extra_dim
        for i, (c, k) in enumerate(zip(ch, ks)):
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU()]
            if i < 2: layers.append(nn.MaxPool1d(2))  # 201 -> 100 -> 50 positions
            c_in = c
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(c_in, lstm, num_layers=1, batch_first=True, bidirectional=True)
        d = 2 * lstm
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.pool_q = nn.Linear(d, 1)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
                                    for _ in range(n_tasks)])
        self.log_var = nn.Parameter(torch.zeros(n_tasks))  # uncertainty weights

    def encode(self, x, extra=None):
        h = self.emb(x) + self.pos[None]
        if extra is not None:
            h = torch.cat([h, extra], -1)
        h = self.cnn(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        h = self.ln(h + self.drop(a))
        att = torch.softmax(self.pool_q(h).squeeze(-1), -1)
        z = torch.cat([(att.unsqueeze(-1) * h).sum(1), h.max(1).values], -1)
        return z, att

    def forward(self, x, extra=None):
        z, _ = self.encode(x, extra)
        return torch.cat([hd(z) for hd in self.heads], -1)

    def loss(self, pred, y):
        """Masked multi-task Huber loss with uncertainty weighting; y may contain NaN."""
        tot = 0.0
        for t in range(y.shape[1]):
            m = ~torch.isnan(y[:, t])
            if m.any():
                l = F.smooth_l1_loss(pred[m, t], y[m, t])
                tot = tot + torch.exp(-self.log_var[t]) * l + self.log_var[t]
        return tot


class GlobalEpistasis(nn.Module):
    """Strong DMS baseline: additive latent phenotype + learned monotone sigmoid link."""
    def __init__(self, n_tasks=2, L=201):
        super().__init__()
        self.eff = nn.Parameter(torch.zeros(n_tasks, L, 20))
        self.lo = nn.Parameter(torch.full((n_tasks,), -5.0))
        self.span = nn.Parameter(torch.full((n_tasks,), 6.0))
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x, extra=None):
        pos = torch.arange(x.shape[1])
        lat = self.eff[:, pos[None, :], x].sum(-1).T  # (B, T); WT residue effect is shared offset
        return self.lo + self.span * torch.sigmoid(lat + 2.0)

    loss = EvoClassNetLite.loss


def build(cls, seed, **kw):
    """Construct a model with weight initialisation controlled by `seed`.

    torch.manual_seed inside fit() is too late: the weights already exist by then, so their values
    depend on whatever consumed the global RNG earlier in the process. Every model in this package
    must be built through this function for a run to be reproducible from its seed.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    return cls(**kw)


def fit(model, X, Y, Xva, Yva, epochs=8, bs=256, lr=2e-3, wd=1e-4, seed=0, log=print, patience=3):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(X); steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y, dtype=torch.float32)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); tl = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = model.loss(model(Xt[idx]), Yt[idx])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tl += loss.item() * len(idx)
        vl = evaluate_loss(model, Xva, Yva)
        log(f"  ep{ep+1}/{epochs} train {tl/n:.4f} val {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval(); Xt = torch.as_tensor(X)
    return torch.cat([model(Xt[i:i + bs]) for i in range(0, len(X), bs)]).numpy()


@torch.no_grad()
def evaluate_loss(model, X, Y):
    p = torch.as_tensor(predict(model, X)); y = torch.as_tensor(Y, dtype=torch.float32)
    return float(np.mean([F.smooth_l1_loss(p[~torch.isnan(y[:, t]), t], y[~torch.isnan(y[:, t]), t]).item() for t in range(y.shape[1])]))
