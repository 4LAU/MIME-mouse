"""w4_pairadv. AMENDMENT 32, registered in step0_prereg.md before this
file existed.

Adversarial refinement of the Pair1 e1 conditional with exact score
function (REINFORCE) gradients: the factorized chain's log probability
of a drawn (s1, th1, dt1) is exactly computable, so no relaxation is
needed. In loop discriminator is a torch logistic regression on the
AMENDMENT 31 twelve features, refit every 100 generator steps on fresh
train rows with disjoint halves per class. It never sees contract
features, the protected scorer, or the protected human eval file.

cmd refine: fine tune from training/w4_pairq1.pt, save w4_pairadv.pt.
cmd gate:   the AMENDMENT 31 revised RF instrument against the refined
            model (draw seeds 30011/30012, split rng(3106)) plus the
            val CE budget check. PASS launches the contract run.
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                       # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS  # noqa: E402
from w4_firsthead import ce_triplet                                # noqa: E402
from w4_pairq import (N_VAL, P1_PATH, VAL_ROWS_SEED, Pair1,        # noqa: E402
                      pair_tokens, splits)
import ledger                                                      # noqa: E402

PA_PATH = "training/w4_pairadv.pt"
CE_BASE = 6.7230          # Pair1 best val sum s1+th1+dt1, AMENDMENT 27
GATE_SEED_D1, GATE_SEED_D2, GATE_SPLIT = 30011, 30012, 3106


def load_data(dev):
    lengths, trained, held = splits()
    trained = trained[lengths[trained] >= 2]
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))
    val = val[lengths[val] >= 2]
    s0, th0, d0, s1, th1, d1 = pair_tokens()
    cond = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    C = torch.from_numpy(cond).to(dev)
    toks = [torch.from_numpy(np.asarray(x, dtype=np.int64)).to(dev)
            for x in (s0, th0, d0, s1, th1, d1)]
    return (torch.from_numpy(trained).to(dev), torch.from_numpy(val).to(dev),
            C, toks)


def draw_logp(pair, c, s0, th0, dt0):
    """One draw per row through the chain, with the exact log probability
    of the drawn triplet. th of a non motion draw is forced to NULL and
    contributes no log probability (deterministic given s), matching the
    ce_triplet convention."""
    h = pair.trunk(c, s0, th0, dt0)
    ls = F.log_softmax(pair.s_head(h), -1)
    s = torch.multinomial(ls.detach().exp(), 1).squeeze(-1)
    lth = F.log_softmax(pair.th_head(pair.th_norm(h + pair.s_emb(s))), -1)
    th = torch.multinomial(lth.detach().exp(), 1).squeeze(-1)
    motion = (s > TICK_CLASS) & (s < S_PAD_CLASS)
    th = torch.where(motion, th, torch.full_like(th, TH_NULL_CLASS))
    ldt = F.log_softmax(pair.dt_head(pair.dt_norm(h + pair.s_emb(s) + pair.th_emb(th))), -1)
    dt = torch.multinomial(ldt.detach().exp(), 1).squeeze(-1)
    logp = (ls.gather(-1, s[:, None]).squeeze(-1)
            + torch.where(motion, lth.gather(-1, th[:, None]).squeeze(-1),
                          torch.zeros_like(h[:, 0]))
            + ldt.gather(-1, dt[:, None]).squeeze(-1))
    return s, th, dt, logp


def dfeat(c, s0, th0, dt0, s, th, dt):
    f = [x.float()[:, None] for x in (s0, th0, dt0, s, th, dt,
                                      s - s0, dt - dt0)]
    return torch.cat([c] + f, -1)


class Disc(torch.nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(12, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.lin((x - self.mu) / self.sd).squeeze(-1)


def fit_disc(Xh, Xd, mu, sd, dev, steps=500):
    d = Disc(mu, sd).to(dev)
    opt = torch.optim.Adam(d.parameters(), lr=1e-2)
    X = torch.cat([Xh, Xd])
    y = torch.cat([torch.ones(len(Xh), device=dev), torch.zeros(len(Xd), device=dev)])
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(d(X), y)
        loss.backward(); opt.step()
    d.eval()
    return d


def logistic_auc(pair, C, toks, rows, dev, seed):
    """Epoch proxy: fresh logistic on disjoint halves of the given rows,
    fit on 80 percent, AUC on the held 20 percent."""
    from sklearn.metrics import roc_auc_score
    S0, TH0, D0, S1, TH1, D1 = toks
    g = np.random.default_rng(seed)
    r = rows[torch.from_numpy(g.permutation(len(rows))).to(dev)]
    h1, h2 = r[:len(r) // 2], r[len(r) // 2:len(r) // 2 * 2]
    with torch.no_grad():
        torch.manual_seed(seed)
        s, th, dt, _ = draw_logp(pair, C[h2], S0[h2], TH0[h2], D0[h2])
        Xh = dfeat(C[h1], S0[h1], TH0[h1], D0[h1], S1[h1], TH1[h1], D1[h1])
        Xd = dfeat(C[h2], S0[h2], TH0[h2], D0[h2], s, th, dt)
    both = torch.cat([Xh, Xd])
    mu, sd = both.mean(0), both.std(0).clamp(min=1e-6)
    ntr = int(len(Xh) * 0.8)
    d = fit_disc(Xh[:ntr], Xd[:ntr], mu, sd, dev)
    with torch.no_grad():
        sc = torch.cat([d(Xh[ntr:]), d(Xd[ntr:])]).cpu().numpy()
    y = np.concatenate([np.ones(len(Xh) - ntr), np.zeros(len(Xd) - ntr)])
    return float(roc_auc_score(y, sc))


def evaluate_ce(pair, C, toks, rows):
    S0, TH0, D0, S1, TH1, D1 = toks
    tot = np.zeros(5)
    with torch.no_grad():
        for c0 in range(0, len(rows), 65536):
            i = rows[c0:c0 + 65536]
            zs, zth, zdt = pair(C[i], S0[i], TH0[i], D0[i], S1[i], TH1[i])
            tot += [float(x) for x in ce_triplet(zs, zth, zdt, S1[i], TH1[i], D1[i])]
    return tot[0] / tot[3], tot[1] / max(tot[4], 1), tot[2] / tot[3]


def cmd_refine(a):
    dev = esp._DEVICE
    tr, va, C, toks = load_data(dev)
    S0, TH0, D0, S1, TH1, D1 = toks
    pk = torch.load(P1_PATH, map_location=dev, weights_only=False)
    pair = Pair1(**pk["config"]).to(dev)
    pair.load_state_dict(pk["model_state_dict"])
    opt = torch.optim.AdamW(pair.parameters(), lr=a.lr, weight_decay=0.01)
    torch.manual_seed(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    print(f"  refine from {P1_PATH} best epoch {pk['best']['epoch']}  lambda {a.lam}"
          f"  lr {a.lr}  epochs {a.epochs}  train rows {len(tr):,}", flush=True)

    disc, mu, sd = None, None, None

    def refit(step):
        nonlocal disc, mu, sd
        i = tr[torch.randperm(len(tr), generator=g, device=dev)[:65536]]
        h1, h2 = i[:32768], i[32768:]
        with torch.no_grad():
            s, th, dt, _ = draw_logp(pair, C[h2], S0[h2], TH0[h2], D0[h2])
            Xh = dfeat(C[h1], S0[h1], TH0[h1], D0[h1], S1[h1], TH1[h1], D1[h1])
            Xd = dfeat(C[h2], S0[h2], TH0[h2], D0[h2], s, th, dt)
        if mu is None:
            both = torch.cat([Xh, Xd])
            mu, sd = both.mean(0), both.std(0).clamp(min=1e-6)
        disc = fit_disc(Xh, Xd, mu, sd, dev)

    best, hist, step = None, [], 0
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr), generator=g, device=dev)]
        pair.train()
        for c0 in range(0, len(perm) - a.batch + 1, a.batch):
            if a.lam > 0 and step % 100 == 0:
                refit(step)
            i = perm[c0:c0 + a.batch]
            zs, zth, zdt = pair(C[i], S0[i], TH0[i], D0[i], S1[i], TH1[i])
            l1 = ce_triplet(zs, zth, zdt, S1[i], TH1[i], D1[i])
            ce = l1[0] / l1[3] + l1[1] / l1[4].clamp(min=1) + l1[2] / l1[3]
            if a.lam > 0:
                s, th, dt, logp = draw_logp(pair, C[i], S0[i], TH0[i], D0[i])
                with torch.no_grad():
                    score = disc(dfeat(C[i], S0[i], TH0[i], D0[i], s, th, dt))
                    adv = score - score.mean()
                loss = ce - a.lam * (adv * logp).mean()
            else:
                # AMENDMENT 33: lam 0 is pure CE continuation.
                loss = ce
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(pair.parameters(), 1.0)
            opt.step(); step += 1
        pair.eval()
        v = evaluate_ce(pair, C, toks, va)
        auc = logistic_auc(pair, C, toks, va, dev, 4000 + ep)
        # AMENDMENT 33: at lam 0 selection is on val CE alone.
        proxy = sum(v) + (10.0 * max(0.0, auc - 0.5) if a.lam > 0 else 0.0)
        hist.append(dict(epoch=ep + 1, step=step, val_s1=v[0], val_th1=v[1],
                         val_dt1=v[2], val_sum=sum(v), val_logauc=auc,
                         proxy=proxy))
        print(f"  epoch {ep + 1:2d}  step {step}  s {v[0]:.4f} th {v[1]:.4f}"
              f" dt {v[2]:.4f}  sum {sum(v):.4f}  logAUC {auc:.4f}"
              f"  proxy {proxy:.4f}", flush=True)
        if best is None or proxy < best["proxy"]:
            best = hist[-1]
            torch.save(dict(config=pk["config"], model_state_dict=pair.state_dict(),
                            hist=hist, best=best, seed=a.seed, lam=a.lam,
                            base=P1_PATH), PA_PATH)
    print(f"  best epoch {best['epoch']} proxy {best['proxy']:.4f}  saved {PA_PATH}")


def cmd_gate(a):
    dev = esp._DEVICE
    tr, va, C, toks = load_data(dev)
    S0, TH0, D0, S1, TH1, D1 = toks
    pk = torch.load(PA_PATH, map_location=dev, weights_only=False)
    pair = Pair1(**pk["config"]).to(dev).eval()
    pair.load_state_dict(pk["model_state_dict"])
    print(f"  gate on {PA_PATH} best epoch {pk['best']['epoch']} lam {pk['lam']}",
          flush=True)

    def draw(seed):
        out = []
        with torch.no_grad():
            for c0 in range(0, len(va), 65536):
                torch.manual_seed(seed + c0)
                i = va[c0:c0 + 65536]
                s, th, dt, _ = draw_logp(pair, C[i], S0[i], TH0[i], D0[i])
                out.append(torch.stack([s, th, dt], -1).cpu().numpy())
        return np.concatenate(out)

    e1_a, e1_b = draw(GATE_SEED_D1), draw(GATE_SEED_D2)
    npc = C[va].cpu().numpy().astype(np.float64)
    nps0, npth0, npd0, nps1, npth1, npd1 = (x[va].cpu().numpy().astype(np.float64)
                                            for x in toks)

    def feats(e1):
        e1 = e1.astype(np.float64)
        return np.concatenate([npc, nps0[:, None], npth0[:, None], npd0[:, None],
                               e1, (e1[:, 0] - nps0)[:, None],
                               (e1[:, 2] - npd0)[:, None]], -1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    def oob_auc(Xa, Xb, seed, tag):
        X = np.concatenate([Xa, Xb])
        y = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
        perm = np.random.default_rng(seed).permutation(len(X))
        rf = RandomForestClassifier(n_estimators=300, n_jobs=28, oob_score=True,
                                    random_state=seed)
        rf.fit(X[perm], y[perm])
        auc = roc_auc_score(y[perm], rf.oob_decision_function_[:, 1])
        print(f"  {tag}: OOB AUC {auc:.4f}", flush=True)
        return auc

    half = np.random.default_rng(GATE_SPLIT).permutation(len(va))
    h1, h2 = half[:len(va) // 2], half[len(va) // 2:]
    Xh = feats(np.stack([nps1, npth1, npd1], -1))
    auc_main = oob_auc(Xh[h1], feats(e1_a)[h2], 3200, "human(h1) vs refined(h2)")
    auc_ctrl = oob_auc(feats(e1_a)[h1], feats(e1_b)[h2], 3201, "control")
    v = evaluate_ce(pair, C, toks, va)
    ce_sum = sum(v)
    c1 = auc_main <= 0.520
    c2 = abs(auc_ctrl - 0.5) <= 0.005
    c3 = ce_sum <= CE_BASE + 0.05
    verdict = "PASS" if (c1 and c2 and c3) else "FAIL"
    print(f"  val CE s {v[0]:.4f} th {v[1]:.4f} dt {v[2]:.4f}  sum {ce_sum:.4f}"
          f"  budget {CE_BASE + 0.05:.4f}")
    print(f"  GATE (a): auc {auc_main:.4f} <= 0.520 {c1}; control ok {c2};"
          f" ce ok {c3}  ==> {verdict}")

    res = dict(auc_main=float(auc_main), auc_control=float(auc_ctrl),
               ce=[float(x) for x in v], ce_sum=float(ce_sum),
               lam=pk["lam"], best_epoch=int(pk["best"]["epoch"]),
               verdict=verdict)
    with open("research/w4_pairadv_gate.json", "w") as fh:
        json.dump(res, fh, indent=1)
    rid = ledger.append_row(
        "w4_pairadv",
        {"phase": "train+gate", "lam": pk["lam"], "epochs": len(pk["hist"]),
         "draw_seeds": [GATE_SEED_D1, GATE_SEED_D2], "split": GATE_SPLIT},
        "ok" if verdict == "PASS" else "failed",
        metrics={"auc_main": res["auc_main"], "auc_control": res["auc_control"],
                 "ce_sum": res["ce_sum"]},
        artifacts=[PA_PATH, "research/w4_pairadv_gate.json"],
        notes=f"AMENDMENT 32 adversarial refinement gate (a). {verdict}."
              f" Registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refine", "gate"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (cmd_refine if a.cmd == "refine" else cmd_gate)(a)


if __name__ == "__main__":
    main()
