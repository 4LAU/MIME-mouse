"""Free re-read of the cached duration probe: is the model uniformly weaker?"""
import sys
from pathlib import Path
import numpy as np
R = Path("/mnt/c/Users/aaron/Code/mouse-trajectory-synthesis")
sys.path[:0] = [str(R), str(R/"research")]
from features import FEATURE_NAMES
d = np.load(R/"research/w3_duration_response_cache.npz")
Xr, hb = d["human"], d["human_band"]
Xm = [d[f"model{k}"] for k in range(3)]
n = len(FEATURE_NAMES); iu = np.triu_indices(n, 1)

def rk(X):
    o = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        r = np.empty(len(X)); r[np.argsort(X[:, j], kind="stable")] = np.arange(len(X))
        o[:, j] = r
    return np.corrcoef(o, rowvar=False)

print(f"{'band':<8}{'human mean |r|':>16}{'model mean |r|':>16}{'ratio':>8}{'signs agree':>13}")
H, M = [], []
for k in range(3):
    h, m = rk(Xr[hb == k])[iu], rk(Xm[k])[iu]
    H.append(h); M.append(m)
    agree = float(np.mean(np.sign(h) == np.sign(m)))
    print(f"{k:<8}{np.abs(h).mean():>16.3f}{np.abs(m).mean():>16.3f}"
          f"{np.abs(m).mean()/np.abs(h).mean():>8.2f}{agree:>12.0%}")
H, M = np.array(H), np.array(M)
print(f"\nper-pair, model |r| as a fraction of human |r|, over all 153 pairs x 3 bands:")
frac = np.abs(M).ravel() / np.maximum(np.abs(H).ravel(), 1e-6)
print(f"  median {np.median(frac):.2f}   share below 1.0: {np.mean(frac < 1):.0%}")
print(f"\nrestricted to pairs where the human coupling is strong (|r| > 0.2):")
s = np.abs(H).ravel() > 0.2
print(f"  {s.sum()} of {len(s)} cells;  median fraction {np.median(frac[s]):.2f}"
      f"   share below 1.0: {np.mean(frac[s] < 1):.0%}")
print(f"\nbiggest absolute misses, band 2 (longest movements):")
gap = np.abs(M[2] - H[2]); o = np.argsort(-gap)[:8]
for t in o:
    i, j = iu[0][t], iu[1][t]
    print(f"  {FEATURE_NAMES[i]:<24}{FEATURE_NAMES[j]:<24}human {H[2][t]:+.3f}  model {M[2][t]:+.3f}")
