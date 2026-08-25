"""w4_firstev aggregation, AMENDMENT 13. Paired differences k minus k=0 across
seeds, the registered read, one ledger row. CPU, nothing generated, nothing
scored, the per seed jsons are the input."""
import argparse, json, glob, re
import numpy as np
import ledger

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="research/w4_firstev_s*.json")
ap.add_argument("--out", default="research/w4_firstev_results.json")
ap.add_argument("--no-ledger", action="store_true")
a = ap.parse_args()

files = sorted(f for f in glob.glob(a.glob) if re.search(r"_s\d+\.json$", f))
runs = [json.load(open(f)) for f in files]
print(f"w4_firstev_agg. {len(runs)} seeds: {[r['seed'] for r in runs]}")
ks = sorted({int(k) for r in runs for k in r["arms"]})
base = np.array([r["arms"]["0"]["contract"] for r in runs])
print(f"  k=0 per seed {np.round(base, 4).tolist()}  mean {base.mean():.4f}  se {base.std(ddof=1)/np.sqrt(len(base)):.4f}")
print()
print("     k   mean contract   paired diff    se    diff/se   frac forced   collapse seeds")
table = {}
for k in ks:
    c = np.array([r["arms"][str(k)]["contract"] for r in runs])
    d = c - base
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    ff = float(np.mean([r["arms"][str(k)]["frac_forced"] for r in runs]))
    col = int(sum(r["arms"][str(k)]["collapse"] for r in runs))
    z = d.mean() / se if se > 0 else float("nan")
    print(f"  {k:4d}     {c.mean():.4f}       {d.mean():+.4f}   {se:.4f}   {z:+6.2f}      {ff:.4f}        {col}/{len(runs)}")
    table[k] = dict(contract=c.tolist(), mean=float(c.mean()), diff=d.tolist(),
                    diff_mean=float(d.mean()), diff_se=float(se), z=float(z),
                    frac_forced=ff, collapse_seeds=col)
d1, s1 = table[1]["diff_mean"], table[1]["diff_se"]
if d1 <= -0.015 and abs(d1) >= 3 * s1:
    verdict = "LOAD BEARING, k=1 drop at or past 0.015 at 3 paired se"
elif abs(d1) <= 0.005:
    verdict = "NOT LOAD BEARING, k=1 within 0.005 of zero"
else:
    verdict = f"BETWEEN, k=1 diff {d1:+.4f} se {s1:.4f}, reported with the number"
print()
print(f"  registered read on k=1: {verdict}")
print(f"  prediction on record: |k=1 move| < 0.01  ->  {'RIGHT' if abs(d1) < 0.01 else 'WRONG'}")
print("  k=2, k=4 are shape only")
out = dict(seeds=[r["seed"] for r in runs], files=files, ks=ks, base=base.tolist(),
           table={str(k): v for k, v in table.items()}, verdict=verdict,
           prediction_right=bool(abs(d1) < 0.01))
json.dump(out, open(a.out, "w"), indent=1)
print(f"  wrote {a.out}")
if not a.no_ledger:
    rid = ledger.append_row(
        "w4_firstev",
        dict(ckpt=runs[0]["ckpt"], temps=runs[0]["temps"], n_rows=runs[0]["n_rows"],
             seeds=out["seeds"], ks=ks, batch=200),
        "ok",
        metrics={f"k{k}_contract_mean": table[k]["mean"] for k in ks}
        | {f"k{k}_diff": table[k]["diff_mean"] for k in ks if k}
        | {f"k{k}_diff_se": table[k]["diff_se"] for k in ks if k}
        | {"k0_se": float(base.std(ddof=1) / np.sqrt(len(base)))},
        artifacts=files + [a.out],
        notes="AMENDMENT 13. forced human prefix, paired per seed. " + verdict
              + ". forced trajectories are diagnostic, not generated, not candidates.",
        tier=1)
    print(f"  ledger {rid}")
