"""w4_firsthead confirm aggregation, AMENDMENT 15. The four registered reads
in order, one ledger row, tier 2 confirming the stage 3 row."""
import argparse, json, glob, re
import numpy as np
import ledger

BASE_MEAN, BASE_SE = 0.5910, 0.0022   # ten seed baseline, w4_tenseed

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="research/w4_fhconfirm_s*.json")
ap.add_argument("--out", default="research/w4_fhconfirm_results.json")
ap.add_argument("--confirms", default="w4_firsthead_2026-08-19T220329+0000_c020f298")
ap.add_argument("--no-ledger", action="store_true")
a = ap.parse_args()

files = sorted(f for f in glob.glob(a.glob) if re.search(r"_s\d+\.json$", f))
runs = [json.load(open(f)) for f in files]
print(f"w4_fhconfirm_agg. {len(runs)} seeds: {[r['seed'] for r in runs]}")
arms = {}
k0 = np.array([r["arms"]["k0"]["contract"] for r in runs])
print(f"\n   arm   per seed{'':38}mean    paired diff     se    diff/se")
for arm in ("k0", "q1", "qT"):
    c = np.array([r["arms"][arm]["contract"] for r in runs])
    d = c - k0
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    z = d.mean() / se if se > 0 else float("nan")
    arms[arm] = dict(contract=c.tolist(), mean=float(c.mean()),
                     se=float(c.std(ddof=1) / np.sqrt(len(c))),
                     diff=d.tolist(), diff_mean=float(d.mean()),
                     diff_se=float(se), z=float(z))
    print(f"   {arm:>3}   {np.round(c, 4).tolist()}   {c.mean():.4f}    "
          f"{d.mean():+.4f}     {se:.4f}   {z:+6.2f}")

print()
# READ 1
gap = arms["k0"]["mean"] - BASE_MEAN
lim = 2 * BASE_SE
r1 = abs(gap) <= lim + 1e-12
print(f"  READ 1  k0 mean {arms['k0']['mean']:.4f} vs baseline {BASE_MEAN} "
      f"(gap {gap:+.4f}, bar {lim:.4f})  -> {'PASS' if r1 else 'STOP, DIAGNOSE'}")
# READ 2
d1, s1 = arms["q1"]["diff_mean"], arms["q1"]["diff_se"]
r2 = d1 <= -0.008 and abs(d1) >= 3 * s1
print(f"  READ 2  PRIMARY q1 minus k0 {d1:+.4f} se {s1:.4f} "
      f"({d1 / s1:+.2f} se)  -> {'TRANSFERS' if r2 else 'DOES NOT MEET THE BAR'}")
# READ 3
dq = np.array(arms["qT"]["contract"]) - np.array(arms["q1"]["contract"])
dq_se = dq.std(ddof=1) / np.sqrt(len(dq))
r3 = dq.mean() <= -0.005 and abs(dq.mean()) >= 2 * dq_se
served = "qT" if (r2 and r3) else "q1"
print(f"  READ 3  qT minus q1 {dq.mean():+.4f} se {dq_se:.4f} "
      f"({dq.mean() / dq_se:+.2f} se)  -> {'qT SERVES' if r3 else 'q1 SERVES'}")
# READ 4
if r1 and r2:
    print(f"  READ 4  NEW HEADLINE  {served} mean {arms[served]['mean']:.4f} "
          f"se {arms[served]['se']:.4f}   (old 0.5910 se 0.0022)")
    verdict = (f"TRANSFERS, served arm {served}, new headline "
               f"{arms[served]['mean']:.4f} se {arms[served]['se']:.4f}")
elif not r1:
    verdict = "K0 OFF BASELINE, stop and diagnose, reads 2 to 4 not taken"
else:
    verdict = f"DOES NOT TRANSFER, q1 diff {d1:+.4f} se {s1:.4f}"
print(f"\n  VERDICT  {verdict}")
out = dict(seeds=[r["seed"] for r in runs], files=files, baseline=[BASE_MEAN, BASE_SE],
           arms=arms, qT_minus_q1=dict(mean=float(dq.mean()), se=float(dq_se)),
           reads=dict(r1=bool(r1), r2=bool(r2), r3=bool(r3)), served=served,
           verdict=verdict)
json.dump(out, open(a.out, "w"), indent=1)
print(f"  wrote {a.out}")
if not a.no_ledger:
    rid = ledger.append_row(
        "w4_firsthead",
        dict(ckpt=runs[0]["ckpt"], q="training/w4_firsthead_q.pt",
             population="make_specs", temps=runs[0]["temps"],
             n_rows=runs[0]["n_rows"], seeds=out["seeds"],
             arms=["k0", "q1", "qT"], batch=200),
        "ok",
        metrics={f"{arm}_mean": arms[arm]["mean"] for arm in arms}
        | {"q1_diff": d1, "q1_diff_se": s1,
           "qT_minus_q1": float(dq.mean()), "qT_minus_q1_se": float(dq_se)},
        artifacts=files + [a.out],
        notes="AMENDMENT 15 confirm on the standard population. " + verdict
              + ". one trajectory per spec, no selection.",
        tier=2, confirms_run_id=a.confirms)
    print(f"  ledger {rid['run_id']}")
