"""w4_firsthead aggregation, AMENDMENT 14 stage 3. Paired q1 and qT minus k0
across seeds, the human k=1 ceiling from w4_firstev on the same rows, the
registered read, one ledger row for the whole arm."""
import argparse, json, glob, re
import numpy as np
import ledger

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="research/w4_firsthead_s*.json")
ap.add_argument("--firstev", default="research/w4_firstev_results.json")
ap.add_argument("--nll", default="research/w4_firsthead_nll.json")
ap.add_argument("--out", default="research/w4_firsthead_results.json")
ap.add_argument("--no-ledger", action="store_true")
a = ap.parse_args()

files = sorted(f for f in glob.glob(a.glob) if re.search(r"_s\d+\.json$", f))
runs = [json.load(open(f)) for f in files]
fe = json.load(open(a.firstev))
nll = json.load(open(a.nll))
print(f"w4_firsthead_agg. {len(runs)} seeds: {[r['seed'] for r in runs]}")
k0 = np.array([r["arms"]["k0"]["contract"] for r in runs])
fe_k0 = np.array([fe["table"]["0"]["contract"][fe["seeds"].index(r["seed"])] for r in runs])
print(f"  k0 here      {np.round(k0, 4).tolist()}  mean {k0.mean():.4f}")
print(f"  k0 w4_firstev {np.round(fe_k0, 4).tolist()}  mean {fe_k0.mean():.4f}"
      f"  (same rows and seeds, GPU nondeterminism only)")
print(f"  human k=1 ceiling, w4_firstev: {fe['table']['1']['diff_mean']:+.4f} se {fe['table']['1']['diff_se']:.4f}")
print()
print("   arm   mean contract   paired diff    se    diff/se   q pad frac   q tick frac   collapse seeds")
table = {}
for arm in ("k0", "q1", "qT"):
    c = np.array([r["arms"][arm]["contract"] for r in runs])
    d = c - k0
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    z = d.mean() / se if se > 0 else float("nan")
    pad = np.mean([r["arms"][arm].get("q_pad_frac", 0.0) for r in runs])
    tick = np.mean([r["arms"][arm].get("q_tick_frac", 0.0) for r in runs])
    col = int(sum(r["arms"][arm]["collapse"] for r in runs))
    print(f"   {arm:>3}     {c.mean():.4f}        {d.mean():+.4f}   {se:.4f}   {z:+6.2f}      {pad:.4f}       {tick:.4f}         {col}/{len(runs)}")
    table[arm] = dict(contract=c.tolist(), mean=float(c.mean()), diff=d.tolist(),
                      diff_mean=float(d.mean()), diff_se=float(se), z=float(z),
                      collapse_seeds=col)
d1, s1 = table["q1"]["diff_mean"], table["q1"]["diff_se"]
if d1 <= -0.010 and abs(d1) >= 3 * s1:
    verdict = "WORKS, q1 drop at or past 0.010 at 3 paired se"
elif abs(d1) <= 0.005:
    verdict = "DOES NOT MOVE, q1 within 0.005 of zero"
else:
    verdict = f"BETWEEN, q1 diff {d1:+.4f} se {s1:.4f}, reported with the number"
print()
print(f"  registered read on q1 (PRIMARY): {verdict}")
print(f"  selected prediction (q1 drop >= 0.010): {'RIGHT' if d1 <= -0.010 else 'WRONG'}")
print(f"  q1 drop as a fraction of the human ceiling: {d1 / fe['table']['1']['diff_mean']:.2f}")
out = dict(seeds=[r["seed"] for r in runs], files=files, k0=k0.tolist(), k0_firstev=fe_k0.tolist(),
           human_k1=fe["table"]["1"], table=table, verdict=verdict,
           nll=dict(ar=nll["ar_pos0"], q=nll["q_pos0"]))
json.dump(out, open(a.out, "w"), indent=1)
print(f"  wrote {a.out}")
if not a.no_ledger:
    rid = ledger.append_row(
        "w4_firsthead",
        dict(ckpt=runs[0]["ckpt"], q="training/w4_firsthead_q.pt", q_d=512, q_epochs=40,
             temps=runs[0]["temps"], n_rows=runs[0]["n_rows"], seeds=out["seeds"],
             arms=["k0", "q1", "qT"], batch=200),
        "ok",
        metrics={f"{arm}_contract_mean": table[arm]["mean"] for arm in table}
        | {f"{arm}_diff": table[arm]["diff_mean"] for arm in ("q1", "qT")}
        | {f"{arm}_diff_se": table[arm]["diff_se"] for arm in ("q1", "qT")}
        | {"nll_ar_pos0_sum": sum(nll["ar_pos0"][h] for h in ("s", "th", "dt")),
           "nll_q_pos0_sum": sum(nll["q_pos0"][h] for h in ("s", "th", "dt")),
           "nll_q_minus_ar_th": nll["q_pos0"]["th"] - nll["ar_pos0"]["th"]},
        artifacts=files + [a.out, a.nll, "training/w4_firsthead_q.pt"],
        notes="AMENDMENT 14. dedicated first event conditional q(e0|cond), event 0 from q, AR "
              "continues through the served sampler, force mask at position 0 only. one trajectory "
              "per row, no selection. " + verdict,
        tier=1)
    print(f"  ledger {rid['run_id']}")
