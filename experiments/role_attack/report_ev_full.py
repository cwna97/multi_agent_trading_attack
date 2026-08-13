"""
Full reviewer-response report: adds median, tail stats (5th pct, CVaR-5%),
benefit column, precise inert definition, and bootstrap block-length
sensitivity, on top of analyze_ev_signed's engine.

Definitions (stated explicitly for the paper):
  * Single-flip marginal Δ_d = V(clean with only day d -> attacked) - V(clean).
  * inert  = Δ_d == 0  (engine no-op: decision changed but no trade occurred;
             the marginals are exactly 0 or clearly nonzero, so the threshold
             is unambiguous — see |Δ|<1e-6 == |Δ|<$1 in the data).
  * harm   = Δ_d < 0 ; benefit = Δ_d > 0 ; inert+harm+benefit = 100%.
  * $/success = mean_{d in S} Δ_d  (S = successful-attack days, pooled over variants).
  * median = median_{d in S} Δ_d.
  * p5 = 5th percentile of Δ_d ; CVaR5 = mean of the worst 5% (most negative) Δ_d.
  * EV = (1/N) Σ_d Δ_d over ALL N decision points, variant-averaged.
         NOTE: EV != ASR x ($/success) — ASR's denominator is *comparable*
         attempts (clean != target) while EV's is *all* attempts, and inert
         successes contribute 0 to the numerator. EV is the mean signed
         portfolio impact per decision the system makes.
"""
import numpy as np
import analyze_ev_signed as A

NBOOT = 10000
BLOCKS = [5, 10, 20]


def pooled_marginals(variants):
    m = []
    for folder, tspec in variants:
        recs = A.variant_attempts(folder, tspec)
        if recs is None:
            continue
        m += [r["marginal"] for r in recs if r["success"]]
    return np.array(m) if m else np.array([0.0])


def dist_stats(m):
    n = len(m)
    mean = m.mean()
    med = np.median(m)
    p5 = np.percentile(m, 5)
    k = max(1, int(np.ceil(0.05 * n)))
    cvar5 = np.sort(m)[:k].mean()          # mean of worst 5% (most negative)
    inert = 100 * np.mean(m == 0.0)
    harm = 100 * np.mean(m < 0.0)
    benefit = 100 * np.mean(m > 0.0)
    return dict(n=n, mean=mean, median=med, p5=p5, cvar5=cvar5,
               inert=inert, harm=harm, benefit=benefit)


def build_variant_records(variants):
    return [A.variant_attempts(f, ts) for f, ts in variants]


if __name__ == "__main__":
    print("=== Distribution / tail stats over successful-attack marginals (pooled variants) ===")
    hdr = (f"{'Attack':22s} {'n':>4s} {'mean':>7s} {'median':>7s} "
           f"{'p5':>7s} {'CVaR5':>8s} {'harm%':>6s} {'inert%':>7s} {'benefit%':>8s}")
    print(hdr)
    for name, variants in A.ATTACKS.items():
        s = dist_stats(pooled_marginals(variants))
        print(f"{name:22s} {s['n']:4d} {s['mean']:7.0f} {s['median']:7.0f} "
              f"{s['p5']:7.0f} {s['cvar5']:8.0f} {s['harm']:6.0f} {s['inert']:7.0f} {s['benefit']:8.0f}")

    print("\n=== EV $/attempt with block-length sensitivity (95% CI) ===")
    print(f"{'Attack':22s} {'EV(pt)':>8s} " + " ".join(f"{'L='+str(L):>18s}" for L in BLOCKS))
    for name, variants in A.ATTACKS.items():
        vrl = build_variant_records(variants)
        if any(v is None for v in vrl):
            print(f"  [WARN] {name} missing"); continue
        row = []
        ev_pt = None
        for L in BLOCKS:
            A.BLOCK = L
            r = A.bootstrap_attack(vrl, nboot=NBOOT)
            ev_pt = r["EV"][0]
            row.append(f"[{r['EV'][1]:6.1f},{r['EV'][2]:6.1f}]")
        print(f"{name:22s} {ev_pt:8.1f} " + " ".join(f"{c:>18s}" for c in row))

    print("\nEV point estimate is block-independent; only the CI width changes with L.")
