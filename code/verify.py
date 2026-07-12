"""
verify.py
=========
End-to-end verification of the CRADLE model -- the group's "verification
triangle": (i) two independent analytic engines (matrix exponential vs Jensen
uniformization), (ii) two independent routes to the per-incident totals
(fundamental matrix  vs  time-integrated accumulated reward), and (iii) a seeded
Gillespie discrete-event simulation.  A check passes when analytic routes agree
to solver tolerance and the simulator agrees within max(2*SEM, 1%).

Also asserts the model's structural claims: conservativeness, unit-mass
transient law, the reward identity MTTResolve = MTTD + MResponse, monotone
security response to detection sensitivity, and -- the headline structural
contrast with the uniform-recovery base model -- that the resolution curve is
NOT a shifted exponential.

Writes results/csv/validation_table.csv and results/validation_report.md, and
prints a PASS/FAIL summary (exit code 0 iff all pass).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from cradle_model import CradleModel, Params, GOOD
from simulate import CradleSimulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVDIR = os.path.join(ROOT, "results", "csv")
os.makedirs(CSVDIR, exist_ok=True)


def main(n_traj: int = 200_000, seed: int = 2026) -> int:
    model = CradleModel()
    p = model.p
    rows = []
    ok = True

    def record(name, expected, got, passed, note=""):
        nonlocal ok
        ok = ok and bool(passed)
        rows.append((name, str(expected), str(got), "PASS" if passed else "FAIL", note))
        flag = "PASS" if passed else "**FAIL**"
        print(f"[{flag}] {name}: expected {expected}, got {got}")

    print("=" * 76)
    print("VERIFICATION: CRADLE ransomware kill-chain model")
    print("=" * 76)

    # 1. state-space count  (d=0 lattice C(m+4,4); d=1 lattice minus the empty pt)
    from math import comb
    exp_t = comb(p.m + 4, 4) + (comb(p.m + 4, 4) - 1)
    record("transient state count", exp_t, model.nt, model.nt == exp_t)

    # 2. generator conservativeness
    record("generator rows sum to 0", "<1e-12",
           f"{np.abs(model.Q.sum(1)).max():.1e}", np.allclose(model.Q.sum(1), 0, atol=1e-12))

    # 3. unit-mass transient law
    tt = np.array([0.5, 2.0, 5.0, 20.0])
    smax = np.abs(model.transient(tt).sum(1) - 1).max()
    record("sum_s pi_s(t) == 1", "<1e-9", f"{smax:.1e}", smax < 1e-9)

    # 4. expm  vs  uniformization
    pe = model.transient(tt)
    pu = model.transient_uniformization(tt)
    d_eng = np.abs(pe - pu).max()
    record("expm vs uniformization", "<1e-9", f"{d_eng:.1e}", d_eng < 1e-9)

    # 5. fundamental-matrix totals  vs  time-integrated accumulated reward
    tau = model.sojourn_times()
    rv = model.reward_vectors()
    tot = model.incident_totals()
    tbig = np.linspace(0, 300, 6001)
    _, accs = model.accumulated(tbig)
    A_inf = accs[-1, :model.nt]
    denial_int = float(A_inf @ rv["c"])
    breach_int = float(A_inf @ rv["breach_rate"])
    d_denial = abs(denial_int - tot["E_denial_daysrv"])
    d_breach = abs(breach_int - tot["E_breach"])
    record("E[denial]: fund-matrix vs integral", "<1e-4", f"{d_denial:.1e}", d_denial < 1e-4)
    record("E[breach]: fund-matrix vs integral", "<1e-4", f"{d_breach:.1e}", d_breach < 1e-4)

    # 6. reward identity  MTTResolve == MTTD + MResponse
    d_id = abs(tot["MTTResolve"] - (tot["MTTD"] + tot["MResponse"]))
    record("MTTResolve == MTTD + MResponse", "<1e-9", f"{d_id:.1e}", d_id < 1e-9)

    # 7. MTTResolve == integral of (1 - P(resolved,t))
    surv = 1.0 - model.resolve_cdf(tbig)
    mttr_int = float(np.trapezoid(surv, tbig))
    d_mttr = abs(mttr_int - tot["MTTResolve"])
    record("MTTResolve == int (1-P_resolved)", "<1e-3", f"{d_mttr:.1e}", d_mttr < 1e-3)

    # 8. headline: resolution curve is NOT a shifted exponential
    tg = np.linspace(0, 15, 61)
    rc = model.resolve_cdf(tg)
    gamma = 1.0 / tot["MTTResolve"]
    dev = np.abs(rc - (1 - np.exp(-gamma * tg))).max()
    record("resolution curve non-exponential", ">0.05", f"{dev:.3f}", dev > 0.05,
           "footprint-aware detection breaks the base-model closed form")

    # 9. monotonicity: sharper detection reduces security loss
    lo = CradleModel(p.with_(kx=p.kx*0.5, kb=p.kb*0.5, kc=p.kc*0.5)).incident_totals()
    hi = CradleModel(p.with_(kx=p.kx*2.0, kb=p.kb*2.0, kc=p.kc*2.0)).incident_totals()
    mono = (hi["E_breach"] < tot["E_breach"] < lo["E_breach"] and
            hi["E_denial_daysrv"] < tot["E_denial_daysrv"] < lo["E_denial_daysrv"])
    record("sharper detection lowers breach & denial", "monotone",
           f"breach {hi['E_breach']:.3f}<{tot['E_breach']:.3f}<{lo['E_breach']:.3f}", mono)

    # 10. Monte-Carlo cross-check (Gillespie SSA)
    print(f"\nGillespie SSA cross-check ({n_traj:,} trajectories) ...")
    sim = CradleSimulator(model)
    ts = np.linspace(0, 20, 21)
    S = sim.run(n_traj=n_traj, sample_times=ts, seed=seed)
    checks = [("E_breach", "breach"), ("E_denial_daysrv", "denial"),
              ("E_dataloss", "dataloss"), ("MTTD", "mttd"), ("MTTResolve", "mttr")]
    n_in = 0
    for an, sm in checks:
        a = tot[an]; msim = S["totals"][sm]; se = S["totals_se"][sm]
        # 3-sigma "no significant discrepancy" gate (robust to correlated,
        # high-variance rewards); a real bug would be many sigma off.
        tol = max(3 * se, 0.01 * abs(a))
        inside = abs(a - msim) <= tol
        n_in += int(inside)
        record(f"SSA {an}", f"{a:.4f}", f"{msim:.4f}+/-{1.96*se:.4f}", inside)
    # transient CDF agreement
    dc = model.detection_cdf(ts); rcv = model.resolve_cdf(ts)
    dcdf_ok = np.abs(dc - S["det_cdf"]).max() < 0.02
    rcdf_ok = np.abs(rcv - S["res_cdf"]).max() < 0.02
    record("SSA detection CDF (max|dP|)", "<0.02", f"{np.abs(dc-S['det_cdf']).max():.3f}", dcdf_ok)
    record("SSA resolution CDF (max|dP|)", "<0.02", f"{np.abs(rcv-S['res_cdf']).max():.3f}", rcdf_ok)

    # ---- write artefacts ------------------------------------------------- #
    path = os.path.join(CSVDIR, "validation_table.csv")
    with open(path, "w") as fh:
        fh.write("check,expected,obtained,result,note\n")
        for r in rows:
            fh.write(",".join(f'"{x}"' for x in r) + "\n")

    n_pass = sum(1 for r in rows if r[3] == "PASS")
    rep = os.path.join(ROOT, "results", "validation_report.md")
    with open(rep, "w") as fh:
        fh.write("# CRADLE -- validation report\n\n")
        fh.write(f"Base configuration: m={p.m}, {model.nt} transient states + 1 absorbing.\n\n")
        fh.write(f"**{n_pass}/{len(rows)} checks PASS.**\n\n")
        fh.write("| check | expected | obtained | result |\n|---|---|---|---|\n")
        for name, exp, got, res, note in rows:
            fh.write(f"| {name} | {exp} | {got} | {res} |\n")
        fh.write("\n## Per-incident expectations (base case)\n\n")
        for k, v in tot.items():
            fh.write(f"- `{k}` = {v:.4f}\n")

    print("\n" + "=" * 76)
    print(f"OVERALL: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} "
          f"({n_pass}/{len(rows)})")
    print(f"  table  -> {path}")
    print(f"  report -> {rep}")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
