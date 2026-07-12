"""
experiments.py
==============
Compute every numerical result of the CRADLE study and write it to
``results/csv/*.csv`` (one file per figure / table) plus a single
``results/csv/headline.json`` of scalar results consumed by ``gen_tables.py``
and ``emit_macros.py``.  ``make_figures.py`` reads only these CSVs, so figures,
tables and in-text numbers all trace back to this one reproducible pass.
"""
from __future__ import annotations

import os
import json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from cradle_model import CradleModel, Params, GOOD
from simulate import CradleSimulator
import optimize as opt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVDIR = os.path.join(ROOT, "results", "csv")
os.makedirs(CSVDIR, exist_ok=True)

H = 20.0                      # analysis horizon (days)
NFINE = 401                   # fine grid for smooth analytic curves
MARKERS = np.arange(0, 21, 2.0)   # simulation-marker times
SSA_TRAJ = 200_000


def save_csv(name, header, cols):
    arr = np.column_stack(cols)
    with open(os.path.join(CSVDIR, name), "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in arr:
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")


def main():
    base = Params()
    cfg = opt.CostConfig()
    model = CradleModel(base)
    idx = model.index
    head = {}

    tf = np.linspace(0, H, NFINE)
    pis, accs = model.accumulated(tf)
    tot = model.incident_totals()
    head["base_totals"] = tot
    head["m"] = base.m
    head["nt"] = model.nt
    head["n"] = model.n

    # ---------- Fig A: kill-chain stage expected counts + SSA markers ------- #
    print("[1/9] transient stage counts + SSA ...")
    sim = CradleSimulator(model)
    S = sim.run(n_traj=SSA_TRAJ, sample_times=MARKERS, seed=2026)
    Ec = {f: model.expected_count(pis, f) for f in ("a", "x", "b", "c")}
    # SSA expected counts at markers
    Ec_sim = {}
    for f, j in (("a", 0), ("x", 1), ("b", 2), ("c", 3)):
        w = np.array([s[j] if s != GOOD else 0 for s in model.all_states], float)
        Ec_sim[f] = S["prob"] @ w
    save_csv("fig_stage_counts.csv",
             ["t", "E_comp", "E_exfil", "E_encr", "E_lock"],
             [tf, Ec["a"], Ec["x"], Ec["b"], Ec["c"]])
    save_csv("fig_stage_counts_sim.csv",
             ["t", "E_comp", "E_exfil", "E_encr", "E_lock"],
             [MARKERS, Ec_sim["a"], Ec_sim["x"], Ec_sim["b"], Ec_sim["c"]])

    # ---------- Fig B: service availability / denial ------------------------ #
    print("[2/9] availability ...")
    avail = 1.0 - Ec["c"] / base.m                    # expected available fraction
    p_deg = 1.0 - model.aggregate(pis, "c", 0, "eq")  # P(>=1 locked)
    p_deg_sim = 1.0 - (S["prob"][:, [idx[s] for s in model.tstates if s[3] == 0]].sum(1))
    save_csv("fig_availability.csv", ["t", "avail_frac", "p_service_denied"],
             [tf, avail, p_deg])
    save_csv("fig_availability_sim.csv", ["t", "p_service_denied"],
             [MARKERS, p_deg_sim])

    # ---------- Fig C: accumulated security-loss decomposition -------------- #
    print("[3/9] accumulated loss decomposition ...")
    rv = model.reward_vectors()
    A = accs[:, :model.nt]
    acc_breach = cfg.c_br * (A @ rv["breach_rate"])
    acc_denial = cfg.c_dn * (A @ rv["c"])
    acc_dataloss = cfg.c_dl * (A @ rv["dataloss_flux"])
    acc_resp = cfg.c_rt * (A @ rv["response"])
    acc_total = acc_breach + acc_denial + acc_dataloss + acc_resp
    save_csv("fig_loss_accum.csv",
             ["t", "breach", "denial", "dataloss", "response", "total"],
             [tf, acc_breach, acc_denial, acc_dataloss, acc_resp, acc_total])

    # ---------- Fig D: detection race vs sensitivity ------------------------ #
    print("[4/9] detection race ...")
    sig_list = [0.5, 1.0, 2.0, 4.0]
    cols = [tf]
    hdr = ["t"]
    for sg in sig_list:
        mm = CradleModel(base.with_(kx=base.kx*sg, kb=base.kb*sg, kc=base.kc*sg))
        cols.append(mm.detection_cdf(tf))
        hdr.append(f"sigma_{sg}")
    save_csv("fig_detection_race.csv", hdr, cols)

    # ---------- Fig E: resolution curve (non-exponential headline) ---------- #
    print("[5/9] resolution curve ...")
    rc = model.resolve_cdf(tf)
    gamma = 1.0 / tot["MTTResolve"]
    rc_exp = 1.0 - np.exp(-gamma * tf)
    rc_sim = S["res_cdf"]
    dev = float(np.abs(rc - rc_exp).max())
    head["resolution_dev_from_exp"] = dev
    save_csv("fig_resolution.csv", ["t", "P_resolved", "exp_fit"], [tf, rc, rc_exp])
    save_csv("fig_resolution_sim.csv", ["t", "P_resolved"], [MARKERS, rc_sim])

    # ---------- Fig F/G: co-design surface, optimum, Pareto ----------------- #
    print("[6/9] co-design optimisation (surface + Pareto) ...")
    surf = opt.grid_surface(base, cfg, nsig=25, ntau=25)
    joint = opt.optimize_joint(base, cfg, surf=surf)
    star = joint["star"]
    head["codesign"] = {k: star[k] for k in
                        ("sigma_D", "tau_b", "J", "S", "O", "MTTD",
                         "E_breach", "E_denial_daysrv", "E_dataloss")}
    # heatmap grid (rows=tau, cols=sigma) -> long format
    with open(os.path.join(CSVDIR, "fig_costsurface.csv"), "w") as fh:
        fh.write("sigma_D,tau_b,J,S,O\n")
        for i, tb in enumerate(surf["tau"]):
            for k, sg in enumerate(surf["sigma"]):
                fh.write(f"{sg:.5g},{tb:.5g},{surf['J'][i,k]:.6g},"
                         f"{surf['S'][i,k]:.6g},{surf['O'][i,k]:.6g}\n")
    par = opt.pareto_frontier(base, cfg)
    save_csv("fig_pareto.csv", ["O", "S"], [par["frontier"][:, 0], par["frontier"][:, 1]])
    save_csv("fig_pareto_cloud.csv", ["O", "S"], [par["all_O"], par["all_S"]])
    blind = opt.cost_of_detection_blind(base, cfg)
    head["blind"] = {"tau_blind": blind["tau_blind"], "J_blind": blind["blind"]["J"],
                     "penalty_frac": blind["penalty_frac"]}

    # ---------- Fig H: lever trade-off 1-D slices --------------------------- #
    print("[7/9] lever slices ...")
    sig_grid = np.linspace(cfg.sig_lo, cfg.sig_hi, 40)
    rows_sig = []
    for sg in sig_grid:
        e = opt.evaluate(sg, star["tau_b"], base, cfg)
        rows_sig.append([sg, e["E_breach"], e["E_denial_daysrv"], e["E_dataloss"],
                         e["MTTD"], e["S"], e["O"], e["J"]])
    rows_sig = np.array(rows_sig)
    save_csv("fig_lever_sigma.csv",
             ["sigma_D", "breach", "denial", "dataloss", "MTTD", "S", "O", "J"],
             [rows_sig[:, k] for k in range(rows_sig.shape[1])])
    tau_grid = np.linspace(cfg.tau_lo, cfg.tau_hi, 40)
    rows_tau = []
    for tb in tau_grid:
        e = opt.evaluate(star["sigma_D"], tb, base, cfg)
        rows_tau.append([tb, e["E_dataloss"], e["S"], e["O"], e["J"]])
    rows_tau = np.array(rows_tau)
    save_csv("fig_lever_tau.csv", ["tau_b", "dataloss", "S", "O", "J"],
             [rows_tau[:, k] for k in range(rows_tau.shape[1])])

    # ---------- Fig I: scaling with MEDC size m ----------------------------- #
    print("[8/9] scaling with m ...")
    scaling = []
    for m in range(2, 7):
        mm = CradleModel(base.with_(m=m))
        t = mm.incident_totals()
        scaling.append([m, mm.nt, t["MTTD"], t["MTTResolve"], t["E_breach"],
                        t["E_denial_daysrv"], t["E_dataloss"]])
    scaling = np.array(scaling, float)
    save_csv("fig_scaling.csv",
             ["m", "nt", "MTTD", "MTTResolve", "E_breach", "E_denial", "E_dataloss"],
             [scaling[:, k] for k in range(scaling.shape[1])])
    head["scaling"] = scaling.tolist()

    # ---------- Fig K: robustness to imperfect restore coverage ------------- #
    print("[9/10] restore-coverage robustness ...")
    cov_grid = np.linspace(1.0, 0.6, 21)
    rows_cov = []
    for cv in cov_grid:
        t = CradleModel(base.with_(cov_r=cv)).incident_totals()
        rows_cov.append([cv, t["MTTResolve"], t["E_denial_daysrv"],
                         t["E_dataloss"], t["E_breach"]])
    rows_cov = np.array(rows_cov)
    save_csv("fig_coverage.csv",
             ["cov_r", "MTTResolve", "E_denial", "E_dataloss", "E_breach"],
             [rows_cov[:, k] for k in range(rows_cov.shape[1])])
    t70 = CradleModel(base.with_(cov_r=0.7)).incident_totals()
    head["coverage"] = {
        "mttr_inflation_at_0p7": t70["MTTResolve"] / tot["MTTResolve"] - 1.0,
        "dataloss_inflation_at_0p7": t70["E_dataloss"] / tot["E_dataloss"] - 1.0}

    # ---------- Fig J: scaled sensitivity ranking --------------------------- #
    print("[10/10] sensitivity ...")
    ss = opt.scaled_sensitivity(base)
    prm_order = opt.SENS_PARAMS
    with open(os.path.join(CSVDIR, "fig_sensitivity.csv"), "w") as fh:
        fh.write("param," + ",".join(opt.SENS_METRICS) + "\n")
        for prm in prm_order:
            fh.write(prm + "," + ",".join(f"{ss[mm][prm]:.5g}"
                                          for mm in opt.SENS_METRICS) + "\n")
    head["sens_dataloss_top"] = sorted(ss["E_dataloss"].items(),
                                       key=lambda kv: -abs(kv[1]))[:3]

    # ---------- headline scalars ------------------------------------------- #
    with open(os.path.join(CSVDIR, "headline.json"), "w") as fh:
        json.dump(head, fh, indent=2)
    print(f"\nAll CSVs + headline.json -> {CSVDIR}")
    print(f"  co-design: sigma_D*={star['sigma_D']:.2f}, tau_b*={star['tau_b']:.1f} h, "
          f"J*={star['J']:.3f}")
    print(f"  detection-blind penalty = {100*blind['penalty_frac']:.1f}%")
    print(f"  resolution deviation from exponential = {dev:.3f}")


if __name__ == "__main__":
    main()
