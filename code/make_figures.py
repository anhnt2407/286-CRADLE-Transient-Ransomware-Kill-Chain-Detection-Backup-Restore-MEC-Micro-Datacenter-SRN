"""
make_figures.py
===============
Render every CRADLE figure (PNG + PDF) from the CSVs written by
``experiments.py``.  Analytic curves are solid lines; independent Gillespie
estimates are overplotted as markers, mirroring the group's numerical-vs-
simulation validation idiom.  Run ``experiments.py`` first.
"""
from __future__ import annotations

import os
import json
import numpy as np

import plotstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVDIR = os.path.join(ROOT, "results", "csv")
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(name):
    """Load a CSV into a dict {column_name: np.ndarray}."""
    path = os.path.join(CSVDIR, name)
    with open(path) as fh:
        hdr = fh.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {h: data[:, i] for i, h in enumerate(hdr)}


def head():
    return json.load(open(os.path.join(CSVDIR, "headline.json")))


# --------------------------------------------------------------------------- #
def fig_stage_counts():
    d = load("fig_stage_counts.csv")
    s = load("fig_stage_counts_sim.csv")
    fig, ax = ps.newfig(ylabel="Expected \\# servers", xlim=(0, 20))
    series = [("E_comp", ps.SKY, "Compromised"), ("E_exfil", ps.ORANGE, "Exfiltrating"),
              ("E_encr", ps.BLUE, "Encrypting"), ("E_lock", ps.VERM, "Locked")]
    for key, c, lab in series:
        ax.plot(d["t"], d[key], color=c, label=lab)
        ax.plot(s["t"], s[key], "o", color=c, ms=3.5, lw=0)
    ax.legend(ncol=2, columnspacing=1.0, handlelength=1.4)
    ps.savefig(fig, "fig01_stage_counts", FIGDIR)


def fig_availability():
    d = load("fig_availability.csv")
    s = load("fig_availability_sim.csv")
    fig, ax = ps.newfig(ylabel="Probability / fraction", xlim=(0, 20))
    ax.plot(d["t"], d["avail_frac"], color=ps.GREEN, label="avail. service fraction")
    ax.plot(d["t"], d["p_service_denied"], color=ps.VERM,
            label="P(service denied)")
    ax.plot(s["t"], s["p_service_denied"], "o", color=ps.VERM, ms=3.5, lw=0)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ps.savefig(fig, "fig02_availability", FIGDIR)


def fig_loss_accum():
    d = load("fig_loss_accum.csv")
    fig, ax = ps.newfig(ylabel="Accumulated expected loss", xlim=(0, 20))
    ax.stackplot(d["t"], d["breach"], d["denial"], d["dataloss"], d["response"],
                 colors=[ps.ORANGE, ps.VERM, ps.BLUE, ps.PURPLE], alpha=0.85,
                 labels=["exfiltration", "service denial", "data loss (RPO)",
                         "response"])
    ax.plot(d["t"], d["total"], color="k", lw=1.3, label="total $S(t)$")
    ax.legend(loc="upper left", ncol=1)
    ps.savefig(fig, "fig03_loss_accum", FIGDIR)


def fig_detection_race():
    d = load("fig_detection_race.csv")
    fig, ax = ps.newfig(ylabel="P(detected by $t$)", xlim=(0, 20))
    cmap = {"sigma_0.5": (ps.GRAY, r"$\sigma_D=0.5$"),
            "sigma_1.0": (ps.SKY, r"$\sigma_D=1$"),
            "sigma_2.0": (ps.BLUE, r"$\sigma_D=2$"),
            "sigma_4.0": (ps.PURPLE, r"$\sigma_D=4$")}
    for key, (c, lab) in cmap.items():
        ax.plot(d["t"], d[key], color=c, label=lab)
    ax.set_ylim(0, 1.02)
    ax.legend(title="detection sensitivity")
    ps.savefig(fig, "fig04_detection_race", FIGDIR)


def fig_resolution():
    d = load("fig_resolution.csv")
    s = load("fig_resolution_sim.csv")
    h = head()
    fig, ax = ps.newfig(ylabel="P(incident resolved by $t$)", xlim=(0, 20))
    ax.plot(d["t"], d["P_resolved"], color=ps.GREEN, label="CRADLE (analytic)")
    ax.plot(s["t"], s["P_resolved"], "o", color=ps.GREEN, ms=3.5, lw=0,
            label="Gillespie SSA")
    ax.plot(d["t"], d["exp_fit"], color="k", ls=":", lw=1.3,
            label=r"$1-e^{-t/\mathrm{MTTR}}$ (memoryless)")
    ax.set_ylim(0, 1.02)
    ax.annotate(f"max gap = {h['resolution_dev_from_exp']:.2f}",
                xy=(3.0, 0.30), fontsize=7.5, color=ps.GRAY)
    ax.legend(loc="lower right")
    ps.savefig(fig, "fig05_resolution", FIGDIR)


def fig_costsurface():
    d = load("fig_costsurface.csv")
    h = head()
    sig = np.unique(d["sigma_D"]); tau = np.unique(d["tau_b"])
    J = np.empty((len(tau), len(sig)))
    si = {v: i for i, v in enumerate(sig)}
    ti = {v: i for i, v in enumerate(tau)}
    for k in range(len(d["J"])):
        J[ti[d["tau_b"][k]], si[d["sigma_D"][k]]] = d["J"][k]
    fig, ax = ps.newfig(xlabel=r"detection sensitivity $\sigma_D$",
                        ylabel=r"backup interval $\tau_b$ (h)")
    im = ax.pcolormesh(sig, tau, J, shading="gouraud", cmap="viridis")
    cs = ax.contour(sig, tau, J, levels=10, colors="w", linewidths=0.4, alpha=0.6)
    st = h["codesign"]
    ax.plot(st["sigma_D"], st["tau_b"], "*", color=ps.VERM, ms=15,
            markeredgecolor="w")
    ax.annotate("co-design\noptimum $(\\sigma_D^\\star,\\tau_b^\\star)$",
                xy=(st["sigma_D"], st["tau_b"]),
                xytext=(st["sigma_D"] + 1.6, st["tau_b"] + 12), fontsize=7,
                color="w", arrowprops=dict(arrowstyle="->", color="w", lw=0.8))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("expected total cost $J$", fontsize=8)
    ps.savefig(fig, "fig06_costsurface", FIGDIR)


def fig_pareto():
    fr = load("fig_pareto.csv")
    cl = load("fig_pareto_cloud.csv")
    h = head()
    fig, ax = ps.newfig(xlabel="operational spend $O$",
                        ylabel="security-dependability loss $S$")
    ax.scatter(cl["O"], cl["S"], s=5, color=ps.GRAY, alpha=0.35, label="feasible designs")
    ax.plot(fr["O"], fr["S"], "-", color=ps.BLUE, lw=1.8, label="Pareto frontier")
    st = h["codesign"]
    ax.plot(st["O"], st["S"], "*", color=ps.VERM, ms=13, markeredgecolor="w",
            label="min-total-cost $J^\\star$")
    ax.legend()
    ps.savefig(fig, "fig07_pareto", FIGDIR)


def fig_lever_sigma():
    d = load("fig_lever_sigma.csv")
    fig, ax = ps.newfig(xlabel=r"detection sensitivity $\sigma_D$",
                        ylabel="per-incident expectation")
    ax.plot(d["sigma_D"], d["breach"], color=ps.ORANGE, label="E[exfiltrated]")
    ax.plot(d["sigma_D"], d["denial"], color=ps.VERM, label="E[denial] (srv-d)")
    ax.plot(d["sigma_D"], d["dataloss"], color=ps.BLUE, label="E[data loss]")
    ax2 = ax.twinx()
    ax2.plot(d["sigma_D"], d["J"], color="k", ls="--", lw=1.3, label="total cost $J$")
    ax2.set_ylabel("total cost $J$")
    ax2.grid(False)
    kstar = int(np.argmin(d["J"]))
    ax2.axvline(d["sigma_D"][kstar], color=ps.GRAY, ls=":", lw=0.8)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper right")
    ps.savefig(fig, "fig08_lever_sigma", FIGDIR)


def fig_lever_tau():
    d = load("fig_lever_tau.csv")
    fig, ax = ps.newfig(xlabel=r"backup interval $\tau_b$ (h)",
                        ylabel="cost component")
    ax.plot(d["tau_b"], d["S"], color=ps.VERM, label="security loss $S$")
    ax.plot(d["tau_b"], d["O"], color=ps.GREEN, label="operational $O$")
    ax.plot(d["tau_b"], d["J"], color="k", lw=1.6, label="total $J$")
    kstar = int(np.argmin(d["J"]))
    ax.axvline(d["tau_b"][kstar], color=ps.GRAY, ls=":", lw=0.9)
    ax.annotate(r"$\tau_b^\star$", xy=(d["tau_b"][kstar], ax.get_ylim()[0]),
                xytext=(d["tau_b"][kstar] + 2, 0.6), fontsize=8)
    ax.legend()
    ps.savefig(fig, "fig09_lever_tau", FIGDIR)


def fig_scaling():
    d = load("fig_scaling.csv")
    fig, ax = ps.newfig(xlabel="MEDC size $m$ (servers)",
                        ylabel="per-incident expectation")
    ax.plot(d["m"], d["MTTResolve"], "o-", color=ps.GREEN, label="MTTResolve (d)")
    ax.plot(d["m"], d["E_breach"], "s-", color=ps.ORANGE, label="E[exfiltrated]")
    ax.plot(d["m"], d["E_denial"], "^-", color=ps.VERM, label="E[denial] (srv-d)")
    ax.plot(d["m"], d["E_dataloss"], "d-", color=ps.BLUE, label="E[data loss]")
    ax.set_ylim(0, None)
    ax.set_xticks(d["m"].astype(int))
    ax.legend(fontsize=7)
    ps.savefig(fig, "fig10_scaling", FIGDIR)


def fig_coverage():
    d = load("fig_coverage.csv")
    fig, ax = ps.newfig(xlabel=r"restore coverage $\mathrm{cov}_r$",
                        ylabel="per-incident expectation")
    ax.plot(d["cov_r"], d["MTTResolve"], "o-", color=ps.GREEN, label="MTTResolve (d)")
    ax.plot(d["cov_r"], d["E_dataloss"], "s-", color=ps.BLUE, label="E[data loss]")
    ax.plot(d["cov_r"], d["E_denial"], "^-", color=ps.VERM, label="E[denial] (srv-d)")
    ax.invert_xaxis()      # coverage worsens to the right
    ax.set_ylim(0, None)
    ax.legend(fontsize=7)
    ps.savefig(fig, "fig12_coverage", FIGDIR)


def fig_sensitivity():
    # tornado of scaled sensitivities on E[dataloss]
    import csv
    rows = list(csv.reader(open(os.path.join(CSVDIR, "fig_sensitivity.csv"))))
    hdr = rows[0]
    col = hdr.index("E_dataloss")
    params = [r[0] for r in rows[1:]]
    vals = np.array([float(r[col]) for r in rows[1:]])
    order = np.argsort(np.abs(vals))
    params = [params[i] for i in order]
    vals = vals[order]
    colors = [ps.VERM if v > 0 else ps.BLUE for v in vals]
    fig, ax = ps.newfig(xlabel=r"scaled sensitivity $ (\theta/M)\,\partial M/\partial\theta$",
                        ylabel="", figsize=(3.4, 3.0))
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([f"$\\{p}$" if p in ("beta", "phi", "rho", "delta0")
                        else p for p in params], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title("drivers of expected data loss", fontsize=8)
    ps.savefig(fig, "fig11_sensitivity", FIGDIR)


def main():
    fig_stage_counts()
    fig_availability()
    fig_loss_accum()
    fig_detection_race()
    fig_resolution()
    fig_costsurface()
    fig_pareto()
    fig_lever_sigma()
    fig_lever_tau()
    fig_scaling()
    fig_coverage()
    fig_sensitivity()
    print(f"Figures -> {FIGDIR}")


if __name__ == "__main__":
    main()
