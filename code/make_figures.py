"""
make_figures.py
===============
Render every CRADLE figure (PNG + PDF) from the CSVs written by
``experiments.py``.  All text is Times New Roman and the palette is monochrome:
series are told apart by line style and by small distinct markers.  Analytic
curves are lines; independent Gillespie estimates are filled markers.  Run
``experiments.py`` first.
"""
from __future__ import annotations

import os
import json
import numpy as np

import plotstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVDIR = os.path.join(ROOT, "results", "csv")
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

SIMKW = dict(linestyle="none", markersize=3.0, markerfacecolor=ps.K1,
             markeredgecolor=ps.K0, markeredgewidth=0.5)


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
    fig, ax = ps.newfig(ylabel="Expected number of servers", xlim=(0, 20))
    keys = [("E_comp", "Compromised"), ("E_exfil", "Exfiltrating"),
            ("E_encr", "Encrypting"), ("E_lock", "Locked")]
    for k, (key, lab) in enumerate(keys):
        ps.series(ax, d["t"], d[key], k, label=lab, every=25)
        ax.plot(s["t"], s[key], marker=ps.MARKERS[k], **SIMKW)
    hs, ls = ax.get_legend_handles_labels()
    hs.append(Line2D([], [], marker="o", **SIMKW))
    ls.append("simulation")
    ax.legend(hs, ls, ncol=2, columnspacing=0.9, handlelength=2.0, frameon=True, framealpha=1.0, edgecolor=ps.K3, facecolor="white", borderpad=0.3)
    ax.set_ylim(0, None)
    ps.savefig(fig, "fig01_stage_counts", FIGDIR)


def fig_availability():
    d = load("fig_availability.csv")
    s = load("fig_availability_sim.csv")
    fig, ax = ps.newfig(ylabel="Expected available fraction", xlim=(0, 20))
    h1 = ps.series(ax, d["t"], d["avail_frac"], 0,
                   label="available service fraction", every=25)
    ax.set_ylim(min(d["avail_frac"]) - 0.004, 1.002)
    ax2 = ax.twinx()
    ax2.grid(False)
    h2 = ps.series(ax2, d["t"], d["p_service_denied"], 2,
                   label="probability of service denial", every=25)
    ax2.plot(s["t"], s["p_service_denied"], marker=ps.MARKERS[2], **SIMKW)
    ax2.set_ylabel("Probability of service denial")
    ax2.set_ylim(0, max(d["p_service_denied"]) * 1.35)
    hs = [h1, h2, Line2D([], [], marker=ps.MARKERS[2], **SIMKW)]
    ax.legend(hs, [h.get_label() for h in hs[:2]] + ["simulation"],
              loc="center right", handlelength=2.0, frameon=True, framealpha=1.0, edgecolor=ps.K3, facecolor="white", borderpad=0.3)
    ps.savefig(fig, "fig02_availability", FIGDIR)


def fig_loss_accum():
    d = load("fig_loss_accum.csv")
    fig, ax = ps.newfig(ylabel="Accumulated expected loss", xlim=(0, 20))
    fills = [ps.K5, ps.K4, ps.K3, ps.K2]
    hatches = ["", "///", "...", "xxx"]
    labels = ["exfiltration", "service denial", "data loss", "response"]
    polys = ax.stackplot(d["t"], d["breach"], d["denial"], d["dataloss"],
                         d["response"], colors=fills, edgecolor=ps.K0,
                         linewidth=0.4)
    for p, hh in zip(polys, hatches):
        p.set_hatch(hh)
    ax.plot(d["t"], d["total"], color=ps.K0, lw=1.4, label="total loss")
    handles = [Patch(facecolor=f, hatch=hh, edgecolor=ps.K0, linewidth=0.4,
                     label=l) for f, hh, l in zip(fills, hatches, labels)]
    handles.append(Line2D([], [], color=ps.K0, lw=1.4, label="total loss"))
    ax.legend(handles=handles, loc="upper left", handlelength=1.6, frameon=True,
              framealpha=1.0, edgecolor=ps.K3, facecolor="white",
              borderpad=0.35)
    ax.set_ylim(0, 1.85)
    ps.savefig(fig, "fig03_loss_accum", FIGDIR)


def fig_detection_race():
    d = load("fig_detection_race.csv")
    fig, ax = ps.newfig(ylabel="Probability of detection by time $t$", xlim=(0, 20))
    labs = [("sigma_0.5", r"$\sigma_D=0.5$"), ("sigma_1.0", r"$\sigma_D=1$"),
            ("sigma_2.0", r"$\sigma_D=2$"), ("sigma_4.0", r"$\sigma_D=4$")]
    for k, (key, lab) in enumerate(labs):
        ps.series(ax, d["t"], d[key], k, label=lab, every=25)
    ax.set_ylim(0, 1.02)
    ax.legend(title="detection sensitivity", handlelength=2.2,
              title_fontsize=7.5, loc="lower right", frameon=True, framealpha=1.0, edgecolor=ps.K3, facecolor="white", borderpad=0.3)
    ps.savefig(fig, "fig04_detection_race", FIGDIR)


def fig_resolution():
    d = load("fig_resolution.csv")
    s = load("fig_resolution_sim.csv")
    h = head()
    fig, ax = ps.newfig(ylabel="Probability of resolution by time $t$", xlim=(0, 20))
    ps.series(ax, d["t"], d["P_resolved"], 0, label="CRADLE (analytic)", every=25)
    ax.plot(s["t"], s["P_resolved"], marker=ps.MARKERS[0], label="Gillespie simulation",
            **SIMKW)
    ax.plot(d["t"], d["exp_fit"], color=ps.K2, ls=":", lw=1.2,
            label=r"$1-e^{-t/\mathrm{MTTR}}$")
    ax.set_ylim(0, 1.02)
    ax.annotate(f"maximum gap {h['resolution_dev_from_exp']:.2f}",
                xy=(6.2, 0.36), fontsize=7.5, color=ps.K1)
    ax.legend(loc="lower right", handlelength=2.2, frameon=True, framealpha=1.0, edgecolor=ps.K3, facecolor="white", borderpad=0.3)
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
    fig, ax = ps.newfig(xlabel=r"Detection sensitivity $\sigma_D$",
                        ylabel=r"Backup interval $\tau_b$ (h)")
    ax.grid(False)
    # log colour scale: the storage term dominates at very short intervals and
    # would otherwise flatten the whole interior basin to one shade
    im = ax.pcolormesh(sig, tau, J, shading="gouraud", cmap="Greys",
                       norm=plt.matplotlib.colors.LogNorm(vmin=J.min(), vmax=J.max()))
    cs = ax.contour(sig, tau, J, levels=[2.0, 2.2, 2.6, 3.2, 4.5, 7.0, 10.0],
                    colors=ps.K0, linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
    st = h["codesign"]
    ax.plot(st["sigma_D"], st["tau_b"], marker="*", markersize=11,
            markerfacecolor=ps.K0, markeredgecolor="w", markeredgewidth=0.8,
            linestyle="none")
    ax.annotate("co-design optimum", xy=(st["sigma_D"], st["tau_b"]),
                xytext=(st["sigma_D"] + 0.9, st["tau_b"] - 11), fontsize=7,
                color=ps.K0,
                arrowprops=dict(arrowstyle="->", color=ps.K0, lw=0.7))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    ticks = [2, 3, 4, 6, 8, 12]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])
    cb.minorticks_off()
    cb.set_label("Expected total cost $J$", fontsize=8)
    ps.savefig(fig, "fig06_costsurface", FIGDIR)


def fig_pareto():
    fr = load("fig_pareto.csv")
    cl = load("fig_pareto_cloud.csv")
    h = head()
    fig, ax = ps.newfig(xlabel="Operational spend $O$",
                        ylabel="Security-dependability loss $S$")
    ax.scatter(cl["O"], cl["S"], s=3, color=ps.K4, alpha=0.7,
               label="feasible designs", edgecolors="none")
    ax.plot(fr["O"], fr["S"], "-", color=ps.K0, lw=1.3, label="Pareto frontier")
    st = h["codesign"]
    ax.plot(st["O"], st["S"], marker="*", markersize=11, linestyle="none",
            markerfacecolor=ps.K0, markeredgecolor="w", markeredgewidth=0.8,
            label="minimum total cost")
    ax.set_xlim(0, 6.2)
    ax.legend(handlelength=1.8)
    ps.savefig(fig, "fig07_pareto", FIGDIR)


def fig_lever_sigma():
    d = load("fig_lever_sigma.csv")
    fig, ax = ps.newfig(xlabel=r"Detection sensitivity $\sigma_D$",
                        ylabel="Per-incident expectation")
    h1 = ps.series(ax, d["sigma_D"], d["breach"], 0, label="datasets exfiltrated")
    h2 = ps.series(ax, d["sigma_D"], d["denial"], 1, label="denial (server-days)")
    h3 = ps.series(ax, d["sigma_D"], d["dataloss"], 2, label="data loss")
    ax2 = ax.twinx()
    ax2.grid(False)
    h4, = ax2.plot(d["sigma_D"], d["J"], color=ps.K0, ls=(0, (6, 2)), lw=1.3,
                   label="total cost $J$")
    ax2.set_ylabel("Total cost $J$")
    kstar = int(np.argmin(d["J"]))
    ax2.axvline(d["sigma_D"][kstar], color=ps.K2, ls=":", lw=0.8)
    ax.legend([h1, h2, h3, h4], [h.get_label() for h in (h1, h2, h3, h4)],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, handlelength=2.2, columnspacing=1.0, frameon=False)
    ps.savefig(fig, "fig08_lever_sigma", FIGDIR)


def fig_lever_tau():
    d = load("fig_lever_tau.csv")
    fig, ax = ps.newfig(xlabel=r"Backup interval $\tau_b$ (h)",
                        ylabel="Cost component")
    ps.series(ax, d["tau_b"], d["dataloss"], 3, label="expected data loss")
    ps.series(ax, d["tau_b"], d["S"], 1, label="security loss $S$")
    ps.series(ax, d["tau_b"], d["O"], 2, label="operational spend $O$")
    ax.plot(d["tau_b"], d["J"], color=ps.K0, lw=1.5, label="total cost $J$")
    kstar = int(np.argmin(d["J"]))
    ax.axvline(d["tau_b"][kstar], color=ps.K2, ls=":", lw=0.8)
    ax.annotate(r"$\tau_b^\star$", xy=(d["tau_b"][kstar], 0.2),
                xytext=(d["tau_b"][kstar] - 8.0, 0.45), fontsize=8)
    ax.set_ylim(0, 6.0)
    ax.legend(handlelength=2.2, frameon=True, framealpha=1.0, edgecolor=ps.K3, facecolor="white", borderpad=0.3)
    ps.savefig(fig, "fig09_lever_tau", FIGDIR)


def fig_scaling():
    d = load("fig_scaling.csv")
    fig, ax = ps.newfig(xlabel="Number of MEC servers",
                        ylabel="Per-incident expectation")
    ps.series(ax, d["m"], d["E_breach"], 0, label="datasets exfiltrated")
    ps.series(ax, d["m"], d["E_denial"], 1, label="denial (server-days)")
    ps.series(ax, d["m"], d["E_dataloss"], 2, label="data loss")
    ax2 = ax.twinx()
    ax2.grid(False)
    h4 = ps.series(ax2, d["m"], d["MTTResolve"], 3,
                   label="mean time to resolution (d)")
    ax2.set_ylabel("Mean time to resolution (d)")
    ax2.set_ylim(3.0, 4.5)
    ax.set_ylim(0, 0.8)
    ax.set_xticks(d["m"].astype(int))
    hs = ax.get_lines() + [h4]
    ax.legend(hs, [h.get_label() for h in hs], fontsize=7, loc="upper center",
              bbox_to_anchor=(0.45, -0.28), ncol=2, handlelength=2.2,
              columnspacing=1.0, frameon=False)
    ps.savefig(fig, "fig10_scaling", FIGDIR)


def fig_coverage():
    d = load("fig_coverage.csv")
    fig, ax = ps.newfig(xlabel=r"Restore coverage $\mathrm{cov}_r$",
                        ylabel="Per-incident expectation")
    ps.series(ax, d["cov_r"], d["E_dataloss"], 0, label="data loss")
    ps.series(ax, d["cov_r"], d["E_denial"], 1, label="denial (server-days)")
    ax2 = ax.twinx()
    ax2.grid(False)
    h3 = ps.series(ax2, d["cov_r"], d["MTTResolve"], 3,
                   label="mean time to resolution (d)")
    ax2.set_ylabel("Mean time to resolution (d)")
    ax.invert_xaxis()      # coverage worsens to the right (twinx shares the axis)
    ax.set_ylim(0.25, 0.95)
    ax2.set_ylim(3.7, 6.6)
    hs = ax.get_lines() + [h3]
    ax.legend(hs, [h.get_label() for h in hs], fontsize=7, loc="upper left",
              handlelength=2.2, frameon=True, framealpha=1.0, edgecolor=ps.K3, facecolor="white", borderpad=0.3)
    ps.savefig(fig, "fig12_coverage", FIGDIR)


SYM = {"beta": r"$\beta$", "betaL": r"$\beta_L$", "eps": r"$\epsilon$",
       "phi": r"$\phi$", "rho": r"$\rho$", "delta0": r"$\delta_0$",
       "kx": r"$\kappa_x$", "kb": r"$\kappa_b$", "kc": r"$\kappa_c$",
       "RTO0": r"$\mathrm{RTO}_0$", "RTO1": r"$\mathrm{RTO}_1$", "wL": r"$w_L$"}


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
    fig, ax = ps.newfig(xlabel=r"Scaled sensitivity $(\theta/M)\,\partial M/\partial\theta$",
                        ylabel="", figsize=(3.4, 3.0))
    for i, v in enumerate(vals):
        ax.barh(i, v, color=ps.K4 if v >= 0 else ps.K2,
                hatch="" if v >= 0 else "///", edgecolor=ps.K0, linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([SYM.get(p, p) for p in params], fontsize=8)
    ax.axvline(0, color=ps.K0, lw=0.6)
    ax.grid(axis="y", visible=False)
    handles = [Patch(facecolor=ps.K4, edgecolor=ps.K0, linewidth=0.5,
                     label="increases loss"),
               Patch(facecolor=ps.K2, hatch="///", edgecolor=ps.K0,
                     linewidth=0.5, label="decreases loss")]
    ax.legend(handles=handles, loc="lower right", handlelength=1.4)
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
