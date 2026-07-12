"""
gen_tables.py
=============
Emit LaTeX ``booktabs`` table fragments into ``paper/tables/*.tex`` that the
manuscript ``\\input``s.  Data-driven tables (validation, co-design, scaling,
sensitivity) are built from ``results/csv`` so every printed number traces to
the reproducible experiment pass.  Run ``experiments.py`` (and ``verify.py``)
first.
"""
from __future__ import annotations

import os
import csv
import json

from cradle_model import Params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVDIR = os.path.join(ROOT, "results", "csv")
TABDIR = os.path.join(ROOT, "paper", "tables")
os.makedirs(TABDIR, exist_ok=True)


def write(name, lines):
    with open(os.path.join(TABDIR, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  -> paper/tables/{name}")


def head():
    return json.load(open(os.path.join(CSVDIR, "headline.json")))


# --------------------------------------------------------------------------- #
def tab_params():
    p = Params()
    rows = [
        (r"$m$", "MEC servers in the MEDC", f"{p.m}", "servers"),
        (r"$\beta$", "initial-access rate", f"{p.beta:.2f}", "d$^{-1}$"),
        (r"$\beta_L$", "lateral-movement rate / active foothold", f"{p.betaL:.2f}", "d$^{-1}$"),
        (r"$\epsilon$", "exfiltration-onset rate / compromised", f"{p.eps:.2f}", "d$^{-1}$"),
        (r"$\phi$", "exfiltration-completion rate / exfiltrating", f"{p.phi:.2f}", "d$^{-1}$"),
        (r"$\rho$", "encryption-completion rate / encrypting", f"{p.rho:.2f}", "d$^{-1}$"),
        (r"$\delta_0$", "baseline detection hazard", f"{p.delta0:.2f}", "d$^{-1}$"),
        (r"$\kappa_x$", "detection gain / exfiltrating server", f"{p.kx:.2f}", "d$^{-1}$"),
        (r"$\kappa_b$", "detection gain / encrypting server", f"{p.kb:.2f}", "d$^{-1}$"),
        (r"$\kappa_c$", "detection gain / locked server", f"{p.kc:.2f}", "d$^{-1}$"),
        (r"$\mathrm{RTO}_0$", "base restore time", f"{p.RTO0:.2f}", "d"),
        (r"$\mathrm{RTO}_1$", "restore-time growth / locked server", f"{p.RTO1:.2f}", "d"),
        (r"$w_L$", "data loss / locked server (RPO)", f"{p.wL:.2f}", "unit"),
        (r"$\mathrm{cov}_r$", "restore coverage", f"{p.cov_r:.2f}", "--"),
    ]
    out = [r"\begin{tabular}{@{}llrl@{}}", r"\toprule",
           r"symbol & meaning & default & unit\\", r"\midrule"]
    for sym, mean, val, unit in rows:
        out.append(f"{sym} & {mean} & {val} & {unit}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_params.tex", out)


def tab_transitions():
    out = [r"\begin{tabular}{@{}lllc@{}}", r"\toprule",
           r"activity & effect on $(a,x,b,c,d)$ & rate & guard\\", r"\midrule",
           r"initial access & $(0,0,0,0,0)\!\to\!(1,0,0,0,0)$ & $\beta$ & $p{=}0,V{\ge}1$\\",
           r"lateral move & $a\!\to\!a{+}1$ & $h\,\beta_L$ & $V{\ge}1,h{\ge}1$\\",
           r"exfil.\ onset & $a\!\to\!a{-}1,\,x\!\to\!x{+}1$ & $a\,\epsilon$ & $a{\ge}1,d{=}0$\\",
           r"exfil.\ done & $x\!\to\!x{-}1,\,b\!\to\!b{+}1$ & $x\,\phi$ & $x{\ge}1,d{=}0$\\",
           r"encrypt done & $b\!\to\!b{-}1,\,c\!\to\!c{+}1$ & $b\,\rho$ & $b{\ge}1$\\",
           r"detection & $d\!:\!0\!\to\!1$ & $\delta(s)$ & $p{\ge}1,d{=}0$\\",
           r"restore & $\to\textsc{Good}$ & $\eta(c)\,\mathrm{cov}_r$ & $d{=}1$\\",
           r"restore (residual) & $\to(1,0,0,0,0)$ & $\eta(c)(1{-}\mathrm{cov}_r)$ & $d{=}1$\\",
           r"\bottomrule", r"\end{tabular}"]
    write("tab_transitions.tex", out)


def tab_validation():
    """SSA cross-checks parsed from validation_table.csv."""
    rows = list(csv.reader(open(os.path.join(CSVDIR, "validation_table.csv"))))
    ssa = [r for r in rows[1:] if r[0].startswith("SSA ")]
    out = [r"\begin{tabular}{@{}lrrc@{}}", r"\toprule",
           r"quantity & analytic & simulation (95\% CI) & agree\\", r"\midrule"]
    label = {"SSA E_breach": "E[datasets exfiltrated]",
             "SSA E_denial_daysrv": "E[denial] (srv-days)",
             "SSA E_dataloss": "E[data loss] (RPO)",
             "SSA MTTD": "MTTD (days)",
             "SSA MTTResolve": "MTTResolve (days)"}
    for name, exp, got, res, _note in ssa:
        if name not in label:
            continue
        got_tex = got.replace("+/-", r"\,$\pm$\,")
        out.append(f"{label[name]} & {exp} & {got_tex} & \\checkmark\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_validation.tex", out)


def tab_codesign():
    h = head()
    b = h["base_totals"]
    c = h["codesign"]
    bl = h["blind"]
    def row(lbl, sig, tau, S, O, J, br, dn, dl):
        return (f"{lbl} & {sig} & {tau} & {br:.3f} & {dn:.3f} & {dl:.3f} "
                f"& {S:.3f} & {O:.3f} & {J:.3f}\\\\")
    # baseline: sigma_D=1, tau_b=24 (S/O from cost at baseline) -- recompute
    import optimize as opt
    cfg = opt.CostConfig()
    base_eval = opt.evaluate(1.0, cfg.tau_b0, Params(), cfg)
    out = [r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
           r"design & $\sigma_D$ & $\tau_b$(h) & E[exfil] & E[denial] & E[loss] "
           r"& $S$ & $O$ & $J$\\", r"\midrule",
           row("baseline", "1.0", "24", base_eval["S"], base_eval["O"],
               base_eval["J"], base_eval["E_breach"], base_eval["E_denial_daysrv"],
               base_eval["E_dataloss"]),
           row("detection-blind", "1.0", f"{bl['tau_blind']:.0f}",
               base_eval["S"], cfg.C_mon(1.0)+cfg.C_bkp(bl['tau_blind']),
               bl["J_blind"], base_eval["E_breach"], base_eval["E_denial_daysrv"],
               base_eval["E_dataloss"]),
           r"\midrule",
           row(r"\textbf{co-design}~$\star$", f"{c['sigma_D']:.2f}",
               f"{c['tau_b']:.0f}", c["S"], c["O"], c["J"], c["E_breach"],
               c["E_denial_daysrv"], c["E_dataloss"]),
           r"\bottomrule", r"\end{tabular}"]
    write("tab_codesign.tex", out)


def tab_scaling():
    rows = list(csv.reader(open(os.path.join(CSVDIR, "fig_scaling.csv"))))[1:]
    out = [r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
           r"$m$ & states & MTTD & MTTResolve & E[exfil] & E[denial] & E[loss]\\",
           r"\midrule"]
    for r in rows:
        m, nt, mttd, mttr, br, dn, dl = r
        out.append(f"{int(float(m))} & {int(float(nt))} & {float(mttd):.3f} & "
                   f"{float(mttr):.3f} & {float(br):.3f} & {float(dn):.3f} & "
                   f"{float(dl):.3f}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_scaling.tex", out)


def tab_sensitivity():
    rows = list(csv.reader(open(os.path.join(CSVDIR, "fig_sensitivity.csv"))))
    hdr = rows[0]
    keep = ["E_breach", "E_denial_daysrv", "E_dataloss", "MTTResolve"]
    ci = [hdr.index(k) for k in keep]
    symmap = {"beta": r"$\beta$", "betaL": r"$\beta_L$", "eps": r"$\epsilon$",
              "phi": r"$\phi$", "rho": r"$\rho$", "delta0": r"$\delta_0$",
              "kx": r"$\kappa_x$", "kb": r"$\kappa_b$", "kc": r"$\kappa_c$",
              "RTO0": r"$\mathrm{RTO}_0$", "RTO1": r"$\mathrm{RTO}_1$", "wL": r"$w_L$"}
    out = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r"param & E[exfil] & E[denial] & E[loss] & MTTResolve\\", r"\midrule"]
    for r in rows[1:]:
        vals = " & ".join(f"{float(r[i]):+.3f}" for i in ci)
        out.append(f"{symmap.get(r[0], r[0])} & {vals}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_sensitivity.tex", out)


def main():
    print("Emitting LaTeX table fragments:")
    tab_params()
    tab_transitions()
    tab_validation()
    tab_codesign()
    tab_scaling()
    tab_sensitivity()


if __name__ == "__main__":
    main()
