"""
optimize.py
===========
Defender co-design and sensitivity analysis for the CRADLE model.

The defender holds two orthogonal investment levers:

  * detection sensitivity  sigma_D >= 1 -- an EDR-tuning multiplier that scales
    the footprint-aware detection coefficients (kx, kb, kc).  Sharper detection
    shortens the stealth window (less exfiltration, fewer locked servers, less
    data loss) but costs more to run (false-positive triage) -> C_mon(sigma_D).

  * backup cadence  tau_b (hours) -- the immutable-backup interval that sets the
    recovery point.  Data loss per Locked server scales with tau_b (RPO), while
    the backup/storage overhead scales with the backup frequency 1/tau_b
    -> C_backup(tau_b).

The per-incident expected total cost is

    J(sigma_D, tau_b) = S(sigma_D, tau_b) + O(sigma_D, tau_b),
    S = c_br E[breach] + c_dn E[denial] + c_dl E[dataloss] + c_rt E[response],
    O = C_mon(sigma_D) + C_backup(tau_b).

``S`` is the incident harm (security-dependability loss); ``O`` is the defender's
operational spend.  We (i) map the whole J surface, (ii) locate the joint
optimum, (iii) trace the security-vs-operational-cost Pareto frontier, (iv) rank
parameters by scaled (elasticity) sensitivity, and (v) quantify the penalty of a
detection-blind backup schedule.  Each evaluation is one exact fundamental-matrix
solve, so the sweeps are cheap.
"""
from __future__ import annotations

import os
# tiny (~140x140) dense solves: one BLAS thread is fastest and avoids the
# catastrophic oversubscription of thousands of sequential parameter sweeps.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from cradle_model import CradleModel, Params


# --------------------------------------------------------------------------- #
#  Cost configuration                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostConfig:
    # security-loss weights (per unit)
    c_br: float = 1.0        # per dataset exfiltrated (confidentiality)
    c_dn: float = 1.0        # per Locked server-day (service denial)
    c_dl: float = 1.0        # per RPO data-loss unit (integrity/recency)
    c_rt: float = 0.5        # per response(restore)-day
    # operational-cost weights
    c_mon: float = 0.30      # monitoring cost per unit detection sensitivity
    mon_exp: float = 1.0     # convexity of monitoring cost
    c_bkp: float = 0.40      # backup/storage cost at the baseline daily cadence
    tau_b0: float = 24.0     # baseline backup interval (h) giving wL = 1.0
    # lever bounds
    sig_lo: float = 1.0
    sig_hi: float = 8.0
    tau_lo: float = 1.0      # 1 h (near-continuous data protection)
    tau_hi: float = 72.0     # 3 days (loose RPO)

    def C_mon(self, sigma_D: float) -> float:
        return self.c_mon * (sigma_D ** self.mon_exp)

    def C_bkp(self, tau_b: float) -> float:
        # storage/compute overhead proportional to backup frequency
        return self.c_bkp * (self.tau_b0 / tau_b)


def params_for(base: Params, sigma_D: float, tau_b: float,
               cfg: CostConfig) -> Params:
    """Map (sigma_D, tau_b) to a concrete Params (scale detection, set RPO)."""
    wL = base.wL * (tau_b / cfg.tau_b0)      # data accrued since last backup ~ RPO
    return base.with_(kx=base.kx * sigma_D, kb=base.kb * sigma_D,
                      kc=base.kc * sigma_D, wL=wL)


# --------------------------------------------------------------------------- #
#  Single-point evaluation                                                     #
# --------------------------------------------------------------------------- #
def evaluate(sigma_D: float, tau_b: float, base: Params, cfg: CostConfig) -> Dict:
    p = params_for(base, sigma_D, tau_b, cfg)
    tot = CradleModel(p).incident_totals()
    S = (cfg.c_br * tot["E_breach"] + cfg.c_dn * tot["E_denial_daysrv"]
         + cfg.c_dl * tot["E_dataloss"] + cfg.c_rt * tot["MResponse"])
    O = cfg.C_mon(sigma_D) + cfg.C_bkp(tau_b)
    out = {"sigma_D": sigma_D, "tau_b": tau_b, "S": S, "O": O, "J": S + O}
    out.update(tot)
    return out


# --------------------------------------------------------------------------- #
#  1-D golden-section (echoes AEGIS/503 cost-optimal interval synthesis)       #
# --------------------------------------------------------------------------- #
def golden_min(f, lo: float, hi: float, tol: float = 1e-4, itmax: int = 100
               ) -> Tuple[float, float]:
    gr = (np.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(itmax):
        if abs(hi - lo) < tol:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    xm = 0.5 * (lo + hi)
    return xm, f(xm)


def optimal_tau(sigma_D: float, base: Params, cfg: CostConfig) -> Tuple[float, float]:
    """Cost-optimal backup interval at a fixed detection sensitivity."""
    return golden_min(lambda tb: evaluate(sigma_D, tb, base, cfg)["J"],
                      cfg.tau_lo, cfg.tau_hi)


def optimal_sigma(tau_b: float, base: Params, cfg: CostConfig) -> Tuple[float, float]:
    """Cost-optimal detection sensitivity at a fixed backup interval."""
    return golden_min(lambda sg: evaluate(sg, tau_b, base, cfg)["J"],
                      cfg.sig_lo, cfg.sig_hi)


# --------------------------------------------------------------------------- #
#  Joint co-design: grid map + coordinate-descent polish                       #
# --------------------------------------------------------------------------- #
def grid_surface(base: Params, cfg: CostConfig, nsig: int = 25, ntau: int = 25
                 ) -> Dict:
    sig = np.linspace(cfg.sig_lo, cfg.sig_hi, nsig)
    tau = np.linspace(cfg.tau_lo, cfg.tau_hi, ntau)
    J = np.empty((ntau, nsig))
    S = np.empty_like(J)
    O = np.empty_like(J)
    for i, tb in enumerate(tau):
        for k, sg in enumerate(sig):
            e = evaluate(sg, tb, base, cfg)
            J[i, k], S[i, k], O[i, k] = e["J"], e["S"], e["O"]
    ii, kk = np.unravel_index(np.argmin(J), J.shape)
    return {"sigma": sig, "tau": tau, "J": J, "S": S, "O": O,
            "argmin": (sig[kk], tau[ii]), "Jmin_grid": J[ii, kk]}


def optimize_joint(base: Params, cfg: CostConfig, iters: int = 8,
                   surf: Dict | None = None) -> Dict:
    """Coordinate descent (alternating golden-section) from the grid minimiser."""
    if surf is None:
        surf = grid_surface(base, cfg)
    sg, tb = surf["argmin"]
    for _ in range(iters):
        tb, _ = optimal_tau(sg, base, cfg)
        sg, _ = optimal_sigma(tb, base, cfg)
    star = evaluate(sg, tb, base, cfg)
    return {"star": star, "surface": surf}


# --------------------------------------------------------------------------- #
#  Pareto frontier  (security loss S  vs  operational spend O)                 #
# --------------------------------------------------------------------------- #
def pareto_frontier(base: Params, cfg: CostConfig, n: int = 900) -> Dict:
    surf = grid_surface(base, cfg, nsig=30, ntau=30)
    pts = np.column_stack([surf["O"].ravel(), surf["S"].ravel()])
    sig = np.repeat(surf["sigma"][None, :], len(surf["tau"]), axis=0).ravel()
    tau = np.repeat(surf["tau"][:, None], len(surf["sigma"]), axis=1).ravel()
    order = np.argsort(pts[:, 0])
    pts, sig, tau = pts[order], sig[order], tau[order]
    frontier, best = [], np.inf
    for (O, S), sg, tb in zip(pts, sig, tau):
        if S < best - 1e-12:
            best = S
            frontier.append((O, S, sg, tb))
    fr = np.array([(O, S) for O, S, _s, _t in frontier])
    return {"frontier": fr,
            "levers": np.array([(sg, tb) for _o, _s, sg, tb in frontier]),
            "all_O": pts[:, 0], "all_S": pts[:, 1]}


# --------------------------------------------------------------------------- #
#  Scaled (elasticity) sensitivity  SS = (theta/M) dM/dtheta   (echoes 190)    #
# --------------------------------------------------------------------------- #
SENS_PARAMS = ["beta", "betaL", "eps", "phi", "rho",
               "delta0", "kx", "kb", "kc", "RTO0", "RTO1", "wL"]
SENS_METRICS = ["E_breach", "E_denial_daysrv", "E_dataloss", "MTTD", "MTTResolve"]


def scaled_sensitivity(base: Params, rel: float = 1e-3) -> Dict[str, Dict[str, float]]:
    base_tot = CradleModel(base).incident_totals()
    out: Dict[str, Dict[str, float]] = {mm: {} for mm in SENS_METRICS}
    for prm in SENS_PARAMS:
        theta = getattr(base, prm)
        hp = CradleModel(base.with_(**{prm: theta * (1 + rel)})).incident_totals()
        hm = CradleModel(base.with_(**{prm: theta * (1 - rel)})).incident_totals()
        for mm in SENS_METRICS:
            dMdt = (hp[mm] - hm[mm]) / (2 * rel * theta)
            M0 = base_tot[mm]
            out[mm][prm] = (theta / M0) * dMdt if abs(M0) > 1e-15 else 0.0
    return out


# --------------------------------------------------------------------------- #
#  Cost of a detection-blind backup schedule                                   #
# --------------------------------------------------------------------------- #
def cost_of_detection_blind(base: Params, cfg: CostConfig) -> Dict:
    """
    A detection-blind operator tunes the backup interval to minimise only the
    (data-loss + backup) sub-cost at the *baseline* detection sensitivity,
    ignoring that sharper detection would reduce the number of Locked servers.
    We compare its realised total J against the co-designed optimum.
    """
    def blind_subcost(tb):
        e = evaluate(cfg.sig_lo, tb, base, cfg)
        return cfg.c_dl * e["E_dataloss"] + cfg.C_bkp(tb)
    tb_blind, _ = golden_min(blind_subcost, cfg.tau_lo, cfg.tau_hi)
    blind = evaluate(cfg.sig_lo, tb_blind, base, cfg)
    co = optimize_joint(base, cfg)["star"]
    penalty = (blind["J"] - co["J"]) / co["J"]
    return {"blind": blind, "codesign": co, "penalty_frac": penalty,
            "tau_blind": tb_blind}


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    base = Params()
    cfg = CostConfig()

    print("=== Joint co-design optimum ===")
    res = optimize_joint(base, cfg)
    st = res["star"]
    print(f"  sigma_D* = {st['sigma_D']:.3f}   tau_b* = {st['tau_b']:.2f} h")
    print(f"  J* = {st['J']:.4f}   (S = {st['S']:.4f},  O = {st['O']:.4f})")
    print(f"  MTTD* = {st['MTTD']:.3f} d   E_breach* = {st['E_breach']:.3f}"
          f"   E_dataloss* = {st['E_dataloss']:.3f}")

    print("\n=== Detection-blind backup schedule penalty ===")
    bl = cost_of_detection_blind(base, cfg)
    print(f"  blind tau_b = {bl['tau_blind']:.2f} h,  J_blind = {bl['blind']['J']:.4f}")
    print(f"  co-design    J* = {bl['codesign']['J']:.4f}")
    print(f"  penalty of ignoring detection coupling = {100*bl['penalty_frac']:.1f}%")

    print("\n=== Top scaled sensitivities on E[dataloss] ===")
    ss = scaled_sensitivity(base)
    ranked = sorted(ss["E_dataloss"].items(), key=lambda kv: -abs(kv[1]))
    for prm, val in ranked[:6]:
        print(f"  {prm:8s}  SS = {val:+.3f}")
