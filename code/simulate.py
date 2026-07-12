"""
simulate.py
===========
Independent Monte-Carlo (Gillespie / SSA) simulator of the CRADLE ransomware CTMC.

Purpose (the group's "verification triangle" third leg): the simulator consumes
ONLY the per-state transition list produced by ``CradleModel.transitions`` and
rebuilds, by pure stochastic simulation,

  * transient state probabilities  P(state, t),
  * the detection and resolution CDFs,
  * per-incident expectations (datasets exfiltrated, locked server-days, data
    loss at restore, mean time-to-detection / -resolution),

so that agreement with the analytic (matrix-exponential + fundamental-matrix)
solution validates the state space, the generator, and the reward algebra all at
once.  Every trajectory starts clean (0,0,0,0,0) and evolves until it is absorbed
in ``Good`` (restore complete) or a horizon ``Tmax`` is reached.  Randomness is
seeded for reproducibility.
"""
from __future__ import annotations

from typing import List

import numpy as np

from cradle_model import CradleModel, GOOD


class CradleSimulator:
    def __init__(self, model: CradleModel | None = None):
        self.model = model if model is not None else CradleModel()
        m = self.model
        self.n = m.n
        self.good = m.index[GOOD]
        # precompute per-state transition targets/rates/kinds (indices)
        self.tgt: List[np.ndarray] = [np.empty(0, int)] * self.n
        self.rate: List[np.ndarray] = [np.empty(0, float)] * self.n
        self.kind: List[List[str]] = [[]] * self.n
        self.total = np.zeros(self.n)
        self.cval = np.array([s[3] if s != GOOD else 0 for s in m.all_states], float)
        self.dval = np.array([s[4] if s != GOOD else -1 for s in m.all_states], float)
        for s in m.tstates:
            a = m.index[s]
            trs = m.transitions(s)
            self.tgt[a] = np.array([m.index[t] for t, _r, _k in trs], int)
            self.rate[a] = np.array([r for _t, r, _k in trs], float)
            self.kind[a] = [k for _t, _r, k in trs]
            self.total[a] = self.rate[a].sum()
        self.cum = [np.cumsum(r) for r in self.rate]

    # ------------------------------------------------------------------ #
    def run(self, n_traj: int, sample_times: np.ndarray,
            Tmax: float | None = None, seed: int = 20260712) -> dict:
        rng = np.random.default_rng(seed)
        st = np.asarray(sample_times, float)
        ns = len(st)
        if Tmax is None:
            Tmax = float(st[-1]) * 4.0 + 50.0
        n = self.n
        m = self.model
        wL = m.p.wL

        occ_state = np.zeros((ns, n))        # count in state at sample time
        # per-incident accumulators (mean + sum of squares for SE)
        acc = {k: 0.0 for k in ("breach", "denial", "dataloss", "mttd",
                                "mresp", "mttr", "detected_ever")}
        sq = {k: 0.0 for k in acc}
        det_by = np.zeros(ns)                # count detected by sample time
        res_by = np.zeros(ns)                # count resolved by sample time

        good = self.good
        tgt, cum, total, kind = self.tgt, self.cum, self.total, self.kind
        cval, dval = self.cval, self.dval

        for _ in range(n_traj):
            t = 0.0
            s = m.index[(0, 0, 0, 0, 0)]
            si = 0
            # incident scalars for this trajectory
            n_breach = 0.0
            denial = 0.0                     # integral of #locked over time
            dataloss = 0.0
            t_detect = np.nan
            t_resolve = np.nan
            # sample state at t=0
            while si < ns and st[si] <= 0.0:
                occ_state[si, s] += 1.0
                si += 1
            done = False
            while not done:
                R = total[s]
                if R <= 0.0:                 # absorbed in Good
                    while si < ns:
                        occ_state[si, s] += 1.0
                        si += 1
                    break
                dt = rng.exponential(1.0 / R)
                t2 = t + dt
                # accumulate denial (=#locked * sojourn) over [t, t2)
                denial += cval[s] * dt
                # snapshot sample times in [t, t2)
                while si < ns and st[si] < t2:
                    occ_state[si, s] += 1.0
                    si += 1
                # choose transition
                u = rng.random() * R
                j = min(int(np.searchsorted(cum[s], u, side="right")),
                        len(tgt[s]) - 1)
                nxt = int(tgt[s][j])
                kd = kind[s][j]
                if kd == "exfil_done":
                    n_breach += 1.0
                elif kd == "detect" and np.isnan(t_detect):
                    t_detect = t2
                elif kd in ("restore", "restore_fail"):
                    dataloss += wL * cval[s]         # data lost at restore instant
                    if kd == "restore":
                        t_resolve = t2
                s = nxt
                t = t2
                if s == good:
                    t_resolve = t2 if np.isnan(t_resolve) else t_resolve
                    # fill any remaining samples in absorbing state
                    while si < ns:
                        occ_state[si, s] += 1.0
                        si += 1
                    done = True
                if t > Tmax:
                    done = True
            # tally detection / resolution CDFs
            if not np.isnan(t_detect):
                det_by += (st >= t_detect)
            if not np.isnan(t_resolve):
                res_by += (st >= t_resolve)
            # incident scalars
            mttd = t_detect if not np.isnan(t_detect) else t_resolve
            mresp = (t_resolve - t_detect) if (not np.isnan(t_detect)
                                               and not np.isnan(t_resolve)) else 0.0
            vals = {"breach": n_breach, "denial": denial, "dataloss": dataloss,
                    "mttd": mttd if not np.isnan(mttd) else 0.0,
                    "mresp": mresp,
                    "mttr": t_resolve if not np.isnan(t_resolve) else Tmax,
                    "detected_ever": 1.0 if not np.isnan(t_detect) else 0.0}
            for k, v in vals.items():
                acc[k] += v
                sq[k] += v * v

        N = float(n_traj)
        prob = occ_state / N
        prob_se = np.sqrt(np.clip(prob * (1 - prob), 0, None) / N)
        out = {"t": st, "prob": prob, "prob_se": prob_se,
               "det_cdf": det_by / N, "res_cdf": res_by / N, "n_traj": n_traj}
        totals, totals_se = {}, {}
        for k in acc:
            mean = acc[k] / N
            var = max(sq[k] / N - mean * mean, 0.0)
            totals[k] = mean
            totals_se[k] = np.sqrt(var / N)
        out["totals"] = totals
        out["totals_se"] = totals_se
        return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    model = CradleModel()
    sim = CradleSimulator(model)
    times = np.linspace(0, 20, 21)
    t0 = time.time()
    out = sim.run(n_traj=120_000, sample_times=times, seed=2026)
    dt = time.time() - t0
    print(f"Gillespie SSA: {out['n_traj']:,} trajectories in {dt:.1f}s\n")

    tot = model.incident_totals()
    print("Per-incident expectations   analytic  vs  simulation (95% CI):")
    pairs = [("MTTD", "mttd"), ("MResponse", "mresp"), ("MTTResolve", "mttr"),
             ("E_breach", "breach"), ("E_denial_daysrv", "denial"),
             ("E_dataloss", "dataloss")]
    for an, sm in pairs:
        a = tot[an]
        m_ = out["totals"][sm]
        ci = 1.96 * out["totals_se"][sm]
        ok = "OK" if abs(a - m_) <= ci + 0.02 * abs(a) else "??"
        print(f"  {an:16s} {a:8.4f}   {m_:8.4f} +/- {ci:.4f}   [{ok}]")

    # transient probability spot check
    dcdf = model.detection_cdf(times)
    rcdf = model.resolve_cdf(times)
    print("\n  t   P(detect) an/sim     P(resolve) an/sim")
    for i in (5, 10, 15):
        print(f"  {times[i]:4.1f}  {dcdf[i]:.3f}/{out['det_cdf'][i]:.3f}"
              f"        {rcdf[i]:.3f}/{out['res_cdf'][i]:.3f}")
