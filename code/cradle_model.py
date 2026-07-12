"""
cradle_model.py
===============
CRADLE -- Crypto-Ransomware Attack: Detection-race, Loss and restore-Economics.

A from-scratch continuous-time Markov / stochastic-reward-net (SRN) model of a
crypto-ransomware *double-extortion* kill-chain on a Multi-access Edge Computing
(MEC) micro data center (MEDC) of ``m`` servers, defended by a footprint-aware
Endpoint-Detection-and-Response (EDR) monitor and an immutable-backup
isolate-and-restore response.

This inherits the transient-CTMC methodology of the RAMS-2021 MEDC model
(project 224: lateral-movement scaling, matrix-exponential transient solution,
accumulated-reward via a block matrix exponential, Gillespie cross-validation,
reward-weighted security loss) and extends it in four load-bearing ways that are
untouched by that model and its siblings:

  1. *Encryption as a recoverable service-denial state* (not an absorbing
     breach): a Locked server denies service but is restorable from backup.
  2. *Ordered exfiltrate-then-encrypt double extortion*: a per-server kill chain
     Vulnerable -> Compromised -> Exfiltrating -> Encrypting -> Locked.
  3. *Footprint-aware detection hazard* keyed on the encryption/exfiltration
     I/O footprint (not the compromised-node count), which breaks the uniform
     recovery closed form of the base model.
  4. *Backup-and-restore economics* with explicit recovery-point (RPO, data
     loss) and recovery-time (RTO) semantics, plus imperfect restore coverage.

State encoding
--------------
A transient state is  s = (a, x, b, c, d)  with

    a = # servers Compromised (foothold, staging; stealthy)
    x = # servers Exfiltrating (actively stealing data -> confidentiality loss)
    b = # servers Encrypting   (crypto in progress; noisy I/O)
    c = # servers Locked       (encryption complete -> data denied, DOWN)
    d in {0, 1}                 phase: 0 = stealth/undetected, 1 = response

Clean/Vulnerable servers  V = m - a - x - b - c;   footprint  p = a + x + b + c;
active (non-locked) foothold  h = a + x + b.  Validity: a,x,b,c >= 0, p <= m; the
stealth phase d=0 exists for every lattice point, the response phase d=1 only for
p >= 1 (a campaign must exist to be responded to).  ``GOOD`` (= incident
resolved / MEDC restored) is a single absorbing state.

Transition rules
----------------
Stealth (d = 0), rates per day, with V = m-p and h = a+x+b:
  Initial access   (p=0, V>=1): (0,0,0,0,0) -> (1,0,0,0,0)      beta
  Lateral movement (V>=1,h>=1): a -> a+1                        h * betaL
  Exfil onset      (a>=1)     : a->a-1, x->x+1                  a * eps
  Exfil complete   (x>=1)     : x->x-1, b->b+1  [+1 breach]     x * phi
  Encrypt complete (b>=1)     : b->b-1, c->c+1                  b * rho
  Detection        (p>=1)     : d: 0 -> 1              delta(s) (footprint-aware)

Response (d = 1): network isolation halts lateral movement, exfiltration and new
encryption onset; host-local crypto already running continues to completion
while the immutable-backup restore races it:
  Encrypt complete (b>=1)     : b->b-1, c->c+1                  b * rho
  Restore          (always)   : -> GOOD           w.p. cov_r    eta(c)
                                -> (1,0,0,0,0)    w.p. 1-cov_r  eta(c)  [residual]

Footprint-aware detection  delta(s) = delta0 + ka*a + kx*x + kb*b + kc*c
State-dependent restore     eta(c)   = 1 / (RTO0 + RTO1 * c)   (RTO grows w/ #locked)

Default parameters -- see ``Params``.  Time unit = days throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

State = Tuple[int, int, int, int, int]     # (a, x, b, c, d)
GOOD = "Good"                              # absorbing: incident resolved / restored


# --------------------------------------------------------------------------- #
#  Parameters                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """Rate parameters of the CRADLE ransomware model (per day)."""
    m: int = 4                 # number of MEC servers in the MEDC

    # --- attacker kill chain -------------------------------------------------
    beta: float = 0.50         # initial access (foothold on the first server)
    betaL: float = 0.40        # lateral movement per active foothold
    eps: float = 0.80          # exfiltration onset per compromised server
    phi: float = 1.00          # exfiltration completion (one dataset stolen)
    rho: float = 2.00          # encryption completion per encrypting server

    # --- footprint-aware detection hazard  delta = delta0 + ka*a+kx*x+kb*b+kc*c
    delta0: float = 0.20       # baseline (routine audit) detection rate
    ka: float = 0.00           # per Compromised server (stealthy -> ~0)
    kx: float = 0.50           # per Exfiltrating server (data-egress signature)
    kb: float = 1.50           # per Encrypting server  (high-entropy I/O -> noisy)
    kc: float = 0.80           # per Locked server      (user-visible outage/alarm)

    # --- backup-and-restore response ----------------------------------------
    RTO0: float = 0.25         # base restore time (days) irrespective of #locked
    RTO1: float = 0.15         # extra restore time per Locked server (RTO growth)
    wL: float = 1.00           # data-loss per Locked server at restore (RPO units)
    cov_r: float = 1.00        # restore coverage (prob. restore is complete)

    # ---- convenience -------------------------------------------------------
    def with_(self, **kw) -> "Params":
        """Return a copy with the given fields replaced (one-line perturbation)."""
        return replace(self, **kw)

    def eta(self, c: int) -> float:
        """State-dependent restore-completion rate (1/RTO), RTO grows with #locked."""
        return 1.0 / (self.RTO0 + self.RTO1 * c)


# --------------------------------------------------------------------------- #
#  Model : state enumeration + generator                                       #
# --------------------------------------------------------------------------- #
class CradleModel:
    """CTMC/SRN of a MEDC ransomware kill-chain with detection + backup-restore."""

    def __init__(self, params: Params | None = None):
        self.p = params if params is not None else Params()
        self.tstates: List[State] = self._enumerate_states(self.p.m)
        self.all_states: List = self.tstates + [GOOD]
        self.index: Dict = {s: n for n, s in enumerate(self.all_states)}
        self.nt = len(self.tstates)          # # transient (non-absorbing) states
        self.n = len(self.all_states)        # + 1 absorbing "Good"
        self.Q = self._build_generator()
        # cache reward-vector arrays over transient states
        self._arr = self._state_arrays()

    # ---- state space --------------------------------------------------- #
    @staticmethod
    def _enumerate_states(m: int) -> List[State]:
        """Transient states: d=0 for every lattice point p<=m; d=1 for p>=1."""
        st: List[State] = []
        for d in (0, 1):
            for a in range(m + 1):
                for x in range(m + 1 - a):
                    for b in range(m + 1 - a - x):
                        for c in range(m + 1 - a - x - b):
                            if d == 1 and (a + x + b + c) == 0:
                                continue     # no response without a campaign
                            st.append((a, x, b, c, d))
        return st

    # ---- rate laws ----------------------------------------------------- #
    def delta(self, s: State) -> float:
        """Footprint-aware EDR detection hazard (0 when there is no footprint)."""
        a, x, b, c, _d = s
        if a + x + b + c == 0:
            return 0.0
        p = self.p
        return p.delta0 + p.ka * a + p.kx * x + p.kb * b + p.kc * c

    def transitions(self, s: State) -> List[Tuple[object, float, str]]:
        """Return [(target, rate, kind), ...] out of transient state s."""
        a, x, b, c, d = s
        p = self.p
        m = p.m
        V = m - (a + x + b + c)              # clean/vulnerable servers
        h = a + x + b                        # active (non-locked) foothold
        out: List[Tuple[object, float, str]] = []

        if d == 0:
            # (1) initial access: first foothold on a clean MEDC
            if (a + x + b + c) == 0 and V >= 1:
                out.append(((1, 0, 0, 0, 0), p.beta, "access"))
            # (2) lateral movement (footprint-scaled, epidemic spread)
            if V >= 1 and h >= 1:
                out.append(((a + 1, x, b, c, 0), h * p.betaL, "lateral"))
            # (3) exfiltration onset: a Compromised server begins staging/egress
            if a >= 1:
                out.append(((a - 1, x + 1, b, c, 0), a * p.eps, "exfil_onset"))
            # (4) exfiltration completion -> encryption onset (double extortion)
            if x >= 1:
                out.append(((a, x - 1, b + 1, c, 0), x * p.phi, "exfil_done"))
            # (5) encryption completion: an Encrypting server becomes Locked
            if b >= 1:
                out.append(((a, x, b - 1, c + 1, 0), b * p.rho, "encrypt_done"))
            # (6) detection: campaign discovered -> response phase
            if (a + x + b + c) >= 1:
                out.append(((a, x, b, c, 1), self.delta(s), "detect"))
        else:  # d == 1 : response (isolate + restore); onsets & lateral halted
            # (5') in-flight host-local encryption still completes (races restore)
            if b >= 1:
                out.append(((a, x, b - 1, c + 1, 1), b * p.rho, "encrypt_done"))
            # (7) restore from immutable backup
            rate = p.eta(c)
            if p.cov_r >= 1.0:
                out.append((GOOD, rate, "restore"))
            else:
                out.append((GOOD, rate * p.cov_r, "restore"))
                out.append(((1, 0, 0, 0, 0), rate * (1.0 - p.cov_r), "restore_fail"))
        return out

    def _build_generator(self) -> np.ndarray:
        """Infinitesimal generator Q (row = from, col = to). dpi/dt = pi Q."""
        n = self.n
        Q = np.zeros((n, n))
        for s in self.tstates:
            a = self.index[s]
            for tgt, rate, _k in self.transitions(s):
                b = self.index[tgt]
                Q[a, b] += rate
                Q[a, a] -= rate
        return Q                              # GOOD row already all zeros

    # ---- per-state arrays (reward building blocks) --------------------- #
    def _state_arrays(self) -> Dict[str, np.ndarray]:
        A = np.array([s[0] for s in self.tstates], float)
        X = np.array([s[1] for s in self.tstates], float)
        B = np.array([s[2] for s in self.tstates], float)
        C = np.array([s[3] for s in self.tstates], float)
        D = np.array([s[4] for s in self.tstates], float)
        eta = np.array([self.p.eta(int(c)) for c in C], float)
        return {"a": A, "x": X, "b": B, "c": C, "d": D,
                "footprint": A + X + B + C, "active": A + X + B, "eta": eta}

    def reward_vectors(self) -> Dict[str, np.ndarray]:
        """Reward-rate vectors over the transient states (length ``nt``)."""
        r = dict(self._arr)
        d0 = (r["d"] == 0)
        r["breach_rate"] = self.p.phi * r["x"] * d0     # datasets stolen / day
        r["response"] = (r["d"] == 1).astype(float)     # response-phase indicator
        r["stealth"] = (r["d"] == 0).astype(float)
        # flux reward collected at restore: expected data-loss = sum eta*wL*c*tau
        r["dataloss_flux"] = np.where(r["d"] == 1, r["eta"] * self.p.wL * r["c"], 0.0)
        return r

    # ---- initial distribution ------------------------------------------ #
    def initial_distribution(self, s0: State = (0, 0, 0, 0, 0)) -> np.ndarray:
        pi0 = np.zeros(self.n)
        pi0[self.index[s0]] = 1.0
        return pi0

    # ---- transient solution -------------------------------------------- #
    @staticmethod
    def _uniform(times: np.ndarray) -> bool:
        if len(times) < 2:
            return False
        dt = np.diff(times)
        return bool(np.all(dt > 0) and np.allclose(dt, dt[0]))

    def transient(self, times: Sequence[float], s0: State = (0, 0, 0, 0, 0)
                  ) -> np.ndarray:
        """
        Transient state probabilities pi(t) for every t in ``times``.
        pi(t) = pi0 . expm(Q t).  For an equally-spaced grid the base-step
        propagator P = expm(Q dt) is formed once and composed (fast, exact).
        Returns array of shape (len(times), n).
        """
        times = np.asarray(times, float)
        pi0 = self.initial_distribution(s0)
        out = np.empty((len(times), self.n))
        if self._uniform(times):
            dt = times[1] - times[0]
            Pdt = expm(self.Q * dt)
            cur = pi0 @ expm(self.Q * times[0]) if times[0] > 0 else pi0.copy()
            out[0] = cur
            for k in range(1, len(times)):
                cur = cur @ Pdt
                out[k] = cur
        else:
            for r, t in enumerate(times):
                out[r] = pi0 @ expm(self.Q * t)
        return out

    def accumulated(self, times: Sequence[float], s0: State = (0, 0, 0, 0, 0)
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Joint solve of pi(t) and the accumulated sojourn A(t)=int_0^t pi(u) du,
        via the augmented linear ODE  d/dt [pi;A] = M [pi;A],
        M = [[Q^T, 0],[I, 0]]  (2n x 2n).  Machine-accurate (no ODE step error).
        """
        times = np.asarray(times, float)
        n = self.n
        M = np.zeros((2 * n, 2 * n))
        M[:n, :n] = self.Q.T
        M[n:, :n] = np.eye(n)
        y0 = np.concatenate([self.initial_distribution(s0), np.zeros(n)])
        pis = np.empty((len(times), n))
        accs = np.empty((len(times), n))
        if self._uniform(times):
            dt = times[1] - times[0]
            Pdt = expm(M * dt)
            cur = (expm(M * times[0]) @ y0) if times[0] > 0 else y0.copy()
            pis[0], accs[0] = cur[:n], cur[n:]
            for k in range(1, len(times)):
                cur = Pdt @ cur
                pis[k], accs[k] = cur[:n], cur[n:]
        else:
            for r, t in enumerate(times):
                y = expm(M * t) @ y0
                pis[r], accs[r] = y[:n], y[n:]
        return pis, accs

    def transient_uniformization(self, times: Sequence[float],
                                 s0: State = (0, 0, 0, 0, 0),
                                 tol: float = 1e-12) -> np.ndarray:
        """
        Independent transient engine (Jensen's uniformization) used to
        cross-check ``transient``.  P = I + Q/Lambda, pi(t)=sum_k Poisson(Lambda t;k) pi0 P^k.
        """
        times = np.asarray(times, float)
        Lam = float(np.max(-np.diag(self.Q))) * 1.02 + 1e-9
        P = np.eye(self.n) + self.Q / Lam
        pi0 = self.initial_distribution(s0)
        out = np.empty((len(times), self.n))
        for r, t in enumerate(times):
            lt = Lam * t
            # right-truncate the Poisson series
            kmax = int(lt + 10 * np.sqrt(lt) + 20)
            vec = pi0.copy()
            pk = np.exp(-lt)
            acc = pk * vec
            for k in range(1, kmax + 1):
                vec = vec @ P
                pk *= lt / k
                acc = acc + pk * vec
                if pk < tol and k > lt:
                    break
            out[r] = acc
        return out

    # ---- fundamental matrix / incident totals -------------------------- #
    def sojourn_times(self, s0: State = (0, 0, 0, 0, 0)) -> np.ndarray:
        """
        Expected time spent in each transient state before absorption in GOOD:
        tau = pi0_T . (-U)^{-1}, where U = Q restricted to transient states.
        (Fundamental-matrix idiom; every incident total is tau . reward.)
        """
        U = self.Q[:self.nt, :self.nt]
        e0 = self.initial_distribution(s0)[:self.nt]
        # solve tau (U^T) = -e0   <=>   (-U^T) tau = e0
        tau = np.linalg.solve(-U.T, e0)
        return tau

    def incident_totals(self, s0: State = (0, 0, 0, 0, 0)) -> Dict[str, float]:
        """Expected per-incident quantities (integrated to resolution)."""
        tau = self.sojourn_times(s0)
        r = self.reward_vectors()
        MTTD = float(tau @ r["stealth"])                 # mean time to detection
        MResp = float(tau @ r["response"])               # mean response duration
        MTTR = MTTD + MResp                               # mean time to resolve
        return {
            "MTTD": MTTD,
            "MResponse": MResp,
            "MTTResolve": MTTR,
            "E_breach": float(tau @ r["breach_rate"]),    # datasets exfiltrated
            "E_denial_daysrv": float(tau @ r["c"]),       # locked server-days
            "E_encrypt_daysrv": float(tau @ r["b"]),      # encrypting server-days
            "E_exposure_daysrv": float(tau @ r["active"]),# foothold server-days
            "E_dataloss": float(tau @ r["dataloss_flux"]),# data lost at restore (RPO)
            "E_maxlocked_proxy": float(tau @ r["c"]) / max(MTTR, 1e-12),
        }

    # ---- first-passage distributions ----------------------------------- #
    def detection_cdf(self, times: Sequence[float],
                      s0: State = (0, 0, 0, 0, 0)) -> np.ndarray:
        """P(detected by t) = 1 - P(still in a stealth state at t)."""
        pis = self.transient(times, s0)
        stealth_idx = [self.index[s] for s in self.tstates if s[4] == 0]
        return 1.0 - pis[:, stealth_idx].sum(axis=1)

    def resolve_cdf(self, times: Sequence[float],
                    s0: State = (0, 0, 0, 0, 0)) -> np.ndarray:
        """P(incident resolved by t) = pi_GOOD(t) (restore-probability curve)."""
        return self.transient(times, s0)[:, self.index[GOOD]]

    # ---- aggregate marginals ------------------------------------------- #
    def aggregate(self, pis: np.ndarray, field: str, thresh: int, mode="ge"
                  ) -> np.ndarray:
        """P(#servers-in-`field` `mode` `thresh`) over time, from pis."""
        j = {"a": 0, "x": 1, "b": 2, "c": 3}[field]
        idx = [self.index[s] for s in self.tstates
               if (s[j] >= thresh if mode == "ge" else s[j] == thresh)]
        return pis[:, idx].sum(axis=1)

    def expected_count(self, pis: np.ndarray, field: str) -> np.ndarray:
        """E[#servers-in-`field`](t)."""
        j = {"a": 0, "x": 1, "b": 2, "c": 3}[field]
        w = np.zeros(self.n)
        for s in self.tstates:
            w[self.index[s]] = s[j]
        return pis @ w

    def prob(self, pis: np.ndarray, s: object) -> np.ndarray:
        return pis[:, self.index[s]]


# --------------------------------------------------------------------------- #
#  Golden-number self-test                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = CradleModel()
    p = model.p
    print(f"CRADLE model: m={p.m}, transient states={model.nt}, total={model.n}")

    # structural sanity
    assert np.allclose(model.Q.sum(axis=1), 0.0, atol=1e-12), "Q rows must sum to 0"
    t = np.linspace(0, 20, 2001)
    pis = model.transient(t)
    assert abs(pis[0].sum() - 1.0) < 1e-12 and abs(pis[-1].sum() - 1.0) < 1e-9
    # cross-check expm vs uniformization
    pu = model.transient_uniformization(np.array([1.0, 5.0, 10.0]))
    pe = model.transient(np.array([1.0, 5.0, 10.0]))
    print(f"expm vs uniformization max|dP| = {np.abs(pu - pe).max():.2e}")

    tot = model.incident_totals()
    print("\nPer-incident expectations (base case, m=4):")
    for k, v in tot.items():
        print(f"  {k:20s} = {v:.4f}")

    # detection / resolution
    tt = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    dcdf = model.detection_cdf(tt)
    rcdf = model.resolve_cdf(tt)
    print("\n  t   P(detected)  P(resolved)")
    for i, tv in enumerate(tt):
        print(f"  {tv:4.1f}   {dcdf[i]:.4f}      {rcdf[i]:.4f}")

    # headline: recovery curve is NOT 1 - e^{-gamma t} (contrast with base 224)
    gamma_eff = 1.0 / tot["MTTResolve"]
    approx = 1.0 - np.exp(-gamma_eff * tt)
    print(f"\n  max |P(resolved) - (1-e^(-t/MTTR))| = {np.abs(rcdf - approx).max():.3f}"
          f"   (non-exponential => footprint-aware detection matters)")
