"""
pytest sanity + reproduction tests for the CRADLE ransomware model.
Run:  cd code && python3 -m pytest -q
"""
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cradle_model import CradleModel, Params, GOOD          # noqa: E402
from simulate import CradleSimulator                          # noqa: E402
import optimize as opt                                        # noqa: E402


def test_state_space_count():
    for m in (2, 3, 4, 5):
        mm = CradleModel(Params(m=m))
        expect = comb(m + 4, 4) + (comb(m + 4, 4) - 1)   # d=0 lattice + d=1 (minus empty)
        assert mm.nt == expect
        assert mm.n == expect + 1
        for (a, x, b, c, d) in mm.tstates:
            assert a >= 0 and x >= 0 and b >= 0 and c >= 0
            assert a + x + b + c <= m
            assert d in (0, 1)
            if d == 1:
                assert a + x + b + c >= 1


def test_generator_conservative_and_absorbing():
    m = CradleModel()
    assert np.allclose(m.Q.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(m.Q[m.index[GOOD]], 0.0)          # Good is absorbing


def test_probabilities_sum_to_one():
    m = CradleModel()
    for t in (0.0, 1.0, 5.0, 20.0):
        assert abs(m.transient(np.array([t]))[0].sum() - 1.0) < 1e-9


def test_expm_vs_uniformization():
    m = CradleModel()
    t = np.array([0.5, 2.0, 8.0])
    assert np.max(np.abs(m.transient(t) - m.transient_uniformization(t))) < 1e-9


def test_fundamental_matrix_matches_integral():
    m = CradleModel()
    tot = m.incident_totals()
    tbig = np.linspace(0, 300, 3001)
    _, accs = m.accumulated(tbig)
    rv = m.reward_vectors()
    A_inf = accs[-1, :m.nt]
    assert abs(float(A_inf @ rv["c"]) - tot["E_denial_daysrv"]) < 1e-4
    assert abs(float(A_inf @ rv["breach_rate"]) - tot["E_breach"]) < 1e-4


def test_reward_identity():
    tot = CradleModel().incident_totals()
    assert abs(tot["MTTResolve"] - (tot["MTTD"] + tot["MResponse"])) < 1e-9


def test_kill_chain_ordering():
    """Encryption is reachable only after exfiltration (double extortion)."""
    m = CradleModel()
    # from a purely-compromised state the only forward moves are lateral or
    # exfil-onset -- never a direct jump to an encrypting server.
    outs = dict((t, k) for t, r, k in m.transitions((1, 0, 0, 0, 0)))
    assert all(k != "encrypt_done" for k in outs.values())
    # exfil-onset then exfil-done reaches an encrypting server
    assert (2, 0, 0, 0, 0) in outs or (1, 1, 0, 0, 0) in outs  # lateral or onset
    assert m.transitions((1, 1, 0, 0, 0))  # exfiltrating -> encrypting exists
    kinds = [k for _t, _r, k in m.transitions((1, 1, 0, 0, 0))]
    assert "exfil_done" in kinds


def test_detection_footprint_aware():
    m = CradleModel()
    assert m.delta((0, 0, 0, 0, 0)) == 0.0                # nothing to detect
    # more footprint -> higher hazard; encrypting is noisier than compromised
    assert m.delta((1, 0, 0, 0, 0)) < m.delta((0, 0, 1, 0, 0))
    assert m.delta((0, 0, 1, 0, 0)) > m.delta((0, 1, 0, 0, 0))  # kb > kx


def test_restore_state_dependent_rto():
    p = Params()
    assert p.eta(0) > p.eta(1) > p.eta(3)                 # RTO grows with #locked


def test_sharper_detection_lowers_loss():
    p = Params()
    base = CradleModel(p).incident_totals()
    sharp = CradleModel(p.with_(kx=p.kx*2, kb=p.kb*2, kc=p.kc*2)).incident_totals()
    assert sharp["E_breach"] < base["E_breach"]
    assert sharp["E_denial_daysrv"] < base["E_denial_daysrv"]
    assert sharp["E_dataloss"] < base["E_dataloss"]


def test_larger_rpo_increases_dataloss():
    p = Params()
    lo = CradleModel(p.with_(wL=0.5)).incident_totals()["E_dataloss"]
    hi = CradleModel(p.with_(wL=2.0)).incident_totals()["E_dataloss"]
    assert hi > lo


def test_resolution_is_non_exponential():
    m = CradleModel()
    tot = m.incident_totals()
    tg = np.linspace(0, 15, 61)
    rc = m.resolve_cdf(tg)
    approx = 1 - np.exp(-tg / tot["MTTResolve"])
    assert np.max(np.abs(rc - approx)) > 0.05


def test_imperfect_restore_absorbs_and_costs_more():
    p = Params(cov_r=0.7)
    m = CradleModel(p)
    # still absorbs w.p. 1 -> probability mass in Good -> 1
    pg = m.transient(np.array([200.0]))[0, m.index[GOOD]]
    assert pg > 1 - 1e-6
    perfect = CradleModel(Params()).incident_totals()["MTTResolve"]
    assert m.incident_totals()["MTTResolve"] > perfect     # residual reinfection


def test_codesign_interior_optimum():
    base, cfg = Params(), opt.CostConfig()
    star = opt.optimize_joint(base, cfg)["star"]
    assert cfg.sig_lo < star["sigma_D"] < cfg.sig_hi
    assert cfg.tau_lo < star["tau_b"] < cfg.tau_hi
    # co-design beats the baseline design
    base_J = opt.evaluate(1.0, cfg.tau_b0, base, cfg)["J"]
    assert star["J"] < base_J


def test_simulation_agrees_with_analytic():
    m = CradleModel()
    sim = CradleSimulator(m)
    ts = np.linspace(0, 20, 21)
    S = sim.run(n_traj=60_000, sample_times=ts, seed=11)
    tot = m.incident_totals()
    for an, sm in [("E_breach", "breach"), ("MTTResolve", "mttr"), ("MTTD", "mttd")]:
        a = tot[an]; msim = S["totals"][sm]; se = S["totals_se"][sm]
        assert abs(a - msim) <= max(4 * se, 0.02 * abs(a))
