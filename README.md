# CRADLE — Transient Security–Dependability Analysis of Crypto-Ransomware Kill-Chains in MEC Micro-Datacenters

*Crypto-Ransomware Attack: **D**etection-race, **L**oss and restore-**E**conomics.*

A from-scratch continuous-time Markov / **stochastic reward net (SRN)** model of a
crypto-ransomware **double-extortion kill-chain** on a Multi-access Edge Computing
(MEC) micro-datacenter (MEDC), defended by a **footprint-aware EDR monitor** and an
**immutable-backup isolate-and-restore** response with explicit **RPO/RTO** data-loss
economics. Everything — model, solvers, simulator, optimisation, figures, tables and
the paper's in-text numbers — is built here in Python and LaTeX and regenerates with
one command.

The model inherits the transient-CTMC methodology of the RAMS-2021 MEDC-under-attack
model (lateral-movement footprint scaling, matrix-exponential transient solution,
accumulated-reward security loss, Gillespie cross-validation) and extends it into a
region untouched by that model and its siblings.

## The idea

A ransomware incident on an MEDC is **survivable**: locked servers are restored from
backup, so the object of study is the *transient* loss of confidentiality, service and
data recency accrued while the attacker's kill-chain
(**Vulnerable → Compromised → Exfiltrating → Encrypting → Locked**) races the
defender's **detect-and-restore** response. Two novelties break prior models:

1. **Encryption is a recoverable service-denial state**, not an absorbing breach.
2. **Detection is footprint-aware** — the EDR hazard grows with the *encryption /
   exfiltration I/O footprint* (`δ(s)=δ₀+κ_a a+κ_x x+κ_b b+κ_c c`), which we **prove** makes
   the resolution law non-memoryless (a departure from uniform-recovery models).
3. **Restore has RPO/RTO economics** — data loss per locked server scales with the
   backup interval; restore time grows with the number of locked servers; restore
   coverage may be imperfect (relapse/re-infection).
4. A **defender co-design** jointly tunes detection sensitivity `σ_D` and backup
   cadence `τ_b` to minimise expected total cost, tracing a security-vs-spend Pareto
   frontier.

## Headline findings

1. **Non-memoryless resolution (Lemma 1).** Because absorption requires *detection then
   restore*, the resolution CDF is S-shaped with zero initial density and departs from
   the best-fit memoryless exponential by **0.21** — a uniform-recovery model cannot
   represent this.
2. **Detection and backups are economic substitutes.** The co-designed optimum
   (`σ_D*≈2.2`, `τ_b*≈32 h`) *relaxes* the backup interval relative to a detection-blind
   operator (25 h): **sharper detection lets you back up less often**, cutting expected
   service denial by **62 %** and expected data loss by **23 %**. Ignoring the coupling
   costs **14 %**.
   Transient service view: availability dips to **98.0 %** at day 3.4 and the probability
   of service denial peaks at **7.5 %** near day 3.2 before restore returns full service.
3. **Per-incident harm saturates with MEDC size** (`m=2…6`): detection resolves the
   incident before the campaign traverses the fleet, so detection latency and backup
   recency — not fleet size — govern ransomware risk.
4. **Restore quality is first-order.** At 70 % restore coverage, relapse inflates the
   mean time-to-resolution by **21 %** and data loss by **43 %**.
5. **Verification triangle passes.** matrix-exponential vs uniformization (~1e-12),
   fundamental-matrix totals vs integrated reward (~1e-15), and an independent Gillespie
   simulation of 200k trajectories (all within 95 % CI). 17/17 checks, 15/15 tests.

## Layout

```
273-CRADLE-.../
  README.md                requirements.txt
  code/
    cradle_model.py        # SRN/CTMC: states, generator, transient (expm),
                           #   accumulated reward (Van Loan block), fundamental
                           #   matrix incident totals, detection/resolution CDFs
    simulate.py            # independent Gillespie/SSA validator (rewards + CDFs + CIs)
    optimize.py            # defender co-design: cost model, grid + golden-section,
                           #   Pareto frontier, scaled sensitivity, detection-blind penalty
    verify.py              # verification triangle -> PASS/FAIL + validation_report.md
    experiments.py         # all result CSVs + results/csv/headline.json
    make_figures.py        # 12 figures (PNG+PDF) from the CSVs
    gen_tables.py          # LaTeX booktabs fragments -> paper/tables/*.tex
    emit_macros.py         # in-text result numbers -> paper/macros.tex
    plotstyle.py           # shared monochrome Times New Roman style
    run_all.py             # verify + experiments + figures + tables + macros + pytest
    tests/test_model.py    # pytest sanity + reproduction suite
  results/
    figures/*.{png,pdf}    # generated figures
    csv/*.csv              # every plotted curve + headline.json + validation_table.csv
    validation_report.md   # generated PASS/FAIL report
  paper/
    main.tex               # IEEEtran journal manuscript (10 pp, 14 figs, 5 tables)
    macros.tex             # auto-generated in-text numbers (do not edit)
    tables/*.tex           # auto-generated booktabs table bodies
```

## Reproduce

```bash
pip install -r requirements.txt
cd code
python3 run_all.py                 # everything: verify + experiments + figures + tables + macros + tests
# or individually:
python3 cradle_model.py            # model self-test (state count, engines, incident totals)
python3 simulate.py                # Gillespie SSA vs analytic
python3 verify.py                  # verification triangle -> must print "ALL CHECKS PASSED (17/17)"
python3 experiments.py             # results/csv/*.csv + headline.json
python3 make_figures.py            # results/figures/*.{png,pdf}
python3 -m pytest tests/ -q        # unit + reproduction tests

# build the paper
cd ../paper && pdflatex main.tex && pdflatex main.tex
```

Time unit is **days** throughout; all randomness in `simulate.py` is seeded.

## Reproduction / validation summary

| route | quantity | agreement |
|---|---|---|
| matrix-exponential vs Jensen uniformization | transient `π(t)` | ~`1e-12` |
| fundamental matrix vs integrated accumulated reward | `E[denial]`, `E[breach]` | ~`1e-15` |
| analytic vs Gillespie SSA (200k) | all per-incident totals | within 95 % CI |
| analytic vs Gillespie SSA (200k) | detection & resolution CDFs | max `|ΔP| < 0.01` |
| structural | `MTTResolve = MTTD + MResponse` | exact |
| structural | resolution curve non-exponential | gap `0.21` (Lemma 1) |

Every number in the paper traces to `results/csv/`; every figure regenerates from those
CSVs; every LaTeX table and in-text number is machine-emitted into `paper/tables/` and
`paper/macros.tex`; and an independent simulator cross-validates the analytic engine.
