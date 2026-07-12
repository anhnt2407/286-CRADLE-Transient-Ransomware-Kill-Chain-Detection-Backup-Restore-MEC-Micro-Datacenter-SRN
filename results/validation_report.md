# CRADLE -- validation report

Base configuration: m=4, 139 transient states + 1 absorbing.

**17/17 checks PASS.**

| check | expected | obtained | result |
|---|---|---|---|
| transient state count | 139 | 139 | PASS |
| generator rows sum to 0 | <1e-12 | 8.9e-16 | PASS |
| sum_s pi_s(t) == 1 | <1e-9 | 4.4e-16 | PASS |
| expm vs uniformization | <1e-9 | 1.9e-12 | PASS |
| E[denial]: fund-matrix vs integral | <1e-4 | 3.2e-15 | PASS |
| E[breach]: fund-matrix vs integral | <1e-4 | 3.9e-15 | PASS |
| MTTResolve == MTTD + MResponse | <1e-9 | 0.0e+00 | PASS |
| MTTResolve == int (1-P_resolved) | <1e-3 | 3.5e-09 | PASS |
| resolution curve non-exponential | >0.05 | 0.212 | PASS |
| sharper detection lowers breach & denial | monotone | breach 0.403<0.566<0.747 | PASS |
| SSA E_breach | 0.5657 | 0.5642+/-0.0028 | PASS |
| SSA E_denial_daysrv | 0.3894 | 0.3854+/-0.0037 | PASS |
| SSA E_dataloss | 0.3823 | 0.3801+/-0.0025 | PASS |
| SSA MTTD | 3.6366 | 3.6308+/-0.0101 | PASS |
| SSA MTTResolve | 3.9440 | 3.9369+/-0.0103 | PASS |
| SSA detection CDF (max|dP|) | <0.02 | 0.002 | PASS |
| SSA resolution CDF (max|dP|) | <0.02 | 0.001 | PASS |

## Per-incident expectations (base case)

- `MTTD` = 3.6366
- `MResponse` = 0.3073
- `MTTResolve` = 3.9440
- `E_breach` = 0.5657
- `E_denial_daysrv` = 0.3894
- `E_encrypt_daysrv` = 0.1912
- `E_exposure_daysrv` = 2.4955
- `E_dataloss` = 0.3823
- `E_maxlocked_proxy` = 0.0987
