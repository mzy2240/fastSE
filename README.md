# 🚀FastSE
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/fastSE.svg)](https://pypi.python.org/pypi/fastSE/)
[![PyPI version fury.io](https://badge.fury.io/py/fastSE.svg)](https://pypi.python.org/pypi/fastSE/)
[![Downloads](https://pepy.tech/badge/fastse)](https://pepy.tech/project/fastse)
[![PyPI license](https://img.shields.io/pypi/l/fastSE.svg)](https://pypi.python.org/pypi/fastSE/)



> A collection of power system computation modules

sparse matrix + aot/jit + klu + custom improved ordering + python
= efficiency in computation and development!

# 🌟Features

* ⚡Jitted and KLU-powered [state estimation](https://github.com/mzy2240/fastSE/blob/master/fastse/se.py)
* ⚡Jitted and KLU-powered [power flow](https://github.com/mzy2240/fastSE/blob/master/fastse/pf.py)
* ⚡Jitted and KLU-powered [continuation power flow](https://github.com/mzy2240/fastSE/blob/master/fastse/cpf.py)
* ⚡Jitted and KLU-powered [temperature-dependent power flow](https://github.com/mzy2240/fastSE/blob/master/fastse/tdpf.py)

# Installation

To install, simply run `pip install fastSE` in your command prompt.

# How to use

Here is one simple example. `solve_se_lm` is a high-level function which
computes derivatives, assemble them as sparse matrix and then calculate the 
estimates using sparse matrix solver. All the low-level functions could also
be imported and used individually.

```python
from fastse import StateEstimator, StateEstimationInput
from fastse import bdd_validation
from scipy.sparse import csr_matrix
import numpy as np

import time
# A 5 bus example from Prof. Overbye's textbook
# node impedance
Ybus = np.array([[3.729 - 49.720j, 0.000 + 0.000j, 0.000 + 0.000j,
        0.000 + 0.000j, -3.729 + 49.720j],
       [0.000 + 0.000j, 2.678 - 28.459j, 0.000 + 0.000j,
        -0.893 + 9.920j, -1.786 + 19.839j],
       [0.000 + 0.000j, 0.000 + 0.000j, 7.458 - 99.441j,
        -7.458 + 99.441j, 0.000 + 0.000j],
       [0.000 + 0.000j, -0.893 + 9.920j, -7.458 + 99.441j,
        11.922 - 147.959j, -3.571 + 39.679j],
       [-3.729 + 49.720j, -1.786 + 19.839j, 0.000 + 0.000j,
        -3.571 + 39.679j, 9.086 - 108.578j]])
Ybus = csr_matrix(Ybus)

# branch impedance
Yf = np.array([[ 3.729-49.720j,  0.000 +0.000j,  0.000 +0.000j,  0.000 +0.000j,
    -3.729+49.720j],
   [ 0.000 +0.000j, -0.893 +9.920j,  0.000 +0.000j,  0.893 -9.060j,
     0.000 +0.000j],
   [ 0.000 +0.000j, -1.786+19.839j,  0.000 +0.000j,  0.000 +0.000j,
     1.786-19.399j],
   [ 0.000 +0.000j,  0.000 +0.000j,  7.458-99.441j, -7.458+99.441j,
     0.000 +0.000j],
   [ 0.000 +0.000j,  0.000 +0.000j,  0.000 +0.000j, -3.571+39.679j,
     3.571-39.459j]])
Yf = csr_matrix(Yf)

Yt = np.array([[-3.729+49.720j,  0.000 +0.000j,  0.000 +0.000j,  0.000 +0.000j,
     3.729-49.720j],
   [ 0.000 +0.000j,  0.893 -9.060j,  0.000 +0.000j, -0.893 +9.920j,
     0.000 +0.000j],
   [ 0.000 +0.000j,  1.786-19.399j,  0.000 +0.000j,  0.000 +0.000j,
    -1.786+19.839j],
   [ 0.000 +0.000j,  0.000 +0.000j, -7.458+99.441j,  7.458-99.441j,
     0.000 +0.000j],
   [ 0.000 +0.000j,  0.000 +0.000j,  0.000 +0.000j,  3.571-39.459j,
    -3.571+39.679j]])
Yt = csr_matrix(Yt)

# branch from and to bus
f = np.array([0, 3, 4, 2, 4])
t = np.array([4, 1, 1, 3, 3])

# slack, pv and pq buses
slack = np.array([0])  # The slack bus does not have to be the 0-indexed bus
pq = np.array([1, 3, 4])
pv = np.array([2])

# measurements
se_input = StateEstimationInput()

se_input.p_inj = np.array([ 3.948e+00, -8.000e+00,  4.400e+00, -6.507e-06, -1.407e-05])
se_input.p_inj_idx = np.arange(len(se_input.p_inj))
se_input.p_inj_weight = np.full(len(se_input.p_inj), 0.01)

se_input.q_inj = np.array([ 1.143e+00, -2.800e+00,  2.975e+00,  6.242e-07,  1.957e-06])
se_input.q_inj_idx = np.arange(len(se_input.q_inj))
se_input.q_inj_weight = np.full(len(se_input.q_inj), 0.01)

se_input.vm_m = np.array([0.834, 1.019, 0.974])
se_input.vm_m_idx = pq
se_input.vm_m_weight = np.full(len(se_input.vm_m), 0.01)

# First time will be slow due to compilation
start = time.time()
estimator = StateEstimator()
v_sol, err, converged, results = estimator.solve_se_lm(se_input, Ybus, Yf, Yt, f, t, slack, pq, pv, flat=True)
print("compilation + execution time:", time.time() - start)
bdd_validation(results, m=len(se_input.measurements), n=Ybus.shape[0] + len(pq) + len(pv))

# But then it will be very performant
start = time.time()
v_sol, err, converged, results = estimator.solve_se_lm(se_input, Ybus, Yf, Yt, f, t, slack, pq, pv, flat=True)
print("Execution time:", time.time() - start)

# Start from previous solution (set flat = False)
start = time.time()
v_sol, err, converged, results = estimator.solve_se_lm(se_input, Ybus, Yf, Yt, f, t, slack, pq, pv, flat=False)
print("Execution time:", time.time() - start)

# False data injection
se_input.vm_m[1] -= 0.025
se_input.vm_m[2] += 0.025
v_sol, err, converged, results = estimator.solve_se_lm(se_input, Ybus, Yf, Yt, f, t, slack, pq, pv)
print("-------------After False Data Injection-------------")
bdd_validation(results, m=len(se_input.measurements), n=Ybus.shape[0] + len(pq) + len(pv))

```

# References

### Temperature Dependent Power Flow (Temperature Dependent Load Flow)

S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-Dependent Power Flow," in IEEE Transactions on Power Systems, vol. 28, no. 4, pp. 4007-4018, Nov. 2013, doi: 10.1109/TPWRS.2013.2266409.

Rahman, Mahbubur et al. “Power handling capabilities of transmission systems using a temperature-dependent power flow.” Electric Power Systems Research (2019): n. pag.

# Acknowledge

This work was supported by the U.S. Department of Energy (DOE) under award DE-OE0000895 and the Sandia National Laboratories’ directed R&D project #222444.
