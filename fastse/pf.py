# fastSE
# Copyright (C) 2022 Zeyu Mao

from kvxopt import matrix, klu
import numpy as np
from fastse.assemble import AC_jacobian
# Import corresponding AOT/JIT functions
import platform
import warnings
PYTHON_VERSION = platform.python_version_tuple()
if PYTHON_VERSION[1] in ['7', '8', '9', '10']:  # pragma: no cover
    try:
        exec(f"from precompile{PYTHON_VERSION[0]}{PYTHON_VERSION[1]} import compute_normf")
    except ImportError or RuntimeError:
        warnings.warn("Fail to load ahead-of-time compiled module")


def nrpf(V, npv, npq, Ybus, Sbus, pv, pq, pvpq, base, tol=None):
    Va = np.angle(V)
    Vm = np.abs(V)
    j1 = 0
    j2 = npv  ## j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  ## j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  ## j5:j6 - V mag of pq buses

    ## evaluate F(x0)
    mis = V * np.conj(Ybus * V) - Sbus
    normF, F = compute_normf(mis[pv], mis[pq], mis[pq])
    itr = 0
    if tol is None:
        tol = 0.1 / base

    CHECK_VAR_LIMIT = False

    while normF > tol and itr < 10:
        itr += 1
        J = AC_jacobian(Ybus, V, pvpq, pq, npv, npq)
        F = matrix(F)
        klu.linsolve(J, F)
        dx = -1 * F
        dx = np.array(dx).ravel()
        ## update voltage
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * np.exp(1j * Va)
        Vm = np.abs(V)  ## update Vm and Va again in case
        Va = np.angle(V)  ## we wrapped around with a negative Vm
        mis = V * np.conj(Ybus * V) - Sbus
        normF, F = compute_normf(mis[pv], mis[pq], mis[pq])

    Scalc = V * np.conj(Ybus * V)
    return Vm, Va, Scalc, normF, itr


def pfsoln(Sbus, V, Cf, Ct, Yf, Yt, base, rates, Qd, Pd, slack, pv):
    """Update bus, gen, branch data
    """
    Vf = Cf * V
    Vt = Ct * V
    If = Yf * V
    It = Yt * V
    Sf = Vf * np.conj(If)
    St = Vt * np.conj(It)
    # Branch losses in MVA
    losses = (Sf + St) * base
    # branch voltage increment
    Vbranch = Vf - Vt

    # Branch power in MVA
    Sfb = Sf * base
    Stb = St * base

    # Branch loading in p.u.
    loading = np.abs(Sfb) / (rates + 1e-9)

    # Update generator reactive power (including slack and pv)
    Qg = Sbus[np.r_[slack, pv]].imag * base + Qd[np.r_[slack, pv]]

    # Update generator real power at slack bus
    Pg_slack = Sbus[slack].real*base + Pd[slack]
    return Sfb, Stb, If, It, Vbranch, loading, losses, Qg, Pg_slack


if __name__ == '__main__':
    from scipy.sparse import csr_matrix
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
    Sbus = np.array([ 3.948+0.000j, -8.000-2.800j,  4.400+5.221j,  0.000+0.000j,
        0.000+0.000j])
    V = np.array([1.000+0.000j, 1.000+0.000j, 1.050+0.000j, 1.000+0.000j,
       1.000+0.000j])
    pv = np.array([2])
    pq = np.array([1, 3, 4])
    pvpq = np.array([2, 1, 3, 4])
    base = np.array([100.000])
    npv = len(pv)
    npq = len(pq)
    start = time.time()
    nrpf(V, npv, npq, Ybus, Sbus, pv, pq, pvpq, base)
    print(time.time()-start)

    start = time.time_ns()
    Vm, _, _, _, _ = nrpf(V, npv, npq, Ybus, Sbus, pv, pq, pvpq, base)
    print('Execution time:', time.time_ns()-start, 'Vm:', Vm)



