# fastSE
# Copyright (C) 2022 Zeyu Mao
import os
import sys
import numpy as np
from numpy import complex128
from numba import njit, config
from collections import namedtuple
from scipy.sparse import csr_matrix
from scipy.optimize import newton
from fastse._aot import *
from plum import dispatch

np.set_printoptions(threshold=sys.maxsize)


result = namedtuple('results', 'temperature diff iterations converged')
tdpf_result = namedtuple(
    'results', 'temperature diff resistance V Sf loss iterations')


def run_tdpf(tas, tc0, rads, winds, V, npv, npq, r0, x, c, tap, shift, f, t, i, nl,
             nb, Sbus, pv, pq, pvpq, base, line_indexes, tran_indexes, rates, tol=1e-3,
             maxiter=50):
    """
    add transformer temperature balance equation
    tc0, rads, winds: only lines
    tas, r0, x, c: all branches (lines and transformers)
    """

    Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
    Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))
    Ysh = csr_matrix((np.zeros(nb, dtype=complex128),
                     (range(nb), range(nb))), (nb, nb))
    # Run conventional power flow first

    r = r0.copy()
    tc = tc0.copy()
    tr0 = np.full(tran_indexes.sum(), 0)

    V, tc, tr, r, Yf, Yt, it = tdpf_loop(V, npv, npq, Sbus, pv, pq, pvpq, base, Cf, line_indexes,
                                         tc, r0, tas, rads, winds, rates, tran_indexes, tr0, r, x, c, tap, shift, f, t, i, nl, nb, Ct, Ysh, tol, maxiter)

    # Compute system losses and remaining variables
    Vf = Cf * V
    Vt = Ct * V
    If = Yf * V
    It = Yt * V
    Sf = Vf * np.conj(If)
    St = Vt * np.conj(It)
    # Branch losses in MVA
    losses = (Sf + St) * base

    t = tas.copy()
    t[line_indexes] = tc
    t[tran_indexes] = tas[tran_indexes] + tr

    return tdpf_result(t, t-tas, r, V, Sf, losses, it)


if __name__ == "__main__":
    # from scipy.sparse import csr_matrix
    #
    # # A 5 bus example from Prof. Overbye's textbook
    # r0 = np.array([0.00150, 0.00900, 0.00450, 0.00075, 0.00225])
    # x = np.array([0.02000, 0.10000, 0.05000, 0.01000, 0.02500])
    # c = np.array([0.00000, 1.72000, 0.88000, 0.00000, 0.44000])
    # tap = np.ones(5, dtype=float)
    # shift = np.zeros(5, dtype=float)
    # f = np.array([0, 3, 4, 2, 4])
    # t = np.array([4, 1, 1, 3, 3])
    # nb = 5  # number of buses
    # nl = 5  # number of branches
    # i = np.r_[range(nl), range(nl)]
    # Sbus = np.array(
    #     [3.948 + 0.000j, -8.000 - 2.800j, 4.400 + 5.221j, 0.000 + 0.000j,
    #      0.000 + 0.000j])
    # V = np.ones(5, dtype=complex128)  # Voltage initial guess
    # pv = np.array([2])
    # pq = np.array([1, 3, 4])
    # pvpq = np.array([2, 1, 3, 4])
    # base = np.array([100.000])
    # npv = len(pv)
    # npq = len(pq)
    # line_indexes = np.array([False, True, True, False, True])
    # # rs = np.full(3, 2.25e-4)
    # # tas = np.full(3, 25)  # assume all ambient temperature are 25 degree
    # tas = np.array([10, 20, 25])
    # tc0 = np.full(3, 25)  # conductor temperature initial guess
    # result = run_tdpf(tas, tc0, V, npv, npq, r0, x, c, tap, shift, f, t, i, nl,
    #          nb, Sbus, pv, pq, pvpq, base, line_indexes)
    # print(result)
    from esa import SAW
    import time

    np.set_printoptions(precision=4, floatmode='fixed')
    sa = SAW(r"C:\Users\test\PWCase\ACTIVSg2000.pwb", CreateIfNotFound=True)
    sa.pw_order = True
    sa.RunScriptCommand('ResetToFlatStart(YES,NO,NO,NO);')

    base = sa.GetParametersMultipleElement('Sim_Solution_Options', ['SBase']).to_numpy(
        float).ravel()

    key = sa.get_key_field_list('bus')
    df = sa.GetParametersMultipleElement('bus', key + ['BusCat', 'BusPUVolt', 'BusRad', 'BusNetMW',
                                                       'BusNetMVR', 'BusGenMW', 'BusGenMVR',
                                                       'BusLoadMW', 'BusLoadMVR', 'GenMVRMax',
                                                       'GenMVRMin', 'BusSS', 'BusSSMW'])
    df['BusNetMW'] = df['BusNetMW'].astype(float)
    df['BusNetMVR'] = df['BusNetMVR'].astype(float)
    df['BusGenMW'] = df['BusGenMW'].astype(float)
    df['BusGenMVR'] = df['BusGenMVR'].astype(float)
    df['BusLoadMW'] = df['BusLoadMW'].astype(float)
    df['BusLoadMVR'] = df['BusLoadMVR'].astype(float)
    df['GenMVRMax'] = df['GenMVRMax'].astype(float)
    df['GenMVRMin'] = df['GenMVRMin'].astype(float)
    df['BusSS'] = df['BusSS'].astype(float)
    df['BusSSMW'] = df['BusSSMW'].astype(float)
    df['BusNum'] = df['BusNum'].astype(int)
    df.fillna(0, inplace=True)

    branch = sa.GetParametersMultipleElement('branch',
                                             sa.get_key_field_list('branch') + ['LineR', 'LineX',
                                                                                'LineC', 'LineTap',
                                                                                'LinePhase',
                                                                                'LineLimMVA',
                                                                                'BranchDeviceType'])
    branch['LineR'] = branch['LineR'].astype(float)
    branch['LineX'] = branch['LineX'].astype(float)
    branch['LineC'] = branch['LineC'].astype(float)
    branch['LineTap'] = branch['LineTap'].astype(float)
    branch['LinePhase'] = branch['LinePhase'].astype(float)
    branch['LineLimMVA'] = branch['LineLimMVA'].astype(float)

    nb = df.shape[0]
    nl = branch.shape[0]

    Ys = 1 / (branch['LineR'].to_numpy() + 1j *
              branch['LineX'].to_numpy())  # series admittance
    Bc = branch['LineC'].to_numpy()  # line charging susceptance
    tap = branch['LineTap'].to_numpy()
    shift = branch['LinePhase'].to_numpy()
    rates = branch['LineLimMVA'].to_numpy()
    tap = tap * np.exp(1j * np.pi / 180 * shift)
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = - Ys / np.conj(tap)
    Ytf = - Ys / tap

    Ysh = (df['BusSSMW'].to_numpy() + 1j * df['BusSS'].to_numpy()) / base

    # lookup table for formatting bus numbers

    def loop_translate(a, d):
        n = np.ndarray(a.shape, dtype=int)
        for k in d:
            n[a == k] = d[k]
        return n

    d = dict()
    for index, value in df['BusNum'].items():
        d[value] = index
    f = branch['BusNum'].to_numpy(dtype=int).reshape(-1)
    f = loop_translate(f, d)
    t = branch['BusNum:1'].to_numpy(dtype=int).reshape(-1)
    t = loop_translate(t, d)
    # connection matrix for line & from buses
    # print(nl, nb, f, t)
    Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
    # connection matrix for line & to buses
    Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))
    i = np.r_[range(nl), range(nl)]  # double set of row indices
    Yf = csr_matrix((np.hstack([Yff.reshape(-1), Yft.reshape(-1)]), (i, np.hstack([f, t]))),
                    (nl, nb))
    Yt = csr_matrix((np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]), (i, np.hstack([f, t]))),
                    (nl, nb))
    Ybus = Cf.T * Yf + Ct.T * Yt + \
        csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    # print(Ybus.toarray())
    # Ybus = sa.get_ybus()

    pv = df.index[df['BusCat'].str.contains("PV")].to_numpy(int)
    pq = df.index[df['BusCat'].str.contains("PQ")].to_numpy(
        int)  # include PQ with Var limit
    pv.sort()
    pq.sort()
    slack = df.index[df['BusCat'] == 'Slack'].to_numpy(int)

    # set up indexing for updating v
    pvpq = np.r_[pv, pq]
    npv = len(pv)
    npq = len(pq)
    # print(npv, npq)
    npvpq = npv + npq

    pvpq_lookup = np.zeros(np.max(Ybus.indices) + 1, dtype=int)
    pvpq_lookup[pvpq] = np.arange(npvpq)

    Sbus = (df['BusNetMW'].to_numpy(dtype=float) + df['BusNetMVR'].to_numpy(
        dtype=float) * 1j) / base
    V = df['BusPUVolt'].to_numpy(complex)

    x = branch['LineX'].to_numpy()
    r = branch['LineR'].to_numpy()
    c = branch['LineC'].to_numpy()
    tap = branch['LineTap'].to_numpy()
    shift = branch['LinePhase'].to_numpy()
    f = branch['BusNum'].to_numpy(dtype=int).reshape(-1)
    f = loop_translate(f, d)
    t = branch['BusNum:1'].to_numpy(dtype=int).reshape(-1)
    t = loop_translate(t, d)
    line_indexes = (branch['BranchDeviceType'] == 'Line').to_numpy()
    tran_indexes = (branch['BranchDeviceType'] == 'Transformer').to_numpy()
    num_lines = line_indexes.sum()

    tc0 = np.full(num_lines, 25.0, dtype=float)
    # tas = np.random.uniform(20, 40, num_lines)
    tas = np.random.uniform(25, 35, nl)

    radiations = np.random.uniform(800, 1200, num_lines)
    winds = np.random.uniform(0, 15, num_lines)

    start = time.time()
    result = run_tdpf(tas, tc0, radiations, winds, V, npv, npq, r, x, c, tap, shift, f, t, i, nl,
                      nb, Sbus, pv, pq, pvpq, base, line_indexes, tran_indexes, rates, 1e-3, 50)

    # print(result.temperature)
    print(result.temperature[tran_indexes][:10])
    print(result.temperature[line_indexes][:10])
