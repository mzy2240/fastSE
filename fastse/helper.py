# fastSE
# Copyright (C) 2022 Zeyu Mao

from scipy.sparse import csr_matrix
import numpy as np


def calculate_branch_impedance(Ys, Bc, tap, shift, nb, nl, f, t):
    """
    Ys = 1/(branch['LineR'].to_numpy() + 1j * branch['LineX'].to_numpy())
    Bc = branch['LineC'].to_numpy()
    tap: off-norminal tap ratio
    shift: phase shift degree
    nb: number of buses
    nl: number of lines
    f,t: adjacency matrix
    """
    tap = tap * np.exp(1j * np.pi / 180 * shift)
    Ytt = (Ys + 1j * Bc / 2)
    Yff = (Ys + 1j * Bc / 2) / (tap * np.conj(tap))
    Yft = - Ys / np.conj(tap)
    Ytf = - Ys / tap
    i = np.hstack([range(nl), range(nl)])  ## double set of row indices
    Yf = csr_matrix((np.hstack([Yff.reshape(-1), Yft.reshape(-1)]), (i, np.hstack([f, t]))), (nl, nb))
    Yt = csr_matrix((np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]), (i, np.hstack([f, t]))), (nl, nb))
    return Yf, Yt


def make_y_bus(GS, BS, base, nb, nl, f, t, Yff, Yft, Ytf, Ytt):
    Ysh = (GS + 1j * BS) / base
    Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))
    i = np.r_[range(nl), range(nl)]  ## double set of row indices
    Yf = csr_matrix((np.hstack([Yff.reshape(-1), Yft.reshape(-1)]), (i, np.hstack([f, t]))), (nl, nb))
    Yt = csr_matrix((np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]), (i, np.hstack([f, t]))), (nl, nb))
    return Cf.T * Yf + Ct.T * Yt + csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))


def generate_admittance_matrix(r, x, c, tap, shift, f, t, i, nl, nb, Cf, Ct, Ysh):
    Ys = 1 / (r + 1j * x)  # series admittance
    Bc = c  # line charging susceptance
    tap = tap
    shift = shift
    tap = tap * np.exp(1j * np.pi / 180 * shift)
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = - Ys / np.conj(tap)
    Ytf = - Ys / tap
    Yf = csr_matrix(
        (np.hstack([Yff.reshape(-1), Yft.reshape(-1)]), (i, np.hstack([f, t]))),
        (nl, nb))
    Yt = csr_matrix(
        (np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]), (i, np.hstack([f, t]))),
        (nl, nb))
    Ybus = Cf.T * Yf + Ct.T * Yt + Ysh
    return Ybus, Yf, Yt

def preprocess(base, df, branch):
    nb = df.shape[0]
    nl = branch.shape[0]

    Ys = 1 / (branch['LineR'].to_numpy() + 1j * branch['LineX'].to_numpy())  # series admittance
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


    d = {}
    for index, value in df['BusNum'].items():
        d[value] = index
    f = branch['BusNum'].to_numpy(dtype=int).reshape(-1)
    f = loop_translate(f, d)
    t = branch['BusNum:1'].to_numpy(dtype=int).reshape(-1)
    t = loop_translate(t, d)
    ## connection matrix for line & from buses
    # print(nl, nb, f, t)
    Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))
    i = np.r_[range(nl), range(nl)]  ## double set of row indices
    Yf = csr_matrix((np.hstack([Yff.reshape(-1), Yft.reshape(-1)]), (i, np.hstack([f, t]))),
                    (nl, nb))
    Yt = csr_matrix((np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]), (i, np.hstack([f, t]))),
                    (nl, nb))
    Ybus = Cf.T * Yf + Ct.T * Yt + csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    # print(Ybus.toarray())
    # Ybus = sa.get_ybus()

    pv = df.index[df['BusCat'].str.contains("PV")].to_numpy(int)
    pq = df.index[df['BusCat'].str.contains("PQ")].to_numpy(int)  # include PQ with Var limit
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

    return V, npv, npq, r, x, c, tap, shift, f, t, i, nl, nb, Sbus, pv, pq, pvpq, base, line_indexes, tran_indexes, rates
