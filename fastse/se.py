# fastSE
# Copyright (C) 2022 Zeyu Mao

import numpy as np
from scipy.sparse import diags, csr_matrix, coo_matrix
try:
    from scipy.sparse.linalg._dsolve import _superlu
except ModuleNotFoundError:
    from scipy.sparse.linalg.dsolve import _superlu
from kvxopt import matrix, spmatrix, spdiag, mul, exp, klu
from fastse.assemble import construct_H_full
from fastse.performance import nb_concatenate, add_diags, mul_diags, diagnoal_max
from fastse.validation import bdd_validation
from fastse.measurements import StateEstimationInput

KVXOPT_ENABLE = True


def Jacobian_SE(Ybus, Yf, Yt, V, f, t, inputs, pvpq, diagV, diagVnorm, *args):
    n = Ybus.shape[0]
    I = Ybus * V
    S = V * np.conj(I)

    # Currently only suitable for p_inj, q_inj and vm cases, but it is REALLY fast
    H_data, H_row, H_col = construct_H_full(Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V), inputs.p_inj_idx,
                                            inputs.q_inj_idx, inputs.vm_m_idx, pvpq)

    h1 = Sf[inputs.p_flow_idx].real if inputs.p_flow_idx.size > 0 else []
    h2 = S[inputs.p_inj_idx].real if inputs.p_inj_idx.size > 0 else []

    h3 = Sf[inputs.q_flow_idx].imag if inputs.q_flow_idx.size > 0 else []
    h4 = S[inputs.q_inj_idx].imag if inputs.q_inj_idx.size > 0 else []

    h5 = np.abs(If[inputs.i_flow_idx]) if inputs.i_flow_idx.size > 0 else []
    h6 = np.abs(V[inputs.vm_m_idx]) if inputs.vm_m_idx.size > 0 else []

    # pack the mismatch vector
    hne = []
    for hi in [h1, h2, h3, h4, h5, h6]:
        if len(hi) > 0:
            hne.append(hi)
    h = nb_concatenate(*hne)

    if KVXOPT_ENABLE:
        H = spmatrix(H_data, H_row, H_col,
                     size=(len(inputs.p_inj_idx) + len(inputs.q_inj_idx) + len(inputs.vm_m_idx), len(pvpq) + n))
        h = matrix(h)
    else:
        H = coo_matrix((H_data, (H_row, H_col)),
                       shape=(len(inputs.p_inj_idx) + len(inputs.q_inj_idx) + len(inputs.vm_m_idx), len(pvpq) + n))

    return H, h


class StateEstimator:

    def __init__(self, Ybus=None, Yf=None, Yt=None, f=None, t=None, slack=None, pq=None, pv=None):
        self.Ybus = Ybus
        self.Yf = Yf
        self.Yt = Yt
        self.f = f
        self.t = t
        self.slack = slack
        self.pq = pq
        self.pv = pv
        self.V = None
        if self.Ybus is not None:
            self.n = self.Ybus.shape[0]
            self.V = np.ones(self.n, dtype=complex)
        if self.pv is not None and self.pq is not None:
            self.pvpq = np.r_[pv, pq]
            self.npvpq = len(self.pvpq)
            self.npq = len(pq)
        if self.slack is not None:
            self.nvd = len(slack)

    def solve_se_lm(self, se_input, Ybus=None, Yf=None, Yt=None, f=None, t=None, slack=None, pq=None, pv=None, flat=False):
        """
        Solve the state estimation problem using the Levenberg-Marquadt method
        """
        if Ybus is not None:
            self.Ybus = Ybus
            self.n = self.Ybus.shape[0]
        if Yf is not None:
            self.Yf = Yf
        if Yt is not None:
            self.Yt = Yt
        if f is not None:
            self.f = f
        if t is not None:
            self.t = t
        if pv is not None and pq is not None:
            self.pv = pv
            self.pq = pq
            self.pvpq = np.r_[pv, pq]
            self.npvpq = len(self.pvpq)
            self.npq = len(pq)
        if slack is not None:
            self.nvd = len(slack)
        if flat or self.V is None:  # flast start
            self.V = np.ones(self.n, dtype=complex)

        # pick the measurements and uncertainties
        z, sigma = se_input.consolidate()

        # compute the weights matrix
        if KVXOPT_ENABLE:
            W = spdiag(matrix(1.0 / np.power(sigma, 2.0)))
            Idn = spdiag(matrix(np.ones(2 * self.n - self.nvd)))  # identity matrix
            z = matrix(z)
            Va = matrix(np.angle(self.V))
            Vm = matrix(np.abs(self.V))
        else:
            W = diags(1.0 / np.power(sigma, 2.0))  # csc_matrix(np.diag(1.0 / np.power(sigma, 2.0)))
            Idn = diags(np.ones(2 * self.n - self.nvd))  # identity matrix
            Va = np.angle(self.V)
            Vm = np.abs(self.V)

        # Levenberg-Marquardt method
        tol = 1e-9
        max_iter = 100
        iter_ = 0

        # x = np.r_[np.angle(V)[pvpq], np.abs(V)]
        lbmda = 0  # any large number
        f_obj_prev = 1e9  # very large number

        converged = False
        err = 1e20
        nu = 2.0

        # first computation of the jacobian and free term
        # initalize the sparse matrix (value not important)
        diagV = diags(self.V).tocsr()
        diagVf = diagV.copy()
        diagVt = diagV.copy()
        diagIf = diagV.copy()
        diagIt = diagV.copy()
        diagVnorm = diagV.copy()
        diagV2 = diagV.copy()
        diagVnorm2 = diagV.copy()
        nl = len(f)
        nb = len(self.V)
        il = np.arange(nl)
        ib = np.arange(nb)
        shape = (nl, nb)
        spVf = csr_matrix((self.V[f], (il, self.f)), shape)
        spVt = csr_matrix((self.V[t], (il, self.t)), shape)
        spVnf = spVf.copy()
        spVnt = spVt.copy()
        H, h = Jacobian_SE(self.Ybus, self.Yf, self.Yt, self.V, self.f, self.t, se_input, self.pvpq, diagV, diagVnorm, diagVf, diagIf, diagVt, diagIt,
                                 diagV2, diagVnorm2, spVf, spVt, spVnf, spVnt)

        while not converged and iter_ < max_iter:

            if KVXOPT_ENABLE:

                # measurements error
                dz = z - h

                # System matrix
                # H1 = H^t·W
                HT = H.T
                HT.V = matrix(mul_diags(HT.V, HT.J, W.V))
                H1 = HT
                H2 = H1 * H

                # set first value of lmbda
                if iter_ == 0:
                    lbmda = 1e-3 * diagnoal_max(H2.V, H2.I, H2.J, H2.size[0])

                # compute system matrix
                temp = lbmda * Idn
                H2.V = matrix(add_diags(H2.V, H2.I, H2.J, temp.V))
                A = H2

                # right hand side
                # H^t·W·dz
                rhs = H1 * dz

                # Solve the increment
                B = matrix(rhs)
                klu.linsolve(A, B)
                dx = B

                # objective function
                f_obj = 0.5 * dz.T * (W * dz)

                # decision function
                rho = (f_obj_prev - f_obj) / (0.5 * dx.T * (lbmda * dx + rhs))
                rho = rho[0]

                # lambda update
                if rho > 0:
                    lbmda = lbmda * max([1.0 / 3.0, 1 - (2 * rho - 1) ** 3])
                    nu = 2.0

                    # modify the solution
                    dVa = dx[:self.npvpq, :]
                    dVm = dx[self.npvpq:, :]
                    Va[matrix(self.pvpq), :] += dVa
                    Vm += dVm
                    self.V = mul(Vm, exp(1j * Va))
                    self.V = np.array(self.V.T).ravel()

                    # update H matrix
                    H, h = Jacobian_SE(self.Ybus, self.Yf, self.Yt, self.V, self.f, self.t, se_input, self.pvpq, diagV, diagVnorm, diagVf, diagIf,
                                             diagVt, diagIt, diagV2, diagVnorm2, spVf, spVt, spVnf, spVnt)

                else:
                    lbmda = lbmda * nu
                    nu = nu * 2

                # compute the convergence
                err = np.linalg.norm(dx, np.Inf)
                converged = err < tol

                # update loops
                f_obj_prev = f_obj
                iter_ += 1

            else:
                # measurements error
                dz = z - h

                # System matrix
                # H1 = H^t·W
                H1 = H.transpose().dot(W)
                H2 = H1.dot(H)

                # set first value of lmbda
                if iter_ == 0:
                    lbmda = 1e-3 * H2.diagonal().max()

                # compute system matrix
                A = H2 + lbmda * Idn

                # right hand side
                # H^t·W·dz
                rhs = H1.dot(dz)

                # Solve the increment
                x, info = _superlu.gssv(A.shape[1], A.nnz, A.data, A.indices, A.indptr, rhs, 1,
                                        options=dict(ColPerm='COLAMD'))
                dx = x.ravel()

                # objective function
                f_obj = 0.5 * dz.dot(W * dz)

                # decision function
                rho = (f_obj_prev - f_obj) / (0.5 * dx.dot(lbmda * dx + rhs))

                # lambda update
                if rho > 0:
                    lbmda = lbmda * max([1.0 / 3.0, 1 - (2 * rho - 1) ** 3])
                    nu = 2.0

                    # modify the solution
                    dVa = dx[:self.npvpq]
                    dVm = dx[self.npvpq:]
                    Va[self.pvpq] += dVa
                    Vm += dVm
                    self.V = Vm * np.exp(1j * Va)

                    # update H matrix
                    H, h = Jacobian_SE(Ybus, Yf, Yt, self.V, f, t, se_input, self.pvpq, diagV, diagVnorm, diagVf, diagIf,
                                             diagVt, diagIt, diagV2, diagVnorm2, spVf, spVt, spVnf, spVnt)

                else:
                    lbmda = lbmda * nu
                    nu = nu * 2

                # compute the convergence
                err = np.linalg.norm(dx, np.Inf)
                converged = err < tol

                # update loops
                f_obj_prev = f_obj
                iter_ += 1

        results = {
            'r': dz,
            'R_inv': W
        }

        return self.V, err, converged, results


if __name__ == '__main__':
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




