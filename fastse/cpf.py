# fastSE
# Copyright (C) 2022 Zeyu Mao

import numpy as np
from numpy import complex128
from fastse.assemble import AC_jacobian, cpf_jacobian
from fastse.performance import cpf_p_jit
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import spsolve
from kvxopt import klu, matrix


class CpfNumericResults:

    def __init__(self):
        self.V = list()
        self.Sbus = list()
        self.lmbda = list()
        self.Sf = list()
        self.St = list()
        self.loading = list()
        self.losses = list()
        self.normF = list()
        self.success = list()

    def add(self, v, sbus, Sf, St, lam, losses, loading, normf, converged):
        self.V.append(v)
        self.Sbus.append(sbus)
        self.lmbda.append(lam)
        self.Sf.append(Sf)
        self.St.append(St)
        self.loading.append(loading)
        self.losses.append(losses)
        self.normF.append(normf)
        self.success.append(converged)

    def __len__(self):
        return len(self.V)


def cpf_p(step, z, V, lam, V_prev, lamprv, pv, pq, pvpq):
    """
    Computes the value of the Current Parametrization Function
    :param parametrization: Value of  option (1: Natural, 2:Arc-length, 3: pseudo arc-length)
    :param step: continuation step size
    :param z: normalized tangent prediction vector from previous step
    :param V: complex bus voltage vector at current solution
    :param lam: scalar lambda value at current solution
    :param V_prev: complex bus voltage vector at previous solution
    :param lamprv: scalar lambda value at previous solution
    :param pv: vector of indices of PV buses
    :param pq: vector of indices of PQ buses
    :param pvpq: vector of indices of PQ and PV buses
    :return: value of the parametrization function at the current point
    """

    """
    #CPF_P Computes the value of the CPF parametrization function.
    #
    #   P = CPF_P(parametrization, STEP, Z, V, LAM, VPRV, LAMPRV, PV, PQ)
    #
    #   Computes the value of the parametrization function at the current
    #   solution point.
    #
    #   Inputs:
    #       parametrization : Value of cpf.parametrization option
    #       STEP : continuation step size
    #       Z : normalized tangent prediction vector from previous step
    #       V : complex bus voltage vector at current solution
    #       LAM : scalar lambda value at current solution
    #       VPRV : complex bus voltage vector at previous solution
    #       LAMPRV : scalar lambda value at previous solution
    #       PV : vector of indices of PV buses
    #       PQ : vector of indices of PQ buses
    #
    #   Outputs:
    #       P : value of the parametrization function at the current point
    #
    #   See also CPF_PREDICTOR, CPF_CORRECTOR.
    #   MATPOWER
    #   Copyright (c) 1996-2015 by Power System Engineering Research Center (PSERC)
    #   by Shrirang Abhyankar, Argonne National Laboratory
    #   and Ray Zimmerman, PSERC Cornell
    #
    #   $Id: cpf_p.m 2644 2015-03-11 19:34:22Z ray $
    #
    #   This file is part of MATPOWER.
    #   Covered by the 3-clause BSD License (see LICENSE file for details).
    #   See http://www.pserc.cornell.edu/matpower/ for more info.
    ## evaluate P(x0, lambda0)
    """
    # ArcLength
    Va = np.angle(V)
    Vm = np.abs(V)
    Va_prev = np.angle(V_prev)
    Vm_prev = np.abs(V_prev)
    P = cpf_p_jit(Va, Vm, Va_prev, Vm_prev, lam, lamprv, step, pv, pq)

    return P


def cpf_p_jac(z, V, lam, Vprv, lamprv, pv, pq, pvpq):
    """
    Computes partial derivatives of Current Parametrization Function (CPF).
    :param parametrization:
    :param z: normalized tangent prediction vector from previous step
    :param V: complex bus voltage vector at current solution
    :param lam: scalar lambda value at current solution
    :param Vprv: complex bus voltage vector at previous solution
    :param lamprv: scalar lambda value at previous solution
    :param pv: vector of indices of PV buses
    :param pq: vector of indices of PQ buses
    :param pvpq: vector of indices of PQ and PV buses
    :return:  partial of parametrization function w.r.t. voltages
              partial of parametrization function w.r.t. lambda
    """

    """
    #CPF_P_JAC Computes partial derivatives of CPF parametrization function.
    #
    #   [DP_DV, DP_DLAM ] = CPF_P_JAC(parametrization, Z, V, LAM, ...
    #                                                   VPRV, LAMPRV, PV, PQ)
    #
    #   Computes the partial derivatives of the continuation power flow
    #   parametrization function w.r.t. bus voltages and the continuation
    #   parameter lambda.
    #
    #   Inputs:
    #       parametrization : Value of cpf.parametrization option.
    #       Z : normalized tangent prediction vector from previous step
    #       V : complex bus voltage vector at current solution
    #       LAM : scalar lambda value at current solution
    #       VPRV : complex bus voltage vector at previous solution
    #       LAMPRV : scalar lambda value at previous solution
    #       PV : vector of indices of PV buses
    #       PQ : vector of indices of PQ buses
    #
    #   Outputs:
    #       DP_DV : partial of parametrization function w.r.t. voltages
    #       DP_DLAM : partial of parametrization function w.r.t. lambda
    #
    #   See also CPF_PREDICTOR, CPF_CORRECTOR.
    #   MATPOWER
    #   Copyright (c) 1996-2015 by Power System Engineering Research Center (PSERC)
    #   by Shrirang Abhyankar, Argonne National Laboratory
    #   and Ray Zimmerman, PSERC Cornell
    #
    #   $Id: cpf_p_jac.m 2644 2015-03-11 19:34:22Z ray $
    #
    #   This file is part of MATPOWER.
    #   Covered by the 3-clause BSD License (see LICENSE file for details).
    #   See http://www.pserc.cornell.edu/matpower/ for more info.
    """

    Va = np.angle(V)
    Vm = np.abs(V)
    Vaprv = np.angle(Vprv)
    Vmprv = np.abs(Vprv)
    dP_dV = 2.0 * (np.r_[Va[pvpq], Vm[pq]] - np.r_[Vaprv[pvpq], Vmprv[pq]])

    if lam == lamprv:   # first step
        dP_dlam = 1.0   # avoid singular Jacobian that would result from [dP_dV, dP_dlam] = 0
    else:
        dP_dlam = 2.0 * (lam - lamprv)

    return dP_dV, dP_dlam


def predictor(V, Ibus, lam, Ybus, Sxfr, pv, pq, step, z, Vprv, lamprv):
    """
    Computes a prediction (approximation) to the next solution of the
    continuation power flow using a normalized tangent predictor.
    :param V: complex bus voltage vector at current solution
    :param Ibus:
    :param lam: scalar lambda value at current solution
    :param Ybus: complex bus admittance matrix
    :param Sxfr: complex vector of scheduled transfers (difference between bus injections in base and target cases)
    :param pv: vector of indices of PV buses
    :param pq: vector of indices of PQ buses
    :param step: continuation step length
    :param z: normalized tangent prediction vector from previous step
    :param Vprv: complex bus voltage vector at previous solution
    :param lamprv: scalar lambda value at previous solution
    :param parametrization: Value of cpf parametrization option.
    :return: V0 : predicted complex bus voltage vector
             LAM0 : predicted lambda continuation parameter
             Z : the normalized tangent prediction vector
    """

    # sizes
    nb = len(V)
    npv = len(pv)
    npq = len(pq)
    pvpq = np.r_[pv, pq]
    nj = npv + npq * 2

    # compute Jacobian for the power flow equations
    J2 = cpf_jacobian(Ybus, V, pvpq, pv, pq, npv, npq, Sxfr, lam, Vprv, lamprv)

    Va_prev = np.angle(V)
    Vm_prev = np.abs(V)

    # compute normalized tangent predictor
    s = np.zeros(npv + 2 * npq + 1)

    # increase in the direction of lambda
    s[npv + 2 * npq] = 1
    s = matrix(s)

    # tangent vector
    klu.linsolve(J2, s)
    z[np.r_[pvpq, nb + pq, 2 * nb]] = np.array(s).ravel()

    # normalize_string tangent predictor  (dividing by the euclidean norm)
    z /= np.linalg.norm(z)

    Va0 = Va_prev
    Vm0 = Vm_prev
    # lam0 = lam

    # prediction for next step
    Va0[pvpq] = Va_prev[pvpq] + step * z[pvpq]
    Vm0[pq] = Vm_prev[pq] + step * z[pq + nb]
    lam0 = lam + step * z[2 * nb]
    V0 = Vm0 * np.exp(1j * Va0)

    return V0, lam0, z


def corrector(Ybus, Ibus, Sbus, V0, pv, pq, lam0, Sxfr, Vprv, lamprv, z, step, tol, max_it,
              verbose, mu_0=1.0, acceleration_parameter=0.5):
    """
    Solves the corrector step of a continuation power flow using a full Newton method
    with selected parametrization scheme.
    solves for bus voltages and lambda given the full system admittance
    matrix (for all buses), the complex bus power injection vector (for
    all buses), the initial vector of complex bus voltages, and column
    vectors with the lists of bus indices for the swing bus, PV buses, and
    PQ buses, respectively. The bus voltage vector contains the set point
    for generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles.
     Uses default options if this parameter is not given. Returns the
     final complex voltages, a flag which indicates whether it converged or not,
     the number of iterations performed, and the final lambda.
    :param Ybus: Admittance matrix (CSC sparse)
    :param Ibus: Bus current injections
    :param Sbus: Bus power injections
    :param V0:  Bus initial voltages
    :param pv: list of pv nodes
    :param pq: list of pq nodes
    :param lam0: initial value of lambda (loading parameter)
    :param Sxfr: [delP+j*delQ] transfer/loading vector for all buses
    :param Vprv: final complex V corrector solution from previous continuation step
    :param lamprv: final lambda corrector solution from previous continuation step
    :param z: normalized predictor for all buses
    :param step: continuation step size
    :param parametrization:
    :param tol: Tolerance (p.u.)
    :param max_it: max iterations
    :param verbose: print information?
    :return: Voltage, converged, iterations, lambda, power error, calculated power
    """

    # initialize
    i = 0
    V = V0
    Va = np.angle(V)
    Vm = np.abs(V)
    lam = lam0             # set lam to initial lam0
    dVa = np.zeros_like(Va)
    dVm = np.zeros_like(Vm)
    dlam = 0

    # set up indexing for updating V
    npv = len(pv)
    npq = len(pq)
    pvpq = np.r_[pv, pq]
    nj = npv + npq * 2

    # j1:j2 - V angle of pv and pq buses
    j1 = 0
    j2 = npv + npq
    # j2:j3 - V mag of pq buses
    j3 = j2 + npq

    # evaluate F(x0, lam0), including Sxfr transfer/loading
    Scalc = V * np.conj(Ybus * V)
    mismatch = Scalc - Sbus - lam * Sxfr
    # F = np.r_[mismatch[pvpq].real, mismatch[pq].imag]

    # evaluate P(x0, lambda0)
    P = cpf_p_jit(V, Vprv, lam, lamprv, step, pv, pq)

    # augment F(x,lambda) with P(x,lambda)
    F = np.r_[mismatch[pvpq].real, mismatch[pq].imag, P]

    # check tolerance
    normF = np.linalg.norm(F, np.Inf)
    converged = normF < tol
    if verbose:
        print('\nConverged!\n')

    # do Newton iterations
    while not converged and i < max_it:

        # update iteration counter
        i += 1

        F = matrix(F)

        # evaluate Jacobian
        J = cpf_jacobian(Ybus, V, pvpq, pv, pq, npv, npq, Sxfr, lam, Vprv, lamprv)

        # compute update step
        klu.linsolve(J, F)
        dx = np.array(F).ravel()
        dVa[pvpq] = dx[j1:j2]
        dVm[pq] = dx[j2:j3]
        dlam = dx[j3]

        # set the restoration values
        prev_Vm = Vm.copy()
        prev_Va = Va.copy()
        prev_lam = lam

        # set the values and correct with an adaptive mu if needed
        mu = mu_0  # ideally 1.0
        back_track_condition = True
        l_iter = 0
        normF_new = 0.0
        while back_track_condition and l_iter < max_it and mu > tol:

            # restore the previous values if we are backtracking (the first iteration is the normal NR procedure)
            if l_iter > 0:
                Va = prev_Va.copy()
                Vm = prev_Vm.copy()
                lam = prev_lam

            # update the variables from the solution
            Va -= mu * dVa
            Vm -= mu * dVm
            lam -= mu * dlam

            # update Vm and Va again in case we wrapped around with a negative Vm
            V = Vm * np.exp(1j * Va)

            # evaluate F(x, lam)
            Scalc = V * np.conj(Ybus * V)
            mismatch = Scalc - Sbus - lam * Sxfr

            # evaluate the parametrization function P(x, lambda)
            P = cpf_p(step, z, V, lam, Vprv, lamprv, pv, pq, pvpq)

            # compose the mismatch vector
            F = np.r_[mismatch[pvpq].real, mismatch[pq].imag, P]

            # check for convergence
            normF_new = np.linalg.norm(F, np.Inf)

            back_track_condition = normF_new > normF
            mu *= acceleration_parameter
            l_iter += 1

            if verbose:
                print('\n#3d        #10.3e', i, normF)

        if l_iter > 1 and back_track_condition:
            # this means that not even the backtracking was able to correct the solution so, restore and end
            Va = prev_Va.copy()
            Vm = prev_Vm.copy()
            V = Vm * np.exp(1.0j * Va)

            return V, converged, i, lam, normF, Scalc
        else:
            normF = normF_new

        converged = normF < tol
        if verbose:
            print('\nNewton''s method corrector converged in ', i, ' iterations.\n')

    if verbose:
        if not converged:
            print('\nNewton method corrector did not converge in  ', i, ' iterations.\n')

    return V, converged, i, lam, normF, Scalc


def continuation_nr(Ybus, Cf, Ct, Yf, Yt, branch_rates, Sbase, Ibus_base, Ibus_target, Sbus_base, Sbus_target,
                    V, bus_installed_power,
                    vd, pv, pq, step=0.01,
                    adapt_step=True, step_min=0.0001, step_max=0.2, error_tol=1e-3, tol=1e-6, max_it=20,
                    qmax_bus=None, qmin_bus=None, base_overload_number=0,
                    verbose=False, call_back_fx=None) -> CpfNumericResults:
    """
    Runs a full AC continuation power flow using a normalized tangent
    predictor and selected approximation_order scheme.
    :param Ybus: Admittance matrix
    :param Cf: Connectivity matrix of the branches and the "from" nodes
    :param Ct: Connectivity matrix of the branches and the "to" nodes
    :param Yf: Admittance matrix of the "from" nodes
    :param Yt: Admittance matrix of the "to" nodes
    :param branch_rates: array of branch rates to check the overload condition
    :param Ibus_base:
    :param Ibus_target:
    :param Sbus_base: Power array of the base solvable case
    :param Sbus_target: Power array of the case to be solved
    :param V: Voltage array of the base solved case
    :param distributed_slack: Distribute the slack?
    :param bus_installed_power: array of installed power per bus
    :param vd: Array of slack bus indices
    :param pv: Array of pv bus indices
    :param pq: Array of pq bus indices
    :param step: Adaptation step
    :param approximation_order: order of the approximation {Natural, Arc, Pseudo arc}
    :param adapt_step: use adaptive step size?
    :param step_min: minimum step size
    :param step_max: maximum step size
    :param error_tol: Error tolerance
    :param tol: Solutions tolerance
    :param max_it: Maximum iterations
    :param stop_at:  Value of Lambda to stop at. It can be a number or {'NOSE', 'FULL'}
    :param control_q: Type of reactive power control
    :param qmax_bus: Array of maximum reactive power per node
    :param qmin_bus: Array of minimum reactive power per node
    :param original_bus_types: array of bus types
    :param base_overload_number: number of overloads in the base situation (used when stop_at=CpfStopAt.ExtraOverloads)
    :param verbose: Display additional intermediate information?
    :param call_back_fx: Function to call on every iteration passing the lambda parameter
    :param logger: Logger instance
    :return: CpfNumericResults instance
    Ported from MATPOWER
        Copyright (c) 1996-2015 by Power System Engineering Research Center (PSERC)
        by Ray Zimmerman, PSERC Cornell,
        Shrirang Abhyankar, Argonne National Laboratory, and Alexander Flueck, IIT
        $Id: runcpf.m 2644 2015-03-11 19:34:22Z ray $
        MATPOWER is covered by the 3-clause BSD License (see LICENSE file for details).
        See http://www.pserc.cornell.edu/matpower/ for more info.
    """

    ########################################
    # INITIALIZATION
    ########################################

    # scheduled transfer
    Sxfr = Sbus_target - Sbus_base
    nb = len(Sbus_base)
    lam = 0
    lam_prev = lam   # lam at previous step
    V_prev = V       # V at previous step
    continuation = True
    cont_steps = 0
    pvpq = np.r_[pv, pq]

    z = np.zeros(2 * nb + 1)
    z[2 * nb] = 1.0

    # compute total bus installed power
    total_installed_power = bus_installed_power.sum()

    # result arrays
    results = CpfNumericResults()

    # Simulation
    while continuation:
        cont_steps += 1

        # prediction for next step -------------------------------------------------------------------------------------
        V0, lam0, z = predictor(V=V,
                                Ibus=Ibus_base,
                                lam=lam,
                                Ybus=Ybus,
                                Sxfr=Sxfr,
                                pv=pv,
                                pq=pq,
                                step=step,
                                z=z,
                                Vprv=V_prev,
                                lamprv=lam_prev)

        # save previous voltage, lambda before updating
        V_prev = V.copy()
        lam_prev = lam

        # correction ---------------------------------------------------------------------------------------------------
        V, success, i, lam, normF, Scalc = corrector(Ybus=Ybus,
                                                     Ibus=Ibus_base,
                                                     Sbus=Sbus_base,
                                                     V0=V0,
                                                     pv=pv,
                                                     pq=pq,
                                                     lam0=lam0,
                                                     Sxfr=Sxfr,
                                                     Vprv=V_prev,
                                                     lamprv=lam_prev,
                                                     z=z,
                                                     step=step,
                                                     tol=tol,
                                                     max_it=max_it,
                                                     verbose=verbose)

        if success:

            # branch values --------------------------------------------------------------------------------------------
            # Branches current, loading, etc
            Vf = Cf * V
            Vt = Ct * V
            If = Yf * V  # in p.u.
            It = Yt * V  # in p.u.
            Sf = Vf * np.conj(If) * Sbase  # in MVA
            St = Vt * np.conj(It) * Sbase  # in MVA

            # Branch losses in MVA
            losses = Sf + St

            # Branch loading in p.u.
            loading = Sf.real / (branch_rates + 1e-9)

            # store series values --------------------------------------------------------------------------------------

            results.add(V, Scalc, Sf, St, lam, losses, loading, normF, success)

            if verbose:
                print('Step: ', cont_steps, ' Lambda prev: ', lam_prev, ' Lambda: ', lam)
                print(V)

            # did not check Q limits
            any_q_control_issue = False

            Qnew = Scalc.imag

            if verbose:
                print('Q controls Ok')

            if abs(lam) < 1e-8:
                # traced the full continuation curve
                if verbose:
                    print('\nTraced full continuation curve in ', cont_steps, ' continuation steps\n')
                continuation = False

            elif (lam < lam_prev) and (lam - step < 0):
                # next step will overshoot

                # modify step-size
                step = lam

                # disable step-adaptivity
                adapt_step = 0

            if adapt_step and continuation:

                # Adapt step size
                fx = np.r_[np.angle(V[pq]), np.abs(V[pvpq]), lam] - np.r_[np.angle(V0[pq]), np.abs(V0[pvpq]), lam0]
                cpf_error = np.linalg.norm(fx, np.Inf)

                if cpf_error == 0:
                    cpf_error = 1e-20

                if cpf_error < error_tol:
                    # Increase step size
                    step = step * error_tol / cpf_error
                    if step > step_max:
                        step = step_max

                else:
                    # Decrease step size
                    step = step * error_tol / cpf_error
                    if step < step_min:
                        step = step_min

            # call callback function
            if call_back_fx is not None:
                call_back_fx(lam)

        else:
            continuation = False
            if verbose:
                print('step ', cont_steps, ' : lambda = ', lam, ', corrector did not converge in ', i, ' iterations\n')

    return results


def run_cpf(Ybus, Cf, Ct, Yf, Yt, rates, base, Sbus, V, bus_installed_power, slack, pv, pq, nb):
    Ibus_base = np.zeros(nb, dtype=complex128)
    Ibus_target = np.zeros(nb, dtype=complex128)
    Sbus_target = Sbus * 2

    result = continuation_nr(Ybus, Cf, Ct, Yf, Yt, rates, base, Ibus_base, Ibus_target, Sbus, Sbus_target,
                             V, bus_installed_power, slack, pv, pq)
    return result

