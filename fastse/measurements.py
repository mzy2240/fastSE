# fastSE
# Copyright (C) 2022 Zeyu Mao

import numpy as np


class StateEstimationInput:
    """State estimation measurements object
    Original author: Santiago Peñate Vera
    Modified by Zeyu Mao
    """

    def __init__(self):
        """
        State estimation inputs constructor
        """

        # Node active power measurements vector of pointers
        self.p_inj = np.array([])

        # Node  reactive power measurements vector of pointers
        self.q_inj = np.array([])

        # Branch active power measurements vector of pointers
        self.p_flow = np.array([])

        # Branch reactive power measurements vector of pointers
        self.q_flow = np.array([])

        # Branch current module measurements vector of pointers
        self.i_flow = np.array([])

        # Node voltage module measurements vector of pointers
        self.vm_m = np.array([])

        # nodes without power injection measurements
        self.p_inj_idx = np.array([])

        # branches without power measurements
        self.p_flow_idx = np.array([])

        # nodes without reactive power injection measurements
        self.q_inj_idx = np.array([])

        # branches without reactive power measurements
        self.q_flow_idx = np.array([])

        # branches without current measurements
        self.i_flow_idx = np.array([])

        # nodes without voltage module measurements
        self.vm_m_idx = np.array([])

        # nodes without power injection measurement weights
        self.p_inj_weight = np.array([])

        # branches without power measurements weights
        self.p_flow_weight = np.array([])

        # nodes without reactive power injection measurements weights
        self.q_inj_weight = np.array([])

        # branches without reactive power measurements weights
        self.q_flow_weight = np.array([])

        # branches without current measurements weights
        self.i_flow_weight = np.array([])

        # nodes without voltage module measurements weights
        self.vm_m_weight = np.array([])

        # total measurements
        self.measurements = None

    def consolidate(self):
        """
        consolidate the measurements into "measurements" and "sigma"
        :return: measurements, sigma
        """

        measurements = np.r_[self.p_flow, self.p_inj, self.q_flow, self.q_inj, self.i_flow, self.vm_m]
        weights = np.r_[
            self.p_flow_weight, self.p_inj_weight, self.q_flow_weight, self.q_inj_weight, self.i_flow_weight, self.vm_m_weight]
        if len(measurements) != len(weights):
            raise Exception("Measurements and weights do not match!")
        self.measurements = measurements
        return measurements, weights

    def clear(self):
        """
        Clear
        """
        self.p_inj.clear()
        self.p_flow.clear()
        self.q_inj.clear()
        self.q_flow.clear()
        self.i_flow.clear()
        self.vm_m.clear()

        self.p_inj_idx.clear()
        self.p_flow_idx.clear()
        self.q_inj_idx.clear()
        self.q_flow_idx.clear()
        self.i_flow_idx.clear()
        self.vm_m_idx.clear()

        self.p_inj_weight.clear()
        self.p_flow_weight.clear()
        self.q_inj_weight.clear()
        self.q_flow_weight.clear()
        self.i_flow_weight.clear()
        self.vm_m_weight.clear()