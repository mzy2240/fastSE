# fastSE
# Copyright (C) 2022 Zeyu Mao

import numpy as np
from scipy.stats import chi2
from rich.console import Console

console = Console()
KVXOPT_ENABLE = True


def bdd_validation(results, m, n, chi2_prob_false=0.05):
    r = results['r']
    R_inv = results['R_inv']
    if KVXOPT_ENABLE:
        J = r.T * (R_inv * r)
        J = J[0]
    else:
        R_inv = R_inv.toarray()
        J = np.dot(r.T, np.dot(R_inv, r))
    test_thresh = chi2.ppf(1 - chi2_prob_false, m - n)
    if J <= test_thresh:
        res = True
        console.print(f"BDD passed with residue {J:.3f}", style="bold green")
    else:
        res = False
        console.print(f"BDD failed with residue {J:.3f}", style="bold red")
    return res, J
