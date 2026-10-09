"""
DRAPE: per-shot Fisher information for the over/under-rotation error (epsilon)
and the phase error (delta), as a function of the operating point delta.

Model (ground truth = the inline DRAPE document):
    The z-basis "success" probability of the DRAPE circuit
        U = Y(pi/2) [ X(-pi/2) X(pi/2) ]^k X(pi/2),   X/Y(+-pi/2)=Z(d) X/Y[+-(pi/2+e)] Z(d)
    is
        P_{|j>} = (cos^2 e / 2) [ (u+v+f cos d)^2 + (v+f sin d)^2 ]
    with
        u = (cos d + sin d) [ S_k - (1+sin e) S_{k-1} ]
        v = (1 - sin e) sin d  S_{k-1}
        f =  sin e             S_{k-1}
    and S_n is the Chebyshev polynomial of the 2nd kind,
        S_n(cos phi) = sin[(n+1) phi] / sin phi,
    evaluated at the characteristic angle
        cos phi = cos^2 d + sin e sin^2 d.

Fisher information of a single z-basis (Bernoulli) shot about a parameter xi:
        F_xi = N (dP/dxi)^2 / [ P (1-P) ],          sigma(xi_hat) >= 1/sqrt(F_xi).
Here N = number of shots (F scales linearly with N).

This module is pure NumPy (no plotting); do the plotting in the notebook that
imports it.  Requires: numpy.
"""

import numpy as np


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def S(n, cos_phi):
    """Chebyshev U_n(cos phi) = sin[(n+1) phi]/sin phi, with the phi->0 limit (= n+1)."""
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    small = np.abs(phi) < 1e-10
    return np.where(small, n + 1.0, np.sin((n + 1) * phi) / np.where(small, 1.0, np.sin(phi)))


def P_j(k, d, e):
    """z-basis success probability P_{|j>} for depth k, phase error d (delta), rotation error e (epsilon)."""
    cos_phi = np.cos(d) ** 2 + np.sin(e) * np.sin(d) ** 2
    Sk, Skm1 = S(k, cos_phi), S(k - 1, cos_phi)
    u = (np.cos(d) + np.sin(d)) * (Sk - (1 + np.sin(e)) * Skm1)
    v = (1 - np.sin(e)) * np.sin(d) * Skm1
    f = np.sin(e) * Skm1
    A = u + v + f * np.cos(d)
    B = v + f * np.sin(d)
    return np.cos(e) ** 2 / 2 * (A ** 2 + B ** 2)


# ----------------------------------------------------------------------
# Per-shot Fisher information (central finite differences; P is smooth in d and e)
# ----------------------------------------------------------------------
def fisher(k, d, e, wrt, N=1.0, h=1e-6):
    """F_wrt where wrt in {'delta','eps'}.  d, e may be scalars or arrays.

    Note: finite-difference step h is fine for the k shown here; for very large k
    (fringe period ~ pi/(k+1)) tighten h or switch to an analytic/complex-step
    derivative.
    """
    P = P_j(k, d, e)
    if wrt == "delta":
        dP = (P_j(k, d + h, e) - P_j(k, d - h, e)) / (2 * h)
    elif wrt == "eps":
        dP = (P_j(k, d, e + h) - P_j(k, d, e - h)) / (2 * h)
    else:
        raise ValueError("wrt must be 'delta' or 'eps'")
    return N * dP ** 2 / (P * (1 - P))


# ----------------------------------------------------------------------
# Closed-form values at the locus delta -> 0 (derived analytically; used to mark plots)
# ----------------------------------------------------------------------
def Fdd_locus(k, e, N=1.0):
    """F_delta,delta at delta=0.  -> 4 N (k+1)^2 as e -> 0 (Heisenberg in k)."""
    return 4 * N * np.cos(e) ** 2 * (1 + k * (1 - 2 * np.sin(e))) ** 2 / (1 + np.sin(e) ** 2)


def Fee_locus(e, N=1.0):
    """F_eps,eps at delta=0.  k-INDEPENDENT; -> 4 N e^2 as e -> 0 (SQL, vanishes at target)."""
    return 4 * N * np.sin(e) ** 2 / (1 + np.sin(e) ** 2)


# ----------------------------------------------------------------------
# Experimental control mapping (the knob you actually sweep):
#     delta = chi (beta0 - beta)   =>   beta/beta0 = 1 - delta/(chi*beta0)
# The locus delta=0 sits at beta/beta0 = 1 (the optimal DRAG coefficient).
# Only the product (chi*beta0) sets the horizontal scale.
# ----------------------------------------------------------------------
def beta_ratio(d, chi_beta0):
    """Map phase error delta -> DRAG sweep beta/beta0."""
    return 1.0 - d / chi_beta0


def delta_of_beta_ratio(r, chi_beta0):
    """Inverse map: DRAG sweep beta/beta0 -> phase error delta."""
    return chi_beta0 * (1.0 - r)


if __name__ == "__main__":
    # quick self-check (no plotting)
    e = 0.05
    print("F_dd(0):  k=1 ->", Fdd_locus(1, e), " k=11 ->", Fdd_locus(11, e))
    print("F_ee(0):  (k-independent) ->", Fee_locus(e))