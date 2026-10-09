"""
drape_estimator
===============
Estimate the over/under-rotation error eps and the optimal DRAG coefficient
beta_0 from a DRAPE beta-sweep, following Nguyen & Nguyen,
"Simultaneous calibration of rotation and phase errors in a single experiment"
(arXiv:2607.19187).

The recipe (Sec. 5 and 8 of notebooks/06_drape_estimator.ipynb):

1. `find_locus`      -- seed beta_0 from the beta at which the k-curves agree
                        most (minimum variance across k); the crossing height
                        cos^2(eps)/2 gives a calibration-free |eps|.
2. `calibrate_chi`   -- one-time free (eps, beta0, chi) fit near the locus; chi,
                        the beta -> delta slope, is a structural constant of the
                        pulse (Eq. 10 of the paper) and is then frozen.
3. `estimate_eps_beta0` -- least-squares fit of (eps, beta0) with chi frozen on
                        a window |beta - beta0| < half_width.
4. `extrapolate_window` -- repeat 3 on a ladder of windows and extrapolate the
                        window to zero, eps(hw) = eps_inf + c hw^2, which removes
                        the bias of the truncated delta(beta) map (Sec. 8).

Conventions: a sweep is an array P of shape (n_k, n_beta) holding P_{|j>} (the
population of the *measured* level, column 2 of the simulation files); row i
corresponds to repetition count ks[i]; beta is in physical units (the plotted
DRAG coefficient; see drape_lib.to_physical).

This module is a reconstruction: the file that the original analysis imported
(`drape_estimator.py`) was not kept in the working folder, only its byte-code.
Its two entry points, `run_on_npz` and `scan_window`, are rebuilt here from the
functions of notebook 06 so that notebooks 05 and 06 run end to end.
"""

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------- model ----

def chebyshev_S(k, cos_phi):
    """Chebyshev polynomial of the 2nd kind S_k(cos phi) = sin((k+1) phi)/sin phi; k+1 as phi -> 0."""
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    sin_phi = np.sin(phi)
    tiny = np.abs(sin_phi) < 1e-9
    return np.where(tiny, k + 1.0, np.sin((k + 1) * phi) / np.where(tiny, 1.0, sin_phi))


def drape_probability(eps, delta, k):
    """P_{|j>} of the DRAPE circuit for one k (paper Eq. 9). eps and delta may be arrays."""
    se, ce = np.sin(eps), np.cos(eps)
    sd, cd = np.sin(delta), np.cos(delta)
    cos_phi = cd ** 2 + se * sd ** 2
    Sk = chebyshev_S(k, cos_phi)
    Skm1 = chebyshev_S(k - 1, cos_phi)
    u = (cd + sd) * (Sk - (1 + se) * Skm1)
    v = (1 - se) * sd * Skm1
    f = se * Skm1
    return (ce ** 2 / 2.0) * ((u + v + f * cd) ** 2 + (v + f * sd) ** 2)


def drape_probability_linear(eps, delta, k):
    """Linear-in-delta model (paper Eq. 11)."""
    return np.cos(eps) ** 2 * (0.5 + delta * (1.0 + k * (1.0 - 2.0 * np.sin(eps))))


def delta_of_beta(beta, beta0, chi, chi2=0.0):
    """Phase map delta(beta) = chi (beta0 - beta) + chi2 (beta0 - beta)^2 (Eq. 10; chi2 only for calibration)."""
    dd = beta0 - np.asarray(beta, float)
    return chi * dd + chi2 * dd ** 2


def predict_curves(eps, beta0, chi, beta, ks, chi2=0.0, linear=False):
    """Model P stacked over the chosen k's: array (len(ks), n_beta)."""
    delta = delta_of_beta(beta, beta0, chi, chi2)
    model = drape_probability_linear if linear else drape_probability
    return np.array([model(eps, delta, int(k)) for k in np.atleast_1d(ks)])


# ------------------------------------------------------------- helpers ----

def ls_stderr(sol):
    """1-sigma parameter errors of a scipy least_squares solution, cov = s^2 (J^T J)^-1."""
    J = sol.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    s_sq = 2.0 * sol.cost / dof
    cov = s_sq * np.linalg.pinv(J.T @ J)
    return np.sqrt(np.abs(np.diag(cov)))


def select_k(P, ks, k_available):
    """Rows of P for the requested repetition counts (row i of P holds k_available[i])."""
    k_available = np.atleast_1d(k_available).astype(int)
    ks = np.atleast_1d(ks).astype(int)
    sorter = np.argsort(k_available)
    pos = np.clip(np.searchsorted(k_available, ks, sorter=sorter), 0, len(k_available) - 1)
    idx = sorter[pos]
    missing = ks[k_available[idx] != ks]
    if missing.size:
        raise ValueError(f"k(s) {list(missing)} not in available set {list(k_available)}")
    return P[idx]


# --------------------------------------------------------------- steps ----

def find_locus(beta, P, ks, k_available):
    """Crossing point: beta0 seed, crossing height and |eps| = arccos(sqrt(2 height))."""
    Psel = select_k(P, ks, k_available) if len(np.atleast_1d(ks)) >= 2 else P
    i0 = Psel.var(axis=0).argmin()
    beta0 = beta[i0]
    height = Psel[:, i0].mean()
    eps_from_height = np.arccos(np.sqrt(np.clip(2 * height, 0.0, 1.0)))
    return beta0, height, eps_from_height


def locus_errors(beta, P, ks, k_available):
    """1-sigma errors of the locus quantities: grid resolution for beta0, across-k spread for the height."""
    Psel = select_k(P, ks, k_available) if len(np.atleast_1d(ks)) >= 2 else P
    i0 = Psel.var(axis=0).argmin()
    beta0_err = abs(np.diff(beta).mean()) / np.sqrt(12)
    height = Psel[:, i0].mean()
    height_err = Psel[:, i0].std(ddof=0) / np.sqrt(Psel.shape[0])
    h = np.clip(2 * height, 1e-9, 1 - 1e-9)
    eps_err = height_err / np.sqrt(h * (1 - h))
    return beta0_err, height_err, eps_err


def calibrate_chi(beta, P_clean, ks, k_available, half_width_chi=0.45, chi_guess=0.057, quadratic=False):
    """One-time free fit of (eps, beta0, chi[, chi2]) near the locus. Returns (chi, chi_err[, chi2])."""
    beta0_seed, _, eps_seed = find_locus(beta, P_clean, ks, k_available)
    mask = np.abs(beta - beta0_seed) < half_width_chi
    bm, Pm = beta[mask], select_k(P_clean, ks, k_available)[:, mask]
    if quadratic:
        sol = least_squares(lambda x: (predict_curves(x[0], x[1], x[2], bm, ks, chi2=x[3]) - Pm).ravel(),
                            [eps_seed, beta0_seed, chi_guess, 0.0], method="lm", max_nfev=20000)
        return sol.x[2], ls_stderr(sol)[2], sol.x[3]
    sol = least_squares(lambda x: (predict_curves(x[0], x[1], x[2], bm, ks) - Pm).ravel(),
                        [eps_seed, beta0_seed, chi_guess], method="lm", max_nfev=20000)
    return sol.x[2], ls_stderr(sol)[2]


def estimate_eps_beta0(beta, P, ks, k_available, chi_frozen, half_width):
    """Fit (eps, beta0) with chi frozen on |beta - beta0_seed| < half_width. Returns a dict with 1-sigma errors."""
    beta0_seed, _, eps_seed = find_locus(beta, P, ks, k_available)
    mask = np.abs(beta - beta0_seed) < half_width
    bm, Pm = beta[mask], select_k(P, ks, k_available)[:, mask]
    sol = least_squares(lambda x: (predict_curves(x[0], x[1], chi_frozen, bm, ks) - Pm).ravel(),
                        [eps_seed, beta0_seed], method="lm", max_nfev=20000)
    err = ls_stderr(sol)
    return {"eps": sol.x[0], "eps_err": err[0], "beta0": sol.x[1], "beta0_err": err[1], "n_pts": int(mask.sum())}


def chi0_extrapolated(beta, P_clean, ks, k_available, hws_chi=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40)):
    """Locus slope chi_0 = lim_{hw -> 0} chi(hw) from quadratic-map calibrations on a window ladder."""
    hwc = np.asarray(hws_chi, float)
    chis = np.array([calibrate_chi(beta, P_clean, ks, k_available, h, quadratic=True)[0] for h in hwc])
    chi0 = np.linalg.lstsq(np.vstack([np.ones_like(hwc), hwc ** 2]).T, chis, rcond=None)[0][0]
    return chi0, hwc, chis


def extrap0(hws, y):
    """Window -> 0 intercept of y(hw) = y_inf + c hw^2; returns (y_inf, |quadratic - quartic intercept|, coefficients)."""
    hw = np.asarray(hws, float)
    c2 = np.linalg.lstsq(np.vstack([np.ones_like(hw), hw ** 2]).T, y, rcond=None)[0]
    c4 = np.linalg.lstsq(np.vstack([np.ones_like(hw), hw ** 2, hw ** 4]).T, y, rcond=None)[0]
    return c2[0], abs(c2[0] - c4[0]), c2


def fit_ladder(beta, P, ks, k_available, chi0, hws):
    """estimate_eps_beta0 across a window ladder; returns eps, beta0, eps_err, beta0_err arrays."""
    r = [estimate_eps_beta0(beta, P, ks, k_available, chi0, h) for h in hws]
    return (np.array([d["eps"] for d in r]), np.array([d["beta0"] for d in r]),
            np.array([d["eps_err"] for d in r]), np.array([d["beta0_err"] for d in r]))


def extrapolate_window(beta, P, ks, k_available, chi0, hws, P_clean=None, n_mc=200, seed=0):
    """
    Converged (eps, beta0) by the window -> 0 extrapolation of Sec. 8.  If a clean
    (noise-free) sweep is given, the statistical error is a Monte-Carlo at the
    noise level of P - P_clean, propagated through the same extrapolation.
    """
    eps_l, b0_l, _, _ = fit_ladder(beta, P, ks, k_available, chi0, hws)
    eps_inf, eps_sys, _ = extrap0(hws, eps_l)
    b0_inf, b0_sys, _ = extrap0(hws, b0_l)
    out = {"eps": eps_inf, "beta0": b0_inf, "eps_sys": eps_sys, "beta0_sys": b0_sys, "chi": chi0}
    if P_clean is not None:
        sigma = float((P - P_clean).std())
        rng = np.random.default_rng(seed)
        mc_eps, mc_b0 = [], []
        for _ in range(n_mc):
            Pb = P_clean + rng.normal(0.0, sigma, P_clean.shape)
            e, b, _, _ = fit_ladder(beta, Pb, ks, k_available, chi0, hws)
            mc_eps.append(extrap0(hws, e)[0]); mc_b0.append(extrap0(hws, b)[0])
        out.update(eps_stat=float(np.std(mc_eps)), beta0_stat=float(np.std(mc_b0)), sigma=sigma)
    return out


# --------------------------------------------------------- entry points ----

def scan_window(beta, Pdata, klist, chi, halfs=(0.05, 0.1, 0.2, 0.45, 0.8, 1.2), verbose=True):
    """Fixed-window estimates for a list of half-widths (the table printed in notebook 05)."""
    rows = []
    if verbose:
        print(f"{'half_width':>10} {'eps':>11} {'eps_std':>11} {'beta0':>11} {'n_pts':>7}")
    for hw in halfs:
        r = estimate_eps_beta0(beta, Pdata, klist, klist, chi, hw)
        rows.append((hw, r["eps"], r["eps_err"], r["beta0"], r["n_pts"]))
        if verbose:
            print(f"{hw:>10.3g} {r['eps']:>11.5f} {r['eps_err']:>11.5f} {r['beta0']:>11.5f} {r['n_pts']:>7d}")
    return rows


def run_on_npz(path, chi=None, half_width=0.45, p_column=2, beta_to_physical=None, verbose=True):
    """
    Full estimate on one simulation/measurement file with keys `beta_range`,
    `drape_pops` (noisy or measured) and optionally `y_fit` (clean), rows k = 1..n.
    chi is calibrated on y_fit when present (else on the data) unless given.
    Returns a dict (eps, beta0, chi, eps_sd, beta0_sd, ...).
    """
    d = np.load(path)
    beta = d["beta_range"]
    if beta_to_physical is not None:
        beta = beta_to_physical(beta)
    Pn = d["drape_pops"][:, :, p_column]
    Pc = d["y_fit"][:, :, p_column] if "y_fit" in d.files else None
    ks = np.arange(1, Pn.shape[0] + 1)
    if chi is None:
        chi = chi0_extrapolated(beta, Pc if Pc is not None else Pn, ks, ks)[0]
    r = estimate_eps_beta0(beta, Pn, ks, ks, chi, half_width)
    out = {"eps": r["eps"], "beta0": r["beta0"], "chi": chi, "eps_sd": r["eps_err"], "beta0_sd": r["beta0_err"],
           "half_width": half_width, "n_pts": r["n_pts"]}
    if verbose:
        print(f"eps = {out['eps']:.5f} +/- {out['eps_sd']:.5f} rad ({np.degrees(out['eps']):.3f} deg),  "
              f"beta0 = {out['beta0']:.5f} +/- {out['beta0_sd']:.5f},  chi = {chi:.5f}")
    return out
