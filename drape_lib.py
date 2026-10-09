"""
drape_lib
=========
Shared definitions for DRAPE protocol simulations on a transmon qubit.

β convention
------------
Internally, β is always stored and passed in **normalized units** — the
form the pulse function (`Omega_y`) multiplies the DRAG correction by.
Physical-unit β (what you put on a plot's x-axis) is

    beta_physical = beta_normalized * (-2π · transmon_anharmonicity)

Use `to_physical()` / `to_normalized()` at I/O boundaries; never sprinkle
the conversion factor in the middle of code.

Module layout
-------------
- Constants                       (anharmonicity, default pulse geometry)
- β-unit conversion               (to_physical, to_normalized)
- Hamiltonian and pulse functions (Hamiltonian, Omega_x, Omega_y, Omega, add_pulse)
- Pulse-level simulations         (simulateDRAPE, simulateAAE, lossDRAPE)
- Gate-level DRAPE                (3-level unitaries, explicit matrices, closed form)
"""

import numpy as np
import qutip as qt


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
TRANSMON_ANHARMONICITY = 0.3071                            # GHz
DELTA_TRANSMON         = -2 * np.pi * TRANSMON_ANHARMONICITY  # rad/ns

DEFAULT_TG        = 20    # ns, gate duration
DEFAULT_SIGMA     = 5     # ns, truncated Gaussian width  (== tg/4)
DEFAULT_REP_RANGE = [1, 3, 5, 7, 9, 11]

DATA_DIR = './data/'


# ----------------------------------------------------------------------
# β-unit conversion
# ----------------------------------------------------------------------
def to_physical(beta_normalized):
    """Normalized β (what the pulse function consumes) → physical β
    (what you plot)."""
    return np.asarray(beta_normalized) * DELTA_TRANSMON


def to_normalized(beta_physical):
    """Physical β → normalized β."""
    return np.asarray(beta_physical) / DELTA_TRANSMON


# ----------------------------------------------------------------------
# Hamiltonian (4-level transmon)
# ----------------------------------------------------------------------
def Hamiltonian(lambdas, delta, Delta):
    """4-level transmon Hamiltonian (Motzoi: 4 levels are sufficient).

    Returns (H0, H1, H1_dag, H2, H2_dag, H3, H3_dag, kets, bras).
    """
    kets = [qt.basis(4, i) for i in range(4)]
    bras = [k.dag() for k in kets]
    ket0, ket1, ket2, ket3 = kets
    bra0, bra1, bra2, bra3 = bras

    l0, l1, l2 = lambdas[0], lambdas[1], lambdas[2]

    H0 = (delta            * ket1 * bra1
          + (2*delta - Delta)   * ket2 * bra2
          + (3*delta - 3*Delta) * ket3 * bra3)

    H1     = (l0/2) * (ket0 * bra1)
    H1_dag = (l0/2) * (ket1 * bra0)
    H2     = (l1/2) * (ket1 * bra2)
    H2_dag = (l1/2) * (ket2 * bra1)
    H3     = (l2/2) * (ket2 * bra3)
    H3_dag = (l2/2) * (ket3 * bra2)

    return H0, H1, H1_dag, H2, H2_dag, H3, H3_dag, kets, bras


# ----------------------------------------------------------------------
# Pulse waveforms
# ----------------------------------------------------------------------
def Omega_x(t, args):
    """Real part of the pulse: truncated Gaussian (main DRAG amplitude)."""
    for item, p in args.items():
        if item == 'clock':
            continue
        if t <= p['start'] or t >= (p['start'] + p['tg']):
            continue
        t_local = t - p['start']
        gauss   = np.exp(-(t_local - p['tg']/2)**2 / (2*p['sigma']**2))
        offset  = np.exp(-p['tg']**2 / (8*p['sigma']**2))
        return p['A'] * (gauss - offset) / (1 - offset)
    return 0


def Omega_y(t, args):
    """Imaginary part / DRAG correction.

    Note: ``p['beta']`` must be in *normalized* units (see module docstring).
    """
    for item, p in args.items():
        if item == 'clock':
            continue
        if t <= p['start'] or t >= (p['start'] + p['tg']):
            continue
        t_local = t - p['start']
        tg, sigma, A = p['tg'], p['sigma'], p['A']

        norm   = 1 - np.exp(-tg**2 / (8*sigma**2))
        gauss  = np.exp(-(t_local - tg/2)**2 / (2*sigma**2))
        offset = np.exp(-tg**2 / (8*sigma**2))

        G  = A * (gauss - offset) / norm
        dG = -A * (t_local - tg/2) / sigma**2 * gauss / norm

        # Raised-cosine window: enforces zero endpoints
        w  = 0.5 * (1 - np.cos(2*np.pi*t_local/tg))
        dw = (np.pi/tg) * np.sin(2*np.pi*t_local/tg)
        return p['beta'] * (dG * w + G * dw)
    return 0


def Omega(t, args):
    """Complex pulse envelope."""
    return Omega_x(t, args) - 1j * Omega_y(t, args)


def add_pulse(pulse_train, pulse_name, *, start, tg, sigma, A, beta):
    """Append one pulse to ``pulse_train`` and advance the clock.

    ``beta`` is in normalized units.
    """
    pulse_train[pulse_name] = {
        'start': start, 'tg': tg, 'sigma': sigma, 'A': A, 'beta': beta
    }
    pulse_train['clock'] += tg


# ----------------------------------------------------------------------
# Pulse-level simulations
# ----------------------------------------------------------------------
def _build_qutip_H(lambdas, gamma):
    """Build the time-dependent Hamiltonian list for qt.sesolve()."""
    Delta = DELTA_TRANSMON
    delta = Delta - gamma * Delta
    H0, H1, H1d, H2, H2d, H3, H3d, kets, bras = Hamiltonian(lambdas, delta, Delta)
    H = [H0,
         [H1,  lambda t, args: np.conjugate(Omega(t, args))],
         [H1d, lambda t, args:              Omega(t, args)],
         [H2,  lambda t, args: np.conjugate(Omega(t, args))],
         [H2d, lambda t, args:              Omega(t, args)],
         [H3,  lambda t, args: np.conjugate(Omega(t, args))],
         [H3d, lambda t, args:              Omega(t, args)]]
    return H, kets, bras


def simulateDRAPE(gamma, rep_range, betas_normalized, A0, lambdas, *,
                  ell=0, tg=DEFAULT_TG, sigma=DEFAULT_SIGMA, verbose=False):
    """Pulse-level simulation of the DRAPE protocol.

    Parameters
    ----------
    gamma : float
        Detuning fluctuation parameter; sets ``delta = Delta - gamma*Delta``.
    rep_range : sequence of int
        Values of ``rep`` in the (U_minus U_plus)^rep block.
    betas_normalized : 1D array
        DRAG coefficients to sweep, in normalized units.
    A0 : float
        Pulse amplitude (uncorrected Rabi calibration).
    lambdas : array-like
        Three transmon coupling lambdas.
    ell : int, default 0
        Number of extra U_plus pulses appended after the DRAPE block (the
        "rotation amplification" extension). ``ell=0`` is the standard DRAPE.

    Returns
    -------
    pops : ndarray, shape (len(rep_range), len(betas_normalized), 3)
        Populations of |0>, |1>, |2> at the end of each pulse train.
    """
    H, kets, _ = _build_qutip_H(lambdas, gamma)
    projectors = [kets[i] * kets[i].dag() for i in range(4)]
    psi0       = kets[1]

    pops = []
    for rep in rep_range:
        pops_vs_beta = []
        for beta in betas_normalized:
            A0_phi = A0 * np.exp(1j * np.pi/2)  # DRAPE protocol fixes phase = π/2

            pulse_train = {'clock': 0}
            add_pulse(pulse_train, 'pulse_init',
                      start=pulse_train['clock'], tg=tg, sigma=sigma, A=A0, beta=beta)

            counter = 0
            for _ in range(rep):                # (U_minus U_plus)^rep
                add_pulse(pulse_train, f'pulse_{counter}',
                          start=pulse_train['clock'], tg=tg, sigma=sigma, A=A0, beta=beta)
                counter += 1
                add_pulse(pulse_train, f'pulse_{counter}',
                          start=pulse_train['clock'], tg=tg, sigma=sigma, A=-A0, beta=beta)
                counter += 1

            for _ in range(4 * ell):            # rotation-amplification extension
                add_pulse(pulse_train, f'pulse_{counter}',
                          start=pulse_train['clock'], tg=tg, sigma=sigma, A=A0, beta=beta)
                counter += 1

            add_pulse(pulse_train, 'map_phase_to_pop',
                      start=pulse_train['clock'], tg=tg, sigma=sigma, A=-A0_phi, beta=beta)

            times   = np.linspace(0, pulse_train['clock'], 10000)
            options = {'store_states': False, 'atol': 1e-10, 'rtol': 1e-10}   # QuTiP 5 options dict
            result  = qt.sesolve(H, psi0, times, e_ops=projectors,
                                 args=pulse_train, options=options)
            pops_vs_beta.append([result.expect[0][-1],
                                 result.expect[1][-1],
                                 result.expect[2][-1],
                                 result.expect[3][-1]])
            if verbose:
                print(f'rep={rep}, beta={beta:.4g}, clock={pulse_train["clock"]}')
        pops.append(pops_vs_beta)
    return np.array(pops)


def lossDRAPE(gamma, pop_data, rep_range, betas_normalized, A0, lambdas):
    """MSE loss for fitting gamma to DRAPE population data."""
    try:
        sim = simulateDRAPE(gamma, rep_range, betas_normalized, A0, lambdas)
        return np.mean((sim - pop_data) ** 2)
    except Exception as e:
        print('simulateDRAPE error:', e)
        return 1e6


def simulateAAE(gamma, N_values, A0, lambdas, beta_normalized, *,
                tg=DEFAULT_TG, sigma=DEFAULT_SIGMA):
    """Amplifying-Amplitude-Error protocol.

    Returns the |2> population at the end of each train (2N+1 U_plus pulses).
    """
    H, kets, _ = _build_qutip_H(lambdas, gamma)
    proj2      = kets[2] * kets[2].dag()
    psi0       = kets[1]

    pops = []
    for N in N_values:
        pulse_train = {'clock': 0}
        add_pulse(pulse_train, 'pulse_init',
                  start=pulse_train['clock'], tg=tg, sigma=sigma, A=A0, beta=beta_normalized)
        counter = 0
        for _ in range(N):
            add_pulse(pulse_train, f'pulse_{counter}',
                      start=pulse_train['clock'], tg=tg, sigma=sigma, A=A0, beta=beta_normalized)
            counter += 1
            add_pulse(pulse_train, f'pulse_{counter}',
                      start=pulse_train['clock'], tg=tg, sigma=sigma, A=A0, beta=beta_normalized)
            counter += 1

        times   = np.linspace(0, pulse_train['clock'], 10000)
        options = {'store_states': False, 'atol': 1e-10, 'rtol': 1e-10}   # QuTiP 5 options dict
        result  = qt.sesolve(H, psi0, times, e_ops=[proj2],
                             args=pulse_train, options=options)
        pops.append(result.expect[0][-1])
    return np.array(pops)


# ----------------------------------------------------------------------
# Gate-level DRAPE (3-level Hilbert space)
# ----------------------------------------------------------------------
# Generalized Pauli matrices acting on the {|1>, |2>} subspace, padded
# with a |0> row/column so they act in the 3-level basis.
#: 3 -> the gates act on {|1>,|2>} inside the 3-level space (notebook 03, 05);
#: 2 -> plain 2x2 Pauli matrices on the {|j>,|j+1>} qubit (the Bloch-sphere pictures of notebook 09).
GATE_DIM = 3

def Lx12():
    if GATE_DIM == 2:
        return qt.sigmax()
    return qt.Qobj(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))

def Ly12():
    if GATE_DIM == 2:
        return qt.sigmay()
    return qt.Qobj(np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]))

def Lz12():
    if GATE_DIM == 2:
        return qt.sigmaz()
    return qt.Qobj(np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]))

# --- Unitary form (qutip expm) --------------------------------------------
def overrotated_Xp2(epsilon, sign):
    s = -1 if sign == 'plus' else +1
    return (s * 1j * (np.pi/2 + epsilon) / 2 * Lx12()).expm()

def overrotated_Yp2(epsilon, sign):
    s = -1 if sign == 'plus' else +1
    return (s * 1j * (np.pi/2 + epsilon) / 2 * Ly12()).expm()

def Z_phase(delta):
    return (- 1j * delta / 2 * Lz12()).expm()

def X_p2(epsilon, delta, sign):
    return Z_phase(delta) * overrotated_Xp2(epsilon, sign) * Z_phase(delta)

def Y_p2(epsilon, delta, sign):
    return Z_phase(delta) * overrotated_Yp2(epsilon, sign) * Z_phase(delta)


def DRAPE_unitary_probs(a0, p0, rep):
    """DRAPE end-state populations using the unitary (qutip expm) gates."""
    ket1 = qt.basis(3, 1)
    ket2 = qt.basis(3, 2)
    psi = X_p2(a0, p0, 'plus') * ket1
    for _ in range(rep):
        psi = X_p2(a0, p0, 'minus') * X_p2(a0, p0, 'plus') * psi
    psi = Y_p2(a0, p0, 'plus') * psi
    return (abs(ket1.dag() * psi))**2, (abs(ket2.dag() * psi))**2

def DRAPE_unitary_probs_beta(b, a0, xi0, b0, rep):
    """DRAPE end-state populations using the unitary (qutip expm) gates."""
    ket1 = qt.basis(3, 1)
    ket2 = qt.basis(3, 2)
    p0 = xi0*(b-b0)
    psi = X_p2(a0, p0, 'plus') * ket1
    for _ in range(rep):
        psi = X_p2(a0, p0, 'minus') * X_p2(a0, p0, 'plus') * psi
    psi = Y_p2(a0, p0, 'plus') * psi
    return (abs(ket1.dag() * psi))**2, (abs(ket2.dag() * psi))**2


# --- Explicit matrix form -------------------------------------------------
def Xpp2_explicit(eps, delta):
    A = np.cos(eps/2) - np.sin(eps/2)
    B = np.cos(eps/2) + np.sin(eps/2)
    return (1/np.sqrt(2)) * qt.Qobj([
        [np.sqrt(2), 0, 0],
        [0, A, -1j*B*np.exp(1j*delta)],
        [0, -1j*B*np.exp(1j*delta), A*np.exp(1j*2*delta)]
    ])

def Xmp2_explicit(eps, delta):
    A = np.cos(eps/2) - np.sin(eps/2)
    B = np.cos(eps/2) + np.sin(eps/2)
    return (1/np.sqrt(2)) * qt.Qobj([
        [np.sqrt(2), 0, 0],
        [0, A, 1j*B*np.exp(1j*delta)],
        [0, 1j*B*np.exp(1j*delta), A*np.exp(1j*2*delta)]
    ])

def Ypp2_explicit(eps, delta):
    A = np.cos(eps/2) - np.sin(eps/2)
    B = np.cos(eps/2) + np.sin(eps/2)
    return (1/np.sqrt(2)) * qt.Qobj([
        [np.sqrt(2), 0, 0],
        [0, A, -B*np.exp(1j*delta)],
        [0, B*np.exp(1j*delta), A*np.exp(1j*2*delta)]
    ])

def Ymp2_explicit(eps, delta):
    A = np.cos(eps/2) - np.sin(eps/2)
    B = np.cos(eps/2) + np.sin(eps/2)
    return (1/np.sqrt(2)) * qt.Qobj([
        [np.sqrt(2), 0, 0],
        [0, A, B*np.exp(1j*delta)],
        [0, -B*np.exp(1j*delta), A*np.exp(1j*2*delta)]
    ])

def XXk(eps, delta, k):
    """Closed form of (X_minus X_plus)^k."""
    cosphi = np.cos(delta)**2+np.sin(eps)*np.sin(delta)**2
    phi = np.arccos(cosphi)
    S_km1   = np.sin(k*phi)     / np.sin(phi) 
    S_km2 = np.sin((k-1)*phi) / np.sin(phi)
    pref  = np.exp(1j * k * 2 * delta)

    m11 = S_km1*np.exp(-1j*delta)*(np.cos(delta)+1j*np.sin(eps)*np.sin(delta))-S_km2
    m22 = S_km1*np.exp(+1j*delta)*(np.cos(delta)-1j*np.sin(eps)*np.sin(delta))-S_km2
    off = S_km1 * np.cos(eps) * np.sin(delta)

    return pref * qt.Qobj([
        [1/pref, 0, 0],
        [0, m11, -off],
        [0, off,  m22]
    ])

def DRAPE_matrix_probs(a0, p0, k):
    """DRAPE end-state populations using the explicit matrix forms."""
    ket1 = qt.basis(3, 1)
    ket2 = qt.basis(3, 2)
    psi  = ket1
    psi  = Xpp2_explicit(a0, p0) * psi
    psi  = XXk(a0, p0, k) * psi
    psi  = Ypp2_explicit(a0, p0) * psi
    return (abs(ket1.dag() * psi))**2, (abs(ket2.dag() * psi))**2

# --- Closed-form / parametric --------------------------------------------
#
# Return convention: every closed-form function returns ``(p_target, p_other)``
# where ``p_target`` is the |2⟩ population (the quantity fit against the data)
# and ``p_other = 1 - p_target`` is the |1⟩ population. At δ = 0 with no
# over-rotation, ``p_target = cos²(ε)/2``.

def DRAPE_closed_form(epsilon, delta, k):
    """Closed-form DRAPE end-state populations.

    Returns (P|2⟩, P|1⟩).
    """
    cosphi = np.cos(delta)**2 + np.sin(epsilon) * np.sin(delta)**2
    # Clip protects against tiny floating-point overshoots that would make
    # arccos return NaN.
    phi    = np.arccos(np.clip(cosphi, -1.0, 1.0))

    # S_k = sin((k+1)φ) / sin(φ).  As φ → 0 this tends to (k+1), but the
    # naïve form gives 0/0.  np.sinc(x) = sin(πx)/(πx) is well-defined at 0,
    # so we use the identity  sin(nφ)/sin(φ) = n · sinc(nφ/π) / sinc(φ/π).
    sinc_phi = np.sinc(phi / np.pi)
    S_k      = (k + 1) * np.sinc((k + 1) * phi / np.pi) / sinc_phi
    S_km1    =  k      * np.sinc( k      * phi / np.pi) / sinc_phi

    u  = (np.cos(delta) + np.sin(delta)) * (S_k - (1 + np.sin(epsilon)) * S_km1)
    v1 = (1 - np.sin(epsilon)) * np.sin(delta) * S_km1
    v2 = np.sin(epsilon) * S_km1

    p_target = 0.5 * np.cos(epsilon)**2 * (
        (u + v1 + v2 * np.cos(delta))**2 + (v1 + v2 * np.sin(delta))**2)
    return p_target, 1 - p_target


def DRAPE_closed_form_linear(epsilon, delta, k):
    """Linear-in-δ expansion of the closed form.

    Returns (P|2⟩, P|1⟩). Used for the linear-stage fit in notebook 04.
    """
    cos_eps2 = np.cos(epsilon)**2
    p_target = (cos_eps2/2) * (1 + 2*delta * (1 + k*(1 - 2*np.sin(epsilon))))
    return p_target, 1 - p_target


def DRAPE_closed_form_beta(beta, k, eps0, xi0, beta0):
    """Closed form reparametrized for ``curve_fit``: ``δ = xi₀·(β - β₀)``.

    Returns the **|2⟩ population only** (scalar, or broadcast over arrays).
    This signature is what stage 2 of notebook 04 fits against.
    """
    delta = xi0 * (beta - beta0)
    p_target, _ = DRAPE_closed_form(eps0, delta, k)
    return p_target