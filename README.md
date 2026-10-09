# DRAPE — simultaneous calibration of rotation and phase errors in a single experiment

Code, data and figures for

> Tien D. Nguyen and Hung Q. Nguyen, *Simultaneous calibration of rotation and phase errors in a single experiment*, [arXiv:2607.19187](https://arxiv.org/abs/2607.19187) (2026).

A symmetric π/2 pulse on a weakly anharmonic qubit realises the gate 𝒳(π/2) = Z(δ) X(π/2 + ε) Z(δ), where the over/under-rotation ε is set by the pulse area alone and the phase error δ depends on ε. DRAPE (DRAG + amplified phase error) is the Ramsey-type sequence Y(π/2) [X(−π/2) X(π/2)]^k X(π/2) read at the fixed axis φ = π/2 while the DRAG coefficient β is swept: the k curves cross at β₀ (the phase correction, estimated with Heisenberg scaling in k) at the height cos²ε/2 (the rotation error, at the standard quantum limit). The experiment is on the |1⟩↔|2⟩ transition of qubit 109 of `ibm_brisbane` with 20 ns Gaussian pulses.

## Layout

| path | what |
|---|---|
| `drape_lib.py` | the model: four-level transmon Hamiltonian, Gaussian/DRAG pulses, pulse-level QuTiP simulations of DRAPE and of the rotation-error (ARE/AAE) sequence, and the three equivalent gate-level descriptions of DRAPE (unitary, explicit matrix, closed form in Chebyshev polynomials — Eq. 9 of the paper) |
| `drape_estimator.py` | the estimator: locate the crossing, calibrate χ (the β→δ slope, Eq. 10), fit (ε, β₀) with χ frozen, window→0 extrapolation; entry points `run_on_npz`, `scan_window` |
| `fisher_information.py` | per-shot Fisher information F_εε and F_δδ of the DRAPE circuit (Fig. 1b) |
| `utility.py` | IQ-discriminator helpers used by notebook 02b |
| `notebooks/` | the analysis, in order (below) |
| `data/measured/` | every measurement used, as plain `.npz` (index in `data/measured/README.json`) |
| `data/simulated/` | pulse-level simulations of the DRAPE and ARE sequences (see below) |
| `figures/` | the figure panels of the paper as exported by notebook 09 |

## Notebooks

| # | notebook | what it does | runtime |
|---|---|---|---|
| 01 | `01_pulse_demo` | builds one DRAG-corrected truncated Gaussian pulse and plots Ω_x, Ω_y | seconds |
| 02 | `02_extract_lambdas` | fits the drive strengths λ₀, λ₁, λ₂ = 0.526, 0.621, 0.753 of the 0–1, 1–2 and 2–3 transitions by overlaying QuTiP Rabi simulations on the measured Rabi oscillations | minutes |
| 02b | `02b_rabi12_discriminator` | the 1–2 Rabi oscillation of the corrected pulse (job `cyb4se57…`) and the post-selected IQ clusters of |1⟩ and |2⟩ | seconds |
| 03 | `03_drape_analytic_models` | checks that the unitary, explicit-matrix, closed-form and linearised DRAPE models agree | seconds |
| 04 | `04_drape_pulse_simulation` | the pulse-level simulation of the DRAPE β-sweep (`drape_lib.simulateDRAPE`) with the detuning γ calibrated on the measured sweep, and the noise injection at the measured noise level | **hours** for the full sweep; the cached results are in `data/simulated/` |
| 05 | `05_parameter_extraction` | linear-stage then full closed-form fits of (ε, χ, β₀) to a simulated sweep; writes `data/simulated/drape_fit_params.npz` | minutes |
| 06 | `06_drape_estimator` | the estimator of the paper step by step, with 1σ errors and the window→0 extrapolation (ε = 0.0174 rad = 0.997°, β₀ = −0.3894) | ~10 min (Monte-Carlo) |
| 07 | `07_chi` | numerical χ = (λ_j/Δ₂) ∫ Ω̇_x cos θ dt for the 20 ns pulse (−4.98·10⁻², theory) vs the fitted −5.03·10⁻² | seconds |
| 08 | `08_aae_validation` | the ARE/AAE sequence simulated with the bare, ε-corrected, β₀-corrected and fully corrected pulse (`drape_lib.simulateAAE`) | ~15 min |
| 09 | `09_paper_figures` | every panel of the paper's figures (`figures/*.pdf`) | minutes |

Run them from the repository root or from `notebooks/` (each notebook moves to the root in its first cell). Outputs are kept in the notebooks, so the numbers can be read without running anything.

## Which data is measured and which is simulated

The paper combines measurements on `ibm_brisbane` q109 (taken between Oct 2024 and Feb 2025 through Qiskit Pulse, which has since been retired from the IBM platform) with pulse-level simulations whose only free parameter, the drift detuning γ = −3.57·10⁻⁴ (relative to Δ₂), was fitted to the measured DRAPE sweep (`drape_results_fit_various_gamma.npz` is that scan). To make the provenance of every curve explicit:

* **measured** (`data/measured/`): the three Rabi oscillations behind λ₀, λ₁, λ₂; the APE fringes of Fig. 2(c) (bare vs DRAG, k = 1, 3, 5, 7); the ARE runs with the bare 0.2512 pulse, with DRAG-Y only, with the rescaled amplitude 0.2365 without DRAG, and with amplitude 0.2365 + DRAG (three repeats); and the DRAPE sweep `drape_50pts_-1.0_-0.2_0.2365_asterisk.npz` (k = 1, 3, …, 11, 50 β values, amplitude 0.2365), which fixes γ and the empirical noise level (`EMPIRICAL_STD` in notebooks 04 and 09). Every job id is listed in `data/measured/README.json`; the raw single-shot IQ data of every job are in the [PulsatingPulseShop](https://github.com/ngdnhtien/PulsatingPulseShop) repository (release `v1.0-data`).
* **simulated** (`data/simulated/`): `drape/<amplitude>/drape_sim_*.npz` are `simulateDRAPE` sweeps at the uncorrected amplitude 0.2390 or 0.2512 (and 0.2300 for under-rotation), with 500 β values and k = 1…11 or 1…21; `drape_pops` = `y_fit` (the clean simulation) + Gaussian noise at the measured noise level, `y_fit` is the clean simulation. `aae_21pts_0_40_0.2390_0.2363.npz` is `simulateAAE` with the bare (0.2390) and the DRAPE-corrected (0.2364) pulse. The DRAPE sweep of Fig. 1(c) and the ARE panel of Fig. 2(d) are drawn from these files (`09_paper_figures.ipynb`), and the estimates ε̂ = 0.997° ± 0.395°, β̂₀ = −0.3894 ± 0.0002 are the estimator's output on `drape/0.2390/drape_sim_500pts_-0.4_-0.37_0.2390_k1to21.npz` and `…_-2.0_1.25_0.2390_k1to11.npz` (notebook 06).

## Reproducing the numbers of the paper

| quantity | paper | where |
|---|---|---|
| λ₀, λ₁, λ₂ | 0.526, 0.621, 0.753 | notebook 02 → `data/measured/lambda_values.npz` |
| χ (theory / empirical) | −4.99·10⁻² / −5.03·10⁻² | notebook 07; `drape_estimator.calibrate_chi` in notebook 06 (the sign depends on the convention δ = χ(β₀ − β) used in the code) |
| ε̂, β̂₀ | 0.997° ± 0.395°, −0.3894 ± 0.0002 | notebook 06, Sections 3, 5 and 8 |
| Fisher information, Fig. 1(b) | F_δδ → 4N(k+1)², F_εε → 4Nε² | `fisher_information.py`, notebook 09 |
| APE δ before/after | 2.521° → 0.0052° | notebook 09, Fig. 2(c) cell, from `data/measured/ape_data.npz` |
| ARE ε before/after | 0.997° → −0.007° | notebooks 08 and 09, Fig. 2(d) |

## Installation

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib, qutip, scikit-learn, jupyter
jupyter lab notebooks/
```

Python ≥ 3.10; QuTiP 5 (the simulations use `qutip.sesolve` with time-dependent coefficients). No Qiskit is needed — the hardware runs are not reproducible on the IBM platform any more, and only their results are used here.

## Citation

```bibtex
@article{nguyen2026drape,
  title   = {Simultaneous calibration of rotation and phase errors in a single experiment},
  author  = {Nguyen, Tien D. and Nguyen, Hung Q.},
  journal = {arXiv preprint arXiv:2607.19187},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`. Tien D. Nguyen, 2026.
