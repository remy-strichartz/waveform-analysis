#!/usr/bin/env python3
"""
optimal_filter.py
=================
Standalone optimal (Wiener) filter analysis for real colored noise.

Estimates the noise power spectrum N(f) from the pre-pulse baseline, then forms
A_hat = Re[sum X(f) S*(f)/N(f)] / sum |S(f)|^2/N(f), weighting each frequency by
its signal-to-noise.  With white noise N(f) is flat and this is a matched filter;
with colored noise it down-weights the noisy frequencies.

The amplitude is evaluated once per event at the trigger's own sub-sample time
(the window is shifted by the fractional peak offset so the pulse aligns with the
template), which makes the estimate unbiased -- no maximization over trial lags.

    python optimal_filter.py --input run00270_ch0.h5 --save-plots
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

import mv_pipeline as P

logger = logging.getLogger("optimal_filter")


def _of_basis(prep: P.Prepared, psd: np.ndarray, config: P.Config):
    """Frequency-domain optimal-filter ingredients shared by the amplitude, chi2
    and resolution routines: window length L, zero-mean signal FFT S(f), the
    matched weight conj(S)/N, and the normalization sum |S|^2/N."""
    L = config.template_pre + config.template_post
    S = np.fft.fft(prep.template - prep.template.mean())   # zero-mean: DC-blind
    weight = np.conj(S) / psd
    denom = float(np.sum((np.abs(S) ** 2) / psd).real)
    return L, S, weight, denom


_SHIFT_PAD = 2      # cubic-resampler taps reach 1 left / 2 right of the sample


def of_aligned_windows(prep: P.Prepared, config: P.Config) -> np.ndarray:
    """Per-event windows sub-sample aligned onto the template, so they are directly
    comparable to it frequency-by-frequency.  Compute once and pass to the amplitude /
    chi2 / residual routines below (each falls back to computing its own).

    The window is cut at the trigger's own sub-sample WINDOW START (events["window_start"]
    = where the template best overlays the record), which is exactly the alignment the
    amplitude estimate assumes.

    The cut is PADDED by _SHIFT_PAD samples on each side and cropped back after the shift.
    _shift zero-fills any resampling tap that runs off the array edge, so an unpadded
    window had its LAST sample forced to zero for every event with a non-integer shift --
    i.e. essentially all of them.  The template is still 1-4% of peak there, so that zero
    was a systematic data-minus-template step at the window edge; in the PSD-whitened
    frequency domain a one-sample step is weighted heavily wherever N(f) is small, and it
    inflated the median per-event chi2 on run00270_ch0 from 0.80 to 0.95.  build_template
    already pads for exactly this reason."""
    L = config.template_pre + config.template_post
    length = prep.corrected.shape[1]
    pad = _SHIFT_PAD
    ws = prep.events["window_start"].to_numpy()
    rows = np.arange(prep.corrected.shape[0])
    # Clip so the PADDED cut stays inside the record, and derive the sub-sample residual
    # from the CLIPPED start (not the raw one) so the shift stays consistent with the window
    # actually cut.  Otherwise an edge pulse (within a template length of either record end)
    # would be mis-aligned by the whole samples the clip removed.  For such a clipped event
    # `frac` may leave [0, 1); _shift resamples any shift correctly.
    hi = max(length - L - pad, pad)
    int_start = np.clip(np.floor(ws).astype(np.int64), pad, hi)
    frac = ws - int_start
    wide = P.extract_at(P.triggered(prep), rows, int_start - pad, L + 2 * pad)
    return P._shift(wide, frac)[:, pad:pad + L]


def optimal_filter_amplitude(prep: P.Prepared, psd: np.ndarray, config: P.Config,
                             windows: np.ndarray | None = None) -> np.ndarray:
    """Wiener amplitude per event, evaluated at the trigger's sub-sample time."""
    _, _, weight, denom = _of_basis(prep, psd, config)
    if windows is None:
        windows = of_aligned_windows(prep, config)
    return np.real(np.fft.fft(windows, axis=1) @ weight) / denom


def optimal_filter_chi2(prep: P.Prepared, psd: np.ndarray, config: P.Config,
                        amplitude: np.ndarray,
                        windows: np.ndarray | None = None) -> np.ndarray:
    """Per-event reduced chi2 of the optimal-filter fit: the noise-weighted
    frequency-domain residual of the data against amplitude*template,
    chi2 = sum_f |X(f) - A.S(f)|^2 / N(f), divided by the degrees of freedom.

    For a clean pulse that matches the template in Gaussian noise this follows a
    reduced-chi2 distribution centered on 1; it rises well above 1 for pile-up, a
    clipped/saturated tail, or any shape anomaly -- the standard, amplitude-
    independent pulse-quality discriminant of optimal-filter analyses.

    Both the windows and the template are zero-meaned, so the DC bin of the residual is
    identically zero and contributes nothing to the sum: it needs no explicit removal (the
    old code subtracted |resid[:,0]|^2/psd[0], which is provably 0 and only suggested
    otherwise).  The DOF count still drops it: L complex bins, minus DC, minus the one
    fitted amplitude = L - 2."""
    L, S, _, _ = _of_basis(prep, psd, config)
    if windows is None:
        windows = of_aligned_windows(prep, config)
    Xz = np.fft.fft(windows - windows.mean(axis=1, keepdims=True), axis=1)
    resid = Xz - amplitude[:, None] * S[None, :]
    chi2 = np.sum(np.abs(resid) ** 2 / psd[None, :], axis=1)
    return chi2 / max(L - 2, 1)


def optimal_filter_resolution(prep: P.Prepared, psd: np.ndarray, config: P.Config):
    """Closure test of the optimal filter: the PREDICTED amplitude noise,
    sigma_A = (sum_f |S|^2/N)^-1/2, against the MEASURED spread of the same
    estimator evaluated on pulse-free baseline windows.  Agreement (ratio ~ 1)
    validates that the template and the noise PSD together describe the data; a
    mismatch points at a wrong PSD normalization or template scale."""
    L, _, weight, denom = _of_basis(prep, psd, config)
    sigma_pred = 1.0 / np.sqrt(denom)
    cap = min(prep.corrected.shape[0], config.psd_cap)
    # Measure the estimator's noise on the SECOND pulse-free block [L:2L], disjoint from the
    # [:L] region the PSD is estimated on, so the closure test is independent: it checks the
    # PSD/template normalization against noise the PSD was NOT fit to (measuring on the PSD's
    # own samples makes meas == pred by construction, testing nothing).  [L:2L] stays early in
    # the guaranteed pulse-free region -- well before the pulse -- so it is not contaminated by
    # the rising edge the way a block flush against noise_stop would be.  Falls back to [:L]
    # when the pre-pulse region is too short to hold two disjoint blocks.
    stop = P.noise_stop(prep.corrected.shape[1], config)
    n0 = L if stop >= 2 * L else 0
    noise_win = prep.corrected[:cap, n0:n0 + L].astype(np.float64)
    a_noise = np.real(np.fft.fft(noise_win, axis=1) @ weight) / denom
    sigma_meas = float(np.std(a_noise))
    return float(sigma_pred), sigma_meas


def optimal_filter_residual(prep: P.Prepared, config: P.Config, amplitude: np.ndarray,
                            windows: np.ndarray | None = None):
    """Per-sample template-fit residual (data - amplitude*template) averaged over
    events, in the optimal filter's DC-blind (zero-mean) space.  Returns the
    sample axis (relative to the peak), the mean residual (systematic shape
    mismatch -- should be ~0) and the residual RMS across events (the noise floor
    plus any event-to-event shape variation the single template cannot absorb)."""
    if windows is None:
        windows = of_aligned_windows(prep, config)
    t0 = prep.template - prep.template.mean()
    wz = windows - windows.mean(axis=1, keepdims=True)
    resid = wz - amplitude[:, None] * t0[None, :]
    return prep.relative, resid.mean(axis=0), resid.std(axis=0)


def plot_noise_psd(psd: np.ndarray, template: np.ndarray, config: P.Config) -> None:
    sig = np.abs(np.fft.fft(template - template.mean())) ** 2
    m = len(psd) // 2 + 1
    f = np.arange(m) / len(psd)
    weight = sig[:m] / psd[:m]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(f, psd[:m] / psd[:m].max(), color="C7", label="noise N(f)")
    ax.semilogy(f, sig[:m] / sig[:m].max(), color="C1", label="signal |S(f)|^2")
    ax.semilogy(f, weight / weight.max(), color="C3", lw=2, label="optimal weight |S|^2 / N")
    ax.set_ylim(1e-4, 2.0)
    ax.set(xlabel="Frequency (cycles/sample)", ylabel="Normalized power",
           title="Optimal-filter frequency content (real noise)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    P._finish(fig, "3_noise_psd", config)


def analyze(config: P.Config) -> dict[str, Any]:
    prep = P.prepare(config)
    config = prep.config          # pick up the auto-estimated pulse window
    windows = of_aligned_windows(prep, config)
    observable = optimal_filter_amplitude(prep, prep.psd, config, windows)
    # Per-event OF chi2 first: it feeds the QC block AND (opt-in) the pileup gate below.
    chi2 = optimal_filter_chi2(prep, prep.psd, config, observable, windows)
    # saturation_qc once: it both excludes clipped events from the fit and feeds the QC block.
    sat = P.saturation_qc(prep, config)
    # Rail-clipped events are a genuinely different pulse shape: they must not set the clean
    # chi2-vs-amplitude trend that the pileup gate thresholds against (see chi2_amplitude_trend).
    clipped = sat["sat_mask"].astype(bool) if sat["clipped_pileup"] else None
    pileup_mask = (P.pileup_mask_from_chi2(chi2, config, amplitude=observable, clipped=clipped)
                   if config.exclude_pileup else None)
    method = P.analyze_observable(observable, prep, "Optimal filter", "C3", config,
                                  sat=sat, pileup_mask=pileup_mask)

    P.print_summary(prep, [method])

    resolution = optimal_filter_resolution(prep, prep.psd, config)
    tw_slope = P.timewalk_slope(observable, prep.events["peak_subsample"].to_numpy())
    trend = P.chi2_amplitude_trend(observable, chi2, clipped)
    P.print_qc(resolution=resolution, chi2_red=chi2, timewalk_slope=tw_slope, saturation=sat,
               chi2_trend=trend)

    if config.show_plots or config.save_plots:
        P.plot_baseline_residuals(prep, config)
        P.plot_template(prep, config)
        plot_noise_psd(prep.psd, prep.template, config)
        if config.mode == "muon":
            P.plot_muon_spectrum(method, config)
        else:
            P.plot_spectrum(method, config)
        P.plot_saturation_qc(sat, config)
        P.plot_of_chi2(observable, chi2, config, peak_scale=method["peak_scale"],
                       clipped=clipped)
        rel, rmean, rrms = optimal_filter_residual(prep, config, observable, windows)
        P.plot_template_residual(rel, rmean, rrms, prep.noise_sigma, config)
        P.plot_timing(prep.events["peak_subsample"].to_numpy(), observable,
                      prep.length, config, peak_scale=method["peak_scale"])
        P.plot_waveform_diagnostics(prep, config)
        P.plot_spectrum_pull(method, config)
    return {"prep": prep, "method": method,
            "qc": {"resolution": resolution, "chi2": chi2, "saturation": sat,
                   "timewalk_slope": tw_slope}}


def main() -> None:
    args = P.build_arg_parser("Optimal (Wiener) filter analysis.").parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    analyze(P.config_from_args(args, script_file=__file__, program="of"))


if __name__ == "__main__":
    main()