#!/usr/bin/env python3
"""
boxcar.py
=========
Standalone boxcar (top-hat integral) analysis for real colored noise.

Integrates the waveform over a top-hat window around the pulse.  The window
length is chosen to maximize the REAL signal-to-noise: the signal is the template
integral, and the noise is the measured ROBUST spread of the same-width integral
over pulse-free segments of the real (colored) noise -- not the white
sigma*sqrt(width) approximation, which is wrong when the noise is correlated, and
not the plain std, which a heavy-tailed baseline drives to a useless answer (see
_integral_noise).

    python boxcar.py --input run00270_ch0.h5 --save-plots
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

logger = logging.getLogger("boxcar")


def _running_integral(rows2d: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral along each row, with a leading zero."""
    cum = np.zeros_like(rows2d)
    cum[:, 1:] = np.cumsum(0.5 * (rows2d[:, :-1] + rows2d[:, 1:]), axis=1)
    return cum


def _integral_noise(integrals: np.ndarray) -> float:
    """Width of the noise-integral distribution -- a ROBUST (MAD) scale, not the plain std.

    This is the noise the window length is optimized against, so its estimator decides the
    window.  np.std is the wrong one on a channel whose baseline is heavy-tailed: a rare
    spike enters the std as its SQUARE, and a wider integration window catches proportionally
    more spikes, so the std grows far faster with width than the noise a typical event
    actually sees.  On run00270_ch9 (baseline spike noise: MAD sigma 1.48 ADC but RMS ~2.4)
    that alone drove the scanned optimum to the SHORTEST window on the grid -- W=3 samples on
    a PMT -- because the std-vs-width curve outran the signal from the very first point.

    Measured (fitted MIP resolution sigma/MPV, the quantity the window is supposed to
    minimise -- lower is better):
        ch9:  W  3 -> 10,   0.72 -> 0.49   (the spike-noise channel; a 32% improvement)
        ch1:  W 39 -> 29,   0.207 -> 0.199
        ch7:  W 39 -> 10,   0.373 -> 0.346
        ch0:  W 28 -> 39,   0.151 -> 0.156  (marginally worse -- ch0's noise is near-Gaussian,
                                             so it had nothing to gain and pays a little
                                             estimator variance)
        ch10 / ch6: unchanged.
    A timing-jitter term was tried here too and does NOT help: the optimal filter's own
    timing resolution is only ~0.03 samples on ch9 (Cramer-Rao, from template + PSD), far too
    small to move the optimum, and with box_pre=1 the window's leading edge sits on the
    pulse's steep rise at EVERY width, so d(integral)/d(time) barely varies with W.  The
    heavy-tailed noise estimator was the whole effect.
    """
    med = float(np.median(integrals))
    return 1.4826 * float(np.median(np.abs(integrals - med)))


def optimal_boxcar_window(prep: P.Prepared, config: P.Config):
    """Window length (samples after the peak) maximizing template_integral /
    measured-colored-noise integral width at that width (see _integral_noise)."""
    template = prep.template
    pk = int(np.argmax(template))
    start = max(pk - config.box_pre, 0)

    # Real colored-noise integral spread vs width, from pulse-free pre-pulse segments.
    length = prep.corrected.shape[1]
    cap = min(prep.corrected.shape[0], config.psd_cap)
    noise = prep.corrected[:cap, :P.noise_stop(length, config)].astype(np.float64)
    cum = _running_integral(noise)

    # Scan out to the template's post-peak extent (a slow pulse's optimum can sit
    # far beyond the fixed box_max, which would silently clip the scan at the
    # grid edge), limited by the pulse-free region the noise estimate comes from.
    hi = max(config.box_max, template.size - 1 - pk)
    hi = min(hi, noise.shape[1] - 1 - config.box_pre)
    grid = np.arange(config.box_min, hi + 1)
    snr = np.empty(grid.size)
    for i, w in enumerate(grid):
        width = config.box_pre + w
        # Integrate the template over the SAME inclusive span the amplitude
        # normalization uses (tlo..thi inclusive = `width` intervals), so the
        # scanned SNR reflects exactly the signal the estimator will integrate.
        stop = min(pk + w, template.size - 1)
        signal = float(np.trapezoid(template[start:stop + 1]))
        integrals = cum[:, width:] - cum[:, :-width]      # every width-sample integral
        noise_w = _integral_noise(integrals)
        snr[i] = signal / noise_w if noise_w > 0 else 0.0
    best = int(grid[int(np.argmax(snr))])
    if best == int(grid[-1]):
        logger.warning("Boxcar SNR is still rising at the scan edge (width %d); the true "
                       "optimum may be longer than the pulse-free noise region allows.",
                       best + config.box_pre)
    if best == int(grid[0]):
        # The SNR curve never turned over: it fell from the very first width.  That means the
        # noise integral is outrunning the signal integral immediately -- strongly correlated
        # (or still heavy-tailed) noise against a pulse whose area is already collected.  The
        # window is then set by the edge of the grid, not by an optimum, and is worth saying
        # out loud rather than shipping a 3-sample "optimal" window in silence.
        logger.warning("Boxcar SNR is already falling at the scan's SHORTEST width (%d "
                       "samples): the optimum is at the grid edge, not at a turnover. The "
                       "noise integral outgrows the signal from the first step -- check the "
                       "baseline for correlated/burst noise before trusting this window.",
                       best + config.box_pre)
    logger.info("Boxcar optimal width = %d samples (colored-noise SNR).", best + config.box_pre)
    return best, grid, snr


def boxcar_amplitude(prep: P.Prepared, box_width: int, config: P.Config) -> np.ndarray:
    """Trapezoidal integral over [peak - box_pre, peak + box_width] per event,
    with fractional endpoints at the trigger's sub-sample peak.  Normalized by the
    template integral so the observable is an amplitude (template = 1), matching
    the optimal-filter scale and making the two estimators directly comparable.

    `events["peak_subsample"]` is now the PHYSICAL pulse peak (window_start +
    argmax(template)), so the data window below and the template normalisation window
    are referenced to the same feature.  They previously were not: the trigger reported
    window_start + template_pre as the "peak" while the normalisation used
    argmax(template), and the two differ by a sample on real data -- a scale error worth
    -1.9% on run00270_ch0 and +4.0% on caen_ch0."""
    cw = P.triggered(prep).astype(np.float64)
    cum = _running_integral(cw)
    peaks = prep.events["peak_subsample"].to_numpy()
    starts, stops = peaks - config.box_pre, peaks + box_width
    n, length = cw.shape
    idx = np.arange(n)

    def antideriv(x):
        x = np.clip(x, 0.0, length - 1.0)
        i = np.floor(x).astype(np.int64)
        frac = x - i
        ip = np.minimum(i + 1, length - 1)
        vi, vip = cw[idx, i], cw[idx, ip]
        edge = vi + frac * (vip - vi)
        return cum[idx, i] + 0.5 * (vi + edge) * frac

    area = antideriv(stops) - antideriv(starts)
    area[~(stops > starts)] = np.nan

    # Normalize by the template integral over the SAME window so that, for a pulse
    # matching the template (data = A * template), the observable equals A -- the
    # same amplitude scale as the optimal filter.
    tmpl = prep.template
    tpk = int(np.argmax(tmpl))
    tlo, thi = max(tpk - config.box_pre, 0), min(tpk + box_width, tmpl.size - 1)
    tnorm = float(np.trapezoid(tmpl[tlo:thi + 1]))
    return area / tnorm if tnorm > 0 else area


def plot_boxcar_scan(grid, snr, best, config: P.Config) -> None:
    width = grid + config.box_pre
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(width, snr, marker="o", ms=3, color="C0",
            label="template / colored-noise integral spread (robust)")
    ax.axvline(best + config.box_pre, ls="--", color="r", label=f"optimal width = {best + config.box_pre} samples")
    ax.set(xlabel="Boxcar window width (samples)", ylabel="signal / noise-integral spread",
           title="Boxcar window optimization (real colored noise)")
    ax.legend(); ax.grid(True, alpha=0.3)
    P._finish(fig, "3_boxcar_scan", config)


def analyze(config: P.Config) -> dict[str, Any]:
    prep = P.prepare(config)
    config = prep.config          # pick up the auto-estimated pulse window
    best, grid, snr = optimal_boxcar_window(prep, config)
    observable = boxcar_amplitude(prep, best, config)

    label = f"Boxcar (W={best + config.box_pre})"
    # saturation_qc once: it both excludes clipped events from the fit and feeds the QC block.
    sat = P.saturation_qc(prep, config)
    method = P.analyze_observable(observable, prep, label, "C0", config, sat=sat)

    P.print_summary(prep, [method], box_width=best + config.box_pre)

    tw_slope = P.timewalk_slope(observable, prep.events["peak_subsample"].to_numpy())
    P.print_qc(timewalk_slope=tw_slope, saturation=sat)

    if config.show_plots or config.save_plots:
        P.plot_baseline_residuals(prep, config)
        P.plot_template(prep, config)
        plot_boxcar_scan(grid, snr, best, config)
        if config.mode == "muon":
            P.plot_muon_spectrum(method, config)
        else:
            P.plot_spectrum(method, config)
        P.plot_saturation_qc(sat, config)
        P.plot_timing(prep.events["peak_subsample"].to_numpy(), observable,
                      prep.length, config, peak_scale=method["peak_scale"])
        P.plot_waveform_diagnostics(prep, config)
        P.plot_spectrum_pull(method, config)
    return {"prep": prep, "method": method, "box_width": best + config.box_pre,
            "qc": {"saturation": sat, "timewalk_slope": tw_slope}}


def main() -> None:
    args = P.build_arg_parser("Boxcar (top-hat integral) analysis.").parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    analyze(P.config_from_args(args, script_file=__file__, program="boxcar"))


if __name__ == "__main__":
    main()