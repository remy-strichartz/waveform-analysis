#!/usr/bin/env python3
"""
compare.py
==========
Run both amplitude estimators on the same real-noise dataset and compare them.

Prepares the data once (load, colored-noise PSD, time-walk-free template,
optimal-filter trigger), then evaluates the optimal (Wiener) filter and the
colored-noise-optimal boxcar through the identical spectrum / separation path
(all in ADC-derived units), and overlays their spectra.

    python compare.py --input run00270_ch0.h5 --save-plots --output-dir results
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (see README)

import numpy as np

from energy_reconstruction import boxcar as BX           # noqa: E402
from energy_reconstruction import mv_pipeline as P       # noqa: E402
from energy_reconstruction import optimal_filter as OF   # noqa: E402

logger = logging.getLogger("compare")


def analyze(config: P.Config) -> dict[str, Any]:
    prep = P.prepare(config)
    config = prep.config          # pick up the auto-estimated pulse window
    # saturation_qc ONCE: it excludes clipped events from both estimators' fits and
    # feeds the QC print/plot below.
    sat = P.saturation_qc(prep, config)

    of_win = OF.of_aligned_windows(prep, config)
    of_obs = OF.optimal_filter_amplitude(prep, prep.psd, config, of_win)
    # OF chi2 up front: it feeds the QC block AND (opt-in) the pileup gate.  Pileup is an
    # event property, so the SAME mask is applied to both estimators' fits.
    chi2 = OF.optimal_filter_chi2(prep, prep.psd, config, of_obs, of_win)
    # Rail-clipped events are a genuinely different pulse shape: they must not set the clean
    # chi2-vs-amplitude trend that the pileup gate thresholds against (chi2_amplitude_trend).
    clipped = sat["sat_mask"].astype(bool) if sat["clipped_pileup"] else None
    pileup_mask = (P.pileup_mask_from_chi2(chi2, config, amplitude=of_obs, clipped=clipped)
                   if config.exclude_pileup else None)
    optimal = P.analyze_observable(of_obs, prep, "Optimal filter", "C3", config,
                                   sat=sat, pileup_mask=pileup_mask)

    best, grid, snr = BX.optimal_boxcar_window(prep, config)
    bx_obs = BX.boxcar_amplitude(prep, best, config)
    boxcar = P.analyze_observable(bx_obs, prep, f"Boxcar (W={best + config.box_pre})", "C0",
                                  config, sat=sat, pileup_mask=pileup_mask)

    methods = [optimal, boxcar]
    P.print_summary(prep, methods, box_width=best + config.box_pre)

    # Quality-control diagnostics (no truth required; pulse quality + reconstruction
    # fidelity + range).  Computed once here, then printed and plotted.
    resolution = OF.optimal_filter_resolution(prep, prep.psd, config)
    aa_bands = P.amplitude_area_outliers(of_obs, bx_obs)
    _, outlier, _, _ = aa_bands
    ok = np.isfinite(of_obs) & np.isfinite(bx_obs)
    amp_area = {"n_outliers": int(outlier.sum()), "n_total": int(ok.sum()),
                "outlier_frac": float(outlier.sum() / max(int(ok.sum()), 1))}
    tw_slope = P.timewalk_slope(of_obs, prep.events["peak_subsample"].to_numpy())
    trend = P.chi2_amplitude_trend(of_obs, chi2, clipped)
    P.print_qc(resolution=resolution, chi2_red=chi2, timewalk_slope=tw_slope,
               amp_area=amp_area, saturation=sat, chi2_trend=trend)

    if config.show_plots or config.save_plots:
        P.plot_baseline_residuals(prep, config)
        P.plot_template(prep, config)
        OF.plot_noise_psd(prep.psd, prep.template, config)
        BX.plot_boxcar_scan(grid, snr, best, config)
        # Both estimators' spectra are plotted (OF first, then boxcar): the MIP
        # lines are near-identical (16_estimator_comparison overlays them), but the
        # boxcar spectrum makes its own fit/cut directly comparable to the OF one.
        # Only the OF pull is kept -- the boxcar pull carried no extra information.
        if config.mode == "muon":
            P.plot_muon_spectrum(optimal, config, name="6_spectrum_optimal")
            P.plot_muon_spectrum(boxcar, config, name="7_spectrum_boxcar")
        else:
            P.plot_spectrum(optimal, config, name="6_spectrum_optimal")
            P.plot_spectrum(boxcar, config, name="7_spectrum_boxcar")
        P.plot_saturation_qc(sat, config)
        P.plot_of_chi2(of_obs, chi2, config, peak_scale=optimal["peak_scale"],
                       clipped=clipped)
        rel, rmean, rrms = OF.optimal_filter_residual(prep, config, of_obs, of_win)
        P.plot_template_residual(rel, rmean, rrms, prep.noise_sigma, config)
        P.plot_timing(prep.events["peak_subsample"].to_numpy(), of_obs,
                      prep.length, config, peak_scale=optimal["peak_scale"])
        P.plot_amplitude_area(of_obs, bx_obs, config, outliers=aa_bands,
                              peak_scale=optimal["peak_scale"])
        P.plot_waveform_diagnostics(prep, config)
        P.plot_spectrum_pull(optimal, config)
        P.plot_estimator_comparison(methods, config)
    return {"prep": prep, "methods": methods,
            "qc": {"resolution": resolution, "chi2": chi2, "saturation": sat,
                   "amp_area": amp_area, "timewalk_slope": tw_slope}}


def main() -> None:
    args = P.build_arg_parser("Compare the optimal filter and boxcar estimators.").parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    analyze(P.config_from_args(args, script_file=__file__, program="compare"))


if __name__ == "__main__":
    main()