#!/usr/bin/env python3
"""
run_stability.py
================
The PER-CHANNEL half of the timing analysis: is this channel's response stable
across the real time axis recovered by event_times.py?

GAIN / BASELINE / NOISE DRIFT.  The pulse-height scale per time block comes from
the house optimal-filter amplitude (mv_pipeline machinery -- the same estimator as
the energy analyses) and is tracked by GAIN-EQUIVARIANT QUANTILES (Q25/Q50/Q75
with bootstrap errors): a pure gain change scales every quantile by the same
factor, so common motion of the three = gain drift, divergence between them = a
spectral-shape change, with NO spectral model assumed.  (That matters here: a
position-smeared paddle spectrum is broad, not a clean Landau line.)  Per-block
baseline level and noise sigma complete the picture -- SiPM gain rides on
temperature through the breakdown voltage, so a diurnal lab cycle would show up
here first, and an EMI/noise disturbance shows up in the noise sigma.

The block noise sigma also DEFINES the run's bad periods (flag_bad_blocks): blocks
whose noise sits a factor above the run median are flagged, merged into time
windows, and written to the JSON, so the energy analyses can exclude them with
`mv_pipeline --exclude-hours LO HI`.  A noise excursion is the right thing to
define the window on, because it is what actually damages the reconstruction: it
moves the PSD the optimal filter is built from and the noise scale the offline
trigger thresholds against, both of which mv_pipeline estimates ONCE for the whole
run.

The run-level quantities -- trigger rate, arrival statistics, DAQ dropouts, dead
time and livetime -- are NOT here.  They read only the times file, so they are
properties of the RUN, identical for every channel of it; they live in
event_times.py and are measured once.  (They used to be recomputed per channel,
which produced three byte-identical copies of one measurement.)

BIAS GUARDS:
  * The quantile trackers apply NO event selection beyond the offline trigger
    (which accepts ~all events, reported), so there is no acceptance window whose
    edges a drift could cross.
  * Gain blocks hold EQUAL EVENT COUNTS (uniform statistical power, no weak
    end-of-run block skewing the drift estimate); each block's extent in time is
    drawn on the plot so unequal spans stay visible.
  * The amplitude observable is template-relative with ONE global template; a
    per-block template would absorb the very drift being measured.  The residual
    worry -- a trigger-efficiency change near threshold faking a drift -- is
    checked numerically: the pulse-height scale is compared to the offline
    threshold in the same units, and the per-block offline-trigger fraction is
    plotted (a threshold effect moves it; a pure gain drift far above threshold
    does not).
  * Baseline drift is reported BOTH as a constant-fit p-value and as a fraction of
    the pulse-height scale.  With ~10^4 events per block the p-value resolves
    drifts of a fraction of an ADC count, which are statistically certain and
    physically irrelevant -- and mv_pipeline subtracts the baseline per event
    anyway, so the analysis is structurally immune to them.  Quoting the size next
    to the significance stops a sub-LSB wobble from reading as a finding.

Examples
--------
    python run_stability.py --input run00270_ch9.h5 --save-plots
    python run_stability.py --input run00270_ch10.h5 --save-plots --bad-noise-factor 1.35
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_dist

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))                            # repo root (see README)

from energy_reconstruction import mv_pipeline as P       # noqa: E402
from energy_reconstruction import optimal_filter as OF   # noqa: E402

logger = logging.getLogger("run_stability")

GAIN_QUANTILES = (25.0, 50.0, 75.0)


# ===========================================================================
# Times file
# ===========================================================================

def load_times(times_path: Path, input_path: Path) -> tuple[np.ndarray, dict]:
    """Per-event times aligned to the waveform file, with hard alignment guards: row
    counts must match and, when both files carry /source_event_index, the indices must
    be identical -- a silent misalignment would scramble every time-resolved result."""
    with h5py.File(times_path, "r") as h5:
        t_rel = np.asarray(h5["event_time_rel_s"][()], dtype=np.float64)
        sei_t = (np.asarray(h5["source_event_index"][()], dtype=np.int64)
                 if "source_event_index" in h5 else None)
        attrs = dict(h5.attrs)
    with h5py.File(input_path, "r") as h5:
        n_wf = h5["waveforms"].shape[0]
        sei_w = (np.asarray(h5["source_event_index"][()], dtype=np.int64)
                 if "source_event_index" in h5 else None)
    if len(t_rel) != n_wf:
        raise ValueError(f"{times_path.name} has {len(t_rel)} rows but "
                         f"{input_path.name} has {n_wf} waveforms.")
    if sei_t is not None and sei_w is not None and not np.array_equal(sei_t, sei_w):
        raise ValueError("source_event_index differs between the times file and the "
                         "waveform file: they come from different conversions.")
    if attrs.get("n_wrap_failed", 0) > 0:
        logger.warning("Times file reports %d failed wrap decisions -- treat "
                       "long-baseline intervals with suspicion.", attrs["n_wrap_failed"])
    return t_rel, attrs


# ===========================================================================
# Gain / baseline / noise per time block
# ===========================================================================

def _sem(x: np.ndarray) -> float:
    return float(np.std(x) / np.sqrt(max(x.size, 1)))


def gain_blocks(t_all: np.ndarray, t_ev: np.ndarray, obs: np.ndarray,
                level_all: np.ndarray, sigma_all: np.ndarray,
                cfg, n_blocks: int, n_boot: int) -> dict:
    """Equal-event-count blocks over the triggered events; baseline/noise and the
    offline-trigger fraction are evaluated on ALL events inside each block's TIME
    window, so every series shares one time axis.

    Baseline/noise block statistics are the MEAN of per-event robust values (per-event
    median / per-event MAD-sigma): the per-event estimators reject the pulse, and the
    block mean keeps sub-LSB resolution that a block median of integer-quantized ADC
    values would destroy.

    (A Landau(x)Gauss MPV cross-check used to run alongside the quantiles.  It was
    valid on every channel tried and merely reproduced them, at the cost of the
    heaviest fit path here -- a guarded Landau fit per block per bootstrap resample --
    so it is gone.  The quantiles are gain-equivariant and assume no spectral model,
    which is the stronger position on a broad paddle spectrum; mv_pipeline still owns
    the Landau fit where it belongs, in the physics.)"""
    finite = np.isfinite(obs)
    obs_f, t_ev_f = obs[finite], t_ev[finite]
    n_ev = obs_f.size

    # t_all is the per-event relative time (monotonic non-decreasing by construction in
    # clock_recovery.recover), so the all-events window for each block is a contiguous
    # slice found by binary search -- O(log n) per block instead of an O(n) boolean mask
    # rebuilt n_blocks times.
    edges_i = np.linspace(0, n_ev, n_blocks + 1).astype(int)
    rows = []
    for b in range(n_blocks):
        o = obs_f[edges_i[b]:edges_i[b + 1]]
        tt = t_ev_f[edges_i[b]:edges_i[b + 1]]
        t_lo, t_hi = float(tt.min()), float(tt.max())

        rng = np.random.default_rng(cfg.seed + b)
        q = np.percentile(o, GAIN_QUANTILES)
        boot_q = np.array([np.percentile(o[rng.integers(0, o.size, o.size)],
                                         GAIN_QUANTILES) for _ in range(n_boot)])
        q_err = boot_q.std(axis=0)

        lo_i = int(np.searchsorted(t_all, t_lo, side="left"))
        hi_i = int(np.searchsorted(t_all, t_hi, side="right"))
        n_in_win = hi_i - lo_i
        lv, sg = level_all[lo_i:hi_i], sigma_all[lo_i:hi_i]
        rows.append({
            "t_mid_h": float(np.median(tt)) / 3600.0,
            "t_lo_h": t_lo / 3600.0, "t_hi_h": t_hi / 3600.0,
            "n_events": int(o.size),
            "q25": q[0], "q25_err": q_err[0],
            "q50": q[1], "q50_err": q_err[1],
            "q75": q[2], "q75_err": q_err[2],
            "baseline_adc": float(np.mean(lv)), "baseline_err": _sem(lv),
            "noise_sigma_adc": float(np.mean(sg)), "noise_sigma_err": _sem(sg),
            "trig_frac": float(o.size / max(n_in_win, 1)),
        })
    blocks = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    # Drift significance: chi2 of the block values against their weighted mean.
    def _const_chi2(y, ye):
        good = np.isfinite(y) & np.isfinite(ye) & (ye > 0)
        if int(good.sum()) < 3:
            return np.nan, np.nan
        wm = np.sum(y[good] / ye[good] ** 2) / np.sum(1.0 / ye[good] ** 2)
        c2 = float(np.sum(((y[good] - wm) / ye[good]) ** 2))
        return c2, float(chi2_dist.sf(c2, int(good.sum()) - 1))

    for key in ("q25", "q50", "q75", "baseline_adc", "noise_sigma_adc"):
        err_key = key.replace("_adc", "") + "_err" if key.endswith("_adc") else key + "_err"
        c2, p = _const_chi2(blocks[key], blocks[err_key])
        blocks[f"{key}_chi2"] = c2
        blocks[f"{key}_p_const"] = p
    return blocks


# ===========================================================================
# Bad-period detection  (the window the energy analyses should exclude)
# ===========================================================================

def flag_bad_blocks(blocks: dict, factor: float) -> tuple[np.ndarray, list[dict]]:
    """Blocks whose noise sigma exceeds `factor` x the run's MEDIAN block sigma, merged
    into contiguous time windows.

    Why noise and not, say, the rate: the rate excess that accompanies a disturbance is
    real but diffuse (it does not localize to a clean window), whereas the noise sigma
    is both sharply localized and the quantity that actually damages the reconstruction
    -- mv_pipeline builds ONE PSD and ONE trigger threshold for the whole run, so a
    window whose noise is well above the run average is filtered and thresholded with a
    model that does not describe it.

    The rule is deliberately blunt and its threshold is a knob (--bad-noise-factor): a
    channel whose noise wanders gently all run (run00270_ch0: 11.4-20.8 ADC, max/median
    = 1.40) is NOT flagged at the 1.5 default, and should not be -- there is no clean
    window to cut.  Nothing here is automatic beyond the flagging: the windows are
    reported, and excluding them is an explicit `--exclude-hours` on the analysis."""
    s = blocks["noise_sigma_adc"]
    med = float(np.median(s))
    bad = s > factor * med

    windows, i = [], 0
    while i < bad.size:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < bad.size and bad[j + 1]:
            j += 1
        windows.append({
            "lo_h": float(blocks["t_lo_h"][i]), "hi_h": float(blocks["t_hi_h"][j]),
            "n_blocks": int(j - i + 1),
            "n_events": int(blocks["n_events"][i:j + 1].sum()),
            "peak_noise_sigma_adc": float(s[i:j + 1].max()),
            "noise_ratio_vs_median": float(s[i:j + 1].max() / med),
        })
        i = j + 1
    return bad, windows


# ===========================================================================
# Plot
# ===========================================================================

def plot_gain_blocks(blocks: dict, bad: np.ndarray, peak_scale: float,
                     thr_ratio: float, config) -> None:
    """Gain, offline-trigger fraction, baseline and noise against real time.

    (The separate 'correlations' figure that used to accompany this one -- block gain
    vs block baseline, block gain vs block rate -- is gone.  Both panels were 15-point
    scatters of series already plotted here against time, and the one correlation that
    looked like a finding, gain vs rate, is confounded: a disturbance that moves both
    makes them correlate by construction.  Reading them off the time axis is both
    honest and sufficient.  The exposure-corrected diurnal fold went with it: ~2 cycles
    in a 50-h run cannot support a measurement, and a single localized disturbance
    necessarily folds onto particular hours-of-day and manufactures a 'diurnal' signal
    -- a confidently-wrong estimator, which is the failure mode this workspace exists
    to avoid.)"""
    th = blocks["t_mid_h"]
    xerr = np.vstack([th - blocks["t_lo_h"], blocks["t_hi_h"] - th])
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    # Gain-equivariant quantiles, each normalized to its own run mean: a pure gain
    # change moves the three curves together; divergence = a shape/population change.
    for i, (qn, col) in enumerate(zip(("q25", "q50", "q75"), ("C0", "C3", "C2"))):
        norm = np.nanmean(blocks[qn])
        axes[0].errorbar(th, blocks[qn] / norm, xerr=xerr if i == 1 else None,
                         yerr=blocks[f"{qn}_err"] / norm, fmt="o-", ms=3.5, lw=0.8,
                         color=col, capsize=2, label=f"{qn.upper()} / run mean")
    axes[0].axhline(1.0, color="gray", lw=0.8)
    axes[0].set(ylabel="Relative pulse-height scale",
                title=f"Gain trackers -- Q50 constant-fit p = {blocks['q50_p_const']:.3g}, "
                      f"pulse height / offline threshold = {thr_ratio:.0f}x")
    axes[0].legend(fontsize=8, ncol=3)

    axes[1].errorbar(th, 100 * blocks["trig_frac"], xerr=xerr, fmt="o", ms=4,
                     color="C2", lw=1.0)
    axes[1].set(ylabel="Offline trigger fraction (%)",
                title="Offline-trigger fraction (a threshold effect would move this with "
                      "the gain; a drift far above threshold does not)")

    base_pp = float(np.nanmax(blocks["baseline_adc"]) - np.nanmin(blocks["baseline_adc"]))
    axes[2].errorbar(th, blocks["baseline_adc"], xerr=xerr, yerr=blocks["baseline_err"],
                     fmt="o", ms=4, color="C1", lw=1.0, capsize=2)
    axes[2].set(ylabel="Baseline (raw ADC)",
                title=f"Baseline level -- peak-to-peak {base_pp:.2f} ADC = "
                      f"{100 * base_pp / peak_scale:.3f}% of the pulse scale "
                      f"[constant-fit p = {blocks['baseline_adc_p_const']:.3g}]")

    axes[3].errorbar(th, blocks["noise_sigma_adc"], xerr=xerr, yerr=blocks["noise_sigma_err"],
                     fmt="o", ms=4, color="C4", lw=1.0, capsize=2)
    axes[3].axhline(float(np.median(blocks["noise_sigma_adc"])), color="gray", lw=0.8,
                    ls="--", label="run median")
    if bad.any():
        for i in np.flatnonzero(bad):
            axes[3].axvspan(blocks["t_lo_h"][i], blocks["t_hi_h"][i], color="C3",
                            alpha=0.15, lw=0)
        axes[3].plot([], [], color="C3", alpha=0.35, lw=8, label="flagged bad period")
    axes[3].set(xlabel="Run time (h)", ylabel="Noise sigma (ADC)",
                title=f"Pre-pulse noise -- constant-fit p = "
                      f"{blocks['noise_sigma_adc_p_const']:.3g}")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Gain / baseline / noise vs real time (equal-count blocks)", fontsize=13)
    fig.tight_layout()
    P._finish(fig, "stab_1_gain", config)


# ===========================================================================
# Summary
# ===========================================================================

def print_stability_summary(blocks: dict, windows: list[dict], peak_scale: float,
                            thr_ratio: float, factor: float, stem: str) -> None:
    q50_adc = blocks["q50"] * peak_scale
    drift_pp = 100.0 * (np.nanmax(q50_adc) - np.nanmin(q50_adc)) / np.nanmean(q50_adc)
    base_pp = float(np.nanmax(blocks["baseline_adc"]) - np.nanmin(blocks["baseline_adc"]))
    sig = blocks["noise_sigma_adc"]
    print()
    print("=" * 72)
    print("Channel stability summary  (run-level rate / dead time: see event_times.py)")
    print("-" * 72)
    print(f"Blocks / events            : {blocks['q50'].size} x "
          f"{int(blocks['n_events'][0]):,} events")
    print(f"Gain drift, Q50 peak-peak  : {drift_pp:.2f} %  "
          f"[constant-fit p = {blocks['q50_p_const']:.3g}]")
    print(f"Pulse height / threshold   : {thr_ratio:.0f}x (threshold bias negligible)")
    print(f"Baseline drift, peak-peak  : {base_pp:.2f} ADC = "
          f"{100 * base_pp / peak_scale:.3f} % of the pulse scale "
          f"[p = {blocks['baseline_adc_p_const']:.3g}]")
    print(f"Noise sigma                : median {np.median(sig):.2f} ADC, "
          f"range {sig.min():.2f}-{sig.max():.2f} "
          f"[p = {blocks['noise_sigma_adc_p_const']:.3g}]")
    print("-" * 72)
    if windows:
        print(f"BAD PERIODS (block noise > {factor:g}x the run median):")
        for w in windows:
            print(f"  {w['lo_h']:6.2f} - {w['hi_h']:6.2f} h   "
                  f"{w['n_blocks']} block(s), {w['n_events']:,} events, "
                  f"peak noise {w['peak_noise_sigma_adc']:.2f} ADC "
                  f"({w['noise_ratio_vs_median']:.2f}x median)")
        excl = " ".join(f"--exclude-hours {w['lo_h']:.2f} {w['hi_h']:.2f}" for w in windows)
        print()
        print("  These events are filtered and thresholded with a PSD and a noise scale")
        print("  built from the whole run, which does not describe them. To exclude them:")
        print(f"    python compare.py --input {stem}.h5 {excl}")
    else:
        print(f"BAD PERIODS: none (no block's noise exceeds {factor:g}x the run median).")
        print("  If the noise wanders without a clean excursion, there is no window to")
        print("  cut; lower --bad-noise-factor to inspect marginal blocks.")
    print("=" * 72)


# ===========================================================================
# All-event baseline & noise (streamed)
# ===========================================================================

def all_event_baseline_noise(config: P.Config, length: int,
                             chunk_events: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Per-event baseline level (raw-ADC pre-pulse median) and noise sigma (MAD of the
    baseline-corrected pre-pulse) for EVERY event in the file, in file-row order -- the
    all-event series the block trackers need but the streaming prepare() no longer
    materializes (it keeps only triggered events).  Read in row-chunks via the same
    iter_waveform_chunks substrate, so peak memory is O(chunk * L), not O(N * L)."""
    stop = P.noise_stop(length, config)
    levels, sigmas = [], []
    for _, block in P.iter_waveform_chunks(config, chunk_events):
        levels.append(np.median(block[:, :stop], axis=1))
        pre_cor = P.remove_baseline(block, config, warn=False)[:, :stop]
        sigmas.append(1.4826 * np.median(
            np.abs(pre_cor - np.median(pre_cor, axis=1, keepdims=True)), axis=1))
    return np.concatenate(levels), np.concatenate(sigmas)


# ===========================================================================
# Driver
# ===========================================================================

def analyze(config: P.Config, times_path: Path, n_blocks: int, n_boot: int,
            bad_noise_factor: float) -> dict:
    t_rel, tattrs = load_times(times_path, config.input_path)

    # ---- house reconstruction for the amplitude time series --------------
    prep = P.prepare(config)
    config = prep.config
    obs = OF.optimal_filter_amplitude(prep, prep.psd, config)
    # Streaming prepare() numbers events 0..n_trig-1 into `corrected`; the row in the
    # source file (hence in the row-aligned times array) is `original_index`.
    ev_rows = prep.events["original_index"].to_numpy()
    t_ev = t_rel[ev_rows]
    peak_scale = float(prep.template.max())            # observable -> ADC pulse height

    # Threshold-bias guard: the offline trigger cuts at trigger_sigma times the response
    # noise; in amplitude units that is trigger_sigma * sigma_A (the OF amplitude noise).
    # The pulse-height scale must sit far above it.
    _, sigma_meas = OF.optimal_filter_resolution(prep, prep.psd, config)
    thr_amp = config.trigger_sigma * sigma_meas
    med_obs = float(np.nanmedian(obs))
    thr_ratio = med_obs / thr_amp if thr_amp > 0 else np.inf
    if thr_ratio < 10:
        logger.warning("Median pulse height is only %.1fx the offline trigger threshold "
                       "-- block trackers may carry selection bias.", thr_ratio)

    # ---- per-event baseline & noise (ALL events, raw ADC), streamed ------
    level_all, sigma_all = all_event_baseline_noise(config, prep.length)

    blocks = gain_blocks(t_rel, t_ev, obs, level_all, sigma_all, config, n_blocks, n_boot)
    bad, windows = flag_bad_blocks(blocks, bad_noise_factor)
    print_stability_summary(blocks, windows, peak_scale, thr_ratio, bad_noise_factor,
                            config.input_path.stem)

    if config.show_plots or config.save_plots:
        plot_gain_blocks(blocks, bad, peak_scale, thr_ratio, config)

    def _clean(v):
        return v.tolist() if isinstance(v, np.ndarray) else v
    result = {"ttt_clock": {k: _clean(v) for k, v in tattrs.items() if k != "note"},
              "peak_scale_adc": peak_scale, "threshold_ratio": float(thr_ratio),
              "blocks": {k: _clean(v) for k, v in blocks.items()},
              "bad_noise_factor": bad_noise_factor,
              "bad_blocks": _clean(bad),
              # The headline product for the rest of the workspace: the time windows
              # to hand to `mv_pipeline --exclude-hours`.
              "bad_windows_h": windows}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = config.output_dir / f"{config.input_path.stem}_stability.json"
    out_json.write_text(json.dumps(result, indent=2, default=float))
    logger.info("Wrote %s", out_json)
    return result


def autobuild_times(input_path: Path, out_path: Path, mid_path: Path | None) -> Path:
    """Recover the per-event time axis in-process when the times file is absent, so one
    `run_stability` call is self-contained.  This is the SAME computation as running
    event_times.py directly (the clock is recovered once per run and written to
    <stem>_times.h5, reusable by every other channel); only the clock QC plots and the
    run-level rate/dead-time analysis are skipped -- run `event_times.py --save-plots`
    for those."""
    from timing_stability import event_times as ET
    ttt, wall, word, bank, sei = ET.load_inputs(input_path, mid_path, None)
    rec = ET.recover(ttt, wall)
    ET.write_output(out_path, ttt, wall, sei, rec, word, bank, input_path)
    logger.info("Auto-recovered clock: %.6f MHz, %d rollovers, %d wrap failures -> %s.  "
                "Run `event_times.py --input %s --save-plots` for the clock QC figures "
                "and the run-level rate / dead-time analysis.", rec["freq_hz"] / 1e6,
                rec["n_wraps_total"], rec["n_fail"], out_path.name, input_path.name)
    return out_path


def main() -> None:
    parser = P.build_arg_parser("Per-channel gain / baseline / noise stability on the "
                                "recovered event-time axis.")
    # --times comes from the shared parser (mv_pipeline.build_arg_parser), which needs
    # the same times file for --exclude-hours; here it is additionally AUTO-BUILT when
    # it does not exist yet, so one run_stability call is self-contained.
    parser.add_argument("--mid", type=Path, default=None,
                        help="MIDAS .mid for wall-clock replay, used only if the times "
                             "file must be auto-built AND the H5 lacks /event_time_unix.")
    parser.add_argument("--n-blocks", type=int, default=15,
                        help="Equal-event-count time blocks for the gain tracking.")
    parser.add_argument("--n-boot", type=int, default=60,
                        help="Bootstrap resamples per block for the tracker errors.")
    parser.add_argument("--bad-noise-factor", type=float, default=1.5,
                        help="Flag a block as a BAD PERIOD when its noise sigma exceeds "
                             "this factor times the run's median block sigma. The flagged "
                             "windows are reported and written to the JSON for "
                             "`mv_pipeline --exclude-hours`.")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    config = P.config_from_args(args, script_file=__file__, program="stability",
                                group="stability")
    # The times file belongs to the same run as the input, so look for it in that run's
    # times/ folder before falling back to a bare-name lookup; when it has to be recovered,
    # it is WRITTEN there too, which is where event_times.py would have put it.
    if args.times is None:
        times = P.find_related(config.input_path,
                               f"{config.input_path.stem}_times.h5")
    else:
        times = P.resolve_input(args.times)
    if not times.exists():
        mid = P.resolve_input(args.mid) if args.mid is not None else None
        logger.info("Times file %s not found -- recovering it now (event_times).", times)
        times.parent.mkdir(parents=True, exist_ok=True)
        autobuild_times(config.input_path, times, mid)
    analyze(config, times, args.n_blocks, args.n_boot, args.bad_noise_factor)


if __name__ == "__main__":
    main()
