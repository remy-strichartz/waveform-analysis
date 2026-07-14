#!/usr/bin/env python3
"""
event_times.py
==============
The RUN-LEVEL half of the timing analysis: recover the real time axis of a run,
then measure everything about the run that only a clock can see and that is a
property of the RUN rather than of a channel.

  1. TIME AXIS.  The digitizer trigger time tag (a free-running counter, modulo
     2^k, that rolls over an unknown number of times between triggers) is
     unwrapped against the MIDAS wall clock (absolute but truncated to whole
     seconds).  Neither clock suffices alone; together they are exact.  The math
     -- and its bias guards, chief among them that the clock frequency is
     MEASURED, never assumed from a datasheet -- lives in
     file_manipulation/clock_recovery.py, shared with the converters so a freshly
     converted file already carries /event_time_rel_s.
  2. TRIGGER RATE & DATA INTEGRITY.  Rate vs time with Poisson errors, a
     uniformity test of the arrival times, and a census of the largest gaps
     against exponential order statistics (a DAQ dropout is a gap far larger than
     Poisson allows; a Poisson fluctuation is not a dropout).
  3. ARRIVAL STATISTICS & DEAD TIME.  Intervals against the exponential law
     (Monte-Carlo KS, so the p-value is valid WITH the rate estimated from the
     same data), lag-1 interval rank correlation (bursts / retriggering), and a
     dead-time bound: dead time removes short intervals entirely, so the smallest
     OBSERVED interval is a hard upper bound, quoted next to the statistical
     sensitivity floor of the run.  Dead time is veto INEFFICIENCY (a muon
     arriving during dead time is unrecorded), so the livetime fraction is the
     headline number -- and it is what hodoscope_efficiency quotes as its
     dead-time systematic.

These are all TRIGGER-level quantities: they read only the times, so they are
properties of the run, identical for every channel of it.  That is why they live
here and not in run_stability, which is per-channel (gain / baseline / noise) --
computing them once per channel produced three identical copies of one
measurement and invited reading them as three confirmations of it.

BIAS GUARDS
-----------
  * Rate and interval statistics use ALL DAQ triggers -- the hardware trigger
    defines the veto livetime, not an offline pulse finder.
  * Exponential and uniformity KS tests use Monte-Carlo p-values (the rate is
    estimated from the same sample, so textbook KS tables would be
    anti-conservative).
  * Every wrap decision carries a margin; marginal decisions and residuals beyond
    the wall clock's truncation-plus-latency budget (clock_recovery.RESID_FAIL_S)
    are counted and plotted.  (Rounding bounds every residual at half a rollover
    BY CONSTRUCTION, so a half-rollover test would be vacuous; the per-run drift
    band is the complementary check.)
  * --selftest runs the identical code path on synthetic (TTT, wall-clock) pairs
    with known truth -- including a long DAQ dropout and readout latency jitter --
    and fails loudly if the recovered times are off.

Output: <input-stem>_times.h5 with per-event `event_time_rel_s` (seconds since
the first event, fine-grained) and `event_time_unix_fine` (absolute, anchored to
the first event's wall second: the +/-1 s anchor uncertainty is irrelevant for
every relative quantity downstream), aligned row-for-row with /waveforms; plus a
<stem>_timing.json with the run-level numbers, and two figures.

Examples
--------
    python event_times.py --input run00270_ch9.h5 --save-plots
    python event_times.py --input run00270_ch9.h5 --mid run00270.mid --save-plots
    python event_times.py --selftest --no-show
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import kstest, spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "file_manipulation"))
import midas_to_h5 as midas  # noqa: E402  (reuse the house MIDAS parser)
from clock_recovery import (RESID_FAIL_S, choose_ttt_word,  # noqa: E402
                            recover, valid_header_rows)
from output_paths import (resolve_input, resolve_output,  # noqa: E402
                          resolve_results_dir, run_dir)

logger = logging.getLogger("event_times")

DEFAULT_WAVEFORM_DIR = _PROJECT_ROOT / "waveform_files"


# ===========================================================================
# Inputs: TTT word and wall clock
# ===========================================================================

def find_header_bank(h5: h5py.File) -> tuple[str, np.ndarray]:
    for name, ds in h5.items():
        if name.startswith("headers_") and isinstance(ds, h5py.Dataset) and ds.ndim == 2:
            return name, np.asarray(ds[()], dtype=np.int64)
    raise KeyError("No /headers_* dataset found; the input must be a midas_to_h5/"
                   "extract_channels output that carries the digitizer header bank.")


def parse_mid_wall(mid_path: Path, source_event_index: np.ndarray) -> np.ndarray:
    """Wall-clock (unix s) for exactly the events kept in the H5, by replaying the
    .mid event headers (payloads are seeked over, so this is fast) with the SAME
    running index midas_to_h5 stored in /source_event_index.  Exact alignment by
    construction -- no size or order heuristics."""
    wanted = set(int(i) for i in source_event_index)
    times: dict[int, int] = {}
    idx = 0
    with open(mid_path, "rb") as fh:
        while True:
            raw = fh.read(midas.EVENT_HEADER_SIZE)
            if len(raw) < midas.EVENT_HEADER_SIZE:
                break
            hdr = midas._parse_event_header(raw)
            fh.seek(hdr["data_size"], 1)
            if idx in wanted:
                times[idx] = hdr["timestamp"]
            idx += 1
    missing = wanted - set(times)
    if missing:
        raise ValueError(f"{len(missing)} events of /source_event_index not found "
                         f"in {mid_path} (first: {sorted(missing)[:5]}).")
    return np.array([times[int(i)] for i in source_event_index], dtype=np.int64)


def load_inputs(input_path: Path, mid_path: Path | None, ttt_word: int | None):
    with h5py.File(input_path, "r") as h5:
        bank, hdr = find_header_bank(h5)
        sei = (np.asarray(h5["source_event_index"][()], dtype=np.int64)
               if "source_event_index" in h5 else None)
        wall_ds = (np.asarray(h5["event_time_unix"][()], dtype=np.int64)
                   if "event_time_unix" in h5 else None)
    # An all-zero header row is an event whose header bank was missing at
    # conversion (the converter zero-fills to keep row alignment).  Its fake
    # tag would corrupt two intervals and permanently offset every later
    # event's time, so refuse LOUDLY rather than recover a silently wrong axis.
    n_zero = int(np.sum(~valid_header_rows(hdr)))
    if n_zero:
        raise SystemExit(
            f"{n_zero} of {len(hdr)} events have an all-zero header row (missing "
            "header bank at conversion); their fake tags would corrupt the "
            "recovery. Re-convert or re-extract with the current file_manipulation "
            "tools, which exclude such rows and fill their times from the wall "
            "clock -- the converted file then carries /event_time_rel_s already.")
    word = choose_ttt_word(hdr, ttt_word)
    ttt = hdr[:, word]

    if wall_ds is not None:
        logger.info("Wall clock: /event_time_unix from the H5.")
        wall = wall_ds
    else:
        if mid_path is None:
            raise SystemExit("The H5 has no /event_time_unix (older conversion); "
                             "pass --mid <run.mid> so the wall clock can be replayed.")
        if sei is None:
            raise SystemExit("The H5 has no /source_event_index; cannot align the "
                             ".mid wall clock. Re-convert with midas_to_h5.py.")
        logger.info("Wall clock: replaying event headers of %s.", mid_path)
        wall = parse_mid_wall(mid_path, sei)
    if len(wall) != len(ttt):
        raise ValueError(f"Wall clock ({len(wall)}) and TTT ({len(ttt)}) row counts differ.")
    if np.any(np.diff(wall) < 0):
        raise ValueError("Wall clock is not monotonic; the file is not in time order.")
    return ttt, wall, word, bank, sei


# ===========================================================================
# Rate, arrival statistics, dead time  (ALL DAQ triggers)
# ===========================================================================

def mc_ks_pvalue(sample: np.ndarray, kind: str, n_sim: int = 400,
                 seed: int = 20260704, ks_cap: int = 100_000) -> tuple[float, float]:
    """KS statistic + Monte-Carlo p-value with the parameter re-estimated in every
    replica, mirroring what was done to the data (Lilliefors-style).

    Scalability: the observed statistic and every MC replica are evaluated at the
    SAME size n_used = min(n, ks_cap).  The KS null distribution depends on that
    size, so the cap must apply to data and replicas alike; beyond ks_cap events the
    sample is randomly sub-sampled (KS on >~10^5 points is both O(n_sim * n log n)
    and hypersensitive -- it rejects on physically negligible deviations -- so a cap
    is the correct behaviour, not merely an optimization).  Replicas are drawn ONE AT
    A TIME, so peak memory is O(n_used).

    The p-value is floored at 1/n_sim rather than reported as an exact 0: with
    n_sim replicas nothing smaller is resolvable, and "p = 0.000" overstates it."""
    rng = np.random.default_rng(seed)
    n = sample.size
    if n > ks_cap:
        sample = sample[rng.choice(n, ks_cap, replace=False)]
    n_used = sample.size

    def _uniform_stat(s):
        lo, hi = s.min(), s.max()
        return kstest((s - lo) / (hi - lo), "uniform").statistic

    def _expon_stat(s):
        return kstest(s / s.mean(), "expon").statistic

    if kind == "uniform":
        stat = _uniform_stat(sample)
        draw, stat_of = (lambda: rng.random(n_used)), _uniform_stat
    elif kind == "expon":
        stat = _expon_stat(sample)
        draw, stat_of = (lambda: rng.exponential(1.0, n_used)), _expon_stat
    else:
        raise ValueError(kind)

    n_ge = sum(1 for _ in range(n_sim) if stat_of(draw()) >= stat)
    return float(stat), float(max(n_ge, 1) / n_sim)


def dead_time_bound(t: np.ndarray) -> dict:
    """Hard upper bound on the DAQ dead time, and the livetime fraction it implies.

    A dead time tau removes ALL intervals below tau, so the smallest interval that
    was actually observed is a hard upper bound on it.  The run can only RESOLVE a
    dead time large enough that >= 3 sub-tau intervals were expected (95% CL):
    tau_sens = 3/((n-1)*rate).  Quoting the two together is the whole point -- a
    bound far below the sensitivity floor would be a fluke, not a measurement.

    Livetime is the veto-relevant number: a muon arriving during dead time is an
    unrecorded, hence unvetoed, muon.  hodoscope_efficiency quotes 1 - livetime as
    its dead-time systematic."""
    n = t.size
    dt = np.diff(t)
    rate = (n - 1) / float(t[-1] - t[0])          # exponential MLE
    dt_min = float(dt.min())
    tau_sens = 3.0 / ((n - 1) * rate)
    return {"rate_hz": rate, "dt_min_s": dt_min, "tau_sens_s": tau_sens,
            "n_below_sens": int(np.sum(dt < tau_sens)),
            "exp_below_sens": float((n - 1) * (1.0 - np.exp(-rate * tau_sens))),
            "dead_time_bound_s": dt_min,
            "livetime_frac_min": 1.0 - rate * dt_min}


def arrival_statistics(t: np.ndarray) -> dict:
    n = t.size
    span = float(t[-1] - t[0])
    dt = np.diff(t)
    rate = (n - 1) / span

    ks_u, p_u = mc_ks_pvalue(t, "uniform")
    ks_e, p_e = mc_ks_pvalue(dt, "expon")
    rho, p_rho = spearmanr(dt[:-1], dt[1:])

    # Gap census: the expected number of the n-1 intervals at or above each observed
    # gap is (n-1)*exp(-rate*gap); a gap with expectation << 1 is a dropout, whereas
    # merely being the largest gap in the run is not evidence of anything.
    order = np.argsort(dt)[::-1][:5]
    gaps = [{"start_h": float(t[i] / 3600.0), "length_s": float(dt[i]),
             "n_expected": float((n - 1) * np.exp(-rate * dt[i]))} for i in order]

    out = {"n": n, "span_s": span,
           "ks_uniform": ks_u, "p_uniform": p_u,
           "ks_expon": ks_e, "p_expon": p_e,
           "lag1_spearman": float(rho), "lag1_p": float(p_rho),
           "gaps": gaps}
    out.update(dead_time_bound(t))
    return out


def binned_rate(t: np.ndarray, n_bins: int) -> dict:
    edges = np.linspace(t[0], t[-1], n_bins + 1)
    counts, _ = np.histogram(t, bins=edges)
    width_s = edges[1] - edges[0]
    expected = t.size / n_bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return {"centers_h": 0.5 * (edges[:-1] + edges[1:]) / 3600.0,
            "rate_hz": counts / width_s,
            "err_hz": np.sqrt(np.maximum(counts, 1.0)) / width_s,
            "chi2": chi2, "dof": n_bins - 1,
            "p": float(chi2_dist.sf(chi2, n_bins - 1))}


def run_level_analysis(t_rel: np.ndarray) -> tuple[dict, dict]:
    ast = arrival_statistics(t_rel)
    n_bins = int(np.clip(round(ast["span_s"] / 3600.0), 12, 60))
    return ast, binned_rate(t_rel, n_bins)


# ===========================================================================
# Plots
# ===========================================================================

def _finish(fig, name: str, outdir: Path, show: bool, save: bool) -> None:
    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_clock_qc(rec, outdir, show, save) -> None:
    """Everything that could catch a BAD clock recovery, in one figure.

    Top row -- the frequency is measured, not assumed: the consistency scan must show
    a single sharp basin, and the zoom locates it against the nearest nominal clock.
    Bottom row -- the recovery succeeded: fine-minus-wall residuals must stay inside a
    narrow band (1-s wall truncation + readout latency jitter) with NO steps of one
    rollover period (a step is exactly what a single wrap error looks like), and the
    wrap margins must cluster near 0 (unambiguous rounding).

    (The old 'tag structure' figure -- raw tag vs event index, tag-phase uniformity,
    raw-difference histogram -- was cut: it illustrated WHY unwrapping is hard but
    could not catch a failure.  A wrong TTT word does not survive the basin test
    below, which is the check that actually has teeth.)"""
    freqs, fom, f = rec["scan_freqs"], rec["scan_fom"], rec["freq_hz"]
    eps, t_h = rec["eps_s"], rec["t_rel"] / 3600.0
    med_eps = np.median(eps)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    axes[0, 0].semilogy(freqs / 1e6, fom, color="C0", lw=0.9)
    axes[0, 0].axvline(f / 1e6, color="C3", lw=1.0, ls="--", label=f"fitted {f / 1e6:.4f} MHz")
    axes[0, 0].set(xlabel="Trial clock frequency (MHz)", ylabel="Median |residual| (s)",
                   title=f"Consistency scan (intervals <= {rec['scan_cap_s']:.0f} s)")
    axes[0, 0].legend()

    near = np.abs(freqs - f) < 25 * (freqs[1] - freqs[0])
    axes[0, 1].semilogy(freqs[near] / 1e6, fom[near], color="C0", marker="o", ms=3)
    axes[0, 1].axvline(f / 1e6, color="C3", lw=1.0, ls="--")
    axes[0, 1].set(xlabel="Trial clock frequency (MHz)", ylabel="Median |residual| (s)",
                   title=f"Basin zoom -- {rec['nominal_mhz']:g} MHz "
                         f"{rec['ppm_vs_nominal']:+.1f} ppm")

    axes[1, 0].plot(t_h, eps - med_eps, ".", ms=1.5, alpha=0.4, color="C0")
    for sign in (+1, -1):
        axes[1, 0].axhline(sign * 0.5 * rec["period_s"], color="C3", ls="--", lw=1.0,
                           label="+/- half rollover (wrap error)" if sign > 0 else None)
    axes[1, 0].set(xlabel="Run time (h)", ylabel="Fine time - wall clock (s, median-centered)",
                   title=f"Residual band [{rec['eps_band_s'][0] - med_eps:+.2f}, "
                         f"{rec['eps_band_s'][1] - med_eps:+.2f}] s (95%)")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].hist(rec["margin"], bins=60, range=(-0.5, 0.5), color="C2", alpha=0.8)
    for sign in (+1, -1):
        axes[1, 1].axvline(sign * 0.35, color="C3", ls="--", lw=1.0)
    axes[1, 1].set(xlabel="Wrap-count margin (fraction of a rollover)", ylabel="Intervals",
                   title=f"Wrap margins ({rec['n_marginal']} marginal, "
                         f"{rec['n_fail']} failed / {rec['n_wraps_total']:,} rollovers)")
    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Trigger-time recovery: quality control", fontsize=13)
    fig.tight_layout()
    _finish(fig, "time_1_clock", outdir, show, save)


def plot_rate_and_deadtime(t: np.ndarray, ast: dict, rb: dict, outdir, show, save) -> None:
    """The run's integrity, in one figure: is the rate stationary, are there
    dropouts, and how much dead time could be hiding?

    The dead-time panel is the cumulative interval distribution against the
    no-dead-time expectation.  It subsumes the plain interval histogram that used to
    sit beside it (same distribution, same exponential overlay, linear axes) while
    also carrying the bound, so only this one is drawn.  The old successive-interval
    scatter is gone too: the lag-1 rank correlation it visualised is a null (rho is
    quoted in the title and the summary), and 15k points of noise cannot show that
    any better than the number does."""
    th = t / 3600.0
    dt = np.diff(t)
    lam = ast["rate_hz"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].plot(th, np.arange(t.size), color="C0", lw=1.0, label="data")
    axes[0].plot([th[0], th[-1]], [0, t.size - 1], color="C3", ls="--", lw=1.0,
                 label="constant rate")
    axes[0].set(xlabel="Run time (h)", ylabel="Cumulative events",
                title=f"Arrival uniformity: MC-KS p = {ast['p_uniform']:.3f}")
    axes[0].legend()

    axes[1].errorbar(rb["centers_h"], rb["rate_hz"] * 3600.0, yerr=rb["err_hz"] * 3600.0,
                     fmt="o", ms=3, color="C0", lw=1.0, capsize=2)
    axes[1].axhline(lam * 3600.0, color="C3", ls="--", lw=1.0,
                    label=f"mean {lam * 3600.0:.1f} /h")
    axes[1].set(xlabel="Run time (h)", ylabel="Trigger rate (events/h)",
                title=f"Binned rate: chi2/dof = {rb['chi2']:.0f}/{rb['dof']}, p = {rb['p']:.3f}")
    axes[1].legend()

    srt = np.sort(dt)
    axes[2].loglog(srt, np.arange(1, srt.size + 1), drawstyle="steps-post", color="C0",
                   label="observed < dt")
    fine = np.logspace(np.log10(max(srt[0] * 0.3, 1e-4)), np.log10(srt[-1]), 200)
    axes[2].loglog(fine, dt.size * (1.0 - np.exp(-lam * fine)), color="C3", ls="--",
                   label="expected (no dead time)")
    axes[2].axvline(ast["dt_min_s"], color="C2", lw=1.2,
                    label=f"hard bound: dead time < {ast['dt_min_s'] * 1e3:.1f} ms")
    axes[2].axvline(ast["tau_sens_s"], color="C7", lw=1.2, ls=":",
                    label=f"95% sensitivity floor {ast['tau_sens_s'] * 1e3:.1f} ms")
    axes[2].set(xlabel="Interval (s)", ylabel="Cumulative intervals",
                title=f"Dead time (intervals: Exp p = {ast['p_expon']:.2f}, "
                      f"lag-1 rho = {ast['lag1_spearman']:+.3f})")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Trigger rate, data integrity and dead time (all DAQ triggers)", fontsize=13)
    fig.tight_layout()
    _finish(fig, "time_2_rate", outdir, show, save)


# ===========================================================================
# Output
# ===========================================================================

def write_output(out_path: Path, ttt, wall, sei, rec, word: int, bank: str,
                 input_path: Path) -> None:
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("event_time_rel_s", data=rec["t_rel"])
        h5.create_dataset("event_time_unix_fine", data=rec["t_unix"])
        h5.create_dataset("wall_time_unix", data=wall.astype(np.uint32))
        h5.create_dataset("ttt_raw", data=ttt.astype(np.uint32))
        if sei is not None:
            h5.create_dataset("source_event_index", data=sei)
        h5.attrs["source_h5"] = str(input_path)
        h5.attrs["ttt_header_bank"] = bank
        h5.attrs["ttt_word"] = word
        h5.attrs["ttt_modulus"] = rec["modulus"]
        h5.attrs["ttt_freq_hz"] = rec["freq_hz"]
        h5.attrs["ttt_rollover_s"] = rec["period_s"]
        h5.attrs["nominal_clock_mhz"] = rec["nominal_mhz"]
        h5.attrs["ppm_vs_nominal"] = rec["ppm_vs_nominal"]
        h5.attrs["n_wrap_marginal"] = rec["n_marginal"]
        h5.attrs["n_wrap_failed"] = rec["n_fail"]
        h5.attrs["resid_band_95_s"] = rec["eps_band_s"]
        h5.attrs["note"] = ("event_time_unix_fine is anchored to the FIRST event's "
                            "wall-clock second: absolute offset uncertain by +/-1 s; "
                            "relative times carry the full TTT precision.")
    logger.info("Wrote %s", out_path)


def write_json(outdir: Path, stem: str, rec, ast: dict, rb: dict) -> Path:
    def _clean(v):
        return v.tolist() if isinstance(v, np.ndarray) else v
    result = {
        "clock": {k: _clean(rec[k]) for k in
                  ("freq_hz", "modulus", "period_s", "nominal_mhz", "ppm_vs_nominal",
                   "n_wraps_total", "n_marginal", "n_fail", "eps_band_s")},
        "arrival": {k: _clean(v) for k, v in ast.items()},
        "binned_rate": {k: _clean(v) for k, v in rb.items()},
    }
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{stem}_timing.json"
    out_json.write_text(json.dumps(result, indent=2, default=float))
    logger.info("Wrote %s", out_json)
    return out_json


def print_summary(rec, ast: dict | None, rb: dict | None, n_events: int) -> None:
    dur = rec["t_rel"][-1]
    print()
    print("=" * 66)
    print("Event-time recovery")
    print("-" * 66)
    print(f"Events                     : {n_events:,}")
    print(f"Run duration               : {dur:,.1f} s  ({dur / 3600:.2f} h)")
    print(f"TTT modulus                : 2^{rec['modulus'].bit_length() - 1}")
    print(f"TTT clock (measured)       : {rec['freq_hz'] / 1e6:.6f} MHz "
          f"({rec['nominal_mhz']:g} MHz {rec['ppm_vs_nominal']:+.1f} ppm)")
    print(f"Rollover period            : {rec['period_s']:.3f} s")
    print(f"Rollovers resolved         : {rec['n_wraps_total']:,}")
    print(f"Marginal wrap decisions    : {rec['n_marginal']}  (|margin| > 0.35)")
    print(f"Failed wrap decisions      : {rec['n_fail']}  (|resid| > {RESID_FAIL_S:g} s "
          f"wall budget)")
    print(f"Residual band (95%)        : [{rec['eps_band_s'][0]:+.2f}, "
          f"{rec['eps_band_s'][1]:+.2f}] s vs wall clock")
    if ast is None:
        print("=" * 66)
        return
    print("-" * 66)
    print("Run-level rate, integrity and dead time (all DAQ triggers)")
    print("-" * 66)
    print(f"Mean trigger rate          : {ast['rate_hz'] * 3600:.2f} events/h "
          f"({ast['rate_hz'] * 1e3:.3f} mHz)")
    print(f"Arrival uniformity (MC-KS) : p = {ast['p_uniform']:.3f}")
    print(f"Binned-rate chi2           : {rb['chi2']:.0f}/{rb['dof']} dof, p = {rb['p']:.3g}")
    print(f"Exponential intervals      : p = {ast['p_expon']:.3f} (MC-KS)")
    print(f"Lag-1 interval correlation : rho = {ast['lag1_spearman']:+.3f} "
          f"(p = {ast['lag1_p']:.2f})")
    g = ast["gaps"][0]
    print(f"Largest gap                : {g['length_s']:.0f} s at {g['start_h']:.1f} h "
          f"(expected {g['n_expected']:.2f} gaps this long)")
    print(f"Dead time (hard bound)     : < {ast['dt_min_s'] * 1e3:.1f} ms "
          f"(sensitivity floor {ast['tau_sens_s'] * 1e3:.1f} ms)")
    print(f"Livetime fraction          : > {100 * ast['livetime_frac_min']:.4f} %  "
          f"[dead-time systematic on any efficiency: "
          f"< {100 * (1 - ast['livetime_frac_min']):.4f} %]")
    if rb["p"] < 0.01:
        print()
        print("  NOTE: the trigger rate is NOT stationary.  Check run_stability's")
        print("        per-channel noise tracker for a matching disturbance, and")
        print("        exclude the affected window with mv_pipeline --exclude-hours.")
    print("=" * 66)


# ===========================================================================
# Self-test (closure on synthetic clocks with known truth)
# ===========================================================================

def selftest(show: bool, save: bool, outdir: Path) -> None:
    """Synthetic (TTT, wall) with known truth through the REAL code path: exponential
    arrivals, a non-round clock frequency, a 2-h DAQ dropout and readout-latency
    jitter (incl. occasional 1.5-s stalls).  The accuracy floor of the method is the
    frequency-fit precision (sub-ppm against a 1-s-truncated wall clock), which drifts
    by span * df/f across the run: the test demands < 2 ppm on f, < 30 ms worst-case
    relative time over the ~22 h span, and zero wrap failures."""
    rng = np.random.default_rng(20260704)
    f_true, modulus, n = 98.7654e6, 1 << 30, 6000
    gaps = rng.exponential(12.0, n - 1)
    gaps[3000] = 7200.0                                  # DAQ dropout
    t_true = np.concatenate([[0.0], np.cumsum(gaps)])
    latency = rng.uniform(0.05, 0.45, n)
    latency[rng.random(n) < 0.01] += 1.5                 # occasional slow readout
    wall = np.floor(t_true + latency + 1.7e9).astype(np.int64)
    ttt = np.mod(123456789 + np.round(t_true * f_true).astype(np.int64), modulus)

    rec = recover(ttt.astype(np.int64), wall)
    err = rec["t_rel"] - t_true
    err -= np.median(err)
    max_err = float(np.max(np.abs(err)))
    ppm = (rec["freq_hz"] - f_true) / f_true * 1e6
    print_summary(rec, None, None, n)
    print(f"SELFTEST truth f = {f_true / 1e6:.4f} MHz | recovered "
          f"{rec['freq_hz'] / 1e6:.4f} MHz ({ppm:+.2f} ppm) | max |t err| = "
          f"{max_err * 1e3:.3f} ms | wrap failures = {rec['n_fail']}")
    plot_clock_qc(rec, outdir, show, save)
    ok = max_err < 30e-3 and abs(ppm) < 2 and rec["n_fail"] == 0
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover fine per-event trigger times from the CAEN trigger time "
                    "tag + MIDAS wall clock, and measure the run-level trigger rate, "
                    "data integrity and dead time.")
    p.add_argument("--input", type=Path, default=None,
                   help="Waveform .h5 carrying /headers_* (bare names are looked up "
                        "in waveform_files/).")
    p.add_argument("--mid", type=Path, default=None,
                   help="Original MIDAS .mid file; needed only when the H5 lacks "
                        "/event_time_unix (older conversions).")
    p.add_argument("--ttt-word", type=int, default=None,
                   help="Header word holding the trigger time tag. Default: auto-detect.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .h5 (default: the run's own recovered time axis, "
                        "waveform_files/<run>/times/<input-stem>_times.h5).")
    p.add_argument("--no-show", "--no-plot", dest="no_show", action="store_true",
                   help="Do not open plot windows.")
    p.add_argument("--save-plots", action="store_true",
                   help="Save the figures + <stem>_timing.json to timing_stability/"
                        "timing_stability_results/times/<input-stem>_times_results[_N]/ "
                        "(the recovered <stem>_times.h5 stays in waveform_files/, in its "
                        "run's times/ folder).")
    p.add_argument("--overwrite", action="store_true",
                   help="Write into <input-stem>_times_results/ even if it exists, overwriting "
                        "the previous run's figures. Default: a re-run gets a fresh _N folder.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the synthetic closure test instead of analysing a file.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _resolve(path: Path | None) -> Path | None:
    """Bare name -> wherever waveform_files/ keeps it; explicit path -> as given."""
    return None if path is None else resolve_input(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    show, save = not args.no_show, args.save_plots

    if args.selftest:
        selftest(show, save, DEFAULT_WAVEFORM_DIR / "event_times_selftest")
        return

    if args.input is None:
        raise SystemExit("Pass --input (or --selftest).")
    input_path = _resolve(args.input)
    mid_path = _resolve(args.mid)
    # The recovered-clock .h5 is a SHARED intermediate that run_stability (and every other
    # channel) looks up by name, so it belongs with the waveform files -- specifically in the
    # times/ folder of the run it was recovered from, which is where find_related looks.
    # Only the figures and the run-level JSON go into this run's results folder.
    out_path = resolve_output(args.output, f"{input_path.stem}_times.h5",
                              into=run_dir(input_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outdir = resolve_results_dir(__file__, input_path.stem, program="times", group="times",
                                 overwrite=args.overwrite)

    ttt, wall, word, bank, sei = load_inputs(input_path, mid_path, args.ttt_word)
    rec = recover(ttt, wall)
    ast, rb = run_level_analysis(rec["t_rel"])
    print_summary(rec, ast, rb, len(ttt))
    write_output(out_path, ttt, wall, sei, rec, word, bank, input_path)
    if save:
        write_json(outdir, input_path.stem, rec, ast, rb)
    plot_clock_qc(rec, outdir, show, save)
    plot_rate_and_deadtime(rec["t_rel"], ast, rb, outdir, show, save)


if __name__ == "__main__":
    main()
