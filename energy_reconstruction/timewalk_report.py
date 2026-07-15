#!/usr/bin/env python3
"""
timewalk_report.py
==================
Diagnose the residual time-walk that compare.py reports as a single number.

`timewalk_slope` is a LINE fit to the binned-median peak-time vs amplitude.  That number
is only meaningful if (a) the trend really is a line, (b) it is stable against its own
binning, and (c) the estimator does not manufacture it.  On run00270 none of those can be
assumed: the PMT channels' trend is curved (the line fits at chi2/dof 8-13), and on the
channels whose amplitude range never reaches MIP scale (ch4, ch7) the slope moves by
several samples/amp when you merely change n_bins.  This tool measures all three, plus the
cause.

    python timewalk_report.py --input run00270_ch0.h5 --save-plots

What it runs, per channel:

  TREND    binned-median peak time vs amplitude WITH per-bin error bars (the shipped
           profile has none), fit by three models -- constant, linear (a + b*A) and
           a + b/A -- compared by AIC.  Also refits the line above `high_amp` only: a
           trend that lives entirely in the low-amplitude bins is not a walk of the
           physics pulses.

  BINNING  the same slope at n_bins 8..30.  A spread comparable to the slope means the
           reported number is an artifact of the binning, not a measurement.

  CLOSURE  a null test: synthetic events with NO true walk by construction (A_i *
           the channel's own template, plus colored noise with its own PSD shape scaled
           to its own noise sigma) pushed through the SAME kernel, trigger and OF
           amplitude.  The measured slope MUST come back ~0.  Run with a deliberately
           mis-centered search window too, since a noise-dominated low-amplitude
           population regressing toward the window center is the obvious way an
           unbiased estimator could still fake a walk.  (Measured: it does not -- every
           scenario returns |slope| < 0.25, so a nonzero slope on real data is real.)

  TEMPLATE rebuild the template from DIM pulses only and from BRIGHT pulses only, re-trigger
           the real data with each, and re-measure the walk.  On run00270's analog bank it
           roughly DOUBLES with a bright template and halves with a dim one.  TEMPLATE
           QUALITY IS REFUTED as the cause (measured 2026-07-14 on ch1 and ch2): degrading
           the bright template to the dim template's quality -- colored noise added at the
           dim averaging level, smearing, or a rebuild from a dim-noise-sized bright SUBSET
           -- leaves the walk at the bright value (ch2: dim -1.06 vs bright -2.76, degraded
           bright -2.76/-2.76/-3.04), and a triage-CLEAN-only rebuild of both bands keeps
           the split (-1.04 / -3.01), so pileup/clipping contamination is out too.
           The swap effect then DECOMPOSES (measured on ch2):
             * the bright-band MEAN template is ~4 samples wider in FWHM than the median
               of the same aligned rows -- a minority tail of events (unresolved pileup /
               afterpulses, bright-biased) pulls the MEAN wide.  Real build artifact, but
               worth only ~10% of the split (median template: walk -2.45 vs mean -2.76);
             * the split SURVIVES width matching (bright MEDIAN, FWHM 58.0 vs dim 58.6,
               still walks -2.45 vs -1.08) and a 0-3 sample smear scan (-2.45 -> -2.35),
               so bulk width and smearing are out;
             * after removing a walk-NEUTRAL ~1.7-sample window-phase offset, the dim-
               and bright-built median templates agree to <~0.7% of peak everywhere
               EXCEPT the EARLY TAIL (+3..+60 samples past the peak, largest ~+18, 1.7%
               of peak) -- but that tail structure is REFUTED as the walk's driver: the
               band-stacked early-tail amplitude trend, measured across 8 channels, does
               not correlate with the shipped walk (Pearson -0.34 / Spearman -0.26;
               signs inconsistent -- ch0 walks -1.18 with a zero tail trend, ch3 has the
               strongest tail trend and the smallest walk).
           So the swap effect's operative template difference is real, reproducible,
           and still UNIDENTIFIED after noise, smear, width, subset size, contamination
           and early-tail content were each individually matched or refuted.  One gap
           worth noting for the next attempt: the CLOSURE test certifies the estimator
           on GAUSSIAN colored noise -- a walk arising from the interplay of the real
           (non-Gaussian, burst/pickup-phase) noise with different template fine
           structure would evade it.

  SHAPE    pulse shape per amplitude band, template-free, WITH THE PICKUP NOTCHED OUT
           (see shape_vs_amplitude -- notching is not optional; un-notched, this panel lies).
           Measured: the RISING EDGE (rise 10-90, crest lag) is amplitude-INDEPENDENT on
           every channel.  Since 2026-07-14 the panel also measures the WIDTH (FWHM and
           the decay-side half-widths), because the template-swap probe found the
           dim/bright-band templates differ in exactly that -- the walk's remaining suspect.

WHAT CAUSES THE WALK IS STILL UNKNOWN, but it is now cornered.  Measured on run00270 and RULED
OUT: the coherent pickup (notch the line out of the data, rebuild the whole noise model, and
70-80% of the walk survives); an amplitude-dependent RISING EDGE (rise and crest lag are flat
once notched -- note that scan never covered the width/decay side, which is exactly where the
template-swap finding above points); a second "fast" pulse species in the dim population (that
was the pickup artifact, not a species); template QUALITY (see TEMPLATE); and LIGHT TRANSIT /
POSITION (2026-07-15).  That last one deserves its epitaph: the analog bank is the 8
fiber-swirl mini-modules of the CUPID prototype panel (arXiv:2505.06129), the trigger
footprint sits at the ch0 corner, and cross-channel amplitudes DO encode muon position
(adjacency reproduced, corr vs layout distance rho=-0.74) -- but at fixed amplitude the OF
peak time is INDEPENDENT of a validated rest-of-bank position proxy (rho 0.00-0.04, n~3000
per band, ch0-ch3), the walk does not collapse within position bands, and 94% of triggered
muons land on ch0's cell so the position lever (~0.1 samples of transit) is 20-40x too small
regardless.  The walk itself is real -- CLOSURE
says the estimator is unbiased -- but small: -2.3 to -3.7 samples across each analog channel's
whole amplitude range, well under one sigma of the ~6.5-sample timing jitter, and harmless for
energy reconstruction.  Don't re-propose the four dead hypotheses without new evidence.

Nothing here is a cut or a correction -- it is a QC report.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mv_pipeline as P
import optimal_filter as OF

logger = logging.getLogger("timewalk_report")

HIGH_AMP = 0.5          # "clear of the noise-trigger cluster", in MIP-scale amplitude units
N_BINS_SCAN = (8, 10, 12, 16, 20, 24, 30)
N_CLOSURE = 20_000


# ---------------------------------------------------------------------------------
# Trend: binned medians WITH error bars, and the competing shapes
# ---------------------------------------------------------------------------------

def trend(amp: np.ndarray, peak: np.ndarray, n_bins: int = 20, min_per_bin: int = 50):
    """Binned-median peak time vs amplitude, with a median error bar per bin.

    Same estimator as mv_pipeline.timewalk_profile (median per amplitude bin, robust to the
    amp~0 noise-trigger cluster) but it also returns sigma_median = 1.2533*sigma_MAD/sqrt(n),
    without which the linear and 1/A models cannot be compared: the low-amplitude bins are
    where the two shapes differ most AND where the medians are most precise, so an unweighted
    fit throws away exactly the discriminating information.  The bin's x is the MEDIAN
    amplitude in it, not the bin center -- inside a steeply-falling spectrum the two differ."""
    ok = np.isfinite(amp) & np.isfinite(peak)
    a, t = amp[ok], peak[ok]
    if a.size <= 10:
        return (np.array([]),) * 4
    lo, hi = np.percentile(a, [1.0, 99.0])
    edges = np.linspace(lo, hi, n_bins + 1)
    c, m, e, n = [], [], [], []
    for i in range(n_bins):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        k = int(sel.sum())
        if k < min_per_bin:
            continue
        ti = t[sel]
        med = float(np.median(ti))
        sig = 1.4826 * float(np.median(np.abs(ti - med)))
        c.append(float(np.median(a[sel])))
        m.append(med)
        e.append(max(1.2533 * sig / np.sqrt(k), 1e-3))
        n.append(k)
    return np.array(c), np.array(m), np.array(e), np.array(n)


def _wls(X, y, e):
    """Weighted least squares -> (params, chi2, dof, aic)."""
    w = 1.0 / e
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    chi2 = float(np.sum(((y - X @ beta) / e) ** 2))
    return beta, chi2, len(y) - X.shape[1], chi2 + 2 * X.shape[1]


def fit_shapes(c, m, e) -> dict[str, Any]:
    """Constant vs linear vs a + b/A, by AIC.

    The 1/A shape is what a FIXED-SIZE (in ADC) perturbation would produce -- its relative
    pull scales as 1/amplitude.  It is in here because the coherent pickup was the natural
    suspect for the analog bank's walk; the fit REFUTES that (1/A wins only on the two PMT
    channels, which carry no pickup, and loses on 6 of the 8 that do).  It stays as the
    curvature test: when 1/A wins, the "walk" lives at low amplitude and the reported slope
    is not describing the physics pulses."""
    one = np.ones_like(c)
    out = {"const": _wls(one[:, None], m, e),
           "linear": _wls(np.column_stack([one, c]), m, e),
           "inv_amp": _wls(np.column_stack([one, 1.0 / np.maximum(c, 1e-6)]), m, e)}
    hi = c >= HIGH_AMP
    out["slope_high_amp"] = (float(_wls(np.column_stack([one[hi], c[hi]]), m[hi], e[hi])[0][1])
                             if int(hi.sum()) >= 3 else float("nan"))
    return out


def binning_stability(amp: np.ndarray, peak: np.ndarray) -> tuple[np.ndarray, float]:
    s = np.array([P.timewalk_profile(amp, peak, n_bins=n)[0] for n in N_BINS_SCAN], float)
    return s, float(np.nanmax(s) - np.nanmin(s))


# ---------------------------------------------------------------------------------
# Reconstruction on an arbitrary block (shared by the closure and template tests)
# ---------------------------------------------------------------------------------

def _reconstruct(corrected, prep, config, kernel, peak_offset, resp_rms):
    """Trigger + OF amplitude + peak time on `corrected`, through the pipeline's own code.

    The synthetic/rebuilt-template runs must not re-implement any of the estimator or the
    test proves nothing about the estimator that ships."""
    fired, wstart, peak, _ = P.filter_trigger(corrected, kernel, config,
                                              resp_rms=resp_rms, peak_offset=peak_offset)
    if int(fired.sum()) < 200:
        return None, None
    sub = dataclasses.replace(
        prep, corrected=np.ascontiguousarray(corrected[fired]),
        events=pd.DataFrame({"event_number": np.arange(int(fired.sum())),
                             "window_start": wstart[fired],
                             "peak_subsample": peak[fired]}))
    amp = OF.optimal_filter_amplitude(sub, prep.psd, config, OF.of_aligned_windows(sub, config))
    return amp, peak[fired]


def _colored_noise(n, length, psd, sigma, rng):
    """Noise with the SHAPE of `psd`, rescaled to RMS `sigma`.

    Rescaling to the measured sigma sidesteps the PSD's normalization convention entirely --
    the closure test only needs the right spectrum and the right total noise power."""
    f_src = np.fft.rfftfreq(psd.size)
    p = np.interp(np.fft.rfftfreq(length), f_src, psd[:f_src.size])
    amp = np.sqrt(np.maximum(p, 1e-30))
    spec = (rng.normal(size=(n, amp.size)) + 1j * rng.normal(size=(n, amp.size))) * amp
    x = np.fft.irfft(spec, n=length, axis=1)
    return x * (sigma / max(float(np.std(x)), 1e-12))


def closure_test(prep, config, kernel, peak_offset, resp_rms, real_amp) -> dict[str, float]:
    """Synthetic events with NO true walk -> the slope the estimator invents on its own."""
    rng = np.random.default_rng(config.seed)
    center = P.pulse_center(prep.length, config)
    T, Lt = prep.template, prep.template.size
    pool = real_amp[np.isfinite(real_amp) & (real_amp > 0)]
    if pool.size < 100:
        return {}
    out = {}
    for tag, mu, jit in (("centered", 0.0, 0.0), ("jitter", 0.0, 3.0),
                         ("offset +3", +3.0, 3.0), ("offset -3", -3.0, 3.0)):
        A = rng.choice(pool, size=N_CLOSURE, replace=True)
        true_peak = np.full(N_CLOSURE, center + mu, float)
        if jit:
            true_peak += rng.normal(0.0, jit, N_CLOSURE)
        start = true_peak - peak_offset
        i0 = np.clip(np.floor(start).astype(int), 0, prep.length - Lt)
        pulses = P._shift(np.tile(T, (N_CLOSURE, 1)), start - np.floor(start)) * A[:, None]
        rec = _colored_noise(N_CLOSURE, prep.length, prep.psd, prep.noise_sigma, rng)
        cols = i0[:, None] + np.arange(Lt)[None, :]
        np.add.at(rec, (np.arange(N_CLOSURE)[:, None], cols), pulses)
        amp, peak = _reconstruct(rec, prep, config, kernel, peak_offset, resp_rms)
        out[tag] = (float("nan") if amp is None
                    else float(P.timewalk_profile(amp, peak)[0]))
    return out


def template_swap(prep, config, real_amp) -> dict[str, float]:
    """Walk on the REAL data with the template rebuilt from dim / from bright pulses.

    template_sigma=0 because the brightness cut has already been made by selecting the
    amplitude band -- letting build_template re-cut would just hand both bands the same
    bright pulses and the test would trivially pass."""
    good = np.isfinite(real_amp)
    if int(good.sum()) < 1000:
        return {}
    q = np.nanpercentile(real_amp[good], [35, 55, 80, 99])
    out = {}
    for tag, (lo, hi) in {"dim": (q[0], q[1]), "bright": (q[2], q[3])}.items():
        sel = good & (real_amp >= lo) & (real_amp <= hi)
        if int(sel.sum()) < 400:
            out[tag] = float("nan")
            continue
        cfg = dataclasses.replace(config, template_sigma=0.0)
        _, tmpl, _, _ = P.build_template(np.ascontiguousarray(prep.corrected[sel]),
                                         prep.noise_sigma, cfg)
        po = int(np.argmax(np.abs(tmpl)))
        kern = P.optimal_filter_kernel(tmpl, prep.psd)
        lo_s = max(P.pulse_center(prep.length, cfg) - po - cfg.search_pad, 0)
        rms = P._response_noise_rms(prep.corrected, kern, lo_s)
        amp, peak = _reconstruct(prep.corrected, dataclasses.replace(prep, template=tmpl),
                                 cfg, kern, po, rms)
        out[tag] = float("nan") if amp is None else float(P.timewalk_profile(amp, peak)[0])
    return out


# ---------------------------------------------------------------------------------
# Shape vs amplitude: the cause the template swap points at, measured template-free
# ---------------------------------------------------------------------------------

def _crossing(y, level, i_hi):
    below = np.where(y[:i_hi + 1] <= level)[0]
    if below.size == 0:
        return np.nan
    i = int(below[-1])
    if i + 1 > i_hi or y[i + 1] == y[i]:
        return float(i)
    return i + (level - y[i]) / (y[i + 1] - y[i])


def _crossing_fall(y, level, i_lo):
    """Sub-sample index of the first downward crossing of `level` at/after i_lo."""
    below = np.flatnonzero(y[i_lo:] <= level)
    if below.size == 0:
        return np.nan
    i = i_lo + int(below[0])
    if i == i_lo or y[i - 1] == y[i]:
        return float(i)
    return (i - 1) + (y[i - 1] - level) / (y[i - 1] - y[i])


def _stack_shape(w):
    """Shape metrics of one amplitude-normalized, half-max-aligned median stack.

    rise (10-90%), crest lag (peak - rising half-max = the LEFT half-width), FWHM, and
    the DECAY-side half-widths (peak -> falling 50% and peak -> falling 1/e).  The decay
    side is measured since 2026-07-14: the template-swap probe showed the dim/bright-band
    templates differ in WIDTH (bright ~5-8% wider FWHM, and the walk tracks it) while the
    rise and crest lag are flat -- so which SIDE carries the width is the open question
    this panel now answers."""
    pk = int(np.argmax(np.median(w, axis=0)))
    t50 = np.array([_crossing(r, 0.5 * max(r[pk], 1e-9), pk) for r in w])
    fin = np.isfinite(t50)
    stack = np.median(P._shift(np.asarray(w[fin], float), t50[fin] - np.median(t50[fin])), axis=0)
    p2 = int(np.argmax(stack))
    if p2 < 3 or p2 + 1 >= stack.size:
        return None
    top = float(stack[p2])
    t10, t50s, t90 = (_crossing(stack, f * top, p2) for f in (0.10, 0.50, 0.90))
    d = float(P._parabolic_delta(np.array([stack[p2 - 1]]), np.array([stack[p2]]),
                                 np.array([stack[p2 + 1]]))[0])
    peak_t = p2 + d
    f50 = _crossing_fall(stack, 0.50 * top, p2)
    f1e = _crossing_fall(stack, top / np.e, p2)
    return {"rise": float(t90 - t10), "lag": float(peak_t - t50s),
            "fwhm": float(f50 - t50s) if np.isfinite(f50) else float("nan"),
            "fall50": float(f50 - peak_t) if np.isfinite(f50) else float("nan"),
            "fall_1e": float(f1e - peak_t) if np.isfinite(f1e) else float("nan")}


def shape_vs_amplitude(prep, config, windows, amp, n_bands: int = 6):
    """Rising-edge shape per amplitude band, measured WITHOUT the template.

    Each pulse is aligned on its OWN half-max leading-edge crossing (a constant-fraction pick:
    no template, no argmax -- an argmax pick would re-phase the pickup, and a template pick
    would make the whole test circular), normalized by its OF amplitude, and median-stacked.
    `peak - halfmax` is the crest lag a single fixed template is forced to assume is constant.

    THE PICKUP MUST BE NOTCHED OUT FIRST, and this is not optional.  On amplitude-SELECTED
    low-SNR analog pulses the ~9.5-sample pickup ripple re-coheres in the median stack and drags
    the argmax a FULL PICKUP PERIOD off the true crest: run00270's low-amplitude bands read a
    crest lag of ~13 samples where the pulse actually has ~23 (23 - 9.5 = 13.5), and the 10-90%
    rise comes out ~10 samples too long.  Un-notched, this fakes a large amplitude-dependent
    pulse shape on every analog channel (ch1's rise appears to run 38 -> 27 samples dim->bright;
    notched, it is 30 -> 29, i.e. flat) -- which in turn fakes a physical cause for the residual
    time-walk, and fakes a second "fast pulse species" in the dim population.  All three of those
    were measured, believed, and are WRONG.

    So the reported numbers are the NOTCHED ones.  The raw values are kept alongside purely so the
    artifact stays visible: a large raw-vs-notched gap means the channel's pickup is strong, not
    that its pulses change shape.  NOTE that an injected-noise control does NOT reproduce this --
    amplitude selection correlates the pickup PHASE, and unselected noise does not."""
    lines = P._pickup_lines(prep.corrected[:, :P.noise_stop(prep.length, config)],
                            config.align_notch_snr)
    ok = np.isfinite(amp) & (amp > 0)
    edges = np.percentile(amp[ok], np.linspace(20, 99, n_bands + 1))
    rows = []
    for i in range(n_bands):
        sel = ok & (amp >= edges[i]) & (amp < edges[i + 1])
        if int(sel.sum()) < 200:
            continue
        w = windows[sel] / amp[sel][:, None]
        raw = _stack_shape(w)
        wn = P._notch_rows(np.ascontiguousarray(w), lines) if lines else w
        got = _stack_shape(np.asarray(wn, float))
        if got is None or raw is None:
            continue
        rows.append({"amp": float(np.mean(amp[sel])), "n": int(sel.sum()),
                     "rise_10_90": got["rise"], "peak_minus_halfmax": got["lag"],
                     "fwhm": got["fwhm"], "fall_50": got["fall50"],
                     "fall_1e": got["fall_1e"],
                     "rise_10_90_raw": raw["rise"], "peak_minus_halfmax_raw": raw["lag"]})
    return rows, [1.0 / c for c, _ in lines]


# ---------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------

def _verdict(fits) -> str:
    d_aic = fits["inv_amp"][3] - fits["linear"][3]
    if d_aic < -2:
        return "CURVED (1/A beats a line): the walk lives at low amplitude, not in the MIP pulses"
    if d_aic > 2:
        return "LINEAR: a genuine amplitude-proportional walk"
    return "AMBIGUOUS: neither shape is preferred"


def analyze(config: P.Config) -> dict[str, Any]:
    prep = P.prepare(config)
    config = prep.config
    windows = OF.of_aligned_windows(prep, config)
    amp = OF.optimal_filter_amplitude(prep, prep.psd, config, windows)
    peak = prep.events["peak_subsample"].to_numpy()

    shipped = P.timewalk_slope(amp, peak)
    c, m, e, n = trend(amp, peak)
    fits = fit_shapes(c, m, e) if c.size >= 4 else None
    scan, spread = binning_stability(amp, peak)

    peak_offset = int(np.argmax(np.abs(prep.template)))
    kernel = P.optimal_filter_kernel(prep.template, prep.psd)
    lo = max(P.pulse_center(prep.length, config) - peak_offset - config.search_pad, 0)
    resp_rms = P._response_noise_rms(prep.corrected, kernel, lo)

    closure = closure_test(prep, config, kernel, peak_offset, resp_rms, amp)
    swap = template_swap(prep, config, amp)
    shape, pickup_periods = shape_vs_amplitude(prep, config, windows, amp)

    print("\n" + "=" * 78)
    print(f"TIME-WALK REPORT -- {config.input_path.stem}")
    print("=" * 78)
    print(f"  Shipped slope (n_bins=12):  {shipped:+.3f} samples/amp")
    if fits is not None:
        dof = max(fits["linear"][2], 1)
        print(f"  Fitted amplitude range:     {c.min():.2f} - {c.max():.2f}")
        print(f"  Walk across that range:     {fits['linear'][0][1] * (c.max() - c.min()):+.2f} samples")
        print(f"  Slope above amp >= {HIGH_AMP}:     {fits['slope_high_amp']:+.3f} samples/amp")
        print(f"  chi2/dof   linear = {fits['linear'][1] / dof:6.1f}   "
              f"a+b/A = {fits['inv_amp'][1] / dof:6.1f}   "
              f"dAIC(1/A - linear) = {fits['inv_amp'][3] - fits['linear'][3]:+.1f}")
        print(f"  -> {_verdict(fits)}")
    print(f"\n  Binning stability (n_bins {N_BINS_SCAN[0]}-{N_BINS_SCAN[-1]}): "
          f"spread = {spread:.3f} samples/amp")
    if abs(shipped) > 1e-9 and spread > 0.5 * abs(shipped):
        print("     WARNING: the spread is a large fraction of the slope -- the reported")
        print("     number is dominated by the binning, not by the data.  Do not quote it.")

    if closure:
        print("\n  CLOSURE (synthetic events with NO true walk; must return ~0):")
        for tag, v in closure.items():
            print(f"     {tag:<12} {v:+.3f}")
        worst = np.nanmax(np.abs(list(closure.values()))) if closure else float("nan")
        print(f"     -> estimator bias <= {worst:.3f} samples/amp; a slope above that on real")
        print("        data is REAL (not an artifact of the trigger or a mis-centered window).")

    if swap:
        print("\n  TEMPLATE SWAP (walk on real data, template rebuilt per amplitude class):")
        print(f"     dim template     {swap.get('dim', float('nan')):+.3f}")
        print(f"     bright template  {swap.get('bright', float('nan')):+.3f}")
        print(f"     shipped template {shipped:+.3f}")
        d, b = swap.get("dim", np.nan), swap.get("bright", np.nan)
        if np.isfinite(d) and np.isfinite(b):
            # The threshold is ABSOLUTE, never a fraction of `shipped`: on a channel whose
            # shipped template is itself broken (ch4/ch7, where the model set is >75% noise)
            # `shipped` is huge, and a fraction-of-shipped threshold would call a real 3x
            # dim/bright split "insensitive" purely because the reference is wrong.
            if abs(b - d) > 0.2:
                print("     -> the walk MOVES with the template's amplitude class: the pulse shape")
                print("        depends on amplitude, so no single template can remove the walk.")
            else:
                print("     -> the walk is insensitive to the template's amplitude class.")
            # Both rebuilt templates agreeing that the walk is far SMALLER than the shipped one
            # indicts the shipped template, not the pulses.
            if abs(shipped) > 2.0 * max(abs(d), abs(b)) + 0.5:
                print("     -> WARNING: BOTH rebuilt templates collapse the walk relative to the")
                print("        shipped one. The shipped TEMPLATE is the outlier here -- suspect a")
                print("        noise-dominated model set (check the template-cleaning warning).")

    if shape:
        per = ", ".join(f"{p:.1f}" for p in pickup_periods) if pickup_periods else "none"
        print(f"\n  SHAPE vs AMPLITUDE (template-free, half-max aligned; pickup NOTCHED)")
        print(f"     notched-out line period(s): {per} samples")
        print(f"     {'<amp>':>7} {'N':>6} | {'rise 10-90':>10} {'crest lag':>9} "
              f"{'FWHM':>7} {'fall50':>7} {'fall1/e':>8} | {'(raw rise)':>10} {'(raw lag)':>9}")
        for r in shape:
            print(f"     {r['amp']:>7.2f} {r['n']:>6} | {r['rise_10_90']:>10.2f} "
                  f"{r['peak_minus_halfmax']:>9.2f} {r['fwhm']:>7.2f} {r['fall_50']:>7.2f} "
                  f"{r['fall_1e']:>8.2f} | {r['rise_10_90_raw']:>10.2f} "
                  f"{r['peak_minus_halfmax_raw']:>9.2f}")
        d_rise = shape[-1]["rise_10_90"] - shape[0]["rise_10_90"]
        d_lag = shape[-1]["peak_minus_halfmax"] - shape[0]["peak_minus_halfmax"]
        d_fwhm = shape[-1]["fwhm"] - shape[0]["fwhm"]
        d_fall = shape[-1]["fall_50"] - shape[0]["fall_50"]
        d_rise_raw = shape[-1]["rise_10_90_raw"] - shape[0]["rise_10_90_raw"]
        print(f"     drift dim -> bright: rise {d_rise:+.2f}  crest lag {d_lag:+.2f}  "
              f"FWHM {d_fwhm:+.2f}  fall50 {d_fall:+.2f}  (samples)")
        if abs(d_rise) < 2.5 and abs(d_lag) < 2.5:
            print("     -> the RISING EDGE is amplitude-independent: the walk is not a "
                  "rise/crest-timing effect.")
        # The width verdict (added 2026-07-14, after the template-swap probe found the
        # dim/bright-band templates differ in WIDTH and the walk tracks it).
        if np.isfinite(d_fwhm) and abs(d_fwhm) >= 1.5:
            if np.isfinite(d_fall) and abs(d_fall) >= 0.7 * abs(d_fwhm):
                print(f"     -> the pulse WIDTH grows {d_fwhm:+.2f} samples dim -> bright and the "
                      f"DECAY side carries it (fall50 {d_fall:+.2f}):")
                print("        an amplitude-dependent decay is the walk's remaining suspect -- it "
                      "matches the template-swap width finding.")
            else:
                print(f"     -> the pulse WIDTH grows {d_fwhm:+.2f} samples dim -> bright but NOT "
                      f"mainly on the decay side (fall50 {d_fall:+.2f}): the crest carries it.")
        elif np.isfinite(d_fwhm):
            print("     -> the stack WIDTH is amplitude-independent here; the template-swap width "
                  "split must then arise in the template BUILD (amplitude weighting / xcorr "
                  "alignment), not in the pulses.")
        if abs(d_rise_raw) > abs(d_rise) + 3.0:
            print(f"     -> NOTE the RAW drift ({d_rise_raw:+.2f}) is an artifact of the pickup ripple")
            print("        dragging the argmax off the crest on the dim bands, NOT a shape change.")
    print("=" * 78 + "\n")

    if (config.show_plots or config.save_plots) and fits is not None:
        plot_timewalk_report(c, m, e, fits, scan, closure, swap, shape, config)
    return {"shipped_slope": shipped, "fits": fits, "binning_spread": spread,
            "closure": closure, "template_swap": swap, "shape": shape}


def plot_timewalk_report(c, m, e, fits, scan, closure, swap, shape, config: P.Config) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.errorbar(c, m, yerr=e, fmt="o", ms=4, color="k", lw=1, label="binned median")
    xs = np.linspace(max(c.min(), 1e-3), c.max(), 200)
    dof = max(fits["linear"][2], 1)
    ax.plot(xs, fits["linear"][0][0] + fits["linear"][0][1] * xs, "r--", lw=1.5,
            label=f"linear (chi2/dof={fits['linear'][1] / dof:.1f})")
    ax.plot(xs, fits["inv_amp"][0][0] + fits["inv_amp"][0][1] / xs, "C0-", lw=1.5,
            label=f"a+b/A (chi2/dof={fits['inv_amp'][1] / dof:.1f})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(HIGH_AMP, color="gray", ls=":", lw=1)
    ax.set(xlabel="Reconstructed amplitude", ylabel="Peak time (samples)",
           title="Trend shape: is the walk a line?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(N_BINS_SCAN, scan, "o-", color="C1")
    ax.axhline(float(np.nanmean(scan)), color="gray", ls=":", lw=1)
    ax.set(xlabel="n_bins", ylabel="Fitted slope (samples/amp)",
           title="Binning stability of the reported slope")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    labels = list(closure) + [k for k in ("dim", "bright") if k in swap]
    vals = [closure[k] for k in closure] + [swap[k] for k in ("dim", "bright") if k in swap]
    cols = ["C7"] * len(closure) + ["C0", "C3"][: len(vals) - len(closure)]
    ax.barh(range(len(vals)), vals, color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([f"closure: {l}" if l in closure else f"template: {l}" for l in labels],
                       fontsize=8)
    ax.set(xlabel="Fitted slope (samples/amp)",
           title="Closure (must be ~0) vs template-class swap")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    if shape:
        a = [r["amp"] for r in shape]
        ax.plot(a, [r["rise_10_90"] for r in shape], "o-", color="C2", label="rise 10-90% (notched)")
        ax.plot(a, [r["peak_minus_halfmax"] for r in shape], "s-", color="C4",
                label="peak - half-max (notched)")
        # Width diagnostics (2026-07-14): the template-swap probe localized the walk's
        # remaining suspect in the pulse WIDTH, so the panel shows which side carries it.
        ax.plot(a, [r["fwhm"] for r in shape], "^-", color="C1", label="FWHM (notched)")
        ax.plot(a, [r["fall_50"] for r in shape], "v-", color="C3",
                label="peak -> falling 50% (notched)")
        # The raw curves are drawn faint on purpose: the gap between them and the notched ones is
        # the pickup artifact, and seeing it is the whole point -- it is what makes an
        # amplitude-independent pulse look like a strongly amplitude-dependent one.
        ax.plot(a, [r["rise_10_90_raw"] for r in shape], "o:", color="C2", alpha=0.4, lw=1,
                label="rise (raw -- pickup artifact)")
        ax.plot(a, [r["peak_minus_halfmax_raw"] for r in shape], "s:", color="C4", alpha=0.4, lw=1,
                label="lag (raw -- pickup artifact)")
        ax.set(xlabel="Mean amplitude in band", ylabel="Samples",
               title="Pulse shape vs amplitude (template-free, pickup notched)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"Time-walk report -- {config.input_path.stem}", fontsize=13)
    fig.tight_layout()
    # The channel goes in the FILE name because it is no longer in the folder name: one
    # report is this program's entire output for a channel, so all of a run's channels
    # share one folder (see main()) and their files must tell themselves apart.
    P._finish(fig, f"{config.input_path.stem}_timewalk_report", config)


def main() -> None:
    args = P.build_arg_parser("Diagnose the residual optimal-filter time-walk.").parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    # `group` keeps this QC report's folders together in energy_reconstruction_results/
    # timewalk/ instead of loose beside the drivers' <mode>_mode/ folders: it has no
    # analysis mode, so it is not mode-routed.  `shared` then puts every channel of one run
    # in ONE folder there -- <dataset>_timewalk_results/, e.g. run00270_timewalk_results/ --
    # because a channel's whole output here is a single figure, and eleven folders holding
    # one plot each are eleven clicks to compare the bank.  A new dataset gets its own
    # folder automatically (the name comes from the input).
    analyze(P.config_from_args(args, script_file=__file__, program="timewalk",
                               group="timewalk", shared=True))


if __name__ == "__main__":
    main()
