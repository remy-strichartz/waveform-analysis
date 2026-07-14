#!/usr/bin/env python3
"""
mv_pipeline.py
==============
Shared pipeline for the scintillator-panel analyzers, built for REAL data:
real colored detector noise and real pulses, with no reliance on injected truth.

Flow: load -> estimate the colored noise power spectrum -> build a time-walk-free
pulse template by exact (frequency-accurate) sub-sample cross-correlation
alignment -> trigger with the noise-weighted (optimal) filter so nearly every
pulse is found regardless of amplitude -> estimate the amplitude at the trigger's
own sub-sample time (no bias) -> fit the gamma/noise population and the muon
(MIP) Landau(x)Gaussian line INDEPENDENTLY, each on its own side of the valley
that separates them -> place the discrimination cut at the equal-area
(balanced-misclassification) point between the two fitted components.
Everything is reported in ADC-derived units (template-relative amplitude, or
absolute ADC via the template scale); there is no conversion to a physical
energy scale.

Both modes fit the SAME MIP line, through the same function (fit_muon_line): a single
Landau(x)Gaussian, upgraded -- only when the BIC says so -- to _langau_tailmix, the same
line with a fluctuating light-yield scale, which gives it a real response tail.  A
fixed-scale Landau(x)Gauss has one eta to set both the core asymmetry and the tail weight,
and on a dim channel (run00270_ch9) it cannot do both; the tail model decouples them.  It
replaced a Gaussian "shoulder" that was reported as a second population at 2.2x MPV and is
not one (see _select_tail for the measurements that settled it).  The modes differ in what
sits BELOW that line and in the fit RANGE it implies -- not in the line.

The tail was briefly REMOVED on 2026-07-13 and restored the same day, so the cost of not
having it is now measured rather than argued: without it, a fixed-scale Landau has to inflate
eta to reach the tail, which drags the peak right.  MPV +15.8% on ch9_clean (0.639 -> 0.759),
+8.3% on ch10_clean, +11.9% on ch6; muon chi2_red 1.01 -> 2.96 on ch9_clean and 1.96 -> 6.09
on ch9 (gamma-muon); the gamma/muon cut moves -43% on ch10.  The quoted MPV DEPENDS on this
model -- it is not cosmetic.  What is cosmetic, and is gone for good, is the "core only"
companion CURVE that plot_muon_spectrum used to draw next to it: it held the fitted `area`
fixed while collapsing the scale spread, so it overshot the data ~1.6x and read as a failed
fit.  The core is not a separate population and is not plotted as one.

Two analysis modes (Config.mode):
  'gamma-muon' (default) -- muon-tagged beam (the usual configuration): the MIP
      line is the DOMINANT peak and a smaller gamma/noise population sits at lower
      amplitude; the two independent fits split at the valley and the cut goes at
      the equal-area balance point (muons above, gamma/noise below).  When the
      spectrum turns out to hold no separable low population (the valley walk runs
      off the edge, or the gamma mixture will not converge), this mode fits the MIP
      line ALONE and reports NO cut -- which is exactly the muon-mode answer.
  'muon' -- cleaned, hodoscope-triggered, mostly-muon data; fit the MIP line and
      summarize the pulse-height distribution.  No gamma continuum and no cut are
      imposed, so no physics the data does not support is forced on it.

The two amplitude estimators live in optimal_filter.py (frequency-domain Wiener
filter) and boxcar.py (colored-noise-optimal top-hat); compare.py runs both.

The noise model (PSD + the trigger's response-noise scale) is built from the ordinary
events: a rare tail of interference-BURST events -- on run00270 ~1% of records where the
coherent pickup carrier swings ~10x its usual amplitude -- is trimmed out first, because
both estimators are event AVERAGES and an average is outlier-dominated (those ~1% inflate
N(f) up to 70x at f~0.02 cycles/sample, inside the signal band).  See noise_event_mask;
--psd-trim-sigma 0 restores the plain all-event mean.  This does NOT improve the MIP
resolution (the noise term is only ~3% of the Landau-dominated MIP width) -- it makes the
noise model CORRECT: it stops the inflated threshold from dropping ~2% of real pulses, and
stops an inflated N(f) from absorbing template mismatch that belongs in the per-event chi2.

Rail-clipped (saturated) events are always dropped from the spectrum FIT (they carry
no valid pulse-height information).  PILEUP events are NOT dropped by default; an
opt-in gate (--exclude-pileup, Config.exclude_pileup) additionally drops events whose
per-event optimal-filter reduced chi2 exceeds pileup_chi2_factor x the chi2 a CLEAN pulse
OF THAT AMPLITUDE would have -- the clean chi2 itself rises as c0 + (A/A0)^2, so a flat
threshold is an amplitude cut in disguise (pileup_mask_from_chi2, chi2_amplitude_trend).
Use it on PILEUP-HEAVY channels (e.g. a low-light PMT like
run00270_ch9) where high-amplitude overlapping pulses pull the muon Landau line up and
bias the gamma-muon cut high; leave it OFF for clean channels (e.g.
run00270_ch0), where it isn't needed and perturbs an already-good cut.  Only the FIT
drops these events -- the muon/gamma population COUNT still uses every event -- and the
gate needs the OF chi2, so it acts in optimal_filter.py and compare.py (boxcar.py
standalone ignores the flag).  Always cross-check that the gated MPV agrees with the
independent muon-mode (--mode muon) MPV.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import landau, rankdata, trim_mean

# The shared triage primitives -- flat-top saturation detector, polarity vote and
# median-pulse extent rule -- live in waveform_triage, so this pipeline calls an
# event "clipped" / a channel "negative" / a pulse "this wide" on exactly the
# grounds triage does (one source of truth).  waveform_triage (and its peakfind
# dependency) live in the sibling preprocessing/ directory, mirroring how the
# other cross-package tools (e.g. file_manipulation/channel_diagnostics.py) put
# it on the path.
_PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "preprocessing"
if str(_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING_DIR))
from waveform_triage import (classify_events, detect_saturation,  # noqa: E402
                             median_pulse_extent, polarity_vote, triage_cuts)

logger = logging.getLogger("mv_pipeline")
_SQRT_2PI = math.sqrt(2.0 * math.pi)
# Mode of the standard Landau (peak of landau.pdf at loc=0, scale=1); used only
# to seed the fit so the loc parameter starts near the desired peak position.
_LANDAU_MODE = float(np.linspace(-2, 2, 4001)[np.argmax(landau.pdf(np.linspace(-2, 2, 4001)))])


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path = field(default_factory=lambda: Path("results"))
    polarity: str = "auto"                 # 'auto' detects the sign from the data; 'positive' (SiPM), 'negative' (PMT)

    # Analysis mode:
    #   'gamma-muon' (default) -- muon-tagged beam: the MIP line is the dominant
    #                peak; the gamma/noise population and the muon line are fit
    #                independently on their own sides of the valley and the cut
    #                goes at the equal-area balance point between them.
    #   'muon'       -- cleaned, hodoscope-triggered, mostly-muon data: fit a single
    #                Landau(x)Gaussian (the MIP line) and summarize the pulse-height
    #                distribution; no gamma continuum and no cut are imposed.
    mode: str = "gamma-muon"

    # Pulse window: by default pulse_center / search_pad / template_pre / template_post
    # are estimated automatically from the data (see estimate_window_params); the values
    # below are only the fallback used with --no-auto-window or when a param is set
    # explicitly on the CLI.  `explicit_window` names the params the user set by hand so
    # auto estimation leaves those alone.
    auto_window: bool = True
    explicit_window: tuple = ()

    pulse_center: int | None = 450         # clean hodoscope pulses peak ~sample 450

    # Template window (samples around the peak) and clean-pulse selection.
    template_pre: int = 100  # 100 samples of flat pre-pulse is sufficient
    template_post: int = 250  # captures decay to ~sample 700 where tail reaches baseline
    template_cap: int = 20_000
    template_sigma: float = 8.0            # bright events used to build a clean template
    # Opt-in ROBUST final template average: trim this fraction from EACH per-sample tail
    # of the aligned pulses before averaging (scipy.stats.trim_mean); 0 (default) keeps
    # the long-standing plain mean.  Motivation (measured 2026-07-14, run00270_ch2): the
    # plain mean is pulled ~4 samples WIDER in FWHM by a minority tail of events
    # (unresolved pileup / afterpulses, bright-biased) that the median of the SAME
    # aligned rows does not see.  A/B before adopting as default: it moves the template,
    # so it moves every canonical number.  See build_template.
    template_trim: float = 0.0
    align_iters: int = 3
    align_lag: int = 6   # covers the tails of the half-max rough alignment (~1 sample RMS)
    # Narrow noise lines (run00270's 9.4-sample pickup) at >= this many times the local
    # broadband PSD floor are notched out of a COPY of each pulse, and the template's
    # half-max picks and alignment shifts are estimated on that copy (see build_template;
    # the raw pulses that get averaged are never filtered).  0 disables the guidance.
    align_notch_snr: float = 4.0

    # Optimal-filter trigger: search half-width around the expected center (covers
    # the pulse-timing jitter; widen for data with less certain pulse position)
    # and the detection threshold in units of the measured response noise.
    # search_pad=90 covers the triage pulse window (samples ~350-550, i.e. +/-100
    # around the 450 center) so every clean pulse falls inside the search range.
    search_pad: int = 90
    trigger_sigma: float = 4.0

    psd_cap: int = 20_000

    # Robust noise model: drop pathological interference-burst events from the PSD and the
    # trigger's response-noise scale (see noise_event_mask).  In units of a MAD sigma on
    # the log pre-pulse power; 0 disables the trim (plain mean over every event).
    psd_trim_sigma: float = 5.0

    box_pre: int = 1                       # boxcar window start = peak - box_pre
    box_min: int = 2
    box_max: int = 30

    # ---------------------------------------------------------------------------------
    # SPECTRUM BINNING -- every scale here is PHYSICAL (a multiple of the MIP amplitude
    # scale, spectrum_scale), never a bin count.
    #
    # The old scheme fixed the bin COUNT (250) and let the RANGE run to the 99.9th
    # percentile.  That makes the bin width a function of the Landau tail's length, which
    # has nothing to do with where the gamma/muon boundary sits.  On a low-light channel
    # the tail reaches 25-35x the MPV, so the entire gamma/valley/MIP structure collapsed
    # into the first ~5 bins; the valley-walk lookahead (then 3% of the bin count) became
    # wider than the structure itself and jumped straight past the peak's left foot into
    # the sparse negative tail -- returning a nonsense negative or NaN cut on run00270
    # ch3-ch7 and on caen's boxcar.  Even on the channels that "worked" the cut then moved
    # by ~18% if you merely changed how far out you plotted the tail.
    #
    # Now: the bin WIDTH is pinned to the MIP scale (scale / bins_per_mip), so the peak
    # always gets the same resolution, and the bin COUNT follows from the range.  The
    # smoothing and the valley lookahead are likewise MIP-scale fractions.  Nothing in the
    # fit is measured in bins any more, so nothing depends on the binning.
    bins_per_mip: int = 50                  # histogram bins across one MIP scale -> the bin WIDTH
    max_spectrum_bins: int = 2000           # safety cap on the derived bin count
    spectrum_lo_pct: float = 0.0            # 0 = keep the true minimum: the gamma/noise pedestal is physics
    spectrum_hi_pct: float = 99.9           # trim only the extreme bright outliers
    # Upper edge also capped at this multiple of the MIP scale.  Beyond ~6x the MPV a MIP
    # line carries almost nothing but pile-up, and the standard practice is to fit the line
    # over a few times its MPV; the cap keeps a pathological tail from setting the range.
    spectrum_hi_scale: float = 6.0
    smoothing_frac: float = 0.03            # spectrum smoothing sigma, in units of the MIP scale
    valley_lookahead_frac: float = 0.08     # valley-walk lookahead, in units of the MIP scale
    # Robust low floor for the MIP-line FIT (the histogram still shows every event -- see
    # spectrum_lo_pct=0).  A handful of far-sub-MIP noise triggers (<1% of events) carry
    # enough counts per bin that a Landau predicting ~0 there blows up the Poisson chi2 and
    # drags the MPV/kills the tail fit; the line is therefore fit only at/above this
    # percentile of the data (the empirical base of the peak).  Applies wherever NOTHING
    # models the pedestal: all of muon mode, and gamma-muon's muon-only fallbacks.  It does
    # NOT apply to a gamma-muon fit that HAS a gamma -- there the pedestal is fit as gamma
    # and the line starts at the valley anyway (_muon_fit_floor).
    muon_fit_lo_pct: float = 0.5
    # A cut whose fitted gamma holds less than this fraction of the events is WARNED about:
    # it separates the MIP line from almost no population at all (run00270_ch9 fits 0.17%).
    # This is a diagnostic, NOT a gate -- the spectrum cannot decide whether a cut is
    # meaningful, and _gamma_evidence has the measurements that show why.
    gamma_frac_warn: float = 0.01
    # The gamma/noise population below the MIP valley (noise pedestal + gamma/EM
    # background) is modeled as a BIC-selected sum of up to this many Gaussians,
    # fit ONLY on the bins below the valley (never across the muon-dominated
    # region) with means/widths bounded to that region, so a component can never
    # sit under the MIP peak.  3 lets a sharp trigger-noise spike, a broad gamma
    # shoulder and one extra feature each get a component when the BIC supports
    # them; the BIC picks fewer on simpler channels.  See fit_spectrum.
    gamma_max_gaussians: int = 3
    # Each gamma Gaussian's width is capped at (gamma region span) / this, so every
    # component decays INSIDE the region it is fit on instead of extrapolating its tail
    # under the MIP peak (which _equal_area_cut would then integrate as real gamma).  3 =>
    # roughly +/-1 sigma inside the region.  See fit_spectrum.
    gamma_sigma_containment: float = 3.0
    # The gamma is fit not just below the valley but UP THROUGH THE OVERLAP, for as long as
    # the fitted muon line stays below this fraction of the smoothed counts (i.e. while the
    # bin is still gamma-dominated, so the gamma's falling tail is measurable there).  The
    # muon enters that fit as a FIXED component.  This is what makes the equal-area balance
    # meaningful: the gamma model is then valid over exactly the range the balance
    # integrates it on.  See _gamma_fit_top.  0.5 = "the muon owns at most half the bin".
    gamma_overlap_frac: float = 0.5
    # A valley is only accepted where the smoothed spectrum has fallen to at most this
    # fraction of the MIP bump height -- a flat spot high on the peak's own flank is not a
    # valley.  See the valley walk in fit_spectrum.
    valley_max_frac: float = 0.5
    # PE-COMB GUARD on the landmark walk (see _comb_period).  A low-light SiPM channel
    # resolves single photoelectrons, so its whole spectrum is modulated by a periodic
    # comb whose inter-PE dips masquerade as valleys.  A comb is declared when the
    # autocorrelation of the envelope-subtracted spectrum rebounds to at least this value
    # at the comb period.  Measured on all run00270 + caen spectra (22 channel/estimator
    # pairs): comb channels score 0.61-0.77, every clean channel <= 0.05 -- so 0.4 sits in
    # an empty gap, and clean channels' landmarks are bit-identical with the guard on.
    comb_rho_min: float = 0.4
    # Bootstrap replicates used to put an ERROR BAR on the gamma/muon cut (see
    # bootstrap_cut).  The cut has no analytic uncertainty -- it falls out of a valley walk,
    # a BIC mixture and an area balance -- so without this it is reported as a bare number
    # and read as exact, though its sampling std is 4-14% depending on the channel.
    # COST: one full spectrum refit per replicate, and _langau rebuilds a Landau PDF +
    # convolution on every curve_fit evaluation, so a fit is ~2 s -- 20 replicates roughly
    # doubles a single-channel run.  20 is deliberately modest: the std of a bootstrap std
    # falls as 1/sqrt(2(N-1)), so N=20 pins it to ~16% of itself, which is far more than
    # enough to tell "+/- 6%" from "+/- 14%" (the decision this number exists to support).
    # Set 0 to disable.
    cut_bootstrap: int = 20
    # RELIABILITY GUARD.  A gamma/muon cut only means something when the data actually
    # holds a MIP line above the noise.  The test is the fitted MIP peak height measured
    # in units of the channel's own baseline noise (mpv * template_peak / noise_sigma) --
    # a scale INDEPENDENT of the spectrum being fit.  The previous guard compared the
    # fitted MPV against the MEDIAN of the same observable, which is self-referential: on
    # a noise-dominated channel the median sits in the noise too, so the MPV tracks it and
    # the test can never fire (it did not fire on run00270_ch4, SNR ~3, where the fit was
    # meaningless).  5 sigma is the usual "this is a signal" bar.
    min_mip_snr: float = 5.0
    # SECOND RELIABILITY GUARD -- is there ROOM for a gamma population below the MIP line?
    #
    # Every event in the spectrum passed the optimal-filter trigger at trigger_sigma times the
    # response noise, so the spectrum is TRUNCATED from below at that threshold (Prepared.
    # trigger_floor, in the observable's own units).  A gamma/noise continuum can therefore
    # only be SEEN in the band between the trigger floor and the MIP valley.  When the MIP
    # line itself sits only a few floors up, that band has collapsed: the low edge of the
    # histogram is the TRIGGER, not the gamma population, and the BIC Gaussian mixture --
    # which will always find something to fit at a hard step -- fits the threshold turn-on
    # and reports it as a gamma.  That is exactly what run00270 ch6 and ch7 were doing: a
    # ~1-bin-wide "gamma" Gaussian sitting on the spectrum's first populated bin, with a
    # gamma chi2_red of 10-19 (a Gaussian fit to a step function), and on ch6 a gamma/muon
    # cut was reported from it.
    #
    # MPV / trigger_floor, measured on run00270 (optimal filter):
    #     room to spare:      ch0 66, ch1 19, ch10 18, ch2 15, ch3 13, ch9 11
    #     borderline:         ch5 7.5, ch6 6.8      <- these have room; their gamma FIT is
    #                                                  what's bad (see gamma_chi2_warn)
    #     trigger-truncated:  ch4 4.5, ch7 3.8      <- no gamma region left at all
    # 6 sits between the two groups.  Note this is NOT a spectrum-shape heuristic (those are
    # refuted -- see _gamma_evidence): the trigger floor comes from the noise model and the
    # trigger setting, independently of the spectrum being fit.
    #
    # Failing this does NOT invalidate the MIP line -- the line is still fit and reported (on
    # ch7 it stands 15 sigma above the OF's own resolution).  It invalidates the CUT, which
    # would be a boundary against a population the trigger already removed.
    min_mip_over_trigger: float = 6.0
    # A fitted gamma continuum whose OWN reduced chi2 exceeds this is WARNED about: the cut is
    # placed by balancing areas under the gamma and muon models, so a gamma that does not
    # describe its own bins cannot place a trustworthy boundary.  Measured on run00270: the
    # channels whose gamma is real fit it well (ch0 1.6, ch1 1.2, ch10 2.4, ch9 3.1, ch3 3.2),
    # while run00270_ch6 -- where the low region is a broad spiky turn-on, not a Gaussian
    # population -- fits at 15-20 and still reports a cut.  This is a DIAGNOSTIC, not a gate:
    # gating on spectrum shape is refuted (see _gamma_evidence), and a legitimately THIN gamma
    # (ch9: 0.1% of events, ~15 counts) scores a meaningless chi2 on a handful of bins.
    gamma_chi2_warn: float = 8.0

    # Waveforms are read and stored as float32: raw ADC is 12-16 bit integer, so
    # baseline subtraction (an integer per-event median) is EXACT in float32's 24-bit
    # mantissa, and the ~7 significant digits far exceed the ~1 LSB noise floor.  This
    # halves the resident memory of the (large) corrected array and the streamed chunks
    # and speeds the BLAS/FFT ~2x.  Fit-critical accumulations (PSD, histograms,
    # curve_fit) still run in float64.  Set to np.float64 to reproduce the old scale.
    dtype: Any = np.float32
    # Rows per chunk in prepare()'s streaming apply pass: peak working memory is
    # ~chunk_events * L * itemsize, so lower it on a memory-tight box and raise it for
    # a bit more throughput.  Does not affect results.
    chunk_events: int = 8192
    # Opt-in pileup gate (OF / compare drivers): drop pileup / shape-anomaly events from
    # the spectrum FIT when their per-event optimal-filter reduced chi2 exceeds
    # pileup_chi2_factor x the median chi2 (a robust, per-channel clean-level estimate).
    # Off by default -- leaves every existing result unchanged.  Only the FIT (and hence
    # the cut) drops them; the full-observable population counts still include them.
    exclude_pileup: bool = False
    pileup_chi2_factor: float = 5.0

    # Good-time-interval gate.  `exclude_hours` is a tuple of (lo_h, hi_h) windows of RUN
    # TIME whose events are dropped BEFORE anything else happens -- they enter neither the
    # noise model (PSD / template / trigger threshold) nor the spectrum.  This exists
    # because that noise model is estimated ONCE for the whole run: a period whose noise
    # sits well above the run average is then optimal-filtered with a PSD that does not
    # describe it, and thresholded against a response-noise scale too small for it, so its
    # events are both mis-weighted and admitted at a lower effective significance.
    # timing_stability/run_stability.py finds and reports such windows (`bad_windows_h` in
    # its JSON, with the exact --exclude-hours line to paste); this is how you act on them.
    # The run-time axis comes from /event_time_rel_s in the input (written at conversion by
    # file_manipulation/clock_recovery.py), else from `times_path`, else from the dataset's
    # recovered axis, waveform_files/<run>/times/<stem>_times.h5.
    exclude_hours: tuple = ()
    times_path: Path | None = None

    seed: int = 20240601
    show_plots: bool = True
    save_plots: bool = False
    full_diagnostics: bool = False         # opt-in validate-once plots (shape params, template stability)


def pulse_center(length: int, config: Config) -> int:
    return length // 2 if config.pulse_center is None else int(config.pulse_center)


def search_region(length: int, config: Config) -> slice:
    """The ONE definition of the pulse search window, pulse_center +/- search_pad
    (upper bound INCLUSIVE), shared by build_template, _clean_model_set and
    saturation_qc.  They previously each wrote the slice out by hand and disagreed by
    one sample on the upper edge, so an event's peak could fall inside the window for
    one stage and outside it for another."""
    center = pulse_center(length, config)
    return slice(max(center - config.search_pad, 0),
                 min(center + config.search_pad + 1, length))


def noise_stop(length: int, config: Config) -> int:
    """End of the guaranteed pulse-free pre-pulse region (before the search window).

    The earliest peak the trigger considers sits at pulse_center - search_pad, and a
    pulse begins RISING up to template_pre samples before its peak -- so the truly
    pulse-free region ends template_pre earlier still.  (Ending it at
    center - search_pad would, for tightly-timed data with a small search_pad, leak
    the leading edge into the noise region and inflate every noise estimate made
    there.)

    It must also be at least one template long, or the PSD / trigger-noise segments
    do not fit.  Both conditions together require

        2 * template_pre + template_post <= pulse_center - search_pad

    which estimate_window_params now enforces.  If a hand-set window violates it we
    RAISE rather than silently returning a region that reaches past the earliest
    possible pulse rise: the old code floored the result at one template length,
    which quietly handed the PSD and the trigger threshold a slice containing pulse
    edge (it was doing exactly that on run00270 ch0/ch1, where the "pulse-free"
    region ran 19 samples past the earliest rise)."""
    center = pulse_center(length, config)
    stop = center - config.search_pad - config.template_pre
    floor = config.template_pre + config.template_post
    if stop < floor:
        raise ValueError(
            f"No pulse-free pre-pulse region: pulse_center({center}) - search_pad"
            f"({config.search_pad}) - template_pre({config.template_pre}) = {stop} samples, "
            f"but one template is {floor} samples (template_pre + template_post). "
            "Move --pulse-center later, or shrink --search-pad / --template-pre / "
            "--template-post (auto-window sizes these to satisfy this automatically).")
    return stop


# ===========================================================================
# Automatic pulse-window estimation
# ===========================================================================

def _oriented_for_estimate(waveforms: np.ndarray, polarity: str) -> np.ndarray:
    """Baseline-subtract (per event, robust median of the first 20% of the record
    taken as the pre-pulse region) and orient so pulses point UP, for the window
    estimate only.  Mirrors remove_baseline's polarity rules; the refined pre-pulse
    baseline is applied by remove_baseline once the window is known."""
    pre = max(int(0.2 * waveforms.shape[1]), 1)
    centered = waveforms - np.median(waveforms[:, :pre], axis=1, keepdims=True)
    if polarity == "negative":
        return -centered
    if polarity == "auto":
        flip = centered.max(axis=1) < -centered.min(axis=1)
        return np.where(flip[:, None], -centered, centered)
    return centered                         # positive (or default)


def estimate_window_params(waveforms: np.ndarray, config: Config,
                           min_sigma: float = 8.0, coverage: float = 0.99,
                           pad: int = 10) -> dict | None:
    """Estimate pulse_center / search_pad / template_pre / template_post from the data.

    Uses the same idea as waveform_triage.recommend_window: collect the peak positions
    of the real pulses (>= min_sigma over a robust noise estimate), center on their
    median, size the search pad from their spread (central `coverage`), and size the
    template from the median pulse's rise/fall.  The four are then clamped so the
    template fits in the record AND the pre-pulse noise region stays long enough
    (pulse_center - search_pad >= template_pre + template_post), so the result is always
    a window the rest of the pipeline can use.  Returns None if there are too few real
    pulses to estimate reliably (caller then keeps the fallback window)."""
    L = waveforms.shape[1]
    sig = _oriented_for_estimate(waveforms, config.polarity)
    sigma = 1.4826 * float(np.median(np.abs(sig - np.median(sig))))
    sigma = max(sigma, 1e-9)

    peak_pos = sig.argmax(axis=1)
    real = sig.max(axis=1) >= min_sigma * sigma
    positions = peak_pos[real]
    if positions.size < 20:
        logger.warning("Auto-window: only %d pulses clear %.0f sigma; keeping the fallback "
                       "window.", int(positions.size), min_sigma)
        return None

    center = int(np.median(positions))
    tail = (1.0 - coverage) / 2.0 * 100.0
    q_lo, q_hi = np.percentile(positions, [tail, 100.0 - tail])
    # A coherent pulse population clusters in time; if the central peak positions instead
    # span more than half the record, the "pulses" are scattered (likely a noise-dominated
    # channel whose argmax lands on noise), so the window estimate -- and the template built
    # from it -- is unreliable.  Warn rather than silently mis-window.
    if (q_hi - q_lo) > 0.5 * L:
        logger.warning("Auto-window: the %.0f%% central peak positions span %.0f samples "
                       "(> half the %d-sample record); the channel may lack a coherent pulse "
                       "(noise-dominated). Check --pulse-center or triage the input.",
                       coverage * 100.0, q_hi - q_lo, L)

    # SEARCH PAD from a ROBUST spread of the peak positions (median +/- k*MAD), not from
    # their outer percentiles.  A ~1% tail of pile-up / after-pulse events puts its ARGMAX
    # hundreds of samples away from the coincidence (on run00270_ch0 the 99.5th percentile
    # is sample 614 while the 99th is 462), so the old central-99% rule was sized entirely
    # by the events it should have ignored -- it demanded ~165 samples and then hit the
    # L//6 clamp, on a pulse whose genuine jitter is MAD ~7 samples.  An over-wide search
    # window is not free: the trigger takes a max over it, so it admits more pure-noise
    # triggers into the spectrum AND it eats the pre-pulse region the PSD needs.  Those
    # late-argmax events are not lost -- the optimal-filter trigger still finds their
    # PRIMARY pulse at the coincidence time.
    mad = 1.4826 * float(np.median(np.abs(positions - center)))
    sp_cap = max(30, L // 6)
    search_pad = int(np.clip(5.0 * max(mad, 1.0) + pad, 10, sp_cap))
    outside = float(np.mean(np.abs(positions - center) > search_pad))
    if outside > 0.02:
        logger.warning("Auto-window: %.1f%% of real-pulse peaks sit outside the estimated "
                       "search pad (+/-%d samples); the pulse timing may be multi-modal. "
                       "Set --search-pad explicitly if pulses are being missed.",
                       100 * outside, search_pad)

    med_pulse = np.median(sig[real], axis=0)
    mpk = int(med_pulse.argmax())
    # Shared extent rule (waveform_triage.median_pulse_extent); the fall side uses a
    # LOWER fraction (2% vs 10%) so the template captures most of the decay tail --
    # a too-short post-peak window leaves tail signal out of the template and
    # inflates the per-event optimal-filter chi2.
    rise, fall = median_pulse_extent(med_pulse, mpk, frac=0.1, fall_frac=0.02)

    center = int(np.clip(center, 1, L - 2))
    template_pre = int(np.clip(rise + pad, 10, center))
    template_post = int(np.clip(fall + pad, 10, L - center - 1))
    search_pad = int(np.clip(search_pad, 10, max(10, center - 1)))

    # Guarantee the region noise_stop actually hands out is pulse-free, which needs
    #     2 * template_pre + template_post <= center - search_pad
    # (one template_pre for the earliest pulse's RISE, then a full template length for the
    # PSD / trigger-noise segments).  The old check only required template_pre +
    # template_post <= center - search_pad -- one template_pre short -- so the window it
    # produced could not be honoured and noise_stop silently clamped, handing the PSD a
    # slice that reached past the earliest possible rise.  Shrink post, then pre, then the
    # search pad, until it fits.
    budget = center - search_pad
    for _ in range(3):
        if 2 * template_pre + template_post <= budget:
            break
        template_post = max(10, budget - 2 * template_pre)
        if 2 * template_pre + template_post <= budget:
            break
        template_pre = max(10, (budget - template_post) // 2)
        if 2 * template_pre + template_post <= budget:
            break
        search_pad = max(10, center - (2 * template_pre + template_post))
        budget = center - search_pad

    return {"pulse_center": center, "search_pad": search_pad,
            "template_pre": template_pre, "template_post": template_post}


def resolve_auto_window(waveforms: np.ndarray, config: Config) -> Config:
    """Return a Config with pulse_center/search_pad/template_pre/template_post filled in
    from the data (when auto_window is on), leaving any the user set explicitly alone."""
    if not config.auto_window:
        return config
    est = estimate_window_params(waveforms, config)
    if est is None:
        return config
    updates = {k: v for k, v in est.items() if k not in config.explicit_window}
    if not updates:
        return config
    resolved = replace(config, **updates)
    logger.info("Auto-window: pulse_center=%s search_pad=%d template_pre=%d template_post=%d "
                "(auto-set: %s; kept explicit: %s).",
                pulse_center(waveforms.shape[1], resolved), resolved.search_pad,
                resolved.template_pre, resolved.template_post,
                ", ".join(updates) or "none", ", ".join(config.explicit_window) or "none")
    return resolved


def resolve_polarity(waveforms: np.ndarray, config: Config) -> Config:
    """Resolve polarity='auto' to ONE concrete 'positive'/'negative' for the whole
    channel, then force that sign downstream -- exactly as if it were given explicitly.
    Polarity is a fixed property of a channel, so it is decided ONCE from the pooled
    data (never per event/window): whichever direction the channel's largest excursions
    from baseline go.  An explicit 'positive'/'negative' is passed through unchanged."""
    if config.polarity != "auto":
        return config
    # The project's ONE polarity vote (waveform_triage.polarity_vote): 95th-pct
    # up- vs down-excursion, so a modest firing fraction still decides the sign.
    detected, up, down, ambiguous = polarity_vote(waveforms)
    if ambiguous:
        logger.warning("Auto-polarity AMBIGUOUS (up %.3g vs down %.3g comparable); using '%s'. "
                       "Pass --polarity explicitly if that is wrong.", up, down, detected)
    else:
        logger.info("Auto-polarity: channel is %s (up %.3g vs down %.3g ADC, 95th-pct excursion).",
                    detected, up, down)
    return replace(config, polarity=detected)


# ===========================================================================
# Shared CLI for the analysis drivers (optimal_filter / boxcar / compare)
# ===========================================================================

_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_WAVEFORM_DIR = _PROJECT_ROOT / "waveform_files"
DEFAULT_OUTPUT_DIR   = _PROJECT_ROOT / "results"

# Shared waveform-file and per-run results-folder conventions
# (see file_manipulation/output_paths.py).
sys.path.insert(0, str(_PROJECT_ROOT / "file_manipulation"))
from output_paths import (find_related, list_waveforms, program_token,  # noqa: E402
                          resolve_input, resolve_results_dir, results_base)

# Drivers whose results are routed into a <mode>_mode subfolder (they have an analysis
# mode).  timewalk_report and run_stability share this module's CLI/config builder but
# have no mode; their results are grouped by their own `group` subfolder
# (energy_reconstruction_results/timewalk/, timing_stability_results/stability/) instead.
_MODE_ROUTED_PROGRAMS = {"compare", "of", "boxcar"}


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """The common argument set shared by every analysis driver, so the three
    drivers expose an identical, consistent CLI."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input", type=Path, default=None,
                   help="Input .h5 file; a bare name is looked up in waveform_files/ "
                        "(its dataset folders and their kind subfolders).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Base output directory; plots go into "
                        "<output-dir>/<input-stem>_<program>_results[_N]/ so runs never "
                        "overwrite each other (a re-run gets a fresh _N folder). Default base: "
                        "the driver's <pipeline>_results/ folder (e.g. energy_reconstruction_results/).")
    p.add_argument("--polarity", choices=["positive", "negative", "auto"], default=None,
                   help="Pulse polarity: 'auto' (default) detects the sign from the data; force "
                        "'positive' (SiPM) or 'negative' (PMT).")
    p.add_argument("--mode", choices=["gamma-muon", "muon"], default=None,
                   help="'gamma-muon' (muon-tagged beam: independent gamma/muon fits split at "
                        "the valley + equal-area cut between them) or 'muon' (cleaned "
                        "hodoscope-triggered MIP pulses: single Landau line, no cut).")
    p.add_argument("--pulse-center", type=int, default=None,
                   help="Expected peak sample. Default: auto-estimated from the data.")
    p.add_argument("--trigger-sigma", type=float, default=None,
                   help="Trigger threshold in units of the measured matched-filter "
                        "response noise. Default: 4.")
    p.add_argument("--template-sigma", type=float, default=None,
                   help="Brightness cut (in units of the baseline noise sigma) for the pulses "
                        "averaged into the template. Default: 8.")
    p.add_argument("--template-trim", type=float, default=None,
                   help="Opt-in robust template average: trim this fraction from EACH "
                        "per-sample tail of the aligned pulses before averaging (0.1 is a "
                        "sensible A/B value). Removes the mean-broadening a minority tail of "
                        "pileup/afterpulse events causes. Default: 0 (plain mean, the "
                        "long-standing behaviour).")
    p.add_argument("--psd-trim-sigma", type=float, default=None,
                   help="Robustness of the noise model: events whose log pre-pulse power "
                        "exceeds median + N MAD sigma are treated as interference bursts and "
                        "excluded from the PSD and the trigger's response-noise scale. "
                        "Pass 0 to disable (plain mean over every event). Default: 5.")
    p.add_argument("--min-mip-snr", type=float, default=None,
                   help="Reliability bar: the fitted MIP line must stand at least this many "
                        "baseline-noise sigma above zero, or the fit is flagged UNRELIABLE and "
                        "no gamma/muon cut is reported. Default: 5.")
    p.add_argument("--min-mip-over-trigger", type=float, default=None,
                   help="Reliability bar: the fitted MIP line must sit at least this many times "
                        "above the TRIGGER FLOOR, or there is no room below it for a gamma "
                        "population (the trigger removed it) and no gamma/muon cut is reported. "
                        "The MIP line is still fit either way. Default: 6.")
    p.add_argument("--overwrite", action="store_true",
                   help="Write into <input-stem>_<program>_results/ even if it exists, "
                        "overwriting the previous run's plots. Default: a re-run gets a fresh "
                        "_N folder and the old results are kept.")
    # Pulse window (samples).  By default these are estimated automatically from the data
    # so the window sits wherever the pulse actually is (e.g. the CAEN hodoscope pulses peak ~338); any value
    # set here is respected and only the rest are auto-filled.  --no-auto-window turns the
    # estimation off and uses these / the built-in fallback (pulse ~450) for all.
    p.add_argument("--no-auto-window", dest="auto_window", action="store_false", default=True,
                   help="Disable automatic pulse-window estimation; use the explicit / fallback values.")
    p.add_argument("--template-pre", type=int, default=None,
                   help="Samples of template kept BEFORE the peak. Default: auto-estimated.")
    p.add_argument("--template-post", type=int, default=None,
                   help="Samples of template kept AFTER the peak. Default: auto-estimated.")
    p.add_argument("--search-pad", type=int, default=None,
                   help="Optimal-filter trigger search half-width around the center. Default: auto-estimated.")
    p.add_argument("--no-show", "--no-plot", dest="no_show", action="store_true",
                   help="Do not open plot windows.")
    p.add_argument("--save-plots", action="store_true", help="Save plots to the output directory.")
    p.add_argument("--full-diagnostics", action="store_true",
                   help="Also produce the opt-in validate-once plots (pulse-shape params, template "
                        "stability). Core QA plots always run.")
    p.add_argument("--chunk-events", type=int, default=None,
                   help="Rows per chunk in the streaming baseline+trigger pass; peak working memory "
                        "scales with this (lower it on a memory-tight box). Does not change results. "
                        "Default: 8192.")
    p.add_argument("--exclude-pileup", action="store_true",
                   help="Exclude pileup / high-chi2 shape-anomaly events from the spectrum FIT "
                        "(per-event OF reduced chi2 > factor x median). Opt-in; off by default. "
                        "OF and compare drivers only (needs the OF chi2).")
    p.add_argument("--pileup-chi2-factor", type=float, default=None,
                   help="Threshold factor for --exclude-pileup: drop events with chi2 > factor x "
                        "median(chi2). Default: 5.")
    p.add_argument("--cut-bootstrap", type=int, default=None, metavar="N",
                   help="Bootstrap replicates used to put an ERROR BAR on the gamma/muon cut "
                        "(resamples the events and refits). The cut's sampling std is 4-14%% "
                        "depending on the channel, so without this a 10%% change between runs "
                        "reads as physics when it is noise. Costs one spectrum refit each "
                        "(~2 s), so it roughly doubles a run. Default: 20; 0 disables.")
    p.add_argument("--exclude-hours", type=float, nargs=2, action="append", default=None,
                   metavar=("LO", "HI"),
                   help="Drop every event whose RUN TIME falls in [LO, HI) hours -- from the "
                        "noise model (PSD / template / trigger threshold) as well as the "
                        "spectrum. Repeatable. Use it on the bad periods that "
                        "timing_stability/run_stability.py reports (a noisy window is filtered "
                        "and thresholded with a whole-run noise model that does not describe "
                        "it). Needs a time axis: /event_time_rel_s in the input, or --times.")
    p.add_argument("--times", type=Path, default=None,
                   help="Times .h5 carrying /event_time_rel_s, row-aligned with the input. Only "
                        "needed when the input itself has no time axis (an older conversion). "
                        "Default: the dataset's own "
                        "waveform_files/<run>/times/<input-stem>_times.h5 if it exists.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def config_from_args(args, script_file: str | None = None, program: str | None = None,
                     group: str | None = None) -> Config:
    """Build a Config from the shared CLI args (input resolved against waveform_files/).

    `script_file` / `program` identify the CALLING driver so its plots land in that
    driver's own per-run results folder: this one builder is shared by the
    energy_reconstruction drivers (compare/optimal_filter/boxcar) AND run_stability,
    which live in different pipelines and want different folders + tokens.  Each
    driver passes its own `__file__` and short token; when omitted they fall back to
    this module's location (energy_reconstruction) and the 'mv' token.  `group` is the
    caller's program subfolder of its default results base (timewalk_report ->
    energy_reconstruction_results/timewalk/, run_stability -> timing_stability_results/
    stability/); like the <mode>_mode level it keeps programs sharing one results folder
    apart, but it is skipped under an explicit --output-dir.
    """
    kwargs: dict[str, Any] = {"show_plots": not args.no_show, "save_plots": args.save_plots,
                              "full_diagnostics": args.full_diagnostics}
    if args.input is not None:
        # Bare filename -> waveform_files/, wherever the layout puts it, so
        # `--input run00270_ch9.h5` finds waveform_files/run00270/channels/run00270_ch9.h5.
        kwargs["input_path"] = resolve_input(args.input)
    else:
        candidates = list_waveforms()
        if not candidates:
            raise FileNotFoundError(f"No .h5 files found in {DEFAULT_WAVEFORM_DIR} or its "
                                    "dataset folders. Pass --input explicitly.")
        kwargs["input_path"] = candidates[0]
        logger.info("No --input given; using %s", kwargs["input_path"])
    if args.polarity is not None: kwargs["polarity"] = args.polarity
    if args.mode is not None: kwargs["mode"] = args.mode
    # Each run's plots go into their OWN per-run results folder
    # <base>/<mode>_mode/<input-stem>_<token>_results[_N], so successive runs never
    # overwrite each other (a re-run gets a fresh _N folder unless --overwrite).  The base
    # is --output-dir if given, else the calling driver's "<pipeline>_results" folder.
    #
    # The <mode>_mode level keeps the two ANALYSES apart: gamma-muon and muon mode fit
    # different models to the same channel and produce same-named plots, so without it a
    # muon-mode run silently lands on top of (or beside, as _1) the gamma-muon run of the
    # same file and the folder no longer says which analysis made it.  Only the three
    # energy-reconstruction DRIVERS have a mode; timewalk_report and run_stability share
    # this builder without one and are grouped by their `group` subfolder instead.
    mode = kwargs.get("mode", Config.mode)
    base = Path(args.output_dir) if args.output_dir is not None else results_base(
        script_file or __file__)
    if (program or program_token(script_file or __file__)) in _MODE_ROUTED_PROGRAMS:
        base = base / f"{mode}_mode"
    elif group is not None and args.output_dir is None:
        base = base / group
    kwargs["output_dir"] = resolve_results_dir(
        script_file or __file__, kwargs["input_path"].stem,
        base=base, program=program,
        overwrite=getattr(args, "overwrite", False))
    if args.trigger_sigma is not None: kwargs["trigger_sigma"] = args.trigger_sigma
    if getattr(args, "template_sigma", None) is not None: kwargs["template_sigma"] = args.template_sigma
    if getattr(args, "template_trim", None) is not None: kwargs["template_trim"] = args.template_trim
    if getattr(args, "psd_trim_sigma", None) is not None: kwargs["psd_trim_sigma"] = args.psd_trim_sigma
    if getattr(args, "min_mip_snr", None) is not None: kwargs["min_mip_snr"] = args.min_mip_snr
    if getattr(args, "min_mip_over_trigger", None) is not None:
        kwargs["min_mip_over_trigger"] = args.min_mip_over_trigger
    if args.chunk_events is not None: kwargs["chunk_events"] = args.chunk_events
    kwargs["exclude_pileup"] = args.exclude_pileup
    if args.pileup_chi2_factor is not None: kwargs["pileup_chi2_factor"] = args.pileup_chi2_factor
    if getattr(args, "cut_bootstrap", None) is not None: kwargs["cut_bootstrap"] = args.cut_bootstrap
    if getattr(args, "exclude_hours", None):
        kwargs["exclude_hours"] = tuple((float(lo), float(hi)) for lo, hi in args.exclude_hours)
    if getattr(args, "times", None) is not None:
        kwargs["times_path"] = resolve_input(args.times)
    # Window params: any set explicitly are recorded so auto-estimation leaves them alone.
    kwargs["auto_window"] = args.auto_window
    explicit = []
    for cli_name, field_name in (("pulse_center", "pulse_center"), ("search_pad", "search_pad"),
                                 ("template_pre", "template_pre"), ("template_post", "template_post")):
        val = getattr(args, cli_name)
        if val is not None:
            kwargs[field_name] = val
            explicit.append(field_name)
    kwargs["explicit_window"] = tuple(explicit)
    return Config(**kwargs)


# ===========================================================================
# Loading, noise, baseline
# ===========================================================================

def _require_h5py(path: Path):
    """Validate the extension and import h5py (kept out of the module top level so
    the pipeline imports without h5py when only its pure-numpy parts are used)."""
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError(f"Unsupported input format '{path.suffix}'; only HDF5 "
                         "(.h5/.hdf5) files are supported.")
    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py is required to read HDF5 files "
                         "(pip install h5py / conda install h5py).")
    return h5py


def _select_dataset(f, h5py):
    """Pick the waveform dataset from an OPEN HDF5 file: prefer one literally named
    'waveforms', else the largest 2-D dataset (else the largest of any) so files
    with different names still work.  Returns (name, h5py.Dataset) -- the dataset is
    NOT read here, so the caller controls how much is pulled into memory."""
    if "waveforms" in f and isinstance(f["waveforms"], h5py.Dataset):
        return "waveforms", f["waveforms"]
    datasets: list = []
    f.visititems(lambda n, o: datasets.append((n, o)) if isinstance(o, h5py.Dataset) else None)
    two_d = [(n, d) for n, d in datasets if d.ndim == 2]
    return max(two_d or datasets, key=lambda nd: int(np.prod(nd[1].shape)))


def _squeeze_channel(block: np.ndarray) -> np.ndarray:
    """Drop a size-1 channel axis produced by extract_channels.py: (m, 1, L) -> (m, L)."""
    if block.ndim == 3 and block.shape[1] == 1:
        return block[:, 0, :]
    if block.ndim != 2:
        raise ValueError(
            f"Waveforms must form a 2-D (N, L) array; got a block of shape {block.shape}. "
            "For multi-channel files use extract_channels.py to select one channel first.")
    return block


def dataset_shape(config: Config) -> tuple[int, int]:
    """(N, L) of the waveform dataset WITHOUT reading it into memory -- lets prepare()
    size its model-building prefix and stream the rest in row-slices."""
    h5py = _require_h5py(config.input_path)
    with h5py.File(config.input_path, "r") as f:
        _, ds = _select_dataset(f, h5py)
        shape = tuple(int(s) for s in ds.shape)
    if len(shape) == 3 and shape[1] == 1:
        return shape[0], shape[2]
    if len(shape) != 2:
        raise ValueError(f"Waveforms must form a 2-D (N, L) dataset; got shape {shape}.")
    return shape[0], shape[1]


def load_waveforms(config: Config, max_events: int | None = None,
                   rows: np.ndarray | None = None) -> np.ndarray:
    """Read the waveform dataset -- or only its first `max_events` rows, or only the
    explicit `rows` -- into an (N, L) float array.

    `max_events` caps the read so the model-building pass need not hold the whole file
    resident; None reads everything (used when a caller genuinely wants the full array).
    `rows` (a sorted, increasing index array) reads exactly those rows and takes
    precedence: it is how prepare() builds its model prefix out of the events a
    good-time-interval gate KEEPS, rather than out of the first rows of the file
    regardless of whether the gate excludes them."""
    h5py = _require_h5py(config.input_path)
    with h5py.File(config.input_path, "r") as f:
        name, ds = _select_dataset(f, h5py)
        if rows is not None:
            logger.info("HDF5: using dataset '%s' %s (%d selected rows).",
                        name, ds.shape, rows.size)
            block = ds[rows] if rows.size else ds[:0]
        else:
            n = ds.shape[0] if max_events is None else min(int(max_events), ds.shape[0])
            logger.info("HDF5: using dataset '%s' %s%s.", name, ds.shape,
                        "" if max_events is None else f" (first {n:,} rows)")
            block = ds[:n]
        waveforms = _squeeze_channel(np.asarray(block, dtype=config.dtype))
    logger.info("Loaded %d waveforms of length %d.", *waveforms.shape)
    return waveforms


# ===========================================================================
# Good-time-interval gate
# ===========================================================================

def event_time_hours(config: Config, n_total: int) -> np.ndarray | None:
    """Per-event RUN TIME in hours, row-aligned with the waveform dataset, or None when
    this input has no time axis at all.

    Looked up in the order that prefers provenance travelling WITH the waveforms:
      1. /event_time_rel_s in the input itself (written at conversion by
         file_manipulation/clock_recovery.py -- the normal case for anything converted
         or re-extracted since);
      2. config.times_path, if given;
      3. <input-stem>_times.h5 (what timing_stability/event_times.py writes), if it happens
         to exist -- looked for BESIDE the input, in its per-run folder, then by bare name.
    Row counts are checked against the waveform dataset in every case: a times file from
    a different conversion would silently scramble the mapping."""
    h5py = _require_h5py(config.input_path)

    def _read(path: Path, key: str) -> np.ndarray | None:
        if not path.exists():
            return None
        with h5py.File(path, "r") as f:
            if key not in f:
                return None
            t = np.asarray(f[key][()], dtype=np.float64)
        if t.size != n_total:
            raise ValueError(f"{path.name}:/{key} has {t.size} rows but the waveform "
                             f"dataset has {n_total}: they are not row-aligned.")
        return t

    t = _read(config.input_path, "event_time_rel_s")
    if t is None and config.times_path is not None:
        t = _read(config.times_path, "event_time_rel_s")
        if t is None:
            raise ValueError(f"{config.times_path} has no /event_time_rel_s dataset.")
    if t is None:
        t = _read(find_related(config.input_path, f"{config.input_path.stem}_times.h5"),
                  "event_time_rel_s")
    return None if t is None else t / 3600.0


def gti_mask(config: Config, n_total: int) -> np.ndarray | None:
    """Boolean keep-mask over FILE ROWS for config.exclude_hours, or None when no
    windows are configured (in which case every caller takes its original path and the
    result is bit-identical to before this gate existed)."""
    if not config.exclude_hours:
        return None
    t_h = event_time_hours(config, n_total)
    if t_h is None:
        raise SystemExit(
            "--exclude-hours needs a per-event time axis, and this input has none.\n"
            "  Re-extract the channel (file_manipulation/extract_channels.py now recovers "
            "/event_time_rel_s\n  from the header bank), or build a times file with "
            "timing_stability/event_times.py and pass --times.")
    keep = np.ones(n_total, dtype=bool)
    for lo, hi in config.exclude_hours:
        keep &= ~((t_h >= lo) & (t_h < hi))
    n_drop = int((~keep).sum())
    if n_drop == n_total:
        raise SystemExit("--exclude-hours excludes every event in the run.")
    logger.info("Time gate: excluding %s -> dropped %d / %d events (%.1f%%); "
                "%d remain over %.2f h.",
                ", ".join(f"[{lo:g}, {hi:g}) h" for lo, hi in config.exclude_hours),
                n_drop, n_total, 100.0 * n_drop / n_total, n_total - n_drop,
                float(t_h[keep].max() - t_h[keep].min()))
    return keep


def iter_waveform_chunks(config: Config, chunk_events: int):
    """Yield (start_row, (m, L) float block) over the WHOLE dataset in row-slices of
    `chunk_events`, holding only one block in memory at a time -- the streaming
    substrate for prepare()'s baseline+trigger pass.  h5py reads exactly the
    requested rows off disk, so peak memory is O(chunk_events * L), not O(N * L)."""
    h5py = _require_h5py(config.input_path)
    with h5py.File(config.input_path, "r") as f:
        _, ds = _select_dataset(f, h5py)
        n_total = ds.shape[0]
        for start in range(0, n_total, chunk_events):
            block = _squeeze_channel(np.asarray(ds[start:start + chunk_events], dtype=config.dtype))
            yield start, block


def adc_dynamic_range(raw: np.ndarray, config: Config) -> float:
    """The digitizer's full-scale headroom from the baseline, in the polarity-corrected
    (pulses-up) space -- i.e. the largest pulse height the ADC can represent at all.

    This is the honest denominator for a "% of full scale" statement, and it is knowable
    from the RAW data, which is why it is computed here (in prepare, before the raw
    prefix is dropped): everything downstream sees only baseline-subtracted waveforms and
    can no longer tell where the ADC's rails are.  Without it, saturation_qc had to fall
    back to "the brightest pulse I happened to see" as its ceiling proxy.

      * negative (PMT) pulses swing DOWN toward the ADC's zero code, so the headroom is
        simply the baseline itself.
      * positive pulses swing UP toward the top code, which is inferred from the observed
        maximum as the smallest 2^bits - 1 that contains it (the ADC codes are a power of
        two; these are 12-bit parts reading up to 4095).  If the data never comes close to
        that inferred top the inference is a guess, so we warn.
    """
    raw = np.asarray(raw, dtype=np.float64)
    pre = max(int(0.2 * raw.shape[1]), 1)
    baseline = float(np.median(np.median(raw[:, :pre], axis=1)))
    if config.polarity == "negative":
        return max(baseline, 1.0)
    top_obs = float(raw.max())
    code_top = next((float(2 ** b - 1) for b in (8, 10, 12, 14, 16) if top_obs <= 2 ** b - 1),
                    top_obs)
    if top_obs < 0.5 * code_top:
        logger.warning("ADC full scale inferred as %.0f from a maximum sample of only %.0f; "
                       "the linearity onset is a guess on this channel.", code_top, top_obs)
    return max(code_top - baseline, 1.0)


def robust_noise_sigma(waveforms: np.ndarray, config: Config) -> tuple[np.ndarray, float]:
    block = waveforms[:, :noise_stop(waveforms.shape[1], config)]
    pooled = (block - np.median(block, axis=1, keepdims=True)).ravel()
    sigma = 1.4826 * np.median(np.abs(pooled - np.median(pooled)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Baseline noise estimate is not positive.")
    logger.info("Noise sigma = %.6g ADC.", sigma)
    return pooled, float(sigma)


def remove_baseline(waveforms: np.ndarray, config: Config, warn: bool = True) -> np.ndarray:
    block = waveforms[:, :noise_stop(waveforms.shape[1], config)]
    centered = waveforms - np.median(block, axis=1)[:, None]
    if config.polarity in ("positive", "negative"):
        # BIAS GUARD: the default is 'positive'; running a PMT (negative) channel
        # without --polarity negative would silently analyze inverted noise.  Warn
        # when the data's dominant excursion disagrees with the declared polarity.
        # `warn` is False on the per-chunk streaming calls (the guard already ran
        # once on the model-building prefix) so the message is not repeated per chunk.
        up = float(np.median(centered.max(axis=1)))
        down = float(np.median(-centered.min(axis=1)))
        looks = "negative" if down > up else "positive"
        if warn and looks != config.polarity:
            logger.warning("Declared polarity '%s' but the data looks '%s' (up %.3g vs down %.3g). "
                           "For a PMT channel pass --polarity negative.", config.polarity, looks, up, down)
        return centered if config.polarity == "positive" else -centered
    if config.polarity == "auto":
        flip = centered.max(axis=1) < -centered.min(axis=1)
        return np.where(flip[:, None], -centered, centered)
    raise ValueError(f"Unknown polarity {config.polarity!r}.")


# ===========================================================================
# Exact sub-sample shift + time-walk-free template
# ===========================================================================

def _shift(windows: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Resample each row at (index + shift) by Catmull-Rom cubic interpolation,
    which is third-order accurate (no amplitude-smearing of a plain linear
    interpolation).  Positions off the edge are zero-filled."""
    m, width = windows.shape
    rows = np.arange(m)[:, None]
    pos = np.arange(width)[None, :] + shifts[:, None]
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    f2, f3 = f * f, f * f * f
    w = (-0.5 * f3 + f2 - 0.5 * f, 1.5 * f3 - 2.5 * f2 + 1.0,
         -1.5 * f3 + 2.0 * f2 + 0.5 * f, 0.5 * f3 - 0.5 * f2)

    def tap(offset):
        idx = i + offset
        valid = (idx >= 0) & (idx <= width - 1)
        return np.where(valid, windows[rows, np.clip(idx, 0, width - 1)], 0.0), valid

    p_m1, _ = tap(-1)
    p_0, v0 = tap(0)
    p_1, v1 = tap(1)
    p_2, _ = tap(2)
    out = w[0] * p_m1 + w[1] * p_0 + w[2] * p_1 + w[3] * p_2
    out[~(v0 & v1)] = 0.0
    return out


def _parabolic_delta(cl: np.ndarray, cm: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Sub-sample offset of the parabola vertex through three equally spaced
    samples (left, middle, right of a discrete maximum), clipped to +/-0.5.

    The zero-denominator case (a flat triple) is skipped with np.divide's `where`, not
    masked afterwards with np.where: np.where evaluates BOTH branches, so the old form
    performed the division by zero anyway and only discarded the result -- which raised
    a live RuntimeWarning on flat-topped records (run00270_ch6)."""
    denom = np.asarray(cl - 2.0 * cm + cr, dtype=float)
    delta = np.zeros_like(denom)
    np.divide(0.5 * (cl - cr), denom, out=delta, where=denom != 0.0)
    return np.clip(np.where(np.isfinite(delta), delta, 0.0), -0.5, 0.5)


def _alignment_shifts(windows: np.ndarray, template: np.ndarray, max_lag: int) -> np.ndarray:
    """Sub-sample shift per row maximizing cross-correlation with the template.

    Integer-lag scores are plain sliced dot products over the overlapping samples
    (a row shifted by a whole lag with zero-fill contributes exactly its overlap),
    so the cubic resampler is never invoked just to pick whole samples; it is used
    once, by the caller, to apply the final fractional shift."""
    m, width = windows.shape
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty((m, lags.size))
    for k, lag in enumerate(lags):
        if lag >= 0:
            corr[:, k] = windows[:, lag:] @ template[:width - lag]
        else:
            corr[:, k] = windows[:, :width + lag] @ template[-lag:]
    best = np.clip(np.argmax(corr, axis=1), 1, lags.size - 2)
    rows = np.arange(m)
    delta = _parabolic_delta(corr[rows, best - 1], corr[rows, best], corr[rows, best + 1])
    return lags[best] + delta


def align_windows(windows: np.ndarray, template: np.ndarray | None = None,
                  iters: int = 3, max_lag: int = 6,
                  guide: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The project's pulse-alignment tool: per-row sub-sample shifts estimated by
    cross-correlation against `template` (parabolic peak interpolation), applied by
    Catmull-Rom resampling, iterated with the template rebuilt from the aligned
    mean each pass.  With no `template` the rows' own mean seeds the first pass;
    pass a robust reference (e.g. a median trace) when junk rows could steer the
    mean.  `guide`, when given, is a same-shape copy of `windows` that the shifts
    are ESTIMATED from (e.g. pickup-notched pulses, see build_template); the shifts
    are always APPLIED to `windows`.  Returns (aligned rows, aligned-mean template)."""
    src = windows if guide is None else guide
    if template is None:
        template = src.mean(axis=0)
    aligned = windows
    for _ in range(max(iters, 0)):
        shifts = _alignment_shifts(src, template, max_lag)
        aligned = _shift(src, shifts)
        template = aligned.mean(axis=0)
    if guide is not None and iters > 0:
        aligned = _shift(windows, shifts)
        template = aligned.mean(axis=0)
    return aligned, template


_NOTCH_MIN_FREQ = 0.02        # below this it's broadband/1-over-f, not a line
_NOTCH_MIN_HALF_WIDTH = 0.004  # cycles/sample; the measured pickup line core
_NOTCH_MAX_HALF_WIDTH = 0.010  # a wider "line" than this is broadband: notch its core only
_NOTCH_MAX_LINES = 6
_NOTCH_MAX_BAND_FRAC = 0.10   # never notch more than this much of the [0, 0.5] band
_NOTCH_DETECT_ROWS = 4000


def _pickup_lines(noise_seg: np.ndarray, snr: float) -> list[tuple[float, float]]:
    """Narrow spectral lines in the pre-pulse noise, as (center_freq, half_width)
    pairs in cycles/sample: PSD bins standing >= `snr` times the local broadband
    floor, grouped into contiguous lines.  The floor is a wide-window low
    QUARTILE, not a median: run00270's pickup line is ~10 bins wide, and a
    narrow median window sits on the line's own top and hides it from the test.
    Strongest lines first, capped in count and total bandwidth so a pathological
    spectrum can never notch away the band wholesale.  Empty = no lines."""
    seg = np.asarray(noise_seg[:_NOTCH_DETECT_ROWS], dtype=np.float64)
    m = seg.shape[1]
    if m < 64:
        return []
    psd = np.mean(np.abs(np.fft.rfft(seg * np.hanning(m), axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(m)
    k = 2 * max(12, m // 24) + 1           # odd window, wider than any plausible line
    padded = np.pad(psd, k // 2, mode="edge")
    floor = np.percentile(np.lib.stride_tricks.sliding_window_view(padded, k),
                          25.0, axis=1)
    hot = np.flatnonzero((psd > snr * floor) & (freqs >= _NOTCH_MIN_FREQ))
    if hot.size == 0:
        return []
    bin_w = float(freqs[1])
    lines = []
    for g in np.split(hot, np.flatnonzero(np.diff(hot) > 1) + 1):
        p = psd[g] - floor[g]
        center = float(np.sum(freqs[g] * p) / np.sum(p))
        half_w = float(np.clip(0.5 * (freqs[g[-1]] - freqs[g[0]]) + bin_w,
                               _NOTCH_MIN_HALF_WIDTH, _NOTCH_MAX_HALF_WIDTH))
        lines.append((float(np.sum(p)), center, half_w))
    lines.sort(reverse=True)
    kept, budget = [], 0.5 * _NOTCH_MAX_BAND_FRAC
    for _, center, half_w in lines[:_NOTCH_MAX_LINES]:
        if 2.0 * half_w <= budget:
            kept.append((center, half_w))
            budget -= 2.0 * half_w
    return kept


def _notch_rows(rows: np.ndarray, lines: list[tuple[float, float]]) -> np.ndarray:
    """Zero the given (center, half_width) frequency bands out of each row.  Used
    only on the alignment GUIDE copy; the pulses that get averaged stay unfiltered."""
    length = rows.shape[1]
    freqs = np.fft.rfftfreq(length)
    mask = np.ones(freqs.size)
    for center, half_w in lines:
        mask[np.abs(freqs - center) < half_w] = 0.0
    out = np.empty_like(rows, dtype=np.float32)
    for s in range(0, rows.shape[0], 4096):
        blk = rows[s:s + 4096]
        out[s:s + len(blk)] = np.fft.irfft(np.fft.rfft(blk, axis=1) * mask,
                                           n=length, axis=1).astype(np.float32)
    return out


def _halfmax_starts(sub: np.ndarray, pk: np.ndarray, pre: int) -> np.ndarray:
    """Window starts from the half-max LEADING-EDGE pick: per row, the last sample
    at/below 50% of the peak, before the peak (same pick as
    hodoscope_efficiency._leading_edge_pos).  Windows are cut so the MEDIAN pulse
    still peaks `pre` samples in, so the window layout matches a peak-aligned cut
    and template_pre/template_post keep their meaning.  Shared by build_template
    and _peak_windows so the diagnostics view the pulses exactly as the template
    build does (an argmax cut would re-phase coherent pickup; see build_template)."""
    length = sub.shape[1]
    thr = 0.5 * sub[np.arange(pk.size), pk]
    grid = np.arange(length)[None, :]
    below = (sub <= thr[:, None]) & (grid <= pk[:, None])
    edge = np.where(below, grid, -1).max(axis=1)
    edge = np.where(edge >= 0, edge, pk)     # degenerate record: fall back to the peak
    rise50 = int(np.median(pk - edge))
    return edge + rise50 - pre


def build_template(corrected: np.ndarray, noise_sigma: float, config: Config):
    """Average bright pulses into a template, roughly aligned on each pulse's
    half-max LEADING-EDGE crossing (a constant-fraction pick) then refined by exact
    sub-sample cross-correlation, so the shape carries no trigger time walk.

    The rough run deliberately avoids the argmax: on a broad-crested pulse the
    argmax lands wherever noise pushes the crest highest, which biases the averaged
    peak high and re-phases event-incoherent periodic pickup into a coherent
    template ripple that the refinement then locks onto (it maximizes correlation
    against a template already carrying the ripple).  The half-max point sits on
    the steep rising edge, where crossing-time jitter is smallest and crest ripple
    barely moves it -- measured on run00270_ch0 this cuts the pickup-band ripple in
    the final template ~8x; on fast PMT pulses the two picks converge to the same
    template.

    On the slow-rise SiPM channels the half-max pick alone is NOT enough: the pickup
    is phase-random event-to-event (resultant |<e^i phi>| ~ 0.004 on run00270_ch1,
    i.e. chance level), so an average whose alignment cannot see the line kills it
    ~sqrt(N)-style -- but a ~46 ADC line on a ~35 ADC/sample rising edge moves the
    half-max crossing by over a sample (a large fraction of the 9.4-sample period)
    and the xcorr refine then locks the phase anyway.  So every pick and shift is
    estimated on a GUIDE copy of the pulses with the detected noise lines notched
    out (_pickup_lines / _notch_rows), and only APPLIED to the raw pulses, whose
    shape is never filtered.  Measured on run00270_ch1: tail ripple 31 -> 1.7 ADC.
    On a line-free channel no notch is found and the build is unchanged."""
    n, length = corrected.shape
    region = search_region(length, config)
    region_start = region.start
    sub = corrected[:min(n, config.template_cap)]

    lines = (_pickup_lines(sub[:, :region_start], config.align_notch_snr)
             if config.align_notch_snr > 0 else [])
    guide = _notch_rows(sub, lines) if lines else sub
    if lines:
        logger.info("Template alignment guided by a pickup-notched copy: line(s) at %s "
                    "cycles/sample.", ", ".join(f"{c:.4f}" for c, _ in lines))

    region_max = guide[:, region].max(axis=1)
    bright = region_max > config.template_sigma * noise_sigma
    if int(bright.sum()) < 50:
        bright = region_max >= np.percentile(region_max, 90.0)
    idx = np.flatnonzero(bright)
    pk = region_start + guide[idx][:, region].argmax(axis=1)

    starts = _halfmax_starts(guide[idx], pk, config.template_pre)

    # Cut the windows PADDED by align_lag on each side, then crop that margin back
    # off after alignment.  align_windows shifts each pulse by up to align_lag
    # samples and zero-fills whatever runs off the window edge; averaging with a
    # fixed denominator then steps the mean DOWN over the last align_lag samples
    # (a spurious ~0.5%-of-peak discontinuity in the decay tail).  Padding keeps
    # every REPORTED sample fully supported by all events during alignment, so the
    # returned [-template_pre, template_post) template carries no edge artifact.
    pad = config.align_lag
    width = config.template_pre + config.template_post
    padded = width + 2 * pad
    starts = starts - pad
    fits = (starts >= 0) & (starts + padded <= length)
    idx, starts = idx[fits], starts[fits]
    take = starts[:, None] + np.arange(padded)[None, :]
    windows = sub[idx[:, None], take]
    aligned, template = align_windows(windows, iters=config.align_iters,
                                      max_lag=config.align_lag,
                                      guide=guide[idx[:, None], take] if lines else None)
    if config.template_trim > 0:
        # Opt-in robust FINAL average (Config.template_trim): the plain mean is pulled
        # wider by a minority tail of events; per-sample trimming removes exactly that
        # while keeping mean-like noise averaging.  Only the final template changes --
        # the alignment iterations keep their plain-mean reference.
        template = trim_mean(aligned, min(float(config.template_trim), 0.45), axis=0)
    core = slice(pad, pad + width)
    template = template[core]
    relative = np.arange(-config.template_pre, config.template_post, dtype=float)
    logger.info("Template built from %d bright pulses.", int(idx.size))
    return relative, template, aligned[:, core].std(axis=0), int(idx.size)


def _rail_clipped_mask(win: np.ndarray, noise_sigma: float, polarity: str,
                       adc_range: float | None = None
                       ) -> tuple[np.ndarray, float, bool]:
    """The project's ONE rail-clipping test on baseline-CORRECTED pulse windows
    (pulses-UP), shared by _clean_model_set and saturation_qc so both stages flag
    a clipped event on identical grounds.

    In corrected space the rail is smeared over a few ADC by each event's own
    baseline estimate, so a genuine rail shows up as a pile-up of peak heights in
    a noise-scaled band below the maximum (1-ADC bins would split it); an
    unclipped spectrum ends in a sparse tail where that band holds ~1 event, so
    require >=3 events and >=0.2% of the sample and a single bright pair can't
    fake a rail.  With a rail found, events with a flat top AT it are flagged via
    waveform_triage.detect_saturation -- fast pulses (PMT / negative polarity) sit
    on the rail for fewer samples, so consec follows triage's sipm=4 / pmt=2
    default.  With no rail nothing is truncated and the mask is empty.  `win`
    must hold real-pulse events only: noise records pile up at the noise floor
    and would be mistaken for the rail.

    `adc_range` is the digitizer's OWN full-scale distance from the baseline
    (Prepared.adc_range).  When no rail pile-up is found it is returned as the
    ceiling, so the linearity onset downstream is anchored to the actual dynamic
    range.  The old fallback was `peak_h.max()` -- the single brightest event seen --
    which makes the "70% of full scale" onset a statistical accident of how bright
    the brightest pulse happened to be.

    The flat-top tolerance handed to detect_saturation is NOISE-scaled (3 sigma
    around the rail) rather than triage's default 1%-of-rail: a relative tolerance
    applied in baseline-CORRECTED space means a ~36 ADC band on run00270_ch0 but only
    ~7 ADC on ch9, i.e. a different physical test per channel.

    Returns (clipped mask, rail estimate, rail found)."""
    peak_h = win.max(axis=1).astype(np.float64)
    tol = max(1.0, 3.0 * noise_sigma)
    top_band = peak_h >= peak_h.max() - tol
    n_top = int(top_band.sum())
    clipped_pileup = n_top >= 3 and n_top / peak_h.size >= 0.002
    if clipped_pileup:
        rail = float(np.median(peak_h[top_band]))
    elif adc_range is not None and np.isfinite(adc_range) and adc_range > 0:
        rail = float(adc_range)
    else:
        rail = float(peak_h.max())
    consec = 2 if polarity == "negative" else 4
    if clipped_pileup:
        rail_tol = float(np.clip(tol / max(rail, 1e-9), 1e-4, 0.5))
        sat_mask, _, _ = detect_saturation(win, saturation_adc=rail, consec=consec,
                                           rail_tol=rail_tol)
        sat_mask = np.asarray(sat_mask, dtype=bool)
    else:
        sat_mask = np.zeros(peak_h.shape, dtype=bool)
    return sat_mask, rail, clipped_pileup


def _clean_model_set(corrected: np.ndarray, noise_sigma: float, config: Config,
                     adc_range: float | None = None) -> np.ndarray:
    """Drop waveform_triage NOISE and PILEUP events, plus rail-CLIPPED events,
    from the TEMPLATE-building prefix.

    `corrected` is per-event baseline-subtracted and oriented pulses-UP
    (remove_baseline output), so triage's shared classifier (classify_events) runs
    with a global baseline of 0 and the pooled noise sigma; its pulse window is the
    optimal-filter search region (pulse_center +/- search_pad).  Each excluded
    class is a shape the averaged template must not carry:
      * NOISE   -- no real pulse in the window, or an oscillating record, that
        would otherwise clear build_template's 8-sigma bright cut;
      * PILEUP  -- a resolved extra pulse smears a secondary bump into the mean
        (and pileup events are bright, so they pass the cut preferentially);
      * CLIPPED -- flat-topped pulses average a flattened, broadened crest into
        the template (~3% wider on run00270_ch0's rail hits).  classify_events'
        own SATURATED label cannot be trusted here -- its integer-value rail
        pile-up is smeared out once each event's baseline is subtracted -- so
        clipping is re-detected with the shared corrected-space rail test
        (_rail_clipped_mask, saturation_qc's exact grounds), on the kept
        real-pulse events only (noise records would fake a rail at the floor).

    The cuts follow the channel's READOUT preset (triage_cuts), resolved from the
    polarity: classify_events' bare defaults ARE the SiPM preset, so passing none --
    as this did -- silently applied SiPM cuts to a PMT channel (ch9/ch10/caen).  The
    one that bites is min_separation=40 on a fast PMT pulse: a genuine second pulse
    25-39 samples after the main one is skipped, and the event survives into the
    template as CLEAN.  Now the CLI and this call classify a channel identically.

    Used for the TEMPLATE ONLY: the noise PSD and the trigger's response-noise
    scale are deliberately built from the FULL prefix (the pickup/oscillation
    dropped here is genuine broadband noise the optimal filter must model;
    cleaning it underestimates N(f), over-sensitizes the trigger and inflates the
    per-event chi2).  The MEASUREMENT pass triggers over every event, so no real
    pulse is discarded here.

    If the cuts would flag nearly everything -- a sign the window or polarity is
    wrong, not that the data is unusable -- the model is built from ALL events and
    a warning is logged rather than starving the template/PSD."""
    n, length = corrected.shape
    region = search_region(length, config)
    lo, hi = max(region.start, 1), region.stop
    cuts = triage_cuts(polarity=config.polarity)
    labels, _, _, _ = classify_events(corrected, 0.0, float(noise_sigma), lo, hi, **cuts)
    keep = (labels != "NOISE") & (labels != "PILEUP")
    n_clipped = 0
    if keep.any():
        kept_idx = np.flatnonzero(keep)
        sat_mask, _, rail_found = _rail_clipped_mask(corrected[kept_idx][:, lo:hi],
                                                     noise_sigma, config.polarity,
                                                     adc_range=adc_range)
        if rail_found:
            keep[kept_idx[sat_mask]] = False
            n_clipped = int(sat_mask.sum())
    n_keep = int(keep.sum())
    dropped = n - n_keep
    # Starvation guard.  The old test (n_keep < min(200, n) or n_keep < 2% of n) was so
    # lax it passed run00270_ch4 -- where triage flagged 13,723 of 14,994 events (91%) as
    # NOISE and the template was averaged from 356 pulses -- without a word.  Discarding
    # most of a channel is either a broken window/polarity or a channel with almost no
    # signal; both deserve to be loud.
    if n_keep < 200 or n_keep < 0.05 * n:
        logger.warning("Template cleaning flagged %d/%d model-prefix events "
                       "(NOISE/PILEUP/clipped), leaving too few clean events; building "
                       "the model from ALL events instead (check --polarity / "
                       "--pulse-center and the pulse window).", dropped, n)
        return corrected
    if dropped > 0.5 * n:
        logger.warning("Template cleaning dropped %d/%d model-prefix events (%.0f%%) as "
                       "NOISE/PILEUP/clipped -- most of the channel. The template is built "
                       "from the %d survivors, but a channel this noise-dominated will not "
                       "support a trustworthy MIP fit; check --polarity / --pulse-center, "
                       "and expect the reliability guard to flag the result.",
                       dropped, n, 100.0 * dropped / max(n, 1), n_keep)
    elif dropped:
        logger.info("Triage cleaned the template set: dropped %d/%d prefix events "
                    "(%d NOISE, %d PILEUP, %d rail-clipped); the template uses the "
                    "remaining %d (PSD and trigger-noise still use the full prefix).",
                    dropped, n, int((labels == "NOISE").sum()),
                    int((labels == "PILEUP").sum()), n_clipped, n_keep)
    return corrected[keep]


# ===========================================================================
# Noise PSD + optimal (noise-weighted) filter kernel
# ===========================================================================

def noise_event_mask(corrected: np.ndarray, config: Config) -> np.ndarray:
    """Keep-mask over events for the NOISE MODEL (PSD + trigger response scale): drop
    pathological interference bursts.

    Both estimators average over events, and an average is outlier-dominated.  On
    run00270 ~1% of events are catastrophic amplitude-modulated pickup bursts -- the
    coherent carrier swings ~10x its usual amplitude, sweeping most of the ADC range --
    and their beat envelope dumps power at LOW frequency.  Averaged in, those ~1% inflate
    N(f) by up to 70x around f~0.02 cycles/sample, which is INSIDE the signal band: the
    optimal filter's 1/N(f) weighting is then crushed at frequencies where the other 99%
    of events actually have good SNR.  A mean over a two-population mixture is nobody's
    noise -- the filter ends up optimal for neither.  So the noise model is built from the
    ordinary population, and the burst events (which pass 2 still triggers on) are left to
    declare themselves through a bad per-event chi2.

    This is NOT the same operation as feeding the triage-cleaned set to the PSD, which is
    deliberately avoided (see prepare): the pickup and oscillation that triage labels NOISE
    are genuine, always-present broadband noise the filter MUST model, and removing them
    under-estimates N(f) and over-sensitizes the trigger.  Here we remove a rare tail of
    outlier events, not a class of ordinary ones.

    The cut is on log total pre-pulse power, centred and scaled by median and MAD: log
    because power is positive and heavy-tailed, MAD because this run's noise genuinely
    switches between two carrier levels over the run and a plain standard deviation would
    be dragged wide by that bimodality.  Self-calibrating, and it assumes nothing about
    WHERE in frequency the pathology sits.  Returns an all-True mask when disabled
    (psd_trim_sigma = 0) or when the statistics do not support a cut.
    """
    n = corrected.shape[0]
    keep = np.ones(n, dtype=bool)
    if config.psd_trim_sigma <= 0 or n < 200:
        return keep
    L = config.template_pre + config.template_post
    v = np.log(np.var(corrected[:, :L].astype(np.float64), axis=1) + 1e-12)
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    if not np.isfinite(mad) or mad <= 0:
        return keep
    keep = v <= med + config.psd_trim_sigma * mad
    dropped = int(n - keep.sum())
    # Refuse to trim a large fraction: that would mean the "outliers" ARE the population
    # (a channel whose noise is genuinely bimodal in power), and the robust model would be
    # built from an unrepresentative half.
    if dropped > 0.05 * n:
        logger.warning("Noise-model trim would drop %d/%d events (%.1f%%) as interference "
                       "bursts -- too many to be a rare pathology, so the trim is SKIPPED "
                       "and the PSD keeps every event. The pre-pulse power on this channel "
                       "is not a tight unimodal distribution; check 14_noise_stationarity.",
                       dropped, n, 100.0 * dropped / n)
        return np.ones(n, dtype=bool)
    if dropped:
        logger.info("Noise model: dropped %d/%d prefix events (%.2f%%) as interference "
                    "bursts (log pre-pulse power > median + %.1f MAD sigma); the PSD and "
                    "the trigger's response-noise scale are built from the remaining %d.",
                    dropped, n, 100.0 * dropped / n, config.psd_trim_sigma, int(keep.sum()))
    return keep


def noise_psd(corrected: np.ndarray, config: Config,
              mask: np.ndarray | None = None) -> np.ndarray:
    """Average two-sided noise power spectrum from pulse-free pre-pulse segments.

    No per-segment mean subtraction: that would zero the DC bin and make the
    1/N(f) weighting blow up at DC.  The real low-frequency noise power is kept,
    so the optimal filter correctly down-weights it.

    `mask` selects the events the average runs over (see noise_event_mask); None uses
    every event.
    """
    L = config.template_pre + config.template_post
    seg = corrected[:, :L].astype(np.float64)
    if mask is not None:
        seg = seg[mask]
    if seg.shape[0] > config.psd_cap:
        seg = seg[np.random.default_rng(config.seed).choice(seg.shape[0], config.psd_cap, replace=False)]
    psd = np.mean(np.abs(np.fft.fft(seg, axis=1)) ** 2, axis=0)
    return np.maximum(psd, psd.max() * 1e-8)


def optimal_filter_kernel(template: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Time-domain optimal filter, IFFT(S(f) / N(f)).  Sliding it over the data
    gives the colored-noise matched-filter response; reduces to the template
    (ordinary matched filter) when N(f) is flat.  The template is zero-meaned so
    the kernel carries no DC and ignores baseline level."""
    S = np.fft.fft(template - template.mean())
    return np.real(np.fft.ifft(S / psd))


# ===========================================================================
# Trigger (slide the optimal kernel; threshold on measured response noise)
# ===========================================================================

def _response_noise_rms(corrected: np.ndarray, kernel: np.ndarray, lo: int,
                        mask: np.ndarray | None = None) -> float | None:
    """RMS of the optimal-filter response over pulse-free pre-pulse segments (the
    trigger's noise scale).  Measured ONCE on the model-building prefix and reused
    for every streamed chunk so the threshold is identical across chunks and matches
    the single-pass trigger; returns None when the pre-pulse region is too short.

    `mask` MUST be the same event mask the PSD used (noise_event_mask): the kernel is
    S/N built from the trimmed PSD, so it carries more low-frequency gain than before,
    and measuring its noise scale on the untrimmed events would let the very bursts the
    PSD excluded inflate the trigger threshold instead -- losing real pulses.  The two
    halves of the noise model have to describe the same population."""
    L = kernel.size
    rows = corrected if mask is None else corrected[mask]
    segs = [rows[:, s:s + L] @ kernel for s in range(0, max(lo - L, 1), L)]
    return float(np.std(np.concatenate(segs))) if segs else None


def filter_trigger(corrected: np.ndarray, kernel: np.ndarray, config: Config,
                   resp_rms: float | None = None, peak_offset: int | None = None):
    """Slide the optimal kernel over a search window centered on the expected pulse
    position; trigger where the peak response exceeds trigger_sigma times the
    response noise measured in a pulse-free region.

    `resp_rms` (measured once on the model prefix) fixes the noise scale so streamed
    chunks all threshold identically; when None it is measured from THIS batch.

    Returns (triggered mask, sub-sample WINDOW START, sub-sample PULSE PEAK, response
    significance).  The two are deliberately distinct:

      * window_start = starts[jbest] + delta is where the template best overlays the
        data.  The optimal filter cuts its window HERE.
      * peak = window_start + peak_offset, where `peak_offset` is the template's OWN
        peak index (argmax(template)).  This is the physical pulse peak, and it is what
        the boxcar integrates around and what plot_timing reports.

    They differ whenever argmax(template) != template_pre, which happens routinely
    (measured d = +1 on run00270_ch0, -1 on caen_ch0).  The old code returned a single
    "peak" fixed at window_start + template_pre: self-consistent for the OF (which only
    ever re-derives the window start from it), but WRONG for the boxcar, which then
    integrated the data around window_start + template_pre while normalising by the
    template around argmax(template) -- a d-sample mismatch worth -1.9% on ch0 and
    +4.0% on caen, i.e. most of the OF-vs-boxcar MPV disagreement."""
    n, length = corrected.shape
    center = pulse_center(length, config)
    if peak_offset is None:
        peak_offset = config.template_pre
    L = kernel.size

    # The window START corresponding to a pulse peaking at `center` is center - peak_offset.
    lo = max(center - peak_offset - config.search_pad, 0)
    hi = min(center - peak_offset + config.search_pad, length - L)
    starts = np.arange(lo, hi)
    rows = np.arange(n)
    resp = np.empty((n, starts.size))
    for j, s in enumerate(starts):
        resp[:, j] = corrected[:, s:s + L] @ kernel

    if resp_rms is None:
        resp_rms = _response_noise_rms(corrected, kernel, lo)
        if resp_rms is None:
            resp_rms = float(np.std(resp))
    resp_rms = max(resp_rms, 1e-9)

    jbest = resp.argmax(axis=1)
    rbest = resp[rows, jbest]
    triggered = rbest > config.trigger_sigma * resp_rms

    jb = np.clip(jbest, 1, starts.size - 2)
    delta = _parabolic_delta(resp[rows, jb - 1], resp[rows, jb], resp[rows, jb + 1])
    window_start = starts[jbest] + delta
    peak = window_start + peak_offset
    return triggered, window_start, peak, rbest / resp_rms


def extract_at(corrected: np.ndarray, rows: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Per-event windows of fixed length starting at integer `starts` (clipped)."""
    starts = np.clip(starts, 0, corrected.shape[1] - length)
    return corrected[rows[:, None], starts[:, None] + np.arange(length)[None, :]].astype(np.float64)


# ===========================================================================
# Prepared bundle
# ===========================================================================

@dataclass
class Prepared:
    config: Config             # the resolved config (with the auto-estimated window)
    n_events: int              # events ANALYZED (triggered + not), i.e. the file minus anything
                               # --exclude-hours dropped; the trigger-rate denominator
    length: int                # samples per waveform
    pooled: np.ndarray
    noise_sigma: float
    corrected: np.ndarray      # baseline-corrected TRIGGERED events only (rows align with `events`)
    events: pd.DataFrame       # event_number = row into `corrected`; original_index = row in the file
    relative: np.ndarray
    template: np.ndarray
    template_std: np.ndarray
    n_template: int
    psd: np.ndarray
    adc_range: float = float("nan")   # digitizer full-scale headroom from baseline (see adc_dynamic_range)
    # The TRIGGER THRESHOLD, expressed in the observable's own (template-relative amplitude)
    # units -- i.e. the lowest pulse amplitude that could possibly have entered `corrected`.
    # Nothing below this exists in ANY spectrum built from these events, because the trigger
    # threw it away before the spectrum was built.  See trigger_floor_amplitude: it is what
    # makes a truncated low edge recognisable as the instrument rather than as physics.
    trigger_floor: float = float("nan")


def trigger_floor_amplitude(template: np.ndarray, psd: np.ndarray, resp_rms: float | None,
                            config: Config) -> float:
    """The trigger's threshold, converted from filter-response units into the AMPLITUDE units
    every observable in this pipeline is normalised to (template = 1).

    filter_trigger fires when the optimal-filter response r = x . kernel exceeds
    trigger_sigma * resp_rms.  For kernel = IFFT(S/N) and the OF amplitude
    A = Re[X . conj(S)/N] / denom  (denom = sum |S|^2/N), the two are the same statistic up
    to a constant:  r = (denom / L) * A.  So the threshold in amplitude units is

        floor = trigger_sigma * resp_rms * L / denom

    Equivalently floor = trigger_sigma * sigma_A, since resp_rms * L / denom IS the optimal
    filter's amplitude resolution -- which is the useful reading: the spectrum is truncated
    at trigger_sigma resolution units, and a channel whose MIP line sits only a few sigma_A
    above that has had its low-amplitude population cut away by the trigger, not by physics.

    Both estimators share this scale (boxcar_amplitude normalises by the template integral
    for exactly that reason), so the floor is meaningful on either spectrum -- though only
    the OF spectrum shows it as a HARD edge; the boxcar re-measures the same events with its
    own noise and so blurs the same truncation into a soft turn-on."""
    if resp_rms is None or not np.isfinite(resp_rms):
        return float("nan")
    S = np.fft.fft(template - template.mean())
    denom = float(np.sum((np.abs(S) ** 2) / psd).real)
    if denom <= 0:
        return float("nan")
    return float(config.trigger_sigma * resp_rms * S.size / denom)


def triggered(prep: Prepared) -> np.ndarray:
    """The triggered events' corrected waveforms, row-aligned with prep.events.

    Since the streaming prepare() keeps ONLY triggered rows, `corrected` already IS that
    array and `events["event_number"]` is just arange(n).  Indexing by it -- which
    saturation_qc, boxcar_amplitude and _peak_windows all used to do -- is an identity
    permutation that fancy-indexing materialises as a full copy (~100 MB per call on
    run00270_ch0, several times per run).  Go through here instead."""
    return prep.corrected


# Prefix used for model building (template / PSD / noise sigma / polarity / window).
# The template and PSD already only consume their first template_cap / psd_cap events,
# so a prefix at least this large reproduces the single-pass template and PSD exactly;
# polarity, the auto-window and the noise sigma are estimated from the same prefix (a
# large sample -> statistically identical to using every event).  When the file has
# fewer rows than this, the prefix IS the whole file and the result is bit-identical to
# the old single-pass prepare().
def _model_prefix_size(config: Config) -> int:
    return max(config.template_cap, config.psd_cap, 20_000)


def prepare(config: Config, chunk_events: int | None = None) -> Prepared:
    """Two-pass, streaming preparation.

    Pass 1 (model): read a bounded prefix and build the noise sigma, time-walk-free
    template, noise PSD and optimal-filter kernel from it -- peak memory O(prefix * L).
    Pass 2 (apply): stream the WHOLE file in row-chunks through baseline subtraction
    and the optimal-filter trigger, retaining only the TRIGGERED events' corrected
    waveforms -- peak memory O(chunk * L) plus the kept events, never the full float
    dataset.  For files that fit under the prefix size this is numerically identical
    to loading everything at once.

    `chunk_events` overrides config.chunk_events for a programmatic call; None (the
    default) uses the configured value (CLI --chunk-events)."""
    chunk_events = config.chunk_events if chunk_events is None else chunk_events
    n_total, length = dataset_shape(config)

    # Good-time-interval gate (no-op unless --exclude-hours was given).  Excluded events
    # are dropped from BOTH passes: they must not reach the noise model either, since the
    # whole point of excluding a noisy window is to keep it out of the PSD and the trigger
    # threshold that every other event is then judged against.
    keep = gti_mask(config, n_total)
    n_analyzed = n_total if keep is None else int(keep.sum())

    # --- Pass 1: model building on a bounded prefix -------------------------------
    prefix_size = _model_prefix_size(config)
    if keep is None:
        prefix = load_waveforms(config, max_events=prefix_size)
    else:
        # Take the first `prefix_size` KEPT rows, not the first rows of the file -- a gate
        # covering the start of the run would otherwise gut (or empty) the model set.
        prefix = load_waveforms(config, rows=np.flatnonzero(keep)[:prefix_size])
    # BIAS GUARD: the noise model (PSD, template, noise sigma, trigger threshold) is built
    # from a bounded prefix and then applied to every event in the run.  That is only sound
    # if the run is stationary in noise -- and timing_stability/run_stability.py exists
    # precisely because it may not be.  When the file is larger than the prefix, the model
    # describes the EARLY run and is extrapolated to the rest, so say so.
    if n_analyzed > prefix_size:
        logger.warning("Noise model (PSD / template / trigger threshold) is built from the "
                       "first %d of %d events -- the EARLY run -- and applied to all of them. "
                       "Check timing_stability/run_stability.py for a noise drift before "
                       "trusting time-integrated results; exclude a bad window with "
                       "--exclude-hours.", prefix_size, n_analyzed)
    config = resolve_polarity(prefix, config)         # decide ONE sign for the whole channel
    config = resolve_auto_window(prefix, config)      # fill in the pulse window from the data
    pooled, noise_sigma = robust_noise_sigma(prefix, config)
    corrected_prefix = remove_baseline(prefix, config)
    # Clean the TEMPLATE-building set with waveform_triage: drop events it labels NOISE
    # (no real pulse in the search window, or an oscillating record) or PILEUP (a
    # resolved extra pulse would smear into the mean), plus rail-CLIPPED events (a flat
    # top averages a broadened crest in; re-detected with the shared corrected-space
    # rail test since the classifier's own rail detector is blind after per-event
    # baseline removal).  ONLY the template uses the cleaned set -- the noise PSD and the
    # trigger's response-noise scale are built from the FULL prefix, because the
    # pickup/oscillation triage drops is genuine broadband noise the optimal filter must
    # model: estimating N(f) or resp_rms from the cleaned set underestimates the noise,
    # over-sensitizes the trigger (it then admits low-amplitude junk that drags the
    # gamma/muon cut down and wrecks the spectrum fit) and inflates the per-event chi2.
    # Pass 2 triggers over EVERY event regardless, so no real pulse is discarded here.
    # The ONE exception is noise_event_mask below: a rare tail of interference-burst
    # events is trimmed from the PSD/resp_rms average.  That is an estimator-robustness
    # fix (a mean is outlier-dominated), NOT the class-dropping described above -- read
    # noise_event_mask before conflating the two.
    adc_range = adc_dynamic_range(prefix, config)
    model_prefix = _clean_model_set(corrected_prefix, noise_sigma, config, adc_range=adc_range)
    relative, template, tstd, n_tmpl = build_template(model_prefix, noise_sigma, config)
    # ONE event mask for BOTH halves of the noise model -- see noise_event_mask and the
    # note in _response_noise_rms on why they cannot disagree.
    noise_mask = noise_event_mask(corrected_prefix, config)
    psd = noise_psd(corrected_prefix, config, mask=noise_mask)
    kernel = optimal_filter_kernel(template, psd)
    # The template's OWN peak index -- NOT template_pre, which is only where the rough
    # half-max cut aimed to put it (they differ by a sample on real data).  Everything
    # that needs the physical pulse peak is referenced to this; see filter_trigger.
    peak_offset = int(np.argmax(template))
    # Fix the trigger's response-noise scale once (measured on the FULL prefix) so every
    # streamed chunk thresholds identically -- reproducing the single-pass trigger.
    center = pulse_center(length, config)
    lo = max(center - peak_offset - config.search_pad, 0)
    resp_rms = _response_noise_rms(corrected_prefix, kernel, lo, mask=noise_mask)
    # The same threshold in the OBSERVABLE's units, so every spectrum below can say where
    # the trigger cut the data off (see trigger_floor_amplitude).
    trig_floor = trigger_floor_amplitude(template, psd, resp_rms, config)
    del prefix, corrected_prefix, model_prefix

    # --- Pass 2: stream baseline + trigger; keep only triggered events ------------
    kept_corr, kept_pidx, kept_psub, kept_wstart, kept_orig = [], [], [], [], []
    for start, block in iter_waveform_chunks(config, chunk_events):
        # The file rows this block covers, minus anything the time gate excludes.  Carrying
        # `rows` (rather than start+sel) keeps original_index a true SOURCE-FILE row, so
        # every row-aligned companion array -- the times, triage labels -- still indexes
        # correctly when the gate has punched holes in the event set.
        rows = np.arange(start, start + block.shape[0], dtype=np.int64)
        if keep is not None:
            in_gti = keep[start:start + block.shape[0]]
            if not in_gti.any():
                continue
            block, rows = block[in_gti], rows[in_gti]
        cblock = remove_baseline(block, config, warn=False)
        fired, wstart, peak, _ = filter_trigger(cblock, kernel, config, resp_rms=resp_rms,
                                                peak_offset=peak_offset)
        sel = np.flatnonzero(fired)
        if sel.size:
            kept_corr.append(cblock[sel])
            kept_pidx.append(np.round(peak[sel]).astype(np.int64))
            kept_psub.append(peak[sel].astype(np.float64))
            kept_wstart.append(wstart[sel].astype(np.float64))
            kept_orig.append(rows[sel])
    if not kept_corr:
        raise ValueError("No window triggered.")
    corrected = np.concatenate(kept_corr)
    n_trig = corrected.shape[0]
    events = pd.DataFrame({
        # event_number indexes `corrected` directly (rows are already the triggered
        # subset, in file order); original_index preserves the row in the source file.
        "event_number": np.arange(n_trig, dtype=np.int64),
        "original_index": np.concatenate(kept_orig),
        # window_start: sub-sample position where the TEMPLATE best overlays the record
        # (what the optimal filter cuts its window at).
        "window_start": np.concatenate(kept_wstart),
        # peak_*: the physical pulse peak = window_start + argmax(template).
        "peak_index": np.concatenate(kept_pidx),
        "peak_subsample": np.concatenate(kept_psub),
    })
    logger.info("Triggered %d / %d analyzed windows (%.1f%%)%s.",
                n_trig, n_analyzed, 100.0 * n_trig / max(n_analyzed, 1),
                "" if keep is None else f"; {n_total - n_analyzed} excluded by the time gate")

    # BIAS GUARD: the trigger only searches pulse_center +/- search_pad.  If the real
    # pulse sits outside that window, events are silently dropped or mis-timed.  Warn
    # when the trigger rate is very low, or triggered peaks pile up at the search edge
    # (a sign the true pulse is at/beyond the window).
    trig_frac = n_trig / max(n_analyzed, 1)
    if trig_frac < 0.3:
        logger.warning("Only %.0f%% of windows triggered; the pulse may not sit near "
                       "--pulse-center=%d (search +/- %d).", 100 * trig_frac, center, config.search_pad)
    peaks = events["peak_subsample"].to_numpy()
    edge_frac = float(np.mean(np.abs(peaks - center) > 0.9 * config.search_pad))
    if edge_frac > 0.2:
        logger.warning("%.0f%% of triggered peaks sit at the edge of the search window "
                       "(--pulse-center=%d +/- %d); the real pulse may be outside it -- "
                       "set --pulse-center.", 100 * edge_frac, center, config.search_pad)

    return Prepared(config, n_analyzed, length, pooled, noise_sigma, corrected, events, relative,
                    template, tstd, n_tmpl, psd, adc_range, trig_floor)


# ===========================================================================
# Gamma/muon spectrum: independent valley-split fits + equal-area cut
# ===========================================================================

def _langau(x, area, loc, eta, sigma):
    """Landau(loc, eta) convolved with Gaussian(sigma) -- the 'langaus' muon line."""
    x = np.asarray(x, dtype=float)
    eta, sigma = max(float(eta), 1e-9), max(float(sigma), 1e-9)
    pad = 8.0 * sigma + 12.0 * eta
    step = max(min(sigma, eta) / 8.0, (x.max() - x.min()) / 4000.0)
    grid = np.arange(x.min() - pad, x.max() + pad + step, step)
    pdf = np.nan_to_num(landau.pdf(grid, loc=loc, scale=eta), nan=0.0, posinf=0.0, neginf=0.0)
    conv = gaussian_filter1d(pdf, sigma / step, mode="constant")
    return area * np.interp(x, grid, conv, left=0.0, right=0.0)


def _sum_of_gaussians(x, *params):
    out = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        a, mu, sig = params[i], params[i + 1], params[i + 2]
        out += a * np.exp(-0.5 * ((x - mu) / max(sig, 1e-9)) ** 2)
    return out


# Quadrature points over the light-yield scale t in _langau_tailmix.  32 is enough:
# the weight t^-index is smooth and the Landau it multiplies is already >= eta wide,
# so the sum converges long before the quadrature error reaches a Poisson error bar.
_TAILMIX_NT = 32


def _langau_tailmix(x, area, mpv1, eta1, sigma1, index, tmax):
    """The MIP line with a REAL response tail: Landau(x)Gauss whose light-yield scale t is
    itself distributed, p(t) ~ t^-index on [1, tmax], instead of being fixed at 1.

    A single Landau(x)Gauss cannot describe run00270_ch9: one eta has to set BOTH the core
    asymmetry and the tail weight, and the data wants a narrow core with a heavy tail.  The
    old fix was to bolt a Gaussian "shoulder" onto the tail and call it a second population
    at ~2x MPV; it is not one (see _select_tail).  Letting the SCALE fluctuate decouples core
    from tail with one shape parameter and describes the same data better with fewer.

    Scaling the light yield by t is exact for the Landau, which is an alpha=1 stable law:
        scale(t) = eta1 * t,  MPV(t) = t*mpv1 + (2/pi)*t*ln(t)*eta1
    (the log term is the alpha=1 stable shift -- it is why the 'shoulder' landed at 2.2x MPV
    rather than 2.0x, which is what made it look like two coincident MIPs).  The resolution
    goes as sqrt(t) (photostatistics).  t -> 1 (tmax -> 1) collapses this EXACTLY back to
    _langau, so the two models are nested and the BIC comparison in _select_tail is honest.

    NOTE ON INTERPRETATION: `index` is a DETECTOR parameter, not a geometric one.  The
    obvious physical candidate -- muon path length, t = sec(theta) -- is REFUTED on this data:
    a flat panel with a cos^2(theta) zenith flux and cos(theta) acceptance predicts index = 5
    with no freedom, and that fits ch9 badly (chi2_red 2.37, tail still +10.5 sigma), while
    the free fit wants index ~ 2, which no zenith distribution can produce (dN/dOmega ~
    cos^k -> index = k+3, so index 2 needs k = -1, flux RISING toward the horizon).  ch0 sees
    the same muons through the same stack and has a far milder tail (its index fit is
    rejected by the BIC).  The tail is per-channel light collection / photostatistics, worst
    on the dim ch9 PMT -- so fit `index`, never fix it at a geometric value."""
    x = np.asarray(x, dtype=float)
    eta1, sigma1 = max(float(eta1), 1e-9), max(float(sigma1), 1e-9)
    tmax = max(float(tmax), 1.0)
    t = np.linspace(1.0, tmax, _TAILMIX_NT)
    w = t ** (-float(index))
    w /= w.sum()
    pad = 8.0 * sigma1 * math.sqrt(tmax) + 12.0 * eta1 * tmax
    step = max(min(sigma1, eta1) / 8.0, (x.max() - x.min()) / 4000.0)
    grid = np.arange(x.min() - pad, x.max() + pad + step, step)
    out = np.zeros_like(grid)
    for ti, wi in zip(t, w):
        scale = eta1 * ti
        mpv_t = ti * mpv1 + (2.0 / math.pi) * ti * math.log(ti) * eta1
        pdf = np.nan_to_num(landau.pdf(grid, loc=mpv_t - _LANDAU_MODE * scale, scale=scale),
                            nan=0.0, posinf=0.0, neginf=0.0)
        out += wi * gaussian_filter1d(pdf, sigma1 * math.sqrt(ti) / step, mode="constant")
    return area * np.interp(x, grid, out, left=0.0, right=0.0)


def spectrum_scale(values) -> float:
    """A robust amplitude scale for the dominant (MIP) population: the median of the
    POSITIVE observable values.  Every length scale in the spectrum fit -- the histogram's
    upper edge, the smoothing width, the valley-walk lookahead -- is expressed as a
    fraction of this, so the fit is invariant to how the histogram happens to be binned.
    It is deliberately model-free (no fit, no landmark) so it can be computed BEFORE any
    of them."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 1.0
    return max(float(np.median(v)), 1e-9)


def _smoothing_bins(scale: float, bin_w: float, n_bins: int, config: Config) -> float:
    """Spectrum smoothing sigma converted from MIP-scale fractions to bins."""
    return float(np.clip(config.smoothing_frac * scale / max(bin_w, 1e-12),
                         1.0, max(2.0, 0.05 * n_bins)))


def build_spectrum(values, config: Config):
    """Histogram the observable with a bin WIDTH pinned to the MIP scale (see Config).

    The bin count is derived from the range, not fixed -- so the MIP peak is resolved to
    the same number of bins on every channel, and the length of the Landau tail no longer
    silently sets the resolution at the gamma/muon boundary.
    """
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    scale = spectrum_scale(a)
    lo, hi = np.percentile(a, [config.spectrum_lo_pct, config.spectrum_hi_pct])
    hi = min(float(hi), float(lo) + config.spectrum_hi_scale * scale)
    lo = float(lo)
    if not hi > lo:
        hi = lo + 1.0
    bin_w = scale / max(config.bins_per_mip, 1)
    n_bins = int(np.clip(round((hi - lo) / max(bin_w, 1e-12)), 50, config.max_spectrum_bins))
    counts, edges = np.histogram(a, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = float(centers[1] - centers[0]) if centers.size > 1 else 1.0
    sm = _smoothing_bins(scale, bin_w, centers.size, config)
    return centers, counts, gaussian_filter1d(counts.astype(float), sm)


def _comb_period(smoothed, muon_bump: int, overlap: int, config: Config) -> int:
    """Period (in bins) of a resolved-photoelectron comb over the region the valley walk
    traverses, or 0 when there is none.

    A low-light SiPM channel (run00270 ch4-ch7, ~5-10 PE per MIP) resolves single
    photoelectrons, so its ENTIRE spectrum -- pedestal edge, "valley", MIP peak -- is
    modulated by a periodic comb at the PE spacing.  That comb defeats the valley walk
    twice over: every inter-PE dip is deep enough to pass the valley_max_frac bar, and
    with the PE spacing wider than the lookahead (ch6: 10 bins vs 4) the walk cannot see
    the lower dip beyond it.  On run00270_ch6's boxcar spectrum the walk stopped in the
    first dip off the MIP peak -- "valley" 0.499 with the MPV at 0.59 -- so the Landau was
    fit on the top half of its own peak, and the low_population guard read the next PE
    peak down (0.44) as a "gamma" population.  The comb is intra-population fine
    structure, not a population boundary; landmarks must be found on its envelope.

    DETECTION.  Autocorrelation of the envelope-subtracted spectrum below the bump.  A
    periodic comb is anti-phased at half its period, so the autocorrelation must first dip
    BELOW ZERO and then rebound to >= comb_rho_min at the period; the residual of a smooth
    spectrum only decays to ~0, and the smoothing correlation among neighbouring bins never
    produces the rebound.  Measured across all 22 run00270 + caen channel/estimator
    spectra: comb channels (ch4, ch5, ch6 both estimators; ch7 boxcar) score rho =
    0.61-0.77 while every clean channel scores <= 0.05 -- the 0.4 bar sits in an empty gap,
    so clean channels are untouched (verified bit-identical landmarks AND fits)."""
    hi = min(len(smoothed), muon_bump + 2 * overlap)
    seg = np.asarray(smoothed[:hi], dtype=float)
    if seg.size < 12:
        return 0
    resid = seg - gaussian_filter1d(seg, 1.5 * max(overlap, 2), mode="nearest")
    resid -= resid.mean()
    denom = float(np.dot(resid, resid))
    if denom <= 0.0:
        return 0
    rho = np.array([np.dot(resid[:-k], resid[k:]) for k in range(1, seg.size // 3 + 1)]) / denom
    neg = np.flatnonzero(rho < 0.0)
    if neg.size == 0:
        return 0
    peak = int(neg[0]) + int(np.argmax(rho[neg[0]:]))
    period = peak + 1                        # rho[i] is lag i+1
    return period if (period >= 3 and rho[peak] >= config.comb_rho_min) else 0


def spectrum_landmarks(centers, smoothed, median_val: float, overlap: int,
                       config: Config) -> dict[str, int]:
    """The ONE set of spectrum landmarks (bin indices): the muon MIP bump, the valley
    below it, and the gamma/noise feature below that.  Used by BOTH fit_spectrum
    (gamma-muon) and fit_muon_spectrum (muon mode's low-population guard), so the two
    modes agree on where the MIP line starts.

    MIP BUMP: the tallest smoothed bin at/above HALF THE COUNT MEDIAN.  In muon-tagged
    data at least half the events are muons, so the MIP bump sits at/above the median; a
    taller-but-narrow noise spike far below it is excluded from the search instead of
    being mistaken for the muon line.  A bare argmax gets this wrong on a low-light
    channel: on the triage-cleaned run00270_ch10, argmax(smoothed) is the residual noise
    spike at 0.13 while the median is 0.84, so muon mode fit its Landau straight ACROSS
    that spike and came out with MPV 0.676 / chi2 3.39 / sigma-over-MPV 97% -- an 18% bias
    against the 0.827 that both the correct fit range and gamma-muon mode agree on.

    VALLEY: walk left from the bump down the smoothed spectrum until EITHER a genuine
    local minimum (a lower value within a small lookahead window keeps the walk going, so
    noise wiggles and short plateaus don't stop it early) OR the level drops below a few %
    of the bump height -- the standard fit-range floor, needed when the smoothing has
    washed a narrow inter-population dip into a monotone slope (ch9's few-bin valley) so no
    local minimum survives.  Anchored at the bump, the walk can never be captured by empty
    histogram-edge bins or by dips INSIDE the gamma structure -- the two failure modes of a
    global argmin below the bump, which is what muon mode used to do.

    The walk must also DESCEND OFF THE PEAK before it is allowed to stop.  A local flat
    spot high on the MIP peak's own rising flank is not a valley, but the plain "no lower
    value within the lookahead -> stop" rule accepted one: on run00270_ch10 at a coarse
    binning the walk halted at 0.80 with the MPV at 0.88, and a "gamma" Gaussian was then
    fit right on top of the MIP peak.  A valley is by definition a place where the density
    has fallen well below the peak, so a local minimum is only accepted once the level is
    under valley_max_frac of the bump height; above that the walk steps on.

    NOTE (2026-07-12): the 5%-of-bump floor being the loop CONDITION -- so the walk aborts
    at a level crossing rather than at a minimum -- looks like the cause of the valley's
    irreproducibility, and it is tempting to make it a stop-at-the-next-local-minimum rule
    instead.  That was TRIED AND MEASURED: it makes ch0 WORSE (cut bootstrap std 10.7% ->
    22.8%), because once below the floor the strict descent slides down ch0's flat gamma
    plateau into the inter-gamma dip at ~0.16 whenever noise makes the plateau monotone.

    PE-COMB GUARD (2026-07-14): on a resolved-photoelectron spectrum the walk above is
    defeated by design -- every inter-PE dip passes the valley_max_frac bar and the PE
    spacing exceeds the lookahead, so the walk stops one PE below the MIP peak (ch6
    boxcar: "valley" 0.499, MPV 0.59).  When _comb_period detects the comb, ALL the
    landmarks are found on a guide smoothed at half the comb period (a Gaussian of
    sigma = period/2 attenuates the comb to < 1% while following its envelope), and the
    valley is then snapped to the raw minimum nearby -- the guide smears the pedestal
    cliff, and the pedestal/first-PE gap is comb-free so the raw minimum is trustworthy
    there.  On a clean spectrum the detection stays silent and nothing changes.

    Keys keep the invariant gamma_peak <= valley <= muon_bump."""
    smoothed = np.asarray(smoothed, dtype=float)
    n = len(smoothed)
    floor_idx = int(np.clip(np.searchsorted(centers, 0.5 * median_val), 0, n - 1))
    muon_bump = floor_idx + int(np.argmax(smoothed[floor_idx:]))

    period = _comb_period(smoothed, muon_bump, overlap, config)
    guide = smoothed
    if period:
        guide = gaussian_filter1d(smoothed, 0.5 * period)
        muon_bump = floor_idx + int(np.argmax(guide[floor_idx:]))

    bump_h = float(guide[muon_bump])
    floor5 = 0.05 * bump_h
    ceil_h = config.valley_max_frac * bump_h
    valley = muon_bump
    while valley > 0 and guide[valley] >= floor5:
        w_lo = max(valley - overlap, 0)
        j = w_lo + int(np.argmin(guide[w_lo:valley]))
        if guide[j] < guide[valley]:
            valley = j
        elif guide[valley] > ceil_h:
            valley = w_lo                  # still on the peak's flank: keep descending
        else:
            break

    if period and valley < muon_bump:
        # The guide smears the pedestal's cliff a few bins leftward; the gap between the
        # pedestal and the first PE peak is itself comb-free, so the RAW minimum within
        # half a period of the guide's valley is the true gap bottom.
        w_lo = max(valley - period // 2, 0)
        w_hi = min(valley + period // 2, muon_bump - 1)
        valley = w_lo + int(np.argmin(smoothed[w_lo:w_hi + 1]))

    # gamma/noise feature at or below the valley -- seeds the low continuum and
    # bounds the crossover search.
    gamma_peak = int(np.argmax(smoothed[: valley + 1])) if valley >= 1 else 0
    return {"gamma_peak": gamma_peak, "muon_bump": muon_bump, "valley": valley}


def _landmark_overlap(scale: float, bin_w: float, n_bins: int, config: Config) -> int:
    """The valley walk's lookahead, in bins, from its MIP-scale fraction (see Config)."""
    return int(np.clip(round(config.valley_lookahead_frac * scale / max(bin_w, 1e-12)),
                       2, max(2, n_bins // 10)))


def _seed_continuum(x, y, max_components: int, mu_hi: float, sig_hi: float,
                    weights=None, smooth_bins: float = 2.0, baseline=None):
    """BIC-selected sum of 1..max_components Gaussians fit to the low-amplitude
    (gamma/noise) region.  Every component's MEAN is bounded to <= `mu_hi` (the
    valley) and its WIDTH to <= `sig_hi`, so each Gaussian stays a genuinely
    low-amplitude feature and cannot bump under the muon MIP peak.

    `baseline` is a FIXED, already-fitted component evaluated on `x` -- the muon Landau.
    The gamma region is extended above the valley into the OVERLAP, where the muon
    contributes real counts; including it as a known additive offset (model = gaussians +
    baseline, Poisson-weighted on the TOTAL) lets the gamma's falling tail be constrained
    by data there instead of being extrapolated or truncated.  This is NOT the "subtract
    the muon and fit the residual" design that failed before: the muon is a fixed known
    term inside a single deterministic fit, not a free component and not an EM iteration.
    None -> no baseline (the old behaviour).

    Component locations are SEEDED at the most prominent local maxima of the
    (lightly re-smoothed) target, so distinct low-side populations -- e.g. a sharp
    noise spike at the threshold plus a broad gamma shoulder -- each get their own
    Gaussian and the BIC over k genuinely decides whether the region holds one
    population or several (linspace fill-in when fewer maxima exist than k).

    `weights` are the per-bin count errors (Poisson sqrt(counts) of the ORIGINAL
    histogram): with them, a modest muon-subtraction residual riding on the
    high-count rising edge is down-weighted by its own large error bar instead of
    out-voting a genuinely significant few-count gamma spike (the statistically
    correct weighting, and the same one the muon fit uses).  The BIC is then on the
    weighted chi2.  Returns (bic, params, k) or None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    base = np.zeros_like(y) if baseline is None else np.asarray(baseline, dtype=float)
    bin_w = float(x[1] - x[0]) if len(x) > 1 else 1.0
    mu_hi = max(mu_hi, float(x[0]) + bin_w)
    span = max(mu_hi - float(x[0]), bin_w)
    # Upper bound on each Gaussian's AMPLITUDE.  A gamma component several times taller than
    # the tallest bin in the region is not a solution, and leaving this open at +inf gives
    # the fit an unbounded direction to wander along (see the x_scale note below).
    amp_hi = max(5.0 * float(np.max(y)) if y.size else 1.0, 1.0)

    # The gamma model is (gaussians + the FIXED muon).  curve_fit always calls the model
    # with the same xdata we hand it, so closing over `base` (aligned to x) is safe.
    def _model(_x, *p):
        return _sum_of_gaussians(_x, *p) + base

    # Candidate component locations: local maxima of the smoothed target (edges
    # included -- a noise spike often sits in the very first bins), tallest first.
    # Seed on the MUON-FREE part of the target (y - baseline): in the overlap bins the
    # muon's own rising edge is the tallest thing there, and seeding a "gamma" component
    # on it is exactly how a Gaussian ends up under the MIP peak.
    # `smooth_bins` comes from the caller's MIP-scale-relative smoothing, so this
    # pre-smoothing is the same PHYSICAL width regardless of the binning.
    ys = gaussian_filter1d(np.maximum(y - base, 0.0), max(float(smooth_bins), 0.5))
    idx = np.arange(len(ys))
    is_max = np.ones(len(ys), dtype=bool)
    is_max[1:] &= ys[1:] >= ys[:-1]
    is_max[:-1] &= ys[:-1] > ys[1:]
    peaks = idx[is_max & (x <= mu_hi) & (ys > 0)]
    peaks = peaks[np.argsort(ys[peaks])[::-1]] if peaks.size else np.array([], dtype=int)
    # A k-Gaussian model costs 3k parameters, so a k that leaves no degrees of freedom is
    # not a model of the data, it is an interpolation of it -- and in practice it does not
    # even converge: on the 14-bin gamma region of ch4/ch5/ch6, k=3 (9 params, dof=5) ran to
    # maxfev and RAISED, so the fit was thrown away and the k=1 fallback used anyway, after
    # ~25 s.  Cap k by what the region can actually constrain; the outcome is unchanged.
    k_max = min(max_components, max((len(x) - 2) // 3, 1))
    best = None
    for k in range(1, k_max + 1):
        seeds = [float(x[j]) for j in peaks[:k]]
        if len(seeds) < k:                      # fill in with evenly-spaced extras
            seeds += list(np.linspace(x[0], mu_hi, (k - len(seeds)) + 2)[1:-1])
        p0, lo, hi = [], [], []
        for mu0 in seeds:
            a0 = float(ys[int(np.clip(np.searchsorted(x, mu0), 0, len(x) - 1))])
            p0 += [max(a0, 1.0), float(mu0), float(np.clip(span / (2 * k), bin_w, sig_hi))]
            lo += [0.0, float(x[0]), 0.5 * bin_w]
            hi += [amp_hi, mu_hi, sig_hi]
        try:
            # x_scale="jac": the three parameters of a Gaussian live on completely different
            # scales (an amplitude in counts, a mean in observable units, a width in a
            # fraction of one), and trf stepping all of them with one yardstick is what made
            # these fits crawl -- a 3-parameter, 110-bin fit on ch6 needed 8,055 function
            # evaluations to converge, and the k=2/3 fits on ch4/ch5/ch6 never converged at
            # all: they ran to maxfev, RAISED, and were silently dropped by the `continue`
            # below, leaving a truncated 0- or 1-Gaussian gamma (and, on ch5-OF / ch10-boxcar,
            # NO CUT AT ALL -- reported as a physics result rather than the fit failure it
            # was).  Rescaling by the jacobian, plus the finite `amp_hi` above (an open
            # amplitude bound is the direction trf wandered off along), converges every fit
            # on every channel in ~10-50 evaluations.  Cuts are unchanged to <=1e-5 wherever
            # the old fit CONVERGED; they only move where it did not.
            params, _ = curve_fit(_model, x, y, p0=p0, bounds=(lo, hi),
                                  sigma=w, x_scale="jac", maxfev=20_000)
        except (RuntimeError, ValueError):
            continue
        chi2_w = float(np.sum(((y - _model(x, *params)) / w) ** 2))
        # BIC penalty uses the number of OBSERVATIONS -- the events in the region -- not
        # the number of BINS.  n_bins is a property of how we chose to histogram, not of
        # how much data we have, so a log(n_bins) penalty made the selected component
        # count depend on the binning: on run00270_ch0 it flipped k between 2 and 3 as the
        # bin width changed, and the two solutions put the cut 20% apart.
        n_obs = max(float(np.sum(y)), 1.0)
        bic = chi2_w + 3 * k * math.log(n_obs)
        if best is None or bic < best[0]:
            best = (bic, list(params), k)
    return best


def _gamma_fit_top(centers, smoothed, mfun, valley: int, mpv_bin: int, config: Config) -> int:
    """Top bin of the GAMMA fit range: the valley, extended UP through the overlap region
    for as long as the muon line does not dominate the histogram.

    The gamma used to be fit only on [start, valley] and then either extrapolated above it
    (unconstrained -- a wide Gaussian could reach under the MIP peak) or truncated at it
    (degenerate -- the equal-area balance then has no gamma area to weigh, so the cut
    collapses onto the valley).  Both are wrong for the same reason: the cut criterion needs
    the gamma model to be valid exactly where the two populations OVERLAP, and the valley is
    not where gamma ends.  On run00270_ch0 the bins from the valley (0.36) up to ~0.45 hold
    121 events against a muon prediction of 2-12 -- over a hundred REAL gamma events the
    model was never allowed to see.

    So the range is extended while the fitted muon stays below `gamma_overlap_frac` of the
    smoothed counts -- i.e. while the bin is still gamma-dominated and the gamma's falling
    tail is therefore MEASURABLE there.  Above that the muon owns the bin and the gamma is
    unconstrained by it, so the range stops.  Capped at the MIP peak, and never shrinks
    below the valley.
    """
    hi = int(np.clip(mpv_bin, valley + 1, len(centers) - 1))
    mu = np.maximum(mfun(centers), 0.0)
    sm = np.maximum(smoothed, 1.0)             # a floor of 1 count: empty bins constrain nothing
    top = int(valley)
    for i in range(int(valley) + 1, hi):
        if mu[i] >= config.gamma_overlap_frac * sm[i]:
            break
        top = i
    return top


def _equal_area_cut(centers, gfun, mfun, lo_idx, hi_idx, gamma_hi: float | None = None):
    """Balanced-misclassification ("equal-area") cut: the amplitude where the gamma
    continuum's area ABOVE the cut (background leaking into the muon sample) equals
    the muon line's area BELOW it (muons lost to the gamma sample).  At that point
    the two misclassification counts cancel, so the muon yield above the cut is
    UNBIASED -- the natural boundary for a counting/spectroscopy measurement, and
    the criterion Remy asked for ("areas left and right of the overlap equal").

    Concretely x* solves  integral_{x*}^{gamma_hi} gamma(x) dx == integral_{bottom}^{x*} muon(x) dx.

    `gamma_hi` is the TOP OF THE GAMMA FIT RANGE (the valley).  The gamma mixture is only
    ever fit on [start, valley]; above the valley it has no data behind it, and a Gaussian
    tail there is an assumption, not a measurement.  Integrating that extrapolation as if
    it were real gamma is what made the cut irreproducible: the balance point was set by
    however far the fitted Gaussians happened to reach into the muon-dominated region, so
    a BIC flip between 2 and 3 components -- driven by nothing but the binning -- swung
    run00270_ch0's cut between 0.36 and 0.55.  Restricting the integral to the gamma
    model's own support removes that term entirely: gamma contributes only where it was
    measured.  (Pass None to integrate to the top of the spectrum, the old behaviour.)

    A consequence worth stating: the balance point now necessarily lands AT OR BELOW the
    valley, because above it the gamma area is zero while the muon area below is not.  The
    cut therefore sits in the region where the two populations actually overlap, which is
    where a discrimination boundary belongs.  It also means the cut no longer counts the
    gamma continuum's true (small) leakage above the valley -- an acknowledged, bounded
    conservatism, traded for a number that does not move when you re-bin the histogram.

    Gamma-area-above is non-increasing and muon-area-below non-decreasing as the cut
    moves right, so their difference is monotone and the balance point is unique.

    The two areas are integrated on a FINE grid (not the histogram bins): a bin-center
    cumulative sum balances the areas only to bin resolution and, because the discrete
    balance falls on a bin EDGE while the interpolation reads bin CENTERS, leaves a
    systematic ~half-bin bias that is large wherever the density at the cut is high (it
    made the muon-left area run ~30% above the gamma-right area on ch9/ch10).  Searched
    within [centers[lo_idx], centers[hi_idx]] (gamma peak .. muon MPV), and never below
    zero -- a NEGATIVE amplitude cut is physically meaningless, and with spectrum_lo_pct=0
    the histogram does start below zero (both estimators return small negative amplitudes
    for noise triggers), so without this floor the search happily balanced its areas out in
    the empty negative tail and returned cuts like -0.43 on run00270_ch6's boxcar.
    Returns the interpolated crossing, or None if there is none in range."""
    lo_x = max(float(centers[int(np.clip(lo_idx, 0, len(centers) - 1))]), 0.0)
    hi_x = float(centers[int(np.clip(hi_idx, 0, len(centers) - 1))])
    if not hi_x > lo_x:
        return None
    xg = np.linspace(float(centers[0]), float(centers[-1]), 8000)
    dx = float(xg[1] - xg[0])
    g = np.maximum(gfun(xg), 0.0)
    if gamma_hi is not None:
        g = np.where(xg <= float(gamma_hi), g, 0.0)   # gamma has no support past its fit range
    m = np.maximum(mfun(xg), 0.0)
    gamma_above = np.cumsum(g[::-1])[::-1] * dx       # integral_x^{gamma_hi} gamma
    muon_below = (np.cumsum(m) - m) * dx              # integral_{bottom}^x muon
    diff = gamma_above - muon_below                   # monotone non-increasing in x
    seg = np.flatnonzero((xg[:-1] >= lo_x) & (xg[:-1] <= hi_x))
    if seg.size == 0:
        return None
    change = seg[(diff[seg] > 0) & (diff[seg + 1] <= 0)]
    if not change.size:
        return None
    i = int(change[0])
    d0, d1, x0, x1 = diff[i], diff[i + 1], xg[i], xg[i + 1]
    return float(x0 - d0 * (x1 - x0) / (d1 - d0)) if d1 != d0 else float(x0)


def _seed_langau(x, y, mpv_seed, bin_width):
    amp_seed = float(np.sum(y) * bin_width)
    best = None
    for eta_frac, sig_frac in ((0.08, 0.04), (0.15, 0.08), (0.25, 0.12)):
        eta0 = max(eta_frac * mpv_seed, 2 * bin_width)
        p0 = [amp_seed, float(mpv_seed) - _LANDAU_MODE * eta0, eta0, max(sig_frac * mpv_seed, 1.5 * bin_width)]
        lo = [0.0, x[0], bin_width, bin_width]
        hi = [np.inf, x[-1] * 1.5, x[-1] - x[0], x[-1] - x[0]]
        try:
            params, _ = curve_fit(_langau, x, y, p0=p0, bounds=(lo, hi), maxfev=20_000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum((y - _langau(x, *params)) ** 2))
        if best is None or rss < best[0]:
            best = (rss, list(params))
    return None if best is None else best[1]


def _reduced_chi2(y_obs, y_model, n_params):
    """Reduced chi2 of a binned fit, using the DATA (Neyman) variance max(y_obs, 1).

    This is deliberately the same variance the fits are MINIMISED under (curve_fit is
    given sigma=sqrt(max(y, 1)) everywhere in this module).  It previously used the MODEL
    (Pearson) variance instead, so every reported chi2_red was not the quantity any fit
    had actually optimised."""
    dof = max(len(y_obs) - n_params, 1)
    y_obs = np.asarray(y_obs, dtype=float)
    return float(np.sum((y_obs - y_model) ** 2 / np.maximum(y_obs, 1.0)) / dof)


def _crossover(centers, model_gamma, model_muon, lo, hi):
    """First value in [lo, hi] where the muon model overtakes the gamma model.
    Never returns a non-positive amplitude (see _equal_area_cut)."""
    x = centers[lo:hi]
    x = x[x >= 0.0]
    if x.size < 2:
        return None
    diff = model_muon(x) - model_gamma(x)
    cross = np.flatnonzero((diff[:-1] <= 0) & (diff[1:] > 0))
    if not cross.size:
        return None
    i = cross[0]
    x0, x1, d0, d1 = x[i], x[i + 1], diff[i], diff[i + 1]
    return float(x0 - d0 * (x1 - x0) / (d1 - d0)) if d1 != d0 else float(x0)


def _gamma_evidence(centers, y, smoothed, gfun, mfun, g_hi: int, k: int,
                    muon_bump: int, valley: int, config: Config) -> dict:
    """How much gamma population is the reported cut actually discriminating?  DIAGNOSTICS --
    these are reported, NOT gated on.  Read the next paragraph before trying to gate on them.

    THE SPECTRUM CANNOT TELL YOU WHETHER A CUT IS MEANINGFUL -- MEASURED 2026-07-13, do not
    re-litigate.  The plan was to auto-select the analysis mode: report a cut only where the
    evidence supports a gamma population, and fall back to the muon-only fit otherwise, so
    --mode need not be specified.  It does not work, because the statistics below do not
    separate the datasets that need a cut from the one that must not have one:

      dataset                          cut?   gamma_frac  valley_contrast  gamma_dbic
      run00270_ch0  (raw)              yes        0.056        0.95           -40450
      run00270_ch9  (raw)              yes        0.0017       0.993           -251
      run00270_ch10 (raw)              yes        0.26         0.56          -145111
      caen_ch0 (hodoscope-tagged)      NO         0.029        0.995            -31

    The negative control lands INSIDE the positive range on every one of them: hodoscope-
    selected muon data carries a bigger fitted gamma fraction (2.9%) than raw ch9 (0.17%),
    which genuinely needs a cut, and its valley contrast (0.995) is indistinguishable from
    ch9's (0.993).  gamma_dbic is a chi2 difference, so it mostly tracks the EVENT COUNT (caen
    has 712 events, ch9 14,875); per-event it inverts, ranking the must-cut channel weakest.
    Left to the spectrum, gamma-muon mode reports a cut at 0.42 on caen_ch0's hodoscope muons
    (MPV 0.61) with no hint that anything is wrong.

    Whether a discrimination cut is meaningful is a fact about the INPUT SELECTION (was the
    gamma population already removed upstream?) -- provenance, which is not in the spectrum.
    So --mode stays an explicit statement of what the data IS.  What these numbers ARE good
    for is telling you how much population the cut rests on: see the gamma_frac warning below.

      gamma_frac      -- fitted gamma area / all events.
      valley_contrast -- (bump - valley) / bump on the smoothed spectrum.
      gamma_dbic      -- does gamma+muon beat the muon alone on the gamma's own bins?
    """
    total = max(float(np.sum(y)), 1.0)
    gamma_frac = float(np.sum(gfun(centers))) / total
    bump, floor = float(smoothed[muon_bump]), float(smoothed[valley])
    contrast = (bump - floor) / bump if bump > 0 else 0.0
    yl, gl, ml = y[:g_hi], gfun(centers[:g_hi]), mfun(centers[:g_hi])
    chi2_with = float(np.sum((yl - (gl + ml)) ** 2 / np.maximum(gl + ml, 1.0)))
    chi2_without = float(np.sum((yl - ml) ** 2 / np.maximum(ml, 1.0)))
    dbic = (chi2_with + 3 * k * math.log(max(g_hi, 2))) - chi2_without
    return {"gamma_frac": gamma_frac, "valley_contrast": float(contrast),
            "gamma_dbic": float(dbic)}


def fit_spectrum(values, config: Config, line_model: dict | None = None):
    """Separate a low-amplitude gamma/noise continuum (BIC-selected sum of
    Gaussians) from the muon Landau(x)Gaussian MIP line and place a discrimination
    cut between them.  In muon-tagged data the MIP line is the DOMINANT peak and the
    gamma/noise is the smaller low-amplitude population.

    TWO-REGION fit, each population fit ONLY on its own side of the valley that
    separates them, on the RAW counts, in one deterministic pass each (exactly the
    CUPID muon-veto gamma/mu procedure: independent left/right fits + a threshold
    between them):

      1. The MIP bump is the tallest smoothed bin at/above half the count median
         (so a taller-but-narrow noise spike at very low amplitude cannot be
         mistaken for it -- in muon-tagged data at least half the events are muons).
      2. The VALLEY is found by walking left from the bump down the smoothed
         spectrum to the first genuine local minimum (lower values within a small
         lookahead window keep the walk going, so noise wiggles don't stop it),
         also stopping once the level falls below a few % of the bump height (the
         standard fit-range floor -- needed when smoothing has washed a narrow dip
         into a monotone slope).  Anchoring the walk at the bump makes it immune
         to the failure modes of a global argmin below the bump: empty
         histogram-edge bins and dips WITHIN the gamma structure can never
         capture it.  If the walk reaches the spectrum edge (no separable low
         population), the muon line alone is fit and no cut is derived.
      3. Muon: Landau(x)Gauss fit to the bins at/above the valley -- turn-on, peak
         and tail all constrained by data on its own side.
      4. Gamma: BIC-selected 1..gamma_max_gaussians Gaussians fit to the bins
         at/below the valley, Poisson-weighted.  Component means are FREE within
         the region and seeded at its local maxima, so a sharp noise spike + a
         broad gamma shoulder are recognized as separate populations when the BIC
         supports them; means/widths are bounded by the valley so no component can
         bump under the MIP peak (it may only decay toward it).

    Cut criterion (`_equal_area_cut`): the BALANCED-MISCLASSIFICATION point where the
    gamma area leaking above the cut equals the muon area lost below it, so the two
    spill-overs cancel and the muon yield above the cut is unbiased -- the standard
    choice for a counting measurement.  Falls back to the density crossover, then the
    smoothed valley, if no balance point exists in range.  Returns fit curves, the
    muon MPV, the muon resolution width, and the cut."""
    centers, counts, smoothed = build_spectrum(values, config)
    y = counts.astype(float)
    bin_w = centers[1] - centers[0]
    finite_vals = np.asarray(values, dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    median_val = float(np.median(finite_vals)) if finite_vals.size else 0.0
    # Landmark scales in MIP-scale fractions, converted to bins HERE (see Config).  The
    # valley-walk lookahead used to be a flat 3% of n_bins, which on a long-tailed
    # low-light channel was wider than the whole gamma/valley/MIP structure.
    scale = spectrum_scale(finite_vals)
    smooth_bins = _smoothing_bins(scale, bin_w, len(centers), config)
    overlap = _landmark_overlap(scale, bin_w, len(centers), config)

    # --- Landmarks (shared with muon mode -- see spectrum_landmarks) -----------------
    # ch0 has no valley to find (see _cut_degenerate below); no tweak to the walk fixes that.
    marks = spectrum_landmarks(centers, smoothed, median_val, overlap, config)
    muon_bump, valley = marks["muon_bump"], marks["valley"]
    valley_x = float(centers[valley])

    out = {"centers": centers, "counts": counts, "smoothed": smoothed,
           "gamma_params": None, "muon_params": None, "tail_params": None, "n_gamma": 0,
           "mpv": float(centers[muon_bump]), "eta": float(bin_w), "sigma": float(bin_w),
           "cut": valley_x, "ok": False, "cut_kind": "valley",
           "valley_cut": valley_x, "crossover_cut": None,
           "equal_area_cut": None, "fit_chi2": np.nan, "reliable": False}

    # --- Independent single-pass fits, each strictly on its own side ---------------
    # The MIP line is fit by fit_muon_line -- the SAME line muon mode fits (Landau(x)Gauss,
    # BIC-upgraded to the response-tail model where the channel needs it).  Only the fit
    # RANGE is this mode's own: the line starts at the valley, and everything below it is
    # the gamma's to model.
    if valley <= 1 or len(centers) - valley < 10:
        # The walk found no separable low population (e.g. an essentially pure muon
        # sample whose handful of low-amplitude events can't anchor a gamma fit):
        # still fit and report the MIP line; no gamma, no cut.
        #
        # With no gamma component, the far sub-MIP pedestal is UNMODELLED, and the few counts
        # it holds would blow up the Poisson chi2 of a Landau that predicts ~0 there.  So
        # floor the fit exactly as muon mode does (Config.muon_fit_lo_pct): this branch IS
        # the muon-only answer, and it should be the same answer --mode muon gives.
        logger.info("No separable low-amplitude population below the MIP line "
                    "(the valley walk reached the spectrum edge): fitting the MIP line "
                    "alone and reporting NO gamma/muon cut.")
        m_lo = max(valley, 0, _muon_fit_floor(centers, finite_vals, config))
        line = fit_muon_line(centers, y, smoothed, m_lo, bin_w, line_model)
        if line is not None:
            # reliable is decided by analyze_observable's MIP-vs-noise SNR test, which uses
            # a scale independent of this spectrum; see Config.min_mip_snr.
            out.update({k: line[k] for k in _LINE_KEYS},
                       cut=np.nan, ok=False, cut_kind="none", reliable=True,
                       fit_lo_x=float(centers[m_lo]),
                       n_low_excluded=int(np.count_nonzero(finite_vals < centers[m_lo])))
        return out

    # Muon: the MIP line on [valley, end].  Fit FIRST -- it both defines how far the gamma
    # can honestly be fit (see _gamma_fit_top) and enters that fit as a fixed component.
    line = fit_muon_line(centers, y, smoothed, valley, bin_w, line_model)
    if line is None:
        return out
    mfun, mpv = line["fun"], line["mpv"]
    mpv_bin = int(np.clip(np.searchsorted(centers, mpv), 1, len(centers) - 1))

    # Gamma: BIC-selected Gaussian mixture on [start, gamma_top] -- raw counts,
    # Poisson-weighted, means still bounded by the valley (the continuum is a low, falling
    # pedestal and must never grow a bump under the MIP peak), but the fit RANGE now runs up
    # through the OVERLAP with the muon entering as a FIXED component (_gamma_fit_top).
    #
    # This is the fix for the criterion, not just the fit.  Confining the gamma to
    # [start, valley] left _equal_area_cut integrating it over bins it had never seen, and
    # both ways out were wrong: extrapolate it (a wide Gaussian reaches under the MIP peak,
    # and the BIC flips between solutions 20% apart on binning alone) or truncate it at the
    # valley (the gamma area above the cut then goes to zero AS the cut approaches the
    # valley, so the balance is satisfied trivially there and the cut collapses onto the
    # valley -- run00270_ch0 had cut == valley in 20/20 bootstrap replicates).  Fitting the
    # gamma over the range the balance actually integrates it on removes the choice.
    #
    # The width cap is the region's span / gamma_sigma_containment (default 3): every
    # component must decay inside the bins it was fit on to within ~1 sigma.  The span is
    # now the LONGER (overlap-inclusive) range, so a broader pedestal is allowed -- but it
    # is allowed because the data there now constrains it, not assumed.
    g_top = _gamma_fit_top(centers, smoothed, mfun, valley, mpv_bin, config)
    g_hi = g_top + 1
    gamma_top_x = float(centers[g_top])
    g_span = max(gamma_top_x - float(centers[0]), 3.0 * bin_w)
    gsig_hi = max(g_span / config.gamma_sigma_containment, 2.0 * bin_w)
    gseed = _seed_continuum(centers[:g_hi], y[:g_hi], config.gamma_max_gaussians,
                            valley_x, gsig_hi,
                            weights=np.sqrt(np.maximum(y[:g_hi], 1.0)),
                            smooth_bins=smooth_bins,
                            baseline=mfun(centers[:g_hi]))
    if gseed is None:
        # The GAMMA mixture failed to converge -- but the MUON fit above did not, and it is
        # the more important of the two.  Keep it, and report the MIP line with no gamma and
        # no cut (exactly what the `valley <= 1` branch does when there is no low population
        # to fit).  Bailing out with a bare `return out` here threw the good muon fit away:
        # muon_params came back None, so fit_chi2 was NaN, the MPV fell back to a raw
        # histogram bin, and plot_spectrum_pull silently produced no figure at all.  That is
        # what run00270 ch4/ch5/ch6 were doing -- reporting an MPV that was never fitted.
        logger.warning("The gamma continuum fit did not converge on this spectrum (its fit "
                       "region is only %d bins wide). Reporting the MIP line alone -- no "
                       "gamma component and NO gamma/muon cut.", g_hi)
        out.update({k: line[k] for k in _LINE_KEYS},
                   cut=np.nan, ok=False, cut_kind="none", reliable=True)
        return out
    _, gpar, k = gseed
    gp_idx = int(np.argmax(_sum_of_gaussians(centers[:g_hi], *gpar)))

    def gfun(x):
        return _sum_of_gaussians(x, *gpar)

    out.update(_gamma_evidence(centers, y, smoothed, gfun, mfun, g_hi, k,
                               muon_bump, valley, config))

    # Final cut: the balanced-misclassification (equal-area) point between the two
    # fitted components; density crossover, then the smoothed valley, as fallbacks.
    # A cut is only ACCEPTED if it is strictly positive and below the MIP peak.  The old
    # test (centers[0] < cut < centers[-1]) was vacuous -- centers[0] is the observable's
    # minimum, which is negative -- and it let through the physically impossible negative
    # cuts seen on run00270 ch4/ch5/ch6's boxcar.
    def _valid(c):
        return c is not None and np.isfinite(c) and 0.0 < c < mpv

    # gamma_hi=gamma_top_x: the gamma area is integrated over exactly the bins the gamma
    # model was FIT on -- which now includes the overlap, so the balance weighs the gamma
    # that genuinely leaks past the cut instead of a truncation (zero) or an extrapolation
    # (unconstrained).  Above gamma_top the muon owns the histogram and the gamma is not
    # measurable, so it still contributes nothing there.
    equal_area = _equal_area_cut(centers, gfun, mfun, gp_idx, mpv_bin, gamma_hi=gamma_top_x)
    crossover = _crossover(centers, gfun, mfun, gp_idx, mpv_bin)
    if _valid(equal_area):
        cut, ok, cut_kind = equal_area, True, "equal-area"
    elif _valid(crossover):
        cut, ok, cut_kind = crossover, True, "crossover"
    elif _valid(valley_x):
        bump, floor = float(smoothed[muon_bump]), float(smoothed[valley])
        valley_clear = muon_bump > valley + 1 and bump > 0 and (bump - floor) / bump > 0.05
        cut, ok, cut_kind = valley_x, valley_clear, "valley"
    else:
        logger.warning("No usable gamma/muon cut: every candidate (equal-area %s, crossover "
                       "%s, valley %.4g) fell outside (0, MPV=%.4g). The two populations are "
                       "not separable in this spectrum -- reporting the MIP fit with NO cut.",
                       "None" if equal_area is None else f"{equal_area:.4g}",
                       "None" if crossover is None else f"{crossover:.4g}", valley_x, mpv)
        cut, ok, cut_kind = np.nan, False, "none"

    # THIN-GAMMA WARNING (not a gate -- see _gamma_evidence for why this cannot be one).  A cut
    # is a boundary between two POPULATIONS; when the fitted gamma holds almost no events there
    # is no second population to be on the other side of it, and the "cut" is just a low
    # landmark on the muon line's turn-on.  run00270_ch9 fits a gamma of 0.17% of events (~25)
    # and still reports a cut at 0.098 -- worth saying out loud, since nothing else in the
    # output reveals how much population the cut actually rests on.
    if ok and out["gamma_frac"] < config.gamma_frac_warn:
        logger.warning("The fitted gamma population is only %.2f%% of events (%.0f of %d): the "
                       "cut at %.4g is separating the MIP line from almost NOTHING. Treat it as "
                       "a threshold landmark, not a gamma/muon discrimination boundary -- and "
                       "check whether this data even has a gamma population to cut on.",
                       100 * out["gamma_frac"], out["gamma_frac"] * float(np.sum(y)),
                       int(np.sum(y)), cut)

    # DEGENERACY GUARD.  The gamma has no support above gamma_top (it is not measurable
    # there), so its area-above falls to zero AS the cut approaches gamma_top.  If the muon
    # also has negligible area below gamma_top, the balance is satisfied trivially AT the
    # boundary -- a root set by the truncation rather than by the two populations.  That is
    # what the old gamma_hi=valley did on every channel with a clean low tail (run00270_ch0:
    # cut == valley in 20/20 bootstrap replicates).  Extending the fit through the overlap
    # should stop it happening; the guard stays so we find out if it ever does again.
    cut_degenerate = bool(cut_kind == "equal-area" and np.isfinite(cut)
                          and abs(cut - gamma_top_x) <= 1.5 * bin_w)
    if cut_degenerate:
        logger.warning(
            "Gamma/muon cut is DEGENERATE: the equal-area root (%.4g) sits on the top of the "
            "gamma's fit range (%.4g), so the balance is satisfied by the gamma running out "
            "of support there rather than by the two populations. The cut is a boundary "
            "artifact -- treat it as a landmark, not a measured boundary.", cut, gamma_top_x)

    # Per-component goodness-of-fit, each on the bins IT was fit on (CUPID-style: independent
    # fits get independent chi2; a total-curve chi2 is not meaningful here).  The gamma's own
    # region now includes the overlap and was fit with the muon as a FIXED component, so the
    # model scored there is gamma + muon -- scoring the gamma alone against counts that
    # genuinely contain muons would report a misfit that is not one.
    gamma_chi2 = _reduced_chi2(y[:g_hi], gfun(centers[:g_hi]) + mfun(centers[:g_hi]), 3 * k)

    # THE GAMMA MUST FIT ITS OWN BINS.  The cut is placed by balancing the AREA under the two
    # fitted models, so a gamma that does not describe the region it was fit on cannot place a
    # trustworthy boundary -- the balance is then weighing a curve against data it misses.
    # This fires on run00270_ch6, where the low region is a broad, spiky threshold turn-on
    # rather than a Gaussian population: the mixture scores chi2_red 15-20 and still hands
    # back a cut (0.177 on the OF, 0.499 on the boxcar -- a 2.8x disagreement, which is the
    # misfit showing through).  Warned, not gated: gating on spectrum shape is refuted (see
    # _gamma_evidence), and a genuinely THIN gamma scores a meaningless chi2 on a few bins.
    # Only when the gamma is not ALREADY flagged thin: a gamma of ~15 events over 7 bins
    # (run00270_ch9's boxcar) scores a large chi2 on Poisson noise alone, and the thin-gamma
    # warning above is the accurate description of that channel.  Firing both just buries it.
    thin_gamma = out.get("gamma_frac", np.nan) < config.gamma_frac_warn
    if (ok and not thin_gamma and np.isfinite(gamma_chi2)
            and gamma_chi2 > config.gamma_chi2_warn):
        logger.warning(
            "The gamma continuum FITS BADLY (chi2_red = %.1f over its %d bins; the bar is "
            "%.1f). The cut at %.4g is an area balance between the muon line and a gamma "
            "model that does not describe the data it was fit to -- so the cut is not "
            "measured, it is an artefact of the misfit. Look at the spectrum: this usually "
            "means the low-amplitude region is a threshold turn-on or a broad spiky "
            "continuum, not the Gaussian population the mixture assumes.",
            gamma_chi2, g_hi, config.gamma_chi2_warn, cut)

    out.update({k_: line[k_] for k_ in _LINE_KEYS},          # muon chi2 is on [valley, end]
               gamma_params=gpar, n_gamma=k, cut=cut, ok=ok,
               cut_kind=cut_kind, valley_cut=valley_x, crossover_cut=crossover,
               equal_area_cut=equal_area, reliable=True, cut_degenerate=cut_degenerate,
               gamma_top=gamma_top_x, gamma_chi2=gamma_chi2)
    return out


def bootstrap_cut(values, config: Config, fit: dict | None = None,
                  n_resamples: int | None = None) -> dict:
    """Statistical uncertainty of the gamma/muon cut, by resampling the EVENTS.

    The cut is a fragile functional of the spectrum -- it comes out of a valley walk, a
    BIC-selected mixture and an area balance, none of which carries an analytic error --
    so the pipeline reported it as a bare number and every downstream comparison silently
    assumed it was exact.  It is not: refitting `n_resamples` bootstrap replicates of the
    same events (resampled with replacement, the standard non-parametric estimate of a
    statistic's sampling distribution) gives std/MPV ~4% on run00270_ch10, ~11% on ch0 and
    ~14% on ch9.  A 10% "shift" in the cut between two pipeline versions is therefore
    usually NOISE, and this has already been mistaken for a result twice.

    The MPV is bootstrapped alongside it -- it is rock stable (<1%) on every channel, which
    is the point: the instability lives in the cut chain, not in the Landau fit, and
    reporting both makes that visible instead of leaving it to be rediscovered.

    Pass the FULL-DATA fit as `fit` and the replicates reuse the line model it selected
    (line_model_of) instead of re-running the BIC selection on every replicate.  That is both
    faster and more correct:
      * CORRECTNESS -- an error bar should describe the sampling spread of the statistic we
        REPORT, which is the cut of the model we chose.  Letting replicates re-select the
        model mixes in the discrete jitter of the SELECTION (a replicate that flips between
        the Landau and the tail line moves the MPV by ~8-15% in one jump), inflating the bar
        with something that is not sampling error of the reported quantity.
      * COST -- _select_tail is a 3-start fit of a quadrature-integrated model.  Re-selecting
        per replicate cost ~16-25 s EACH (5-8 min per channel at the default 20); reusing the
        choice replaces it with one short _refit_tail seeded at the full-data solution.

    Returns {} when disabled (config.cut_bootstrap <= 0) or when too few replicates
    converge to say anything.
    """
    n = int(config.cut_bootstrap if n_resamples is None else n_resamples)
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if n <= 1 or a.size < 50:
        return {}
    line_model = line_model_of(fit) if fit is not None else None
    rng = np.random.default_rng(config.seed)
    cuts, mpvs = [], []
    # The replicate fits must not LOG.  fit_spectrum warns about degenerate cuts and
    # unseparable spectra, and n copies of that would bury the one warning that matters --
    # the one from the real fit.
    was_disabled = logger.disabled
    logger.disabled = True
    try:
        for _ in range(n):
            rep = a[rng.integers(0, a.size, a.size)]
            try:
                f = fit_spectrum(rep, config, line_model)
            except Exception:                  # a pathological replicate must not kill the run
                continue
            c, m = f.get("cut", np.nan), f.get("mpv", np.nan)
            if f.get("ok") and np.isfinite(c):
                cuts.append(float(c))
            if np.isfinite(m):
                mpvs.append(float(m))
    finally:
        logger.disabled = was_disabled
    if len(cuts) < max(5, n // 4):
        logger.info("Cut bootstrap: only %d/%d replicates yielded a cut; not reporting an "
                    "uncertainty.", len(cuts), n)
        return {}
    cuts = np.asarray(cuts)
    mpvs = np.asarray(mpvs) if mpvs else np.array([np.nan])
    return {"cut_std": float(cuts.std(ddof=1)),
            "cut_lo": float(np.percentile(cuts, 16)),
            "cut_hi": float(np.percentile(cuts, 84)),
            "mpv_std": float(np.nanstd(mpvs, ddof=1)) if mpvs.size > 1 else np.nan,
            "n_resamples": int(cuts.size)}


# ===========================================================================
# The MIP line (ONE implementation, shared by both modes) + the muon-only spectrum
# ===========================================================================

def _tail_bounds(mpv0: float, bin_w: float):
    """Parameter bounds for _langau_tailmix (area, mpv1, eta1, sigma1, index, tmax)."""
    return ([0.0, 0.2 * mpv0, bin_w, bin_w, 1.0, 1.0],
            [np.inf, 2.0 * mpv0, mpv0, mpv0, 12.0, 8.0])


def _refit_tail(x, y, bin_w, mpv_seed, seed):
    """Refit _langau_tailmix from a KNOWN-GOOD start, with NO multi-start and NO BIC gate.

    Here the model CHOICE is an input, not a decision: the caller has already established (on
    the full data) that this channel's line has a tail, and only wants the tail line's
    PARAMETERS on this particular set of counts.  Seeding at the full-data solution means one
    short fit instead of _select_tail's three long ones, which is what makes the bootstrap
    affordable again (see bootstrap_cut).  Returns (params, mpv, chi2_red), or None if the
    refit fails -- in which case the caller keeps the plain Landau."""
    lo, hi = _tail_bounds(max(float(mpv_seed), 2.0 * bin_w), bin_w)
    p0 = [float(np.clip(v, l, h)) for v, l, h in zip(seed, lo, hi)]
    try:
        p, _ = curve_fit(_langau_tailmix, x, y, p0=p0, bounds=(lo, hi),
                         sigma=np.sqrt(np.maximum(y, 1.0)), maxfev=40_000)
    except (RuntimeError, ValueError):
        return None
    fine = np.linspace(x[0], x[-1], 4000)
    mpv = float(fine[np.argmax(_langau_tailmix(fine, *p))])
    return list(p), mpv, _reduced_chi2(y, _langau_tailmix(x, *p), 6)


def _select_tail(x, y, langau_params, bin_w, mpv_seed):
    """Upgrade the muon line from Landau(x)Gauss (4 params) to _langau_tailmix (6) when the
    channel's response carries a heavy tail the fixed-scale line cannot reach -- kept only
    when it lowers the BIC, so a channel with a clean line is left on the simpler model.
    The two are NESTED (tmax -> 1 recovers _langau exactly), so the BIC comparison is honest.

    Returns (tail_params, mpv, chi2_red) when the tail model wins, else None.

    HISTORY -- read before "restoring" the shoulder.  This replaced `_select_shoulder`, which
    bolted a Gaussian onto the tail and reported it as a secondary population at ~2.2x MPV on
    run00270_ch9.  It is not a population.  Measured 2026-07-12:
      * A masked-sideband fit (train on core + far tail, 1.7-2.9x MPV held out) shows the
        excess does NOT close above the window: the pull stays at +4 to +8 sigma from 2.9x
        out to ~3.8x.  A bump would return to zero there.  It is a tail.
      * The shoulder events are ordinary muon pulses: peak-aligned mean waveform matches the
        MIP band's to ~1% of pulse height, same peak-height/OF-amplitude ratio (239 vs 241,
        so the OF is not inflating them), same triage pileup rate (1.3% vs 1.5%), 1.98x peak
        height, 1.96x charge, flat over the 50-h run.  Not pileup, not saturation, not a
        filter artifact.  Accidental coincidence is out by ~6 orders of magnitude (0.083 Hz
        trigger rate).
      * The shoulder's fitted AREA was ~18% of events -- meaningless, degenerate with the
        tail.  It was never a population fraction.
    On ch9 this model beats the shoulder by dBIC = -108 with one FEWER parameter and is the
    only one leaving a flat tail residual (+0.9 sigma vs +2.0).

    The tail is NOT cosmetic: the quoted MPV depends on it.  Measured by deleting it outright
    (2026-07-13, reverted the same day) -- MPV +15.8% on ch9_clean, +8.3% on ch10_clean,
    +11.9% on ch6; muon chi2_red 1.01 -> 2.96 on ch9_clean; the ch10 gamma/muon cut -43%."""
    n = len(x)
    if n < 12:
        return None
    w = np.sqrt(np.maximum(y, 1.0))

    def _chi2(model_y):
        return float(np.sum((y - model_y) ** 2 / np.maximum(model_y, 1.0)))

    base_chi2 = _chi2(_langau(x, *langau_params))
    mpv0 = max(float(mpv_seed), 2.0 * bin_w)
    lo, hi = _tail_bounds(mpv0, bin_w)

    # MULTI-START, and one of the starts is the NESTED one.  The tail model contains the
    # plain Landau (tmax -> 1), so at its optimum it can never fit worse -- but a single
    # heuristic seed does land in a worse local minimum on some channels (ch2: base chi2
    # 1362, one-seed tail fit 1889).  Rejecting on that would mean "the optimizer failed",
    # not "the channel has no tail", and the BIC gate would be reading noise.  Starting one
    # fit AT the base solution (with a whisker of scale spread) floors the search there, so a
    # rejection now genuinely means the extra freedom bought nothing.
    a_b, loc_b, eta_b, sig_b = langau_params
    starts = [
        # nested: the base Landau(x)Gauss itself, barely perturbed
        [a_b, loc_b + _LANDAU_MODE * eta_b, eta_b, sig_b, 4.0, 1.05],
        # narrow core + real spread (the base fit had to inflate eta/sigma to reach the tail,
        # which is exactly the misfit this model exists to undo)
        [float(np.sum(y) * bin_w), 0.9 * mpv0, 0.08 * mpv0, 0.2 * mpv0, 4.0, 2.0],
        # heavy tail, very narrow core (where the dim PMT channels land)
        [float(np.sum(y) * bin_w), 0.8 * mpv0, 0.05 * mpv0, 0.15 * mpv0, 2.0, 3.0],
    ]
    best = None
    for s in starts:
        p0 = [float(np.clip(v, l, h)) for v, l, h in zip(s, lo, hi)]
        try:
            p, _ = curve_fit(_langau_tailmix, x, y, p0=p0, bounds=(lo, hi),
                             sigma=w, maxfev=200_000)
        except (RuntimeError, ValueError):
            continue
        c = _chi2(_langau_tailmix(x, *p))
        if best is None or c < best[0]:
            best = (c, p)
    if best is None:
        return None
    tail_chi2, p = best
    if (tail_chi2 + 6 * math.log(n)) >= (base_chi2 + 4 * math.log(n)):
        return None
    fine = np.linspace(x[0], x[-1], 4000)
    mpv = float(fine[np.argmax(_langau_tailmix(fine, *p))])
    return list(p), mpv, _reduced_chi2(y, _langau_tailmix(x, *p), 6)


# The keys fit_muon_line contributes to a fit result.  Both modes copy exactly these onto
# their own result dict, so a fit is described by the same line the same way either way.
_LINE_KEYS = ("muon_params", "tail_params", "mpv", "eta", "sigma", "fit_chi2")


def line_model_of(fit: dict) -> dict:
    """The line-model CHOICE a completed fit made (tail or no tail), in the form
    fit_muon_line takes back as `line_model`.  Lets a refit of the same channel reuse the
    choice instead of re-running the BIC selection."""
    tp = fit.get("tail_params")
    return {"tail": list(tp) if tp else None}


def _muon_fit_floor(centers, finite_vals, config: Config) -> int:
    """Lowest bin the MIP line may be fit from when NOTHING models the pedestal below it
    (muon mode, and gamma-muon's muon-only fallbacks).  The displayed histogram is untouched
    (spectrum_lo_pct=0): this only keeps the far sub-MIP noise triggers -- a handful of events
    the Landau predicts ~0 counts for -- out of the Poisson chi2, where they would otherwise
    drag the fit.  Where a gamma component IS fit, it models those bins and this floor does
    not apply."""
    if not finite_vals.size:
        return 0
    floor_x = float(np.percentile(finite_vals, config.muon_fit_lo_pct))
    return int(np.clip(np.searchsorted(centers, floor_x), 0, len(centers) - 1))


def fit_muon_line(centers, y, smoothed, fit_lo: int, bin_w: float,
                  line_model: dict | None = None) -> dict | None:
    """THE MIP line fit -- the single implementation, used by BOTH modes.

    Landau(x)Gauss on the bins [fit_lo, end], Poisson-weighted, followed by the BIC-gated
    upgrade to _langau_tailmix (see _select_tail).  Returns None when the line cannot even
    be seeded; otherwise the fitted line, plus `fun`: the fitted curve as a callable, which
    gamma-muon mode needs as a FIXED component of the gamma fit and of the cut criterion.

    `line_model` (from line_model_of) SKIPS the BIC selection and imposes an already-made
    choice: {"tail": None} fits the plain Landau, {"tail": [...]} refits the tail line from
    those parameters (_refit_tail).  Bootstrap replicates pass the full-data choice, so they
    resample the DATA and not the model -- see bootstrap_cut.

    WHY THIS EXISTS.  The two modes each fit this line themselves, and they were not fitting
    the same thing: gamma-muon fit a bare 4-parameter _langau, while muon mode fit that and
    then ran the tail upgrade on top of it.  Where the tail wins it moves the MPV by several
    percent, so the two modes' MPVs were not comparable -- while the module docstring asks
    you to cross-check exactly those two numbers against each other.

    Only the fit RANGE still differs by mode, and legitimately so: gamma-muon starts the line
    at the valley and models the pedestal below it as gamma, whereas muon mode has no gamma
    component and so floors the fit at config.muon_fit_lo_pct instead (see Config).  The
    caller picks `fit_lo`; the LINE is the same one either way."""
    xf, yf = centers[fit_lo:], y[fit_lo:]
    if xf.size < 5:
        return None
    mpv_seed = float(xf[int(np.argmax(smoothed[fit_lo:]))])
    seed = _seed_langau(xf, yf, mpv_seed, bin_w)
    if seed is None:
        return None
    lo = [0.0, xf[0], bin_w, bin_w]
    hi = [np.inf, xf[-1] * 1.5, xf[-1] - xf[0], xf[-1] - xf[0]]
    try:
        par, _ = curve_fit(_langau, xf, yf, p0=seed, bounds=(lo, hi),
                           sigma=np.sqrt(np.maximum(yf, 1.0)), maxfev=40_000)
        params = list(par)
    except (RuntimeError, ValueError):
        params = list(seed)
    fine = np.linspace(xf[0], xf[-1], 4000)
    line = {"muon_params": params, "tail_params": None,
            "mpv": float(fine[np.argmax(_langau(fine, *params))]),
            "eta": float(params[2]), "sigma": float(params[3]),
            "n_params": 4,
            "fit_chi2": _reduced_chi2(yf, _langau(xf, *params), 4),
            "fun": (lambda x, p=params: _langau(x, *p))}
    # Optional response-tail upgrade: some channels (the dim ch9 PMT) have a tail a
    # fixed-scale Landau(x)Gauss cannot reach, because one eta must set both the core
    # asymmetry and the tail weight.  _langau_tailmix lets the light-yield scale fluctuate,
    # which decouples them; it is nested in _langau, so a clean channel keeps the simple
    # model.  eta/sigma then describe the CORE (at scale t=1) and the MPV is the peak of the
    # full mixture -- it can move several % from the single-Landau value, which is the point.
    #
    # The model is BIC-SELECTED here (_select_tail) unless the caller already made the choice.
    if line_model is None:
        tail = _select_tail(xf, yf, params, bin_w, mpv_seed)
    elif line_model.get("tail"):
        tail = _refit_tail(xf, yf, bin_w, mpv_seed, line_model["tail"])
    else:
        tail = None
    if tail is not None:
        tpar, mpv_t, chi2_t = tail
        line.update(tail_params=list(tpar), mpv=mpv_t, eta=float(tpar[2]),
                    sigma=float(tpar[3]), n_params=6, fit_chi2=chi2_t,
                    fun=(lambda x, p=list(tpar): _langau_tailmix(x, *p)))
    return line


def fit_muon_spectrum(values, config: Config):
    """Fit the muon/MIP line (fit_muon_line) to the WHOLE pulse-height spectrum.
    This is the right model for cleaned, hodoscope-triggered, mostly-muon data: no
    gamma continuum is added and no discrimination cut is made.  Returns the fit
    curve, the MPV (in the ADC-derived observable units), the Landau width eta
    (intrinsic energy-loss straggling) and the Gaussian sigma (detector smearing)."""
    centers, counts, smoothed = build_spectrum(values, config)
    y = counts.astype(float)
    bin_w = centers[1] - centers[0]

    # BIAS GUARD: on data that was NOT triage-cleaned, low-amplitude noise/gamma
    # triggers form a distinct population BELOW the MIP line -- a bump, then a valley,
    # then the Landau rise.  Left in the fit they distort the Landau turn-on (and, when
    # they dominate, drag the MPV into the gap between the populations).  The Landau's
    # OWN turn-on rises monotonically into the peak, so a genuine secondary population is
    # identified by a PROMINENT low-amplitude bump below a valley below the MIP peak;
    # when found, fit the MIP line only ABOVE the valley, using the SAME hardened landmarks
    # the gamma-muon mode uses (spectrum_landmarks).  This is independent of whether the
    # noise population is the dominant mode, so it also cleans muon-tagged data whose MIP
    # peak dominates but still carries a noise shoulder -- no pre-triage of the input
    # required.
    #
    # These landmarks used to come from a bare argmax + global-argmin (find_spectrum_
    # landmarks), which is exactly what broke on a low-light channel: on the triage-cleaned
    # run00270_ch10 the argmax IS the residual noise spike (0.13, with the median at 0.84),
    # so `mb` was the spike, the guard never fired, and the Landau was fit straight across
    # the spike -- MPV 0.676, chi2 3.39, sigma/MPV 97%, an 18% bias low.  The shared
    # landmarks floor the bump search at half the median and walk the valley down from it.
    finite_vals = np.asarray(values, dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    median_val = float(np.median(finite_vals)) if finite_vals.size else 0.0
    fit_lo, n_below = 0, 0
    mode_x = float(centers[int(np.argmax(smoothed))])   # raw dominant bin -- for the warning only
    scale = spectrum_scale(finite_vals)
    overlap = _landmark_overlap(scale, bin_w, len(centers), config)
    marks = spectrum_landmarks(centers, smoothed, median_val, overlap, config)
    gp, vy, mb = marks["gamma_peak"], marks["valley"], marks["muon_bump"]
    # A real secondary bump clears the valley floor by a healthy margin (>=2x); the
    # bare Landau turn-on pins the valley at the first bin (gp == vy, ratio ~1) and is
    # left untouched.
    low_population = (mb > vy > gp
                      and smoothed[gp] >= 2.0 * max(smoothed[vy], 1.0))
    if low_population:
        fit_lo = vy
        strongly_bimodal = bool(median_val > 0 and mode_x < 0.5 * median_val)
        msg = ("Muon spectrum carries a low-amplitude population below the MIP line: "
               "fitting the Landau only above the valley at %.4g.")
        if strongly_bimodal:
            logger.warning(msg + " The noise population DOMINATES (mode %.3g << median %.3g); "
                           "triage-clean the input for the cleanest line.",
                           float(centers[fit_lo]), mode_x, float(np.median(finite_vals)))
        else:
            logger.info(msg, float(centers[fit_lo]))
    # Robust FIT floor: exclude the far sub-MIP noise pedestal from the Landau fit even when
    # it is not a prominent bump the valley logic above catches (see _muon_fit_floor).
    fit_lo = max(fit_lo, _muon_fit_floor(centers, finite_vals, config))
    n_below = int(np.count_nonzero(finite_vals < centers[fit_lo])) if fit_lo > 0 else 0
    out = {"centers": centers, "counts": counts, "smoothed": smoothed,
           "muon_params": None, "gamma_params": None, "tail_params": None, "n_gamma": 0,
           "mpv": float(centers[int(np.argmax(smoothed[fit_lo:])) + fit_lo]),
           "eta": float(bin_w), "sigma": float(bin_w),
           "ok": False, "fit_chi2": np.nan, "cut": np.nan,
           "fit_lo_x": float(centers[fit_lo]), "n_low_excluded": int(n_below)}
    line = fit_muon_line(centers, y, smoothed, fit_lo, bin_w)
    if line is None:
        return out
    out.update({k: line[k] for k in _LINE_KEYS}, ok=True)
    return out


def muon_summary(values, fit,
                 peak_scale: float = np.nan, charge_scale: float = np.nan) -> dict[str, Any]:
    """Model-free + fit-based summary of the (mostly-muon) pulse-height
    distribution, entirely in ADC-derived units.  Reports the MPV, the central
    location (mean/median), the observed spread (robust 16-84 percentile
    half-width), and the two width components of the fitted line.

    The observable is template-relative (A=1 == a template-shaped pulse), which is
    invariant to a common gain change -- useless for a GAIN SWEEP.  So we also report
    the MPV in ABSOLUTE units by multiplying by the template's own ADC scale:
    `peak_scale` = template peak height (ADC) and `charge_scale` = template integral
    (ADC.samples).  MPV_peak_adc / MPV_charge_adc DO scale with the gain and are the
    right y-values to plot against bias voltage."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    p16, p50, p84 = (float(x) for x in np.percentile(v, [16, 50, 84]))
    half_spread = 0.5 * (p84 - p16)
    mpv = fit["mpv"]
    return {
        "n": int(v.size),
        "mpv_obs": mpv,
        "mean_obs": float(v.mean()),
        "median_obs": p50,
        "halfspread_obs": half_spread,
        "spread_frac": half_spread / mpv if mpv > 0 else np.nan,
        "eta_obs": fit["eta"], "eta_frac": fit["eta"] / mpv if mpv > 0 else np.nan,
        "sigma_obs": fit["sigma"], "res_mpv": fit["sigma"] / mpv if mpv > 0 else np.nan,
        # Absolute (gain-tracking) scale: observable * template ADC scale.
        "peak_scale": float(peak_scale), "charge_scale": float(charge_scale),
        "mpv_peak_adc": mpv * peak_scale, "mpv_charge_adc": mpv * charge_scale,
    }


# ===========================================================================
# Discrimination
# ===========================================================================

def discrimination_report(observable, fit) -> dict[str, Any]:
    """Muon-like / gamma-like population split at the fitted cut.

    When there IS no cut (`fit["ok"]` false: the spectrum has no separable gamma
    population, or every cut candidate was non-physical) this reports `ok=False` and
    leaves the counts as None.  It used to compare the observable against NaN, which is
    False everywhere, and so reported "0 muon-like, N gamma-like (0.0% muon-like)" -- a
    definite-looking measurement that was really just the absence of a cut.  Callers must
    check `ok` before reading the counts."""
    n_total = int(np.isfinite(observable).sum())
    cut = float(fit["cut"]) if fit["ok"] else np.nan
    if not np.isfinite(cut):
        return {"cut": np.nan, "ok": False, "n_total": n_total,
                "n_muon_like": None, "n_gamma_like": None, "muon_fraction": np.nan}
    flagged = np.isfinite(observable) & (observable > cut)
    n_muon = int(flagged.sum())
    return {"cut": cut, "ok": True, "n_total": n_total,
            "n_muon_like": n_muon, "n_gamma_like": n_total - n_muon,
            "muon_fraction": 100.0 * n_muon / max(n_total, 1)}


def chi2_amplitude_trend(amplitude: np.ndarray, chi2: np.ndarray,
                         clipped: np.ndarray | None = None) -> tuple[float, float]:
    """The CLEAN per-event OF chi2 as a function of amplitude: chi2_red(A) = c0 + (A/A0)^2.
    Returns (c0, A0).

    WHY THIS SHAPE.  The per-event chi2 is not amplitude-independent, and it is not supposed
    to be.  If the pulses are not all EXACTLY the template shape, write

        data = A * (template + delta) + noise          delta = fractional shape mismatch

    The optimal filter's residual is then  A*delta + noise: the mismatch part scales with the
    amplitude, the noise part does not.  Squaring and noise-weighting gives

        chi2_red(A) = c0 + (A / A0)^2

    with c0 the additive-noise floor (1.0 when the PSD is right) and A0 the amplitude at which
    the shape mismatch equals the noise.  So a chi2-vs-amplitude band that RISES is the
    generic signature of a real detector whose pulses vary in shape -- NOT of a broken fit,
    and NOT of a biased amplitude (the OF amplitude is a linear estimator; chi2 is computed
    after it and never feeds back into it).

    Measured on run00270 (the A^2 model reproduces ch0's binned median chi2 to within 0.041
    over a 1.01 -> 2.04 range, so this is the law, not a parameterisation):

        channel   c0      A0      chi2 at the MIP
        ch0       0.99    1.63    ~1.25      <- c0 ~ 1: the PSD/noise model is correct
        ch10      1.47    0.81    ~2.1
        ch9       5.89    0.43    ~9         <- c0 >> 1 is ch9's ~1-LSB noise, a KNOWN offset

    **A0 is a template-fidelity metric** -- lower A0 = the template describes the pulses less
    well -- and it ranks the channels exactly as the independent pulse-shape-jitter finding
    does (ch0 best, ch9 worst).

    Fit is on the amplitude-binned MEDIAN chi2 (robust to the pileup tail) of the UNCLIPPED
    events (`clipped`, from saturation_qc's sat_mask): rail-clipped pulses are a genuinely
    different shape and produce the big high-amplitude arc that would otherwise dominate the
    fit.  Degenerate input -> (median chi2, inf), i.e. a flat trend, which makes every caller
    fall back to the old amplitude-independent behaviour."""
    a = np.asarray(amplitude, dtype=float)
    c = np.asarray(chi2, dtype=float)
    ok = np.isfinite(a) & np.isfinite(c) & (a > 0)
    if clipped is not None:
        ok &= ~np.asarray(clipped, dtype=bool)
    flat = (float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else 1.0, np.inf)
    if ok.sum() < 200:
        return flat
    a, c = a[ok], c[ok]
    edges = np.percentile(a, np.linspace(2, 98, 15))
    cen, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (a >= lo) & (a < hi)
        if s.sum() >= 30:
            cen.append(float(np.median(a[s])))
            med.append(float(np.median(c[s])))
    if len(cen) < 5:
        return flat
    cen, med = np.asarray(cen), np.asarray(med)
    G = np.vstack([np.ones_like(cen), cen ** 2]).T
    try:
        c0, k = np.linalg.lstsq(G, med, rcond=None)[0]
    except np.linalg.LinAlgError:
        return flat
    c0 = float(max(c0, 1e-6))
    if not np.isfinite(k) or k <= 0:
        return c0, np.inf
    return c0, float(1.0 / math.sqrt(k))


def expected_chi2(amplitude, c0: float, A0: float) -> np.ndarray:
    """The clean chi2 an event of this amplitude is EXPECTED to have (chi2_amplitude_trend)."""
    a = np.asarray(amplitude, dtype=float)
    if not np.isfinite(A0) or A0 <= 0:
        return np.full(a.shape, c0, dtype=float)
    return c0 + (a / A0) ** 2


def pileup_mask_from_chi2(chi2: np.ndarray, config: Config,
                          amplitude: np.ndarray | None = None,
                          clipped: np.ndarray | None = None) -> np.ndarray:
    """Flag pileup / shape-anomaly events as events whose per-event OF reduced chi2 exceeds
    pileup_chi2_factor x the chi2 a CLEAN pulse OF THAT AMPLITUDE would have.  Returns a
    boolean mask (True = flagged).  The OF and compare drivers pass this to
    analyze_observable when config.exclude_pileup is set.

    THE THRESHOLD IS AMPLITUDE-DEPENDENT, and it has to be.  It used to be a single global
    number, chi2 > factor x median(chi2).  But the clean chi2 itself RISES with amplitude --
    chi2_red(A) = c0 + (A/A0)^2, see chi2_amplitude_trend -- so a flat threshold is really an
    AMPLITUDE cut wearing a goodness-of-fit costume.  Measured on run00270_ch9 (which has NO
    clipped events at all, so none of this was saturation):

        amplitude   0.5-0.7   1.3-1.7   2.0-3.1   3.1+
        flagged        0.1%      3.3%      19%     55%

    It was eating the Landau tail: 370 events flagged, MPV pulled +2.9%.  Scaling the
    threshold by the expected chi2 flags 70 instead, and the ones it drops are the ones that
    are actually anomalous FOR THEIR AMPLITUDE.

    `amplitude` omitted -> falls back to the old global-median threshold (a caller with no
    amplitude to hand cannot do better).  `clipped` (saturation_qc's sat_mask) is excluded
    from the TREND fit only -- clipped events are still flagged if they exceed the threshold,
    which they will, since their shape really is wrong."""
    chi2 = np.asarray(chi2, dtype=float)
    finite = np.isfinite(chi2)
    if not finite.any():
        return np.zeros(chi2.shape, dtype=bool)
    if amplitude is None:
        med = float(np.median(chi2[finite]))
        if not np.isfinite(med) or med <= 0:
            return np.zeros(chi2.shape, dtype=bool)
        return finite & (chi2 > config.pileup_chi2_factor * med)
    c0, a0 = chi2_amplitude_trend(amplitude, chi2, clipped)
    thresh = config.pileup_chi2_factor * expected_chi2(amplitude, c0, a0)
    return finite & np.isfinite(thresh) & (chi2 > thresh)


def _unclipped_for_fit(observable, prep: Prepared, config: Config, sat: dict | None = None,
                       pileup_mask: np.ndarray | None = None):
    """Hard-clipped (rail-saturated) events pile up at a fixed peak height, which in the
    observable lands as a spurious spike on the high side (near rail/template-peak).  A
    Landau tail cannot describe it, so it distorts the muon line and wrecks the spectral
    chi2 without carrying real shape information.  When saturation_qc detects a genuine
    rail, drop the flat-top-clipped events (saturation_qc's sat_mask -- triage's exact
    criterion) for FITTING only; no rail detected (e.g. triage-cleaned data) -> none are
    dropped on that account.  `pileup_mask` (opt-in, from pileup_mask_from_chi2) drops
    pileup / high-chi2 shape-anomaly events from the fit as well -- these are mostly
    high-amplitude overlapping pulses that pull the muon Landau line up and bias the
    crossover cut (see the gamma-muon-cut-pileup-bias finding).  `sat` is a precomputed
    saturation_qc result (the drivers compute it once anyway); computed here if omitted.
    Returns (observable_fit, n_clipped, n_pileup) where n_pileup counts events dropped
    for pileup that were not ALREADY dropped as clipped."""
    observable = np.asarray(observable, dtype=float)
    if sat is None:
        sat = saturation_qc(prep, config)
    n = observable.shape[0]
    clip = np.zeros(n, dtype=bool)
    if sat["clipped_pileup"] and sat["sat_mask"].shape[0] == n:
        clip = sat["sat_mask"].astype(bool)
    pile = np.zeros(n, dtype=bool)
    if pileup_mask is not None and np.asarray(pileup_mask).shape[0] == n:
        pile = np.asarray(pileup_mask, dtype=bool)
    drop = clip | pile
    return observable[~drop], int(clip.sum()), int((pile & ~clip).sum())


def analyze_observable(observable, prep: Prepared, label: str, color: str, config: Config,
                       sat: dict | None = None, pileup_mask: np.ndarray | None = None) -> dict:
    """Fit the spectrum (all in ADC-derived observable units); in gamma-muon mode
    also report the discrimination cut.  Rail-clipped (saturated) events are excluded
    from the FIT in both modes -- they carry no valid pulse-height information.  Two
    observable arrays are kept on the result: m["observable"] is the FULL set (used by
    plot_amplitude_area to surface the saturation outliers, and, in gamma-muon mode, by
    the muon/gamma population counts -- bright saturated pulses are genuine muons on the
    muon-like side of the cut), while m["observable_fit"] is the clip-excluded set the
    spectrum was fit on (used by plot_estimator_comparison so its histogram matches the
    fitted line and the width it reports)."""
    observable = np.asarray(observable, dtype=float)
    m: dict[str, Any] = {"label": label, "color": color, "observable": observable}
    obs_fit, n_clip, n_pileup = _unclipped_for_fit(observable, prep, config, sat, pileup_mask)
    m["observable_fit"] = obs_fit
    m["n_clip_excluded"] = n_clip
    m["n_pileup_excluded"] = n_pileup
    # Absolute ADC scales for the template-relative observable: A=1 corresponds to a
    # template-shaped pulse, so peak height = A*template.max() and charge = A*integral.
    # Kept on the method so the spectrum / pull plots can add an ADC-reading twin axis.
    peak_scale = float(prep.template.max())
    charge_scale = float(np.trapezoid(prep.template))
    m["peak_scale"] = peak_scale
    m["charge_scale"] = charge_scale
    if config.mode == "muon":
        m["fit"] = fit_muon_spectrum(obs_fit, config)
        m["fit"]["n_clip_excluded"] = n_clip
        m["fit"]["n_pileup_excluded"] = n_pileup
        _apply_reliability_guard(m["fit"], prep, config, label)
        m["summary"] = muon_summary(obs_fit, m["fit"],
                                    peak_scale=peak_scale, charge_scale=charge_scale)
        return m
    m["fit"] = fit_spectrum(obs_fit, config)
    m["fit"]["n_clip_excluded"] = n_clip
    m["fit"]["n_pileup_excluded"] = n_pileup
    _apply_reliability_guard(m["fit"], prep, config, label)
    # Error bar on the cut (resampling the events).  Only worth computing when there IS a
    # cut; a channel the reliability guard rejected has nothing to put an error bar on.
    if m["fit"].get("ok") and np.isfinite(m["fit"].get("cut", np.nan)):
        # Pass the fit: the replicates reuse ITS line model rather than re-selecting one.
        m["fit"].update(bootstrap_cut(obs_fit, config, m["fit"]))
    # The cut is FIT on the pileup-excluded set, but the muon/gamma population COUNT uses
    # the full observable: pileup events are genuine high-amplitude muons and sit on the
    # muon-like side of the cut, so they belong in the count even though they'd distort the fit.
    m["disc"] = discrimination_report(observable, m["fit"])
    return m


def _apply_reliability_guard(fit: dict, prep: Prepared, config: Config, label: str) -> None:
    """TWO independent questions, both answered on scales that come from the WAVEFORMS and
    the trigger -- never from the spectrum being fit (a self-referential test can't fire; the
    guard this replaced compared the MPV against the MEDIAN of its own observable, so on a
    noise-dominated channel the median sat in the noise too, the MPV tracked it, and
    run00270_ch4 -- pulse SNR ~3, 91% of events triaged NOISE -- sailed through with a cut).

      1. IS THERE A PULSE?   mip_snr = mpv * template_peak / noise_sigma, the MIP peak height
         in units of the raw per-sample baseline noise.  This is a SINGLE-SAMPLE SNR: "would
         you see this pulse by eye in one sample".  It is deliberately conservative and it is
         what stops ch4.

      2. IS THERE ROOM FOR A GAMMA?   mip_over_trigger = mpv / trigger_floor.  Every event
         here passed the trigger, so the spectrum is truncated from below (see Config.
         min_mip_over_trigger).  A gamma population is only VISIBLE between the trigger floor
         and the MIP; when the MIP sits a few floors up, that band is gone and the histogram's
         low edge is the threshold turn-on, which the BIC mixture then happily fits AS a
         gamma.  This is what stops ch6 and gives ch7 the right reason.

    Also reports mip_snr_obs = mpv / sigma_A: the line's significance in the OBSERVABLE that
    is actually being fit, where sigma_A = trigger_floor / trigger_sigma is the optimal
    filter's amplitude resolution.  It is NOT a gate -- it is the number that keeps the other
    two honest.  The two SNRs answer different questions and can disagree by a lot: on
    run00270_ch7 the line is 2.2 sigma over the raw sample noise but 12.6 sigma over the OF's
    own resolution, because the filter integrates 338 samples.  Both are true.  Reporting
    only the first made "no MIP line to speak of" the printed reason for a line that is, in
    the spectrum being fit, perfectly well resolved.

    Mutates `fit` in place: sets the three numbers and `reliable`, and clears `ok` when
    unreliable, so no cut is reported for a channel that has nothing to cut."""
    mpv = fit.get("mpv", np.nan)
    peak_scale = float(prep.template.max())
    sigma = float(prep.noise_sigma)
    floor = float(prep.trigger_floor)
    snr = float(mpv * peak_scale / sigma) if (sigma > 0 and np.isfinite(mpv)) else np.nan
    over = float(mpv / floor) if (floor > 0 and np.isfinite(mpv)) else np.nan
    sigma_a = floor / config.trigger_sigma if (floor > 0 and config.trigger_sigma > 0) else np.nan
    fit["mip_snr"] = snr
    fit["mip_over_trigger"] = over
    fit["trigger_floor"] = floor
    fit["mip_snr_obs"] = float(mpv / sigma_a) if (sigma_a > 0 and np.isfinite(mpv)) else np.nan

    has_pulse = bool(np.isfinite(snr) and snr >= config.min_mip_snr)
    # NaN floor (no pulse-free region to measure resp_rms on) -> cannot test; don't refuse.
    has_room = bool((not np.isfinite(over)) or over >= config.min_mip_over_trigger)
    fit["reliable"] = has_pulse and has_room
    if fit["reliable"]:
        return
    fit["ok"] = False
    if not has_pulse:
        logger.warning(
            "%s: the fitted MIP line stands only %.1f sigma above the raw per-sample baseline "
            "noise (MPV %.4g x template peak %.0f ADC / noise sigma %.3g; the bar is %.1f). "
            "In the filtered observable it is %.1f sigma -- but a pulse this small in the raw "
            "trace is not a trustworthy MIP line, and NO gamma/muon cut is reported. Check "
            "--polarity / --pulse-center, or accept that the channel is too dim.",
            label, snr, mpv, peak_scale, sigma, config.min_mip_snr, fit["mip_snr_obs"])
    if not has_room:
        logger.warning(
            "%s: the MIP line (MPV %.4g) sits only %.1fx above the TRIGGER FLOOR (%.4g = "
            "%.1f x the optimal filter's amplitude resolution); the bar is %.1fx. The "
            "gamma/noise population lives BELOW the MIP line, and the trigger has already "
            "removed everything under the floor -- so there is no gamma population left in "
            "this spectrum to cut against, and the histogram's low edge is the threshold "
            "turn-on, not physics. The MIP line itself is still fit and reported; NO "
            "gamma/muon cut is. Lower --trigger-sigma to recover the low-amplitude events, "
            "or accept that this channel cannot support the cut.",
            label, mpv, over, floor, config.trigger_sigma, config.min_mip_over_trigger)


def print_muon_summary(prep: Prepared, methods: list[dict], box_width: int | None = None) -> None:
    """Summary for cleaned, hodoscope-triggered, mostly-muon data: the MIP line
    location and the pulse-height distribution (ADC units) -- no gamma/muon split."""
    print("\n" + "=" * 66)
    print("Waveform analysis  (muon mode: hodoscope-selected MIP pulses)")
    print("=" * 66)
    print(f"Triggered events:  {len(prep.events):,} / {prep.n_events:,}")
    print(f"Noise sigma (ADC): {prep.noise_sigma:.4g}")
    if box_width is not None:
        print(f"Boxcar window:     {box_width} samples")
    print("\nMuon (MIP) line  [all values in ADC-derived observable units]")
    for m in methods:
        s, f = m["summary"], m["fit"]
        tp = f.get("tail_params")
        shape = "Landau(x)Gauss + response tail" if tp else "Landau(x)Gauss"
        print(f"  {m['label']:<18} MPV   = {s['mpv_obs']:.4g} obs"
              f"   ({shape} chi2_red={f['fit_chi2']:.2f})")
        if tp:
            print(f"  {'':<18} + BIC-selected response tail: light-yield scale t ~ "
                  f"t^-{tp[4]:.2f} on [1, {tp[5]:.2f}]")
            print(f"  {'':<18}   (a per-channel light-collection tail, NOT a second "
                  f"population and NOT muon path length -- see _langau_tailmix)")
        if f.get("n_clip_excluded", 0):
            print(f"  {'':<18} (fit excludes {f['n_clip_excluded']:,} rail-clipped events)")
        if f.get("n_pileup_excluded", 0):
            print(f"  {'':<18} (fit excludes {f['n_pileup_excluded']:,} pileup / high-chi2 events)")
        if f.get("n_low_excluded", 0):
            print(f"  {'':<18} (fit starts above the valley at {f['fit_lo_x']:.4g}: "
                  f"{f['n_low_excluded']:,} low-amplitude noise events excluded)")
        print(f"  {'':<18} MPV (absolute, gain-tracking) = {s['mpv_peak_adc']:.4g} ADC peak"
              f"  |  {s['mpv_charge_adc']:.4g} ADC.samp charge")
        print(f"  {'':<18} median= {s['median_obs']:.4g} obs"
              f"   |  mean = {s['mean_obs']:.4g} obs")
        print(f"  {'':<18} spread (16-84, half) = {s['halfspread_obs']:.4g} obs "
              f"({100*s['spread_frac']:.1f}% of MPV)")
        print(f"  {'':<18} Landau eta/MPV = {100*s['eta_frac']:.1f}%  (energy-loss straggling)")
        print(f"  {'':<18} Gauss sigma/MPV = {100*s['res_mpv']:.1f}%  (detector resolution at MIP)")
    print("=" * 66)


def print_summary(prep: Prepared, methods: list[dict], box_width: int | None = None) -> None:
    if all("summary" in m for m in methods):
        return print_muon_summary(prep, methods, box_width)
    print("\n" + "=" * 66)
    print("Waveform analysis  (all values in ADC-derived observable units)")
    print("=" * 66)
    print(f"Triggered events:  {len(prep.events):,} / {prep.n_events:,}")
    print(f"Noise sigma (ADC): {prep.noise_sigma:.4g}")
    if box_width is not None:
        print(f"Boxcar window:     {box_width} samples")
    print("\nMuon (MIP) line")
    for m in methods:
        f = m["fit"]
        mpv = f["mpv"]
        width = f["sigma"] / mpv if mpv > 0 else np.nan
        tp = f.get("tail_params")
        shape = "Landau(x)Gauss + response tail" if tp else "Landau(x)Gauss"
        print(f"  {m['label']:<18} MPV={mpv:.4g} obs   ({shape} chi2_red={f['fit_chi2']:.2f})\n"
              f"  {'':<18} line-width sigma/MPV = {100*width:.1f}% "
              f"({'core, at scale t=1' if tp else 'Landau(x)Gauss'}; upper bound)")
        if tp:
            print(f"  {'':<18} + BIC-selected response tail: light-yield scale t ~ "
                  f"t^-{tp[4]:.2f} on [1, {tp[5]:.2f}]  (see _langau_tailmix)")
        # BOTH SNRs: the raw single-sample one (the pulse-exists gate) and the one in the
        # observable actually being fit.  They answer different questions and can differ by
        # the filter's whole noise gain -- 2.2 vs 12.6 sigma on run00270_ch7.
        snr, snr_obs = f.get("mip_snr", np.nan), f.get("mip_snr_obs", np.nan)
        if np.isfinite(snr):
            print(f"  {'':<18} MIP / raw baseline noise  = {snr:.1f} sigma  (one sample)")
        if np.isfinite(snr_obs):
            print(f"  {'':<18} MIP / filter resolution   = {snr_obs:.1f} sigma  (this observable)")
        over = f.get("mip_over_trigger", np.nan)
        if np.isfinite(over):
            short = over < prep.config.min_mip_over_trigger
            print(f"  {'':<18} MIP / trigger floor       = {over:.1f}x  "
                  f"(floor = {f.get('trigger_floor', np.nan):.4g} obs)"
                  + ("   [no room below it for a gamma population]" if short else ""))
        if not f.get("reliable", True):
            print(f"  {'':<18} [UNRELIABLE -- see the warning above; no cut reported]")
        excl = [f"{f[k]:,} {t}" for k, t in (("n_clip_excluded", "rail-clipped"),
                ("n_pileup_excluded", "pileup/high-chi2")) if f.get(k, 0)]
        if excl:
            print(f"  {'':<18} (fit excludes {'; '.join(excl)} events)")
    print("\nGamma/muon separation")
    for m in methods:
        d, f = m["disc"], m["fit"]
        # A missing cut is reported AS missing, WITH which guard refused it.  It used to be
        # printed as "cut=nan: 0 muon-like, N gamma-like (0.0% muon-like)" -- which reads like
        # a measurement but is only the absence of one.
        if not d.get("ok", False):
            over = f.get("mip_over_trigger", np.nan)
            if np.isfinite(over) and over < prep.config.min_mip_over_trigger:
                why = (f"the MIP line is only {over:.1f}x the trigger floor -- the trigger "
                       f"already removed the gamma population")
            elif not f.get("reliable", True):
                why = "no MIP line above the raw baseline noise"
            else:
                why = "the spectrum holds no separable gamma population"
            print(f"  {m['label']:<18} NO CUT DERIVED ({why}); "
                  f"{d['n_total']:,} events left unclassified")
            continue
        # The cut is quoted WITH its bootstrap error bar (bootstrap_cut).  A bare number
        # invites reading a 10% change between runs as physics; on these channels 10% IS
        # the noise.
        std = f.get("cut_std", np.nan)
        err = f" +/- {std:.2g} ({100*std/d['cut']:.0f}%)" if (np.isfinite(std) and d['cut']) else ""
        print(f"  {m['label']:<18} cut={d['cut']:.4g}{err} ({f.get('cut_kind', 'equal-area')}): "
              f"{d['n_muon_like']:,} muon-like, "
              f"{d['n_gamma_like']:,} gamma-like ({d['muon_fraction']:.1f}% muon-like)")
        # How much population does the cut actually rest on?  A cut against a ~0% gamma is a
        # threshold landmark, not a discrimination boundary (see _gamma_evidence).
        gfrac = f.get("gamma_frac", np.nan)
        if np.isfinite(gfrac):
            thin = "   [THIN: the cut separates the MIP line from almost nothing]" \
                if gfrac < prep.config.gamma_frac_warn else ""
            print(f"  {'':<18} fitted gamma population = {100*gfrac:.2f}% of events{thin}")
        # The cut balances AREAS under the two models, so a gamma that misfits its own bins
        # cannot place it.  Say so next to the number, not only in the log.
        gchi = f.get("gamma_chi2", np.nan)
        if np.isfinite(gchi):
            # A THIN gamma's chi2 is Poisson noise on a handful of bins -- don't call that a
            # bad fit on top of the THIN flag it already carries.
            bad = ("   [BAD FIT: this cut is an artefact of the misfit, not a measurement]"
                   if (gchi > prep.config.gamma_chi2_warn
                       and gfrac >= prep.config.gamma_frac_warn) else "")
            print(f"  {'':<18} gamma fit chi2_red = {gchi:.1f}{bad}")
        if f.get("cut_degenerate"):
            print(f"  {'':<18} [DEGENERATE: the equal-area root sits on the valley "
                  f"({f.get('valley_cut', np.nan):.4g}) -- the muon model has no area below "
                  f"it, so the\n  {'':<18}  balance is trivial. The cut is the valley walk's "
                  f"answer, with the valley walk's noise.]")
    print("=" * 66)


# ===========================================================================
# Plots
# ===========================================================================

def _finish(fig, name, config: Config) -> None:
    if config.save_plots:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    if config.show_plots:
        plt.show()
    plt.close(fig)


def _add_adc_axis(ax, scale: float | None, label: str = "Peak amplitude (ADC)",
                  which: str = "x"):
    """Overlay a twin axis that reads the template-relative observable in absolute ADC.

    The observable A is normalized so A=1 is a template-shaped pulse (both the optimal
    filter and the boxcar share this scale), so peak height = A * template.max() -- a
    single linear factor.  A secondary axis with that transform lets the spectrum /
    amplitude axes be read directly in ADC without rescaling the plotted data.  No-op
    when `scale` is missing or non-physical (e.g. a degenerate template)."""
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return None
    fwd, inv = (lambda a: a * scale), (lambda v: v / scale)
    if which == "x":
        sec = ax.secondary_xaxis("top", functions=(fwd, inv))
        sec.set_xlabel(label)
    else:
        sec = ax.secondary_yaxis("right", functions=(fwd, inv))
        sec.set_ylabel(label)
    return sec


def plot_baseline_residuals(prep: Prepared, config: Config) -> None:
    s = prep.noise_sigma
    span = max(6.0 * s, 4.0)
    # Real ADC data is integer-quantized.  When the noise sigma is only a few ADC
    # counts, sub-LSB bins scatter the (integer) residuals into a misleading
    # "picket fence" that towers over any continuous Gaussian and makes the fit
    # look broken.  Bin at integer-ADC (1 LSB) width, centered on integers, so each
    # bar is exactly one ADC value and its density is directly comparable to the
    # Gaussian (whose integral over a 1-ADC bin equals the expected bar height).
    lsb_limited = s < 3.0
    if lsb_limited:
        # Noise ~ 1 LSB: embrace the discrete structure -- integer bars straight
        # from the (median-subtracted) pooled residuals, one ADC value per bar.
        res = np.asarray(prep.pooled, dtype=float)
        edges = np.arange(-np.ceil(span) - 0.5, np.ceil(span) + 1.5, 1.0)
    else:
        # Continuous-Gaussian regime.  Draw the residuals from the SAME pooled,
        # MEDIAN-subtracted data that the noise_sigma (and Gaussian overlay) come
        # from, so the histogram and the fit describe one consistent quantity.
        # On integer ADC the per-event median is itself an integer sample value, so
        # one sample per event lands on exactly 0 -- a small (~few %) pinning spike
        # in the 0 bin over the Gaussian.  That bump is accepted here as the price
        # of median/median self-consistency (mean subtraction would remove it but
        # would fit a median sigma to a mean-subtracted histogram).  ~120 bins when
        # the span allows.
        res = np.asarray(prep.pooled, dtype=float)
        width = max(1.0, np.round(span / 60.0))
        edges = np.arange(-np.ceil(span) - 0.5, np.ceil(span) + width + 0.5, width)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(res, bins=edges, density=True, color="C0", alpha=0.7, label="Baseline residuals")
    x = np.linspace(-span, span, 400)
    ax.plot(x, np.exp(-0.5 * (x / s) ** 2) / (s * _SQRT_2PI), color="red", lw=2,
            label=f"Gaussian (sigma = {s:.3g})")
    if lsb_limited:
        # With noise ~ 1 LSB the data occupies only a handful of integer ADC values, so
        # the CONTINUOUS Gaussian curve dips through the empty space between integers and
        # reads as a poor fit even when the noise is perfectly Gaussian.  The apples-to-
        # apples reference is the Gaussian's probability MASS integrated over each 1-ADC
        # bin (== the expected bar height at density scale); overlay it at the integer
        # centers so the bars can be judged against the correct discrete expectation.
        centers = np.arange(-np.ceil(span), np.ceil(span) + 1.0)
        cdf = lambda z: 0.5 * (1.0 + erf(z / (s * math.sqrt(2.0))))
        expected = cdf(centers + 0.5) - cdf(centers - 0.5)
        ax.plot(centers, expected, "D", color="k", ms=5, zorder=5,
                label="Gaussian integrated per 1-ADC bin")
    # Gaussianity check.  noise_sigma is a ROBUST (MAD) estimate of the CORE width -- the
    # right scale for triggering/template thresholds because it ignores tails.  When the
    # residuals are a clean Gaussian, ~68.3% sit within +/-sigma and the RMS equals sigma.
    # A spike-dominated channel (e.g. a PMT with frequent excursions) is leptokurtic: a
    # sharp core plus heavy tails, so far fewer than 68.3% land within +/-sigma and the RMS
    # runs well above sigma.  Report both so the overlay's "poor fit" is correctly read as
    # non-Gaussian noise, not a binning artifact.
    rms = float(np.std(res))
    within1 = float(np.mean(np.abs(res) <= s))
    heavy_tailed = within1 < 0.62 or rms > 1.3 * s
    ax.text(0.02, 0.97, f"RMS = {rms:.3g} ADC\nwithin +/-sigma: {within1*100:.0f}% (Gaussian 68%)"
            + ("\nheavy-tailed / spike noise" if heavy_tailed else ""),
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.85))
    note = "  (integer-ADC bins: noise ~ 1 LSB)" if lsb_limited else ""
    ax.set(xlabel="Baseline-subtracted ADC", ylabel="Probability density",
           title="Baseline noise residuals (real noise)" + note)
    ax.legend(); ax.grid(True, alpha=0.3)
    _finish(fig, "0_baseline_residuals", config)


def plot_template(prep: Prepared, config: Config) -> None:
    sem = prep.template_std / math.sqrt(max(prep.n_template, 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(prep.relative, prep.template, label="Cross-correlation-aligned average")
    ax.fill_between(prep.relative, prep.template - sem, prep.template + sem, alpha=0.35,
                    label=f"+/-1 standard error (N={prep.n_template:,})")
    ax.axvline(0, ls="--", color="gray", label="Peak")
    ax.set(xlabel="Samples relative to peak", ylabel="Corrected ADC", title="Pulse template")
    ax.legend(); ax.grid(True)
    _finish(fig, "1_template", config)


def plot_spectrum(method: dict, config: Config, name=None) -> None:
    fit = method["fit"]
    centers, counts = fit["centers"], fit["counts"]
    excl = [t for k, t in (("n_clip_excluded", "rail-clipped"),
            ("n_pileup_excluded", "pileup")) if fit.get(k, 0)]
    data_label = f"Data ({', '.join(excl)} excluded)" if excl else "Data"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(centers, counts, where="mid", color="C0", lw=1.0, label=data_label)
    # CUPID-style presentation: the gamma and muon fits are INDEPENDENT (each fit
    # only on its own side of the valley), so each is drawn with its own chi2 and
    # no total curve is shown -- a summed curve would double-count in the overlap
    # region where both tails contribute.
    # ... but only when the cut machinery stands behind it: with a reliability guard
    # refusing the cut (fit["reliable"] False), whatever the mixture latched onto below
    # the valley is the threshold turn-on or the MIP line's own flank, not a gamma
    # population (run00270 ch4/ch5: near-delta Gaussians on single noisy bins), and
    # drawing it reads as a claimed population.  Named in the legend; the title names
    # the guard that refused.
    if fit["gamma_params"] is not None:
        if fit.get("reliable", True):
            g = _sum_of_gaussians(centers, *fit["gamma_params"])
            gchi = fit.get("gamma_chi2", np.nan)
            ax.plot(centers, g, "--", color="red", lw=1.8,
                    label=f"gamma fit ({fit['n_gamma']} Gauss, chi2_red={gchi:.2f})")
        else:
            ax.plot([], [], " ", label="gamma fit not drawn (guard refused the cut)")
    mu_fn = _muon_curve(fit)
    if mu_fn is not None:
        shape = "Landau(x)Gauss + tail" if fit.get("tail_params") else "Landau(x)Gauss"
        ax.plot(centers, mu_fn(centers), "-.", color="orange", lw=1.8,
                label=f"muon {shape} (chi2_red={fit['fit_chi2']:.2f})")
    if fit["ok"]:
        cut_kind = fit.get("cut_kind", "valley")
        std = fit.get("cut_std", np.nan)
        # Draw the cut's BOOTSTRAP uncertainty as a band, not just a line.  A bare line reads
        # as an exact boundary; on ch0/ch9 the band is ~10-14% wide (see bootstrap_cut), which
        # is the honest width of the answer and the thing a reader needs to see.
        lab = f"Cut ({cut_kind}) = {fit['cut']:.4g}"
        if np.isfinite(std):
            lab += f" +/- {std:.2g}"
            ax.axvspan(fit["cut"] - std, fit["cut"] + std, color="k", alpha=0.10,
                       label=f"Cut +/- 1 sigma (bootstrap, n={fit.get('n_resamples', 0)})")
        if fit.get("cut_degenerate"):
            lab += "  [degenerate: = valley]"
        ax.axvline(fit["cut"], ls=":", color="k", lw=2, label=lab)
    else:
        # Say so on the figure, rather than just omitting the line and letting the plot
        # look like a normal result that happens to have no cut drawn on it.
        ax.plot([], [], " ", label="NO CUT DERIVED")
    # THE TRIGGER FLOOR.  Everything below this was thrown away before the spectrum existed,
    # so the histogram's low edge is an instrument artefact wherever the floor is close to it.
    # Drawing it is what makes a trigger-truncated spectrum (run00270 ch6/ch7: the "gamma" is
    # a Gaussian fit to this very step) readable as such instead of looking like a population.
    floor = fit.get("trigger_floor", np.nan)
    if np.isfinite(floor) and floor > 0:
        over = fit.get("mip_over_trigger", np.nan)
        ax.axvline(floor, ls="--", color="C2", lw=1.5,
                   label=f"trigger floor = {floor:.4g}"
                         + (f" (MPV = {over:.1f}x it)" if np.isfinite(over) else ""))
        ax.axvspan(min(centers[0], floor), floor, color="C2", alpha=0.07)
    # Scale the y-axis from the DATA counts (not the fit curve): a fit that
    # overshoots near the threshold turn-on otherwise inflates the top limit and
    # leaves the histogram squashed in the lower third of an empty frame.
    ax.set_yscale("log"); ax.set_ylim(0.5, max(int(counts.max()), 1) * 3)
    # Name the guard that actually refused, not "the noise" -- on a trigger-truncated channel
    # the MIP line can be perfectly well resolved in this observable and still support no cut.
    snr, over = fit.get("mip_snr", np.nan), fit.get("mip_over_trigger", np.nan)
    if fit.get("reliable", True):
        unreliable = ""
    elif np.isfinite(over) and over < config.min_mip_over_trigger:
        unreliable = (f"\n[NO CUT: MIP line is only {over:.1f}x the trigger floor -- the "
                      f"trigger removed the gamma population it would be cut against]")
    else:
        unreliable = (f"\n[NO CUT: MIP line only {snr:.1f} sigma over the raw baseline noise "
                      f"-- too dim to trust]")
    ax.set(xlabel="Observable (proportional to energy)", ylabel="Counts/bin",
           title=f"Spectrum ({method['label']}): gamma continuum + muon line" + unreliable)
    ax.legend(loc="upper right"); ax.grid(True, which="both", alpha=0.3)
    _add_adc_axis(ax, method.get("peak_scale"))
    _finish(fig, name or f"spectrum_{method['label'].split()[0].lower()}", config)


def plot_muon_spectrum(method: dict, config: Config, name=None) -> None:
    """Pulse-height spectrum for mostly-muon data: histogram + single
    Landau(x)Gaussian (the MIP line), MPV marked.  No continuum, no cut."""
    fit, s = method["fit"], method["summary"]
    centers, counts = fit["centers"], fit["counts"]
    excl = [t for k, t in (("n_clip_excluded", "rail-clipped"),
            ("n_pileup_excluded", "pileup")) if fit.get(k, 0)]
    data_label = (f"Data (muon-selected, {', '.join(excl)} excluded)" if excl
                  else "Data (muon-selected)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(centers, counts, where="mid", color="C0", lw=1.0, label=data_label)
    lo_x = fit.get("fit_lo_x", float(centers[0]))
    if fit["muon_params"] is not None:
        # Draw the line only over the FITTED range (above the low-side valley cut, if any)
        # so the curve isn't shown claiming to describe the excluded noise population.
        drawn = centers >= lo_x
        # ONE curve: the line that was actually fitted (tail model where the BIC chose it).
        # There is deliberately no "core only" companion curve -- it held the fitted `area`
        # fixed while switching the scale spread off, so it overshot the data ~1.6x and read
        # as a failed fit sitting next to a good one.  The core is not a separate population.
        shape = "Landau(x)Gauss + response tail" if fit.get("tail_params") else "Landau(x)Gauss"
        ax.plot(centers[drawn], _muon_curve(fit)(centers[drawn]), "-", color="C3", lw=1.8,
                label=f"muon {shape} (chi2_red={fit['fit_chi2']:.2f})")
    if fit.get("n_low_excluded", 0):
        ax.axvline(lo_x, ls="-.", color="C1", lw=1.5,
                   label=f"fit start (valley) = {lo_x:.4g}")
    ax.axvline(fit["mpv"], ls="--", color="k", lw=1.5, label=f"MPV = {fit['mpv']:.4g}")
    ax.axvline(s["median_obs"], ls=":", color="gray", lw=1.5, label=f"median = {s['median_obs']:.4g}")
    # Scale the y-axis from the DATA counts (not the Landau curve, which can spike
    # near the turn-on and otherwise leaves the histogram squashed in an empty frame).
    ax.set_yscale("log"); ax.set_ylim(0.5, max(int(counts.max()), 1) * 3)
    ax.set(xlabel="Pulse height (proportional to energy)", ylabel="Counts/bin",
           title=f"Muon pulse-height spectrum ({method['label']})")
    ax.legend(loc="upper right"); ax.grid(True, which="both", alpha=0.3)
    _add_adc_axis(ax, method.get("peak_scale"))
    _finish(fig, name or f"spectrum_muon_{method['label'].split()[0].lower()}", config)


# ===========================================================================
# Quality-control diagnostics (estimator- and physics-agnostic; work on real
# data with no truth -- pulse quality, reconstruction fidelity, timing, range)
# ===========================================================================

def saturation_qc(prep: Prepared, config: Config) -> dict[str, Any]:
    """Linearity / saturation guard on the events entering the analysis.  Works in
    the polarity-corrected space where pulses are positive.  Estimates the clipping
    level (a value where per-event peaks pile up, else the largest peak seen as an
    ADC-ceiling proxy), then flags the genuinely CLIPPED events with
    waveform_triage.detect_saturation -- the same flat-top test (>= `consec`
    consecutive samples within rail_tol of the rail) triage uses -- so both stages
    call an event saturated on identical grounds.  A tall clean pulse that merely
    grazes the rail has no flat top and is NOT flagged.  Separately reports the
    upper, potentially non-linear part of the dynamic range: SiPMs become
    non-linear well below hard saturation, by convention near ~70% of full scale.

    Depends only on `prep` (whose config carries the resolved window), so the result
    is computed once per Prepared and cached: analyze_observable's clip-exclusion and
    the drivers' QC print/plot calls all reuse the same dict."""
    cached = getattr(prep, "_sat_qc", None)
    if cached is not None:
        return cached
    # Only the TRIGGERED events enter the analysis; non-triggered noise records would
    # otherwise pile up at the noise floor and be mistaken for the ADC rail.  prep.corrected
    # already holds exactly those (see triggered()).
    length = prep.corrected.shape[1]
    # Measure each event in the pulse SEARCH window (where the triggered peak sits)
    # rather than over the whole record, so a second pile-up pulse elsewhere in the
    # record cannot be mistaken for this event's peak (and hence for the ADC rail).
    # The flat-top test below runs on this SAME slice, for consistency with peak_h.
    win = triggered(prep)[:, search_region(length, config)]
    peak_h = win.max(axis=1).astype(np.float64)
    # Rail estimate + flat-top clip flags via the shared corrected-space test
    # (_rail_clipped_mask) -- the same grounds _clean_model_set uses to keep
    # clipped events out of the template.
    sat_mask, rail, clipped_pileup = _rail_clipped_mask(win, prep.noise_sigma,
                                                        config.polarity,
                                                        adc_range=prep.adc_range)
    # The non-linearity onset is 70% of the digitizer's FULL SCALE (prep.adc_range), not
    # 70% of the tallest pulse in the file.  With no clipping the old code used the
    # brightest event as its ceiling, which made "% above the onset" a statistical
    # accident of how bright that one event happened to be.
    full_scale = (float(prep.adc_range) if np.isfinite(prep.adc_range) and prep.adc_range > 0
                  else rail)
    onset = 0.70 * full_scale
    result = {"peak_h": peak_h, "rail": rail, "full_scale": full_scale, "onset": onset,
              "clipped_pileup": bool(clipped_pileup),
              "sat_mask": sat_mask,
              "n_at_rail": int(sat_mask.sum()),
              "frac_above_onset": float(np.mean(peak_h > onset)),
              "n_above_onset": int(np.sum(peak_h > onset))}
    prep._sat_qc = result
    return result


def plot_saturation_qc(qc: dict, config: Config, name="8_saturation_qc") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(qc["peak_h"], bins=200, color="C0", alpha=0.8)
    rail_lbl = (f"measured rail = {qc['rail']:.0f}" if qc["clipped_pileup"]
                else f"ADC full scale = {qc['full_scale']:.0f} (no clipping)")
    ax.axvline(qc["rail"], ls="--", color="r", lw=2, label=rail_lbl)
    ax.axvline(qc["onset"], ls=":", color="C1", lw=2,
               label=f"~70% of full scale = {qc['onset']:.0f}")
    ax.set_yscale("log")
    tail = (f"{qc['n_at_rail']} clipped at rail" if qc["clipped_pileup"]
            else "no hard clipping detected")
    ax.set(xlabel="Pulse peak height (corrected ADC)", ylabel="Events",
           title=(f"Saturation / linearity guard  "
                  f"({qc['frac_above_onset']*100:.2f}% above onset, {tail})"))
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    _finish(fig, name, config)


def plot_of_chi2(amplitude: np.ndarray, chi2_red: np.ndarray, config: Config,
                 name="9_of_chi2", peak_scale: float | None = None,
                 clipped: np.ndarray | None = None) -> None:
    """Per-event optimal-filter goodness-of-fit: the reduced-chi2 histogram, and chi2 vs
    amplitude WITH the fitted clean trend chi2_red(A) = c0 + (A/A0)^2 drawn on it.

    The right-hand panel used to be captioned "flat band = clean", which is wrong and reads a
    healthy plot as a defect.  The clean band is NOT flat: any shape mismatch between the
    pulses and the template contributes a residual proportional to the amplitude, so chi2
    rises quadratically (see chi2_amplitude_trend).  What matters is whether the events FOLLOW
    that trend -- so the trend is drawn, and A0 (the template-fidelity scale) is reported."""
    a = np.asarray(amplitude, float); c = np.asarray(chi2_red, float)
    clip = (np.asarray(clipped, bool) if clipped is not None
            else np.zeros(a.shape, dtype=bool))
    ok = np.isfinite(a) & np.isfinite(c)
    a, c, clip = a[ok], c[ok], clip[ok]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    hi = float(np.percentile(c, 99.9)) if c.size else 5.0
    ax0.hist(c, bins=np.linspace(0, max(hi, 2.0), 100), color="C0", alpha=0.8)
    ax0.axvline(1.0, ls="--", color="r", label="chi2_red = 1 (ideal)")
    med = float(np.median(c)) if c.size else np.nan
    ax0.axvline(med, ls=":", color="k", label=f"median = {med:.2f}")
    # log-y: the interesting physics (pile-up/saturation) lives in the sparse
    # high-chi2 tail, which a linear scale crushes against the axis.
    ax0.set_yscale("log")
    ax0.set(xlabel="Reduced chi2", ylabel="Events", title="OF goodness-of-fit")
    ax0.legend(); ax0.grid(True, alpha=0.3)
    c0, a0 = (chi2_amplitude_trend(a, c, clip) if a.size else (np.nan, np.inf))
    title = "OF chi2 vs amplitude"
    if a.size:
        rng = np.random.default_rng(config.seed)
        pick = (rng.choice(a.size, 6000, replace=False) if a.size > 6000 else np.arange(a.size))
        # Draw the CLIPPED events separately: they are a different pulse shape, and they are
        # what makes the high-amplitude arc.  Lumping them in makes the arc look like a
        # reconstruction failure rather than the saturation it is.
        cl = clip[pick]
        ax1.scatter(a[pick][~cl], c[pick][~cl], s=4, alpha=0.2, color="C0", label="events")
        if cl.any():
            ax1.scatter(a[pick][cl], c[pick][cl], s=6, alpha=0.4, color="C1",
                        label="rail-clipped (saturated)")
        # cap the x-range to the bulk: a few high-amplitude outliers otherwise
        # autoscale the linear axis out to ~14 and crush the whole cloud into a
        # narrow vertical stripe on the left, hiding the chi2-vs-amplitude trend.
        # 99.5th pct keeps a bit more of the bright tail in view than the bulk-only 99th.
        ax1.set_xlim(0, max(float(np.percentile(a, 99.5)), np.finfo(float).eps))
        # THE CLEAN TREND.  The band is not flat and is not meant to be: a fractional template
        # mismatch contributes a residual proportional to A, so chi2 = c0 + (A/A0)^2.  Drawing
        # it turns "why does this rise?" into a measurement -- events should FOLLOW this curve;
        # the ones that sit above it are the genuine shape anomalies.
        if np.isfinite(c0):
            xs = np.linspace(0, ax1.get_xlim()[1], 200)
            lab = f"clean trend: {c0:.2f} + (A/{a0:.2f})$^2$" if np.isfinite(a0) else \
                  f"clean level = {c0:.2f} (no trend)"
            ax1.plot(xs, expected_chi2(xs, c0, a0), "-", color="C3", lw=2, label=lab)
            ax1.plot(xs, config.pileup_chi2_factor * expected_chi2(xs, c0, a0), "--",
                     color="C3", lw=1.2, alpha=0.7,
                     label=f"pileup gate ({config.pileup_chi2_factor:g}x trend)")
        if np.isfinite(a0):
            title += f"   [A0 = {a0:.2f}: template fidelity, higher = better]"
    # log-y to spread the upward-scattering anomalies; a log axis needs a
    # positive floor (0 is invalid), so clamp just below the clean band.
    lo = max(float(np.percentile(c, 0.5)), 0.1) if c.size else 0.1
    ax1.set_yscale("log")
    ax1.set_ylim(lo, max(hi, 2.0))
    # a narrow chi2 range can span <1 decade (only 10^1 lands a labelled major
    # tick), leaving the axis nearly blank; label the minor ticks too.
    ax1.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(2, 0.5)))
    ax1.set(xlabel="Reconstructed amplitude", ylabel="Reduced chi2", title=title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    _add_adc_axis(ax1, peak_scale)
    _finish(fig, name, config)


def plot_template_residual(relative: np.ndarray, resid_mean: np.ndarray, resid_rms: np.ndarray,
                           noise_sigma: float, config: Config, name="10_template_residual") -> None:
    """Per-sample residual of the optimal-filter fit (data - A*template) averaged
    over events.  A faithful template gives a mean residual ~0 everywhere and a
    residual RMS at the noise floor; structure at the peak/edge reveals shape
    mismatch, smearing or baseline droop."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(relative, resid_mean, color="C3", lw=1.5, label="mean residual (systematic)")
    ax.fill_between(relative, resid_mean - resid_rms, resid_mean + resid_rms,
                    color="C0", alpha=0.3, label="+/-1 residual RMS (per sample)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(noise_sigma, ls=":", color="k", lw=1, label=f"noise sigma = {noise_sigma:.3g}")
    ax.axhline(-noise_sigma, ls=":", color="k", lw=1)
    ax.set(xlabel="Samples relative to peak", ylabel="Residual (corrected ADC)",
           title="Template-fit residual (data - A*template)")
    ax.legend(); ax.grid(True, alpha=0.3)
    _finish(fig, name, config)


def timewalk_profile(observable: np.ndarray, peak_subsample: np.ndarray,
                     n_bins: int = 12, min_per_bin: int = 20):
    """Binned-median peak-time vs amplitude, plus the slope of a line through those
    medians.  Returns (slope, bin_centers, bin_medians).

    A per-event OLS fit of peak-time on amplitude is dominated by the low-amplitude
    NOISE-TRIGGER cluster: those events have no real pulse, so their reconstructed
    peak scatters across the whole search window at amp~0, acting as high-leverage
    points that inflate and even sign-flip the slope (measured -1.28 vs ~-0.6 for the
    real trend on run00270_ch0).  Taking the MEDIAN peak-time in each amplitude bin
    and fitting those is robust to that cluster, deterministic (unlike a Theil-Sen
    sub-sample, whose value drifts with the draw) and O(n).  Units: samples per
    amplitude unit; ~0 when the sub-sample trigger has removed the walk.  Slope is
    NaN with fewer than two populated bins."""
    a = np.asarray(observable, float)
    t = np.asarray(peak_subsample, float)
    ok = np.isfinite(a) & np.isfinite(t)
    a, t = a[ok], t[ok]
    if a.size <= 10:
        return float("nan"), np.array([]), np.array([])
    lo, hi = np.percentile(a, [1.0, 99.0])
    if not hi > lo:
        return float("nan"), np.array([]), np.array([])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers, meds = [], []
    for i in range(n_bins):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if int(sel.sum()) >= min_per_bin:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            meds.append(float(np.median(t[sel])))
    centers, meds = np.asarray(centers), np.asarray(meds)
    if centers.size < 2:
        return float("nan"), centers, meds
    return float(np.polyfit(centers, meds, 1)[0]), centers, meds


def timewalk_slope(observable: np.ndarray, peak_subsample: np.ndarray) -> float:
    """Residual time-walk QC number: the slope of the binned-median peak-time vs
    amplitude (see timewalk_profile); ~0 when the sub-sample trigger has removed the
    walk.  The single number behind plot_timing's right panel."""
    return timewalk_profile(observable, peak_subsample)[0]


def plot_timing(peak_subsample: np.ndarray, amplitude: np.ndarray, length: int,
                config: Config, name="11_timing", peak_scale: float | None = None) -> None:
    """Trigger-timing diagnostics: the reconstructed sub-sample peak time relative
    to the expected center (should be a tight, centered cluster), and peak time vs
    amplitude.  A non-zero slope of the latter is residual time-walk -- the bias
    the sub-sample optimal-filter peak is meant to remove."""
    center = pulse_center(length, config)
    t = np.asarray(peak_subsample, float) - center
    a = np.asarray(amplitude, float)
    ok = np.isfinite(t) & np.isfinite(a)
    t, a = t[ok], a[ok]
    # Sub-threshold noise triggers (no real pulse) latch onto the search-window
    # edges, scattering their reconstructed time to +/-search_pad.  A plain std is
    # yanked up by that junk tail, so headline the robust core width (1.4826*MAD)
    # and separately report the tail fraction so the QC still exposes the junk.
    t_med = float(np.median(t)) if t.size else np.nan
    robust_sig = 1.4826 * float(np.median(np.abs(t - t_med))) if t.size else np.nan
    tail_cut = max(5.0 * robust_sig, 5.0) if np.isfinite(robust_sig) else np.inf
    tail_frac = float(np.mean(np.abs(t - t_med) > tail_cut)) if t.size else 0.0
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    ax0.hist(t, bins=80, color="C0", alpha=0.8)
    ax0.axvline(t_med, ls="--", color="r",
                label=f"median = {t_med:.3f}, robust sigma = {robust_sig:.3f} samp\n"
                      f"raw std = {np.std(t):.3f} ({tail_frac*100:.1f}% noise-trigger tail)")
    # log-y: the distribution is a sharp core over a ~1%-level noise-trigger tail
    # scattered across the search window; a linear axis shows only the core, hiding
    # exactly the structure (tail fraction, satellite bumps) this QC panel exists for.
    ax0.set_yscale("log"); ax0.set_ylim(bottom=0.5)
    ax0.set(xlabel="Peak time - expected center (samples)", ylabel="Events",
            title="Reconstructed pulse-time distribution")
    ax0.legend(); ax0.grid(True, alpha=0.3)
    if a.size > 10:
        rng = np.random.default_rng(config.seed)
        pick = (rng.choice(a.size, 6000, replace=False) if a.size > 6000 else np.arange(a.size))
        ax1.scatter(a[pick], t[pick], s=4, alpha=0.2, color="C0")
        # cap the x-range to the bulk: a few high-amplitude outliers otherwise
        # autoscale the linear axis out and crush the real amplitude spectrum into
        # a narrow vertical stripe, hiding the time-walk trend (the top ADC axis,
        # a live secondary_xaxis, rescales with it automatically).
        ax1.set_xlim(0, max(float(np.percentile(a, 99)), np.finfo(float).eps))
        # Robust binned-median trend (matches the print_qc time-walk number); a plain
        # OLS line here would be yanked by the amp~0 noise-trigger cluster.
        slope, bc, bm = timewalk_profile(a, t)
        if bc.size:
            ax1.plot(bc, bm, color="red", lw=1.5, marker="o", ms=3, label="binned median")
            if np.isfinite(slope):
                intercept = float(np.mean(bm) - slope * np.mean(bc))
                xs = np.array([a.min(), a.max()])
                ax1.plot(xs, slope * xs + intercept, ls="--", color="red", lw=1.2,
                         label=f"robust time-walk slope = {slope:.2e} samp/amp")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set(xlabel="Reconstructed amplitude", ylabel="Peak time - center (samples)",
            title="Time-walk check (flat = unbiased)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    _add_adc_axis(ax1, peak_scale)
    _finish(fig, name, config)


def amplitude_area_outliers(of_obs: np.ndarray, bx_obs: np.ndarray, n_mad: float = 5.0):
    """Flag events off the amplitude-vs-area line by a robust cut on the ratio
    (median +/- n_mad * MAD).  Off-band events are pile-up, residual saturation or
    shape anomalies.  Returns (ratio, outlier_mask, lo, hi)."""
    of_obs, bx_obs = np.asarray(of_obs, float), np.asarray(bx_obs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = of_obs / bx_obs
    finite = np.isfinite(ratio)
    med = float(np.median(ratio[finite])) if finite.any() else np.nan
    mad = 1.4826 * float(np.median(np.abs(ratio[finite] - med))) if finite.any() else np.nan
    lo, hi = med - n_mad * mad, med + n_mad * mad
    outlier = finite & ((ratio < lo) | (ratio > hi))
    return ratio, outlier, lo, hi


def plot_amplitude_area(of_obs: np.ndarray, bx_obs: np.ndarray, config: Config,
                        name="12_amplitude_area", outliers: tuple | None = None,
                        peak_scale: float | None = None) -> dict[str, Any]:
    """Amplitude (optimal filter) vs area (boxcar) correlation -- tightly linear
    for a single pulse shape.  Off-band events (highlighted) are pile-up / residual
    saturation / shape anomalies, the classic SiPM amplitude-charge quality cut.
    `outliers`: optionally the amplitude_area_outliers tuple a driver already
    computed for print_qc, so it is not recomputed here."""
    of_obs, bx_obs = np.asarray(of_obs, float), np.asarray(bx_obs, float)
    ratio, outlier, lo, hi = (outliers if outliers is not None
                              else amplitude_area_outliers(of_obs, bx_obs))
    ok = np.isfinite(of_obs) & np.isfinite(bx_obs)
    inlier = ok & ~outlier
    rng = np.random.default_rng(config.seed)
    idx = np.flatnonzero(inlier)
    pick = rng.choice(idx, 6000, replace=False) if idx.size > 6000 else idx
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    ax0.scatter(bx_obs[pick], of_obs[pick], s=4, alpha=0.2, color="C0", label="in-band")
    if outlier.any():
        ax0.scatter(bx_obs[outlier], of_obs[outlier], s=10, color="C3",
                    label=f"outliers (n={int(outlier.sum())})")
    ax0.set(xlabel="Area (boxcar)", ylabel="Amplitude (optimal filter)",
            title="Amplitude vs area correlation")
    ax0.legend(); ax0.grid(True, alpha=0.3)
    # Both estimators share the A=1 template scale, so the OF-amplitude (y) axis reads
    # directly in ADC via the template peak; label it so the scatter can be read absolutely.
    _add_adc_axis(ax0, peak_scale, which="y")
    r = ratio[np.isfinite(ratio)]
    rlo, rhi = np.percentile(r, [0.5, 99.5])
    ax1.hist(r, bins=np.linspace(rlo, rhi, 120), color="C0", alpha=0.8)
    ax1.axvline(lo, ls=":", color="C3"); ax1.axvline(hi, ls=":", color="C3",
                label="outlier band")
    ax1.set(xlabel="Amplitude / area ratio", ylabel="Events",
            title="Shape-consistency ratio")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    _finish(fig, name, config)
    return {"n_outliers": int(outlier.sum()), "n_total": int(ok.sum()),
            "outlier_frac": float(outlier.sum() / max(int(ok.sum()), 1))}


def print_qc(resolution: tuple[float, float] | None = None,
             chi2_red: np.ndarray | None = None,
             timewalk_slope: float | None = None,
             amp_area: dict | None = None,
             saturation: dict | None = None,
             chi2_trend: tuple[float, float] | None = None) -> None:
    """Compact console report of the quality-control diagnostics."""
    print("\n" + "-" * 66)
    print("Quality-control diagnostics")
    print("-" * 66)
    if resolution is not None:
        pred, meas = resolution
        ratio = meas / pred if pred else float("nan")
        print(f"  OF amplitude resolution: predicted {pred:.4g}, measured {meas:.4g} "
              f"(meas/pred = {ratio:.2f})")
    if chi2_red is not None and len(chi2_red):
        c = np.asarray(chi2_red, float)
        c = c[np.isfinite(c)]
        print(f"  OF goodness-of-fit:      median chi2_red = {np.median(c):.2f}, "
              f"90th pct = {np.percentile(c, 90):.2f}")
    if chi2_trend is not None:
        c0, a0 = chi2_trend
        if np.isfinite(a0):
            print(f"  OF chi2 vs amplitude:    chi2_red(A) = {c0:.2f} + (A/{a0:.2f})^2  "
                  f"(the band RISES -- this is template mismatch, not a bad fit;")
            print(f"  {'':<24} A0 = {a0:.2f} is the template-fidelity scale, higher = better)")
        else:
            print(f"  OF chi2 vs amplitude:    flat at {c0:.2f} (no amplitude trend)")
    if timewalk_slope is not None:
        print(f"  Timing:                  time-walk slope = {timewalk_slope:.2e} samples/amp")
    if amp_area is not None:
        print(f"  Amplitude-area band:     {amp_area['n_outliers']:,} / {amp_area['n_total']:,} "
              f"off-band ({100*amp_area['outlier_frac']:.2f}%)")
    if saturation is not None:
        clip = (f"{saturation['n_at_rail']} clipped at rail {saturation['rail']:.0f}"
                if saturation["clipped_pileup"] else "no hard clipping")
        print(f"  Saturation/linearity:    {saturation['n_above_onset']:,} above ~70% of the "
              f"{saturation['full_scale']:.0f} ADC full scale "
              f"({100*saturation['frac_above_onset']:.2f}%); {clip}")
    print("-" * 66)


# ===========================================================================
# Extended diagnostics (P2/P3): pulse-shape parameters, noise stationarity,
# spectrum pulls, estimator comparison, template stability, baseline droop.
# All estimator- and truth-agnostic; they run on real data.
# ===========================================================================

def _peak_windows(prep: Prepared, config: Config, cap: int, bright_only: bool = False):
    """Per-event (integer) windows cut on the half-max LEADING-EDGE pick, with every
    pick and the brightness selection estimated on a pickup-NOTCHED guide copy -- the
    same two-track scheme build_template uses -- so these diagnostics view the pulses
    exactly as the template build does (an argmax cut, or an un-notched pick, re-phases
    coherent pickup into the averaged / PCA'd shapes; see build_template).  Optionally
    only bright pulses.  Returns (windows, per-row peak index within the window, guide),
    where `guide` is the notched copy of the same rows -- and IS `windows` itself on a
    channel with no detectable line, so such channels are bit-identical to the old
    behaviour.

    MEASURE SHAPE ON THE GUIDE; AVERAGE/ALIGN THE RAW ROWS (whose shape is never
    filtered) USING SHIFTS ESTIMATED FROM IT.  The notch is not cosmetic: on the analog
    bank the ~9.5-sample pickup rides the slow rising edge and the crest, so the raw peak
    sits on a ripple crest and the raw 10-90 / FWHM / decay crossings read the ripple as
    pulse structure.  Un-notched, that fakes an amplitude-dependent pulse shape on
    channels whose pulses are in fact shape-constant -- measured on run00270, Spearman rho
    raw -> notched: ch1 FWHM -0.29 -> +0.01 and decay -0.39 -> -0.04; ch0 rise -0.22 ->
    -0.05 (the "taller pulses rise faster" trend was the pickup, not slew).  The control is
    the line-free PMT ch9, which does not move (+0.02 / -0.03 / -0.05 raw vs +0.02 / -0.01
    / -0.03 notched), so the notch is not distorting pulses.  What survives on ch1 is a
    residual rise trend (-0.24): the 10-90% crossings of DIM pulses are genuinely
    noise-inflated, which is a property of the measurement at low SNR, not a shape trend --
    read it off the median profile, not the rho.  Same measurement, same reasoning as
    timewalk_report.shape_vs_amplitude, whose docstring has the full trap."""
    pk = prep.events["peak_index"].to_numpy()
    pre, post = config.template_pre, config.template_post
    length = prep.corrected.shape[1]
    sub = triggered(prep)
    # Guide construction copied from build_template: lines detected in the pre-pulse
    # region, notched out of a copy that the picks and the brightness cut run on.
    lines = (_pickup_lines(sub[:, :search_region(length, config).start],
                           config.align_notch_snr)
             if config.align_notch_snr > 0 else [])
    guide = _notch_rows(sub, lines) if lines else sub
    starts = _halfmax_starts(guide, pk, pre)
    width = pre + post
    fits = (starts >= 0) & (starts + width <= length)
    sub, starts, pk = sub[fits], starts[fits], pk[fits]
    take = starts[:, None] + np.arange(width)[None, :]
    rows = np.arange(sub.shape[0])[:, None]
    win = sub[rows, take]
    gwin = guide[fits][rows, take] if lines else win
    pk_in = pk - starts
    if bright_only:
        keep = gwin.max(axis=1) > config.template_sigma * prep.noise_sigma
        if int(keep.sum()) >= 50:
            win, pk_in = win[keep], pk_in[keep]
            gwin = gwin[keep] if lines else win
    if win.shape[0] > cap:
        sel = np.random.default_rng(config.seed).choice(win.shape[0], cap, replace=False)
        win, pk_in = win[sel], pk_in[sel]
        gwin = gwin[sel] if lines else win
    return win, pk_in, gwin


def _cross(y: np.ndarray, level: float, start: int, rising: bool) -> float:
    """Sub-sample index where normalized pulse y first/last crosses `level`."""
    if rising:
        idx = np.flatnonzero(y[:start + 1] >= level)
        if idx.size == 0:
            return np.nan
        i = idx[0]
        if i == 0:
            return 0.0
        y0, y1 = y[i - 1], y[i]
        return (i - 1) + (level - y0) / (y1 - y0) if y1 != y0 else float(i)
    idx = np.flatnonzero(y[start:] >= level)
    if idx.size == 0:
        return np.nan
    i = start + idx[-1]
    if i >= y.size - 1:
        return float(i)
    y0, y1 = y[i], y[i + 1]
    return i + (level - y0) / (y1 - y0) if y1 != y0 else float(i)


def _shape_param_arrays(prep: Prepared, config: Config, cap: int) -> dict[str, np.ndarray]:
    """Per-event peak amplitude plus shape parameters (10-90% rise, FWHM, 1/e decay
    time from the peak), row-aligned, NaN where a crossing is undefined.  The single
    measurement loop behind pulse_shape_params and pulse_shape_vs_amplitude, which
    differ only in how they filter the arrays.

    Measured on the pickup-NOTCHED guide, not the raw window: the raw peak (which
    normalizes every crossing here) rides a ripple crest and the crossings read the
    ripple as pulse structure -- see _peak_windows for the measured size of that
    artifact.  On a line-free channel the guide IS the raw window."""
    _, pks, win = _peak_windows(prep, config, cap)
    amp, rise, fwhm, decay = [], [], [], []
    for w, pk in zip(win, pks):
        m = w.max()
        if m <= 0:
            continue
        pk = int(pk)
        y = w / m
        t10, t90 = _cross(y, 0.1, pk, True), _cross(y, 0.9, pk, True)
        lh, rh = _cross(y, 0.5, pk, True), _cross(y, 0.5, pk, False)
        t1e = _cross(y, 1.0 / math.e, pk, False)
        amp.append(m); rise.append(t90 - t10); fwhm.append(rh - lh); decay.append(t1e - pk)
    return {k: np.asarray(v, float) for k, v in
            (("amp", amp), ("rise", rise), ("fwhm", fwhm), ("decay", decay))}


def _template_shape_ref(template: np.ndarray) -> dict[str, float]:
    """The template's own rise/FWHM/decay (same measurement as the per-event loop);
    the reference lines drawn by plot_shape_params and plot_shape_vs_amplitude."""
    tn = template / template.max()
    pk = int(np.argmax(tn))
    return {"rise": _cross(tn, 0.9, pk, True) - _cross(tn, 0.1, pk, True),
            "fwhm": _cross(tn, 0.5, pk, False) - _cross(tn, 0.5, pk, True),
            "decay": _cross(tn, 1.0 / math.e, pk, False) - pk}


def pulse_shape_params(prep: Prepared, config: Config, cap: int = 8000) -> dict[str, np.ndarray]:
    """Per-event pulse-shape parameters: 10-90% rise time, FWHM, and 1/e decay
    time (samples from the peak to where the tail falls to 1/e).  Bimodality or
    long tails here flag shape populations / pile-up that a single template hides.
    Each parameter is filtered independently (an event whose rise crossing is
    undefined still contributes its decay)."""
    out = _shape_param_arrays(prep, config, cap)
    return {k: v[np.isfinite(v) & (v > 0)] for k, v in out.items() if k != "amp"}


def plot_shape_params(prep: Prepared, config: Config, name="13_shape_params") -> None:
    params = pulse_shape_params(prep, config)
    ref = _template_shape_ref(prep.template)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title in zip(axes, ("rise", "fwhm", "decay"),
                              ("10-90% rise time", "FWHM", "1/e decay time")):
        v = params[key]
        lo = hi = None
        if v.size:
            # Robust x-window (bulk +/- 4 IQR): these params have a heavy tail of
            # noise-inflated crossings (the 99.5th pct can sit ~5x the bulk), which a
            # plain percentile clip still lets stretch the axis into an unreadable
            # spike.  Matches plot_shape_vs_amplitude's window so the two agree.
            q25, q75 = np.percentile(v, [25, 75])
            iqr = max(q75 - q25, 1e-6)
            lo = max(float(v.min()), q25 - 4.0 * iqr)
            hi = min(float(v.max()), q75 + 4.0 * iqr)
            ax.hist(v, bins=np.linspace(lo, hi, 80), color="C0", alpha=0.8)
            ax.axvline(float(np.median(v)), ls="--", color="k", label=f"median = {np.median(v):.1f}")
        ax.axvline(ref[key], ls=":", color="C3", label=f"template = {ref[key]:.1f}")
        ax.set(xlabel="Samples", ylabel="Events", title=title)
        ax.legend(); ax.grid(True, alpha=0.3)
        if lo is not None:      # explicit, so the template marker can't re-stretch the axis
            ax.set_xlim(min(lo, ref[key]), max(hi, ref[key]))
    fig.suptitle("Pulse-shape parameter distributions  (pickup-notched)", fontsize=13)
    fig.tight_layout()
    _finish(fig, name, config)


def pulse_shape_vs_amplitude(prep: Prepared, config: Config, cap: int = 8000) -> dict[str, np.ndarray]:
    """Per-event pulse-shape params (rise / FWHM / 1/e decay) paired ROW-ALIGNED with
    the pulse amplitude, so shape can be correlated with pulse height.

    Amplitude is each window's OWN peak height (ADC, baseline-subtracted, oriented up,
    pickup-notched) -- the very value that normalizes the shape measurement -- so a decay
    time set by the electronics/scintillator time constants shows up as
    amplitude-independent, while one that tracks pulse height shows a trend.  Unlike
    pulse_shape_params (which filters each parameter separately), all four arrays here
    share one finite mask and stay row-aligned for the correlation.

    The notch is what makes the answer trustworthy on the analog bank; without it this
    function reports the pickup ripple as an amplitude-dependent pulse shape (see
    _peak_windows)."""
    out = _shape_param_arrays(prep, config, cap)
    good = np.isfinite(out["amp"]) & (out["amp"] > 0)
    for k in ("rise", "fwhm", "decay"):
        good &= np.isfinite(out[k]) & (out[k] > 0)
    return {k: v[good] for k, v in out.items()}


def plot_shape_vs_amplitude(prep: Prepared, config: Config, name="15_shape_vs_amplitude") -> None:
    """Pulse-shape parameters vs pulse AMPLITUDE -- the direct test of whether the decay
    time (and rise / FWHM) is a fixed detector/electronics constant or scales with pulse
    height.

    Each panel is a 2-D density of one shape parameter against the per-event peak
    amplitude (log x), with the binned MEDIAN PROFILE (red solid) and the template's own
    value (brick-red dashed) overlaid, and the Spearman rank correlation in the title.  A FLAT
    profile / |rho| ~ 0 => the parameter is amplitude-independent (set by the shaping /
    scintillator time constants); a sloped profile / large |rho| => it is amplitude-
    correlated (e.g. amplitude-dependent shaping, or saturation stretching the tail).

    Measured on pickup-notched pulses (_peak_windows).  Read the raw-vs-notched numbers
    there before trusting any rho this plot ever printed un-notched: the ripple alone is
    worth rho ~ -0.2 to -0.4 on the analog channels."""
    d = pulse_shape_vs_amplitude(prep, config)
    amp = d["amp"]
    ref = _template_shape_ref(prep.template)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    if amp.size < 20:
        for ax in axes:
            ax.text(0.5, 0.5, "too few pulses to correlate", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle("Pulse shape vs amplitude", fontsize=13)
        fig.tight_layout(); _finish(fig, name, config); return

    a_lo, a_hi = np.percentile(amp, [0.5, 99.5])
    a_lo = max(a_lo, 1e-6)
    edges = np.logspace(np.log10(a_lo), np.log10(a_hi), 13)      # log-spaced amplitude bins
    centers = np.sqrt(edges[:-1] * edges[1:])
    for ax, key, title in zip(axes, ("rise", "fwhm", "decay"),
                              ("10-90% rise time", "FWHM", "1/e decay time")):
        v = d[key]
        # Robust y-window (bulk +/- 4 IQR): keeps the real spread/trend but crops the
        # thin tail of noise-inflated low-amplitude crossings that would otherwise
        # stretch the axis and squash the dense band.
        q25, q75 = np.percentile(v, [25, 75])
        iqr = max(q75 - q25, 1e-6)
        y_lo = max(float(v.min()), q25 - 4.0 * iqr)
        y_hi = q75 + 4.0 * iqr
        shown = (amp >= a_lo) & (amp <= a_hi) & (v >= y_lo) & (v <= y_hi)
        hb = ax.hexbin(amp[shown], v[shown], gridsize=40, xscale="log", bins="log",
                       cmap="viridis", mincnt=1)
        # Profile over the SAME robust-window events the hexbin shows, so the red line
        # summarizes the displayed bulk and stays in-frame.  Over ALL events the
        # lowest-amplitude bins are dominated by noise-inflated crossings, whose
        # median shoots far above the y-window and the line exits the top-left.
        prof = np.full(centers.size, np.nan)
        for i in range(edges.size - 1):
            sel = (amp >= edges[i]) & (amp < edges[i + 1]) & shown
            if int(sel.sum()) >= 20:
                prof[i] = np.median(v[sel])
        ax.plot(centers, prof, color="red", lw=2.0, marker="o", ms=3, label="median profile")
        ax.axhline(ref[key], ls="--", color="C3", lw=1.5, label=f"template = {ref[key]:.1f}")
        # Spearman rho over ALL paired events (rank correlation: robust to the log axis
        # and to any monotone non-linearity in the trend).
        rho = float(np.corrcoef(rankdata(amp), rankdata(v))[0, 1])
        ax.set(xscale="log", xlabel="Pulse amplitude (ADC)", ylabel="Samples",
               title=f"{title}   (Spearman ρ={rho:+.2f})")
        ax.set_ylim(y_lo, y_hi)
        ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)
        fig.colorbar(hb, ax=ax, label="log₁₀ count")
    fig.suptitle("Pulse-shape parameters vs amplitude, pickup-notched  "
                 "(flat profile = constant; sloped = amplitude-correlated)", fontsize=13)
    fig.tight_layout()
    _finish(fig, name, config)


def plot_noise_stationarity(prep: Prepared, config: Config, name="14_noise_stationarity") -> None:
    """Is the noise stationary across the run and as modelled?  Noise RMS vs event
    index (drift), pre-pulse autocorrelation (correlation length), and the noise
    PSD of the first vs second half of the run (spectral stationarity).  (The
    baseline LEVEL vs event index was dropped: flat to <1 ADC on every run, and
    per-event tilt/level are already covered by 18_baseline_droop.)
    The two half-run PSDs share one normalization, so a taller line in one half is a
    real difference rather than a per-half rescaling artifact."""
    L = config.template_pre + config.template_post
    stop = noise_stop(prep.corrected.shape[1], config)
    rms = np.std(prep.corrected[:, :stop], axis=1)
    nb = min(60, max(rms.size // 100, 5))
    edges = np.linspace(0, rms.size, nb + 1).astype(int)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rms_b = np.array([rms[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])

    cap = min(prep.corrected.shape[0], config.psd_cap)
    seg = prep.corrected[:cap, :stop].astype(np.float64)
    seg = seg - seg.mean(axis=1, keepdims=True)
    f = np.fft.rfft(seg, axis=1)
    ac = np.fft.irfft((f * np.conj(f)).real, n=seg.shape[1], axis=1).mean(axis=0)
    ac = ac / ac[0]
    maxlag = min(60, ac.size - 1)

    n = prep.corrected.shape[0]
    h = n // 2
    psd1 = np.mean(np.abs(np.fft.rfft(prep.corrected[:h, :L], axis=1)) ** 2, axis=0)
    psd2 = np.mean(np.abs(np.fft.rfft(prep.corrected[h:, :L], axis=1)) ** 2, axis=0)
    fr = np.fft.rfftfreq(L)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(centers, rms_b, color="C1")
    axes[0].set(xlabel="Event index", ylabel="Pre-pulse RMS (ADC)",
                title="Noise RMS vs event index")
    axes[1].plot(np.arange(maxlag + 1), ac[:maxlag + 1], color="C2", marker="o", ms=3)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set(xlabel="Lag (samples)", ylabel="Autocorrelation",
                title="Pre-pulse noise autocorrelation")
    # Normalize BOTH halves by the SAME reference so the comparison is fair: a taller
    # spectral line (or a raised floor) in one half is then a real difference, not the
    # artifact of dividing each half by its own max (which makes a half that grew a
    # sharp line look uniformly "lower").
    ref = max(psd1.max(), psd2.max())
    axes[2].semilogy(fr, psd1 / ref, color="C0", label="first half")
    axes[2].semilogy(fr, psd2 / ref, color="C3", alpha=0.7, label="second half")
    axes[2].set(xlabel="Frequency (cycles/sample)", ylabel="PSD (shared normalization)",
                title="Noise PSD stationarity (run halves, common scale)")
    axes[2].legend()
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle("Noise stationarity & correlation", fontsize=13)
    fig.tight_layout()
    _finish(fig, name, config)


def _muon_curve(fit: dict):
    """The fitted MIP line as a callable: the response-tail model when the BIC selected it,
    else the plain Landau(x)Gauss.  EITHER mode's fit may carry a tail -- both fit the line
    through fit_muon_line -- so nothing downstream may assume a bare _langau."""
    tp = fit.get("tail_params")
    if tp:
        return lambda x: _langau_tailmix(x, *tp)
    mp = fit.get("muon_params")
    return (lambda x: _langau(x, *mp)) if mp is not None else None


def _spectrum_model(fit: dict):
    """Reconstruct the fitted spectrum model on the bin centers (works for the
    single muon line and for the gamma + muon pair).  NOTE: in gamma-muon
    mode the two components are fit INDEPENDENTLY on their own sides of the
    valley, so their sum slightly double-counts in the overlap region -- the pull
    there reads a little negative even for good fits."""
    mu = _muon_curve(fit)
    if mu is None:
        return None
    gp = fit.get("gamma_params")
    if gp is not None:
        return lambda x: _sum_of_gaussians(x, *gp) + mu(x)
    return mu


def plot_spectrum_pull(method: dict, config: Config, name=None) -> None:
    """Normalized fit residual (data - model)/sqrt(model) vs pulse height -- shows
    WHERE the spectrum fit misfits rather than only the aggregate chi2."""
    fit = method["fit"]
    model_fn = _spectrum_model(fit)
    if model_fn is None:
        return
    centers, counts = fit["centers"], fit["counts"].astype(float)
    model = model_fn(centers)
    pull = (counts - model) / np.sqrt(np.maximum(model, 1.0))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhspan(-2, 2, color="C0", alpha=0.12); ax.axhspan(-1, 1, color="C0", alpha=0.18)
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(centers, pull, color="C3", lw=0.9)
    ax.set(xlabel="Observable (proportional to energy)", ylabel="(data - model)/sqrt(model)",
           title=f"Spectrum fit pull ({method['label']}; chi2_red={fit['fit_chi2']:.2f})")
    ax.grid(True, alpha=0.3)
    _add_adc_axis(ax, method.get("peak_scale"))
    tag = method["label"].split()[0].lower()
    _finish(fig, name or f"19_spectrum_pull_{tag}", config)


def plot_estimator_comparison(methods: list[dict], config: Config, name="16_estimator_comparison") -> None:
    """Truth-free estimator comparison: each method's spectrum scaled to its own
    MPV and overlaid, so the narrower MIP line (better resolution) is visible
    directly.  The legend reports each method's fractional MIP width."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in methods:
        # Use the clip-excluded set the spectrum was fit on (falls back to the full set),
        # so the overlaid shape and the fractional width reported below describe the SAME
        # population as the fitted MIP line.
        obs = np.asarray(m.get("observable_fit", m["observable"]), float)
        mpv = m["fit"].get("mpv")
        if not mpv or not np.isfinite(mpv):
            continue
        x = obs[np.isfinite(obs)] / mpv
        x = x[(x > 0) & (x < 3)]
        # Name the width for what it IS.  Both modes used to print "width ~ X% MPV" while
        # reporting two different quantities: the model-free 16-84 half-spread in muon mode
        # and the fitted Landau(x)Gauss sigma in gamma-muon mode.
        if "summary" in m:
            width, wname = m["summary"]["spread_frac"], "16-84 half-spread"
        else:
            width = m["fit"]["sigma"] / mpv if mpv > 0 else np.nan
            wname = "Landau(x)Gauss sigma"
        ax.hist(x, bins=np.linspace(0, 3, 160), histtype="step", lw=1.8, color=m["color"],
                density=True, label=f"{m['label']} ({wname} ~ {100*width:.1f}% MPV)")
    ax.axvline(1.0, ls="--", color="gray", label="MPV")
    ax.set(xlabel="Observable / MPV", ylabel="Probability density",
           title="Estimator comparison (spectra scaled to each MPV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    _finish(fig, name, config)


def plot_template_stability(prep: Prepared, config: Config, name="17_template_stability") -> None:
    """Template reproducibility and shape variability.  Left: templates built from
    two disjoint halves of the bright events overlaid (+ their difference) -- they
    should coincide.  Right: the singular-value spectrum of the aligned pulses; a
    dominant first component means one template captures almost all the shape, the
    rest quantifies residual event-to-event variation.

    The windows are sub-sample aligned to the analysis template with the same
    cross-correlation tool build_template uses (align_windows), so the halves and
    the PCA see the pulses exactly as the template build does -- jitter broadening
    would otherwise masquerade as shape variation in both.  That equivalence is why
    the shifts are estimated on the pickup-notched GUIDE and applied to the raw rows
    (align_windows(guide=...)), exactly as build_template does: aligning on the raw
    pulses locks the halves and the PCA to the pickup phase, which would validate an
    alignment path the pipeline stopped using.  The rows that get averaged and
    decomposed are the RAW ones, so the residual variance still contains the real
    pickup structure riding on the tail -- that is a property of the pulses, and the
    template build's own spread (template_std) carries it too."""
    win, _, guide = _peak_windows(prep, config, cap=config.template_cap, bright_only=True)
    aligned, _ = align_windows(win, template=prep.template, iters=1,
                               max_lag=config.align_lag,
                               guide=None if guide is win else guide)
    norm = aligned / aligned.max(axis=1, keepdims=True)
    pk = int(np.argmax(prep.template))
    rel = np.arange(-pk, norm.shape[1] - pk)
    h = norm.shape[0] // 2
    t1, t2 = norm[:h].mean(axis=0), norm[h:].mean(axis=0)
    M = norm - norm.mean(axis=0)
    sv = np.linalg.svd(M, compute_uv=False)
    var = sv ** 2 / np.sum(sv ** 2)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    ax0.plot(rel, t1, color="C0", lw=1.5, label=f"half A (n={h:,})")
    ax0.plot(rel, t2, color="C3", lw=1.0, alpha=0.8, label=f"half B (n={norm.shape[0]-h:,})")
    ax0.plot(rel, t1 - t2, color="C2", lw=1.0, label="A - B (difference)")
    ax0.axhline(0, color="gray", lw=0.8)
    ax0.set_xlim(-config.template_pre, config.template_post)
    ax0.set(xlabel="Samples relative to peak", ylabel="Normalized amplitude",
            title="Split-half template stability")
    ax0.legend(); ax0.grid(True, alpha=0.3)
    k = min(19, var.size)
    ax1.semilogy(np.arange(1, k + 1), var[:k], marker="o", color="C0")
    ax1.set(xlabel="Principal component", ylabel="Variance fraction",
            title=f"Shape PCA (PC1 = {100*var[0]:.1f}% of variation)")
    ax1.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _finish(fig, name, config)


def plot_baseline_droop(prep: Prepared, config: Config, name="18_baseline_droop") -> None:
    """Baseline-shape diagnostic: per-event pre-pulse slope (tilt).  AC coupling or
    droop shows up as a non-zero center; symmetric +/- lobes are the signature of a
    coherent pre-pulse swing (pickup); a one-sided heavy tail is a preceding pulse's
    decay leaking into the pre-pulse window (pileup).  Two panels: the CORE (central
    99%, linear) resolves the near-zero shape that the full range would crush into one
    bar, and a FULL-RANGE log-y view shows EVERY event, so the outlier tail the core
    panel trims is still visible and counted rather than silently dropped.  (The
    median-full-record panel was dropped: the pulse shape / tail return are already
    shown by 1_template and 10_template_residual.)"""
    stop = noise_stop(prep.corrected.shape[1], config)
    pre = prep.corrected[:, :stop].astype(np.float64)
    x = np.arange(stop) - (stop - 1) / 2.0
    slopes = (pre * x).sum(axis=1) / np.sum(x ** 2)
    med = float(np.median(slopes))

    lo, hi = np.percentile(slopes, [0.5, 99.5])
    n_out = int(np.count_nonzero((slopes < lo) | (slopes > hi)))
    smin, smax = float(slopes.min()), float(slopes.max())
    if smax <= smin:                       # degenerate (all slopes equal): give the bins width
        smin, smax = smin - 0.5, smax + 0.5

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    # Core: central 99% on a linear scale resolves the shape near zero (droop offset,
    # symmetric pickup lobes) that the full range crushes into a single bar.
    ax0.hist(slopes, bins=np.linspace(lo, hi, 100), color="C0", alpha=0.8)
    ax0.axvline(0, ls="--", color="gray")
    ax0.axvline(med, ls=":", color="C3", label=f"median = {med:.2e} ADC/sample")
    ax0.set(xlabel="Pre-pulse slope (ADC/sample)", ylabel="Events",
            title=f"Baseline tilt: core (central 99%; {n_out:,} outliers off-panel)")
    ax0.legend(); ax0.grid(True, alpha=0.3)
    # Full range, log y: keeps the core shape AND makes the sparse (often one-sided,
    # pileup-driven) outlier tail visible -- no event is dropped here.
    ax1.hist(slopes, bins=np.linspace(smin, smax, 120), color="C0", alpha=0.8)
    ax1.axvline(0, ls="--", color="gray")
    ax1.axvline(med, ls=":", color="C3")
    ax1.set_yscale("log")
    ax1.set(xlabel="Pre-pulse slope (ADC/sample)", ylabel="Events",
            title="Baseline tilt: full range (log y, all events)")
    ax1.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _finish(fig, name, config)


def plot_waveform_diagnostics(prep: Prepared, config: Config) -> None:
    """Estimator-agnostic extended diagnostics, shared by every driver.

    Noise-stationarity and baseline-droop are QA (they detect drift over the run)
    and always run.  Pulse-shape-parameter and template-stability plots are
    validate-once exploration and run only with full_diagnostics (--full-diagnostics)."""
    plot_noise_stationarity(prep, config)      # QA: noise drift over the run
    plot_baseline_droop(prep, config)          # QA: baseline drift over the run
    if config.full_diagnostics:
        plot_shape_params(prep, config)
        plot_shape_vs_amplitude(prep, config)      # decay/rise/FWHM vs amplitude: constant or correlated?
        plot_template_stability(prep, config)