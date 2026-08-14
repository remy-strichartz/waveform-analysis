# energy_reconstruction

Amplitude ("energy") reconstruction for the scintillator-panel waveform data, built
for real colored noise and real pulses — no injected truth. One shared library
(`mv_pipeline.py`) plus three thin drivers and one standalone QC report.

Dependencies: `pip install -r requirements.txt` from the repo root. scipy >= 1.16 is a hard
floor — `scipy.stats.landau` is what fits the MIP line. (On the analysis machine the deps
live in miniconda rather than the bare `python` on PATH; see the root README.)

## Files

| file | role |
|---|---|
| `mv_pipeline.py` | shared pipeline: loading, noise model, template, trigger, spectrum fits, cut, QC, plots. Library only — no `main`. |
| `optimal_filter.py` | driver: frequency-domain Wiener (optimal-filter) amplitude. |
| `boxcar.py` | driver: colored-noise-optimal top-hat integral amplitude. |
| `compare.py` | **the production driver**: runs both estimators on one preparation and compares them. |
| `timewalk_report.py` | standalone QC report on the residual time-walk (see below — retained deliberately). |
| `event_scanner.py` | interactive per-event browser: step through events on the OF chi2-vs-amplitude or time-walk plane (pileup-gated / off-time / all selections) and see each waveform with its A·template fit. `--export N` for headless PNGs. |
| `run_batch.py` | **the batch driver**: runs the whole canonical sweep and writes its record (see below). |
| `make_summary.py` | rebuilds `final_run_summary.json` from the batch logs; called by `run_batch.py`. |
| `tests/test_energy_reconstruction.py` | synthetic end-to-end regression tests (no real data touched). |

## Pipeline stages, in order

Everything below happens inside `P.prepare()` + `analyze()`; the order is load-bearing
and enforced by the code (e.g. `noise_stop` raises if the window leaves no pulse-free
pre-pulse region).

1. **Good-time-interval gate** (`--exclude-hours`, off by default): drops events
   *before* anything else, so an excluded noisy period enters neither the noise model
   nor the spectrum.
2. **Model pass on a bounded, run-spanning subsample** (up to 20k kept events; when
   the file is larger, contiguous slabs spread evenly across the run — a run-average
   model, never an early-run one): auto polarity
   (`polarity_vote`) → auto pulse window (`estimate_window_params`) → robust noise
   sigma → per-event baseline subtraction → ADC dynamic range → triage-cleaned
   template set (NOISE/PILEUP/rail-clipped dropped **for the template only**) →
   pickup-notch-guided, half-max-aligned template (`build_template`) → burst-trimmed
   noise PSD + matched-filter kernel + response-noise scale (`noise_event_mask`,
   PSD and trigger threshold deliberately use the **full** prefix population) →
   trigger floor in observable units.
3. **Apply pass, streamed**: every event is baseline-subtracted and triggered with the
   fixed kernel/threshold; only triggered events are kept
   (`events["original_index"]` maps each back to its source-file row).
4. **Amplitude estimation**: optimal filter (sub-sample-aligned windows, evaluated at
   the trigger's own time — no maximization bias) and boxcar (trapezoidal integral,
   template-integral normalized so both share the template=1 amplitude scale).
5. **Spectrum fit** (`--mode`):
   * `gamma-muon` (default, muon-tagged beam): independent CUPID-style fits — BIC
     Gaussian-mixture gamma below the valley (fit through the overlap with the muon
     line as a fixed component), `fit_muon_line` (Landau⊗Gauss, BIC-upgraded response
     tail) above — then the **equal-area cut** between them, with a bootstrap error
     bar (`--cut-bootstrap`, default 20).
   * `muon` (triage-cleaned / hodoscope-selected data): the same MIP line, no gamma,
     no cut. The MPV carries its own bootstrap error bar (`bootstrap_mpv`, same
     `--cut-bootstrap` knob; ~1.2% at run00270 statistics — a percent-level "shift"
     between runs is usually this noise).

   Before either fit, `--gain-correct` (opt-in) removes the measured per-block gain
   drift from the observable: block medians (gain-equivariant, `run_stability`'s
   estimator) rescale events to the run-average gain, with refusals when the drift is
   consistent with constant or the motion is a shape change rather than a gain change
   (see `gain_correction`). It needs the `/event_time_rel_s` axis and leaves behind a
   quoted residual gain systematic — an analysis choice for the provenance line, like
   `--mode`.
   Rail-clipped events are always excluded from the *fit*; `--exclude-pileup`
   optionally gates high-chi2 events with an amplitude-aware threshold.
   Two reliability guards (`min_mip_snr`, `min_mip_over_trigger`) refuse a cut when
   there is no trustworthy MIP line or no room below it for a gamma population.
6. **QC + plots**: resolution closure, per-event chi2 (+ the rising
   `c0 + (A/A0)^2` clean trend — expected, not a defect), time-walk slope,
   amplitude-vs-area band, saturation/linearity, noise stationarity, baseline tilt.

**Two invariants the diagnostics must hold to** (both were violated and fixed
2026-07-14; a fix to an estimator has to land with the diagnostic that watches it):

* **A diagnostic must describe the population the thing it validates was built
  from.** The resolution closure compares `sigma_pred` (from the burst-**trimmed**
  PSD) against the measured spread, so the spread is measured over that same
  `noise_event_mask` population. Measuring it over every event with a plain `np.std`
  — itself outlier-dominated — reported the ~1% interference bursts as a broken
  noise model: ch0 read **meas/pred = 1.51** where the honest answer is **0.99**.
* **Anything that measures pulse SHAPE or crest timing on the analog bank must
  notch the pickup first.** `_peak_windows` (plots 13/15/17) now estimates its
  picks and brightness cut on a notched guide and measures shape on it, the same
  two-track scheme `build_template` uses. Un-notched it read the ~9.5-sample ripple
  as pulse structure and faked an amplitude-dependent shape: ch1 FWHM Spearman
  ρ −0.29 → **+0.01**, ch1 decay −0.39 → **−0.04**, ch0 rise −0.22 → **−0.05** (the
  old "taller pulses rise faster / bandwidth-slew" reading was the pickup). The
  line-free PMT ch9 is the control and does not move. See
  `timewalk_report.shape_vs_amplitude` for the full trap.

## Production commands

Results land in `energy_reconstruction_results/<mode>_mode/<stem>_compare_results/`
(`--overwrite` keeps one canonical folder per channel; without it a re-run gets a
fresh `_N` suffix).

**Re-run the whole thing with `run_batch.py`** — 10 gamma-muon compares, three
muon-mode compares (hodoscope-tagged caen_ch0 plus the two triage-cleaned PMT
exports; `--mode` is provenance, and theirs says the gamma population is already
gone), 11 timewalk reports (~80 min):

```bash
python run_batch.py            # the canonical sweep
python run_batch.py --dry-run  # list jobs, check inputs
```

It writes the *record*, not just the plots: `logs/<job>.log`, `logs/manifest.txt`
(commands, exit codes, wall times, library + git versions), `logs/COMPLETE.txt` — which
exists **only if all 24 jobs exited 0**, so treat its absence as "the batch did not
finish" — and `final_run_summary.json`, rebuilt from those logs by `make_summary.py`.

**Prefer it to a hand-run `compare.py` after any change that moves a printed QC
number.** The summary and the manifest are *derived* artifacts that no pipeline stage
writes, so a bare re-run refreshes the figures and leaves the numbers of record stale
beside them — which is exactly what happened after the closure fix below: the plots were
current while `final_run_summary.json` still quoted the pre-fix `meas/pred = 1.51`.
Batching the sweep and the summary into one command is what stops them separating.

Individual channels, if you need one:

```bash
python compare.py --input run00270_ch0.h5 --save-plots --no-show --overwrite

# The triage-cleaned exports are waveform-qc's output, and they are .h5 -- gitignored, so
# they ship with neither repo and exist only where triage was run.  $TRIAGE_RESULTS points
# at that tree; it is the same variable run_batch.py reads, which does this for you.
# (PowerShell: $env:TRIAGE_RESULTS)
python compare.py --mode muon --save-plots --no-show --overwrite \
    --input "$TRIAGE_RESULTS/run00270_ch9_triage_results/run00270_ch9_clean.h5"
```

`--mode` is **provenance, not shape**: whether a gamma/muon cut is meaningful is a
fact about the input selection (was the gamma population already removed upstream?),
and it is *measured* that the spectrum alone cannot decide it — so say what the data
is. `ch4` and `ch7` correctly refuse a cut (trigger-truncated); expect the guards to
say so.

Quoting rules established by measurement (2026-07-14):

* **ch9's MPV**: quote the ch9_clean muon-mode value only. The raw gamma-muon fit
  sits ~15% low (χ²=2.07 vs 0.99 clean) and `--exclude-pileup` does *not* reconcile
  it (0.543 → 0.533) — the raw line is fit-range/tail-model dominated, and its "cut"
  is a THIN-flagged threshold landmark.
* **ch4–ch7 are resolved-photoelectron spectra** (2026-07-14): these low-light
  SiPM channels resolve single PE, so the *whole* spectrum — including the MIP
  peak — is a periodic comb at the PE spacing (ch6: 0.2 of the MIP scale, i.e.
  ~5 PE per MIP). Their dimness is geometry, not defect (2026-07-15,
  arXiv:2505.06129): they are the mini-modules of the panel farthest from the
  trigger footprint at the ch0 corner (~30 cm effective attenuation length,
  18× brightness span ch0→ch7). The old "gamma" fits on these channels were fitting PE teeth,
  and the valley walk stopped in an inter-PE dip (ch6 boxcar "valley" 0.499 with
  the MPV at 0.59 — the Landau was fit on the top half of its own peak).
  `spectrum_landmarks` now detects the comb (`_comb_period`, autocorrelation
  dip-then-rebound, `comb_rho_min`) and finds landmarks on its envelope; clean
  channels are measured bit-identical. ch6's old cuts (OF 0.238 / boxcar 0.458)
  were comb artifacts and are gone: the OF now refuses, the boxcar cut is the
  pedestal threshold. A Landau over a resolved-PE comb honestly fits at
  χ² ≈ 5–13 — that is the comb, not a fit problem.

### `--template-trim` A/B (2026-07-14, trim 0.1 vs 0, all 13 datasets)

Uniform fidelity gains, zero physics cost, one bookkeeping cost:

* residual time-walk **down 10–30%** on every analog channel (caen −70%);
* template-fidelity scale A0 **up 4–28%** on 12 of 13; caen median χ² −19%;
* resolution closure, line-fit χ², and the **absolute** (gain-tracking) MPV
  unchanged (155.2→155.1 / 165.1→165.2 ADC on the clean PMTs);
* **but** every template-*relative* number (MPV_obs, cuts, trigger floor) shifts
  ~+4…+37%, because trimming clips the right-skewed amplitude mixture and lowers
  the template's own scale — a unit change, not a physics change.

Verdict: a real improvement whose adoption as *default* re-bases every canonical
template-relative number. Left **opt-in** deliberately; adopt at a natural
analysis breakpoint (one commit + one full batch re-run), not mid-comparison.

### Modelling muon mode's sub-MIP pedestal instead of truncating it — REFUTED (2026-07-30)

The proposal, and it is a reasonable one: muon mode *truncates* the sub-MIP pedestal
out of the line fit (`muon_fit_lo_pct` = 0.5, or the valley when the landmark walk
finds one), where gamma-muon mode *models* its low population. A nuisance component
under the MIP peak biases the MPV whatever it is physically made of — gamma, EM or
digitiser noise — so the pedestal should be modelled and the MPV allowed to respond.
Provenance decides whether a *cut* means anything; it should not decide the *MPV*.

Built and measured: a bounded Gaussian mixture on `[trigger_floor, truncation point]`,
extended up through the overlap (`_gamma_fit_top`), means bounded by the truncation
point and widths by `gamma_sigma_containment` — i.e. exactly the gamma fit's machinery
minus the cut — then the line refit over the full range with that pedestal fixed and
its tail-model choice inherited so any MPV shift is attributable to the pedestal alone.

**It does not work. Three regimes, and it loses in all three.**

*Truth closure* (synthetic Landau⊗Gauss, true MPV 0.68502, censored at a trigger floor,
3 seeds; the pedestal is generated **as a Gaussian**, so this test is rigged in the
model's favour). MPV error, truncated vs modelled:

| scenario | truncated | modelled | pedestal fraction fitted (true) |
|---|---|---|---|
| `overlap` — 24% separable low population, tail into the turn-on | +0.55 / −0.07 / +0.66 % | +0.28 / +0.55 / +0.63 % | 0.236–0.241 (0.244) ✓ |
| `clean` — muons only | +0.36 / +0.72 / +0.66 % | +0.67 / +0.04 / +0.53 % | 0.000 (0.000) ✓ |
| `censored` — pedestal below the floor, no overlap | +0.76 / +0.46 / +1.08 % | +0.58 / +0.42 / +0.62 % | 0.042 (0.043) ✓ |
| `merged` — 33%, **no valley**, 4.5k events under the line | **−10.33 / −10.19 / −10.61 %** | **identical — declined to engage** | — |

The mixture recovers the true pedestal fraction every time and never invents a
population on clean data, so it is *correct*; the MPV differences in the first three
rows are ±0.7%, random-signed — noise, not a gain.

*Real data* (the three muon-mode channels of record, both estimators):

| channel | truncated region | MPV shift | pedestal | χ² on `[trunc_lo, end]` |
|---|---|---|---|---|
| `caen_ch0` OF | valley, 28 ev (3.9%) | −0.60% | k=1, 2.8% | 0.405 → 0.406 |
| `caen_ch0` boxcar | valley, 26 ev (3.7%) | +1.09% | k=1, 3.5% | 0.444 → 0.430 |
| `ch9_clean` OF | 0.5 pct, 76 ev (0.5%) | **−10.27%** | k=1, **0.00%** | 0.993 → **1.755** |
| `ch9_clean` boxcar | 0.5 pct, 88 ev (0.6%) | **−8.65%** | k=1, **0.00%** | 0.926 → **1.434** |
| `ch10_clean` OF | valley, 2501 ev (20%) | −0.25% | k=2, 15.4% | 0.843 → 0.862 |
| `ch10_clean` boxcar | valley, 2433 ev (19%) | −0.05% | k=2, 15.4% | 1.026 → 1.045 |

It also pushes the two estimators *apart* on `caen_ch0` (5.3% → 7.0%), and never
improves the line's own χ² on the range the truncated fit already owned.

**The mechanism, measured.** Everything turns on whether the landmark walk found a
valley, i.e. whether the low counts are a *population* or a *smear*:

* **Valley present** (`caen_ch0`, `ch10_clean`, synthetic `overlap`/`censored`) — the
  mixture fits the population properly and the MPV moves ≤1%. The truncated fit was
  already unbiased, because the Landau's turn-on above the valley has plenty of data
  to pin the MPV and the pedestal's density there is small next to the line.
* **No valley** (`ch9_clean`) — the truncated region is *11 bins* holding
  `[7, 9, 6, 1, 2, 1, 0, 2, 3, 8, 21]`: a sparse, non-monotone smear with a hole in
  it, not a Gaussian anything. The bounded mixture fits it to amplitude ≈ 0 (fraction
  0.0000), and the full-range refit then has to absorb 11 bins where the Landau
  predicts ~0 counts — so the line is dragged down 9–10%. **This is precisely the
  failure `muon_fit_lo_pct` exists to prevent, reintroduced.** The truncation is not a
  crude stand-in for a pedestal model; it is protecting against something no bounded
  mixture can absorb.
* **Genuinely merged** (synthetic `merged`) — here truncation *is* badly biased
  (−10.2 to −10.6%, and the user's concern is real), and the method cannot help:
  with no valley, the mean bound sits at the trigger floor, so there is nothing for a
  low-side component to grip and the machinery declines on all three seeds.

**The fundamental trap.** Leave the mixture unbounded and a wide Gaussian reaches
under the MIP peak, degenerate with the Landau's own turn-on (the failure
`_gamma_fit_top` documents). Bound it, and it cannot reach the contamination that
actually matters. The overlapping fraction is **not identifiable from the spectrum
shape**: the part of the low population that overlaps the line is not separately
visible, and the part that *is* visible is the sparse sub-MIP smear. This is why
upstream selection (hodoscope tag, triage) beats fitting — it is independent
information, and a Gaussian extrapolation is not.

Verdict: **reverted, not shipped opt-in.** A gated version (accept the pedestal only
where a valley exists) is possible and would be harmless, but it would buy nothing —
that is exactly the regime already measured at ≤1%. Do not re-chase this. Harness:
`ab_pedestal.py` / `closure_pedestal.py` in the session scratchpad.

## Tests

```bash
python tests\test_energy_reconstruction.py
```

Synthetic end-to-end regression suite (trigger completeness, row alignment through
the streaming and GTI paths, OF amplitude proportionality + resolution closure,
**time-walk closure** — the estimator must not invent a walk on walk-free events —
chi2 normalization, two-population cut placement checked against truth, muon-mode /
gamma-muon MPV agreement, boxcar/OF scale agreement, determinism, loud failure on
invalid windows/inputs). All randomness in the pipeline is seeded (`Config.seed`);
two runs of the same command are bit-identical.

## timewalk_report.py — status (2026-07-14)

Retained **deliberately**; do not delete it as cleanup. It answers five questions
about the residual OF time-walk on run00270's analog bank, and the headline one is
still open:

* **Answered**: the estimator does not invent the walk (CLOSURE — now also protected
  by `tests/test_energy_reconstruction.py::test_timewalk_closure`); the shipped
  slope's trend shape and binning stability are characterized per channel; the pulse
  shape is amplitude-independent once the pickup is notched (the un-notched
  measurement is a known trap — read the docstring before measuring shape/crest
  timing on analog channels).
* **Killed hypotheses** (do not re-propose without new evidence): coherent pickup,
  amplitude-dependent pulse shape, a second pulse species, and (2026-07-15) light
  transit / position — the panel geometry is real (fiber-cluster layout confirmed
  from cross-channel amplitude correlations), but at fixed amplitude the OF peak
  time does not depend on a validated position proxy, and 94% of triggered muons
  land on ch0's cell, so the position lever (~5–10 cm ≈ 0.1 samples of transit) is
  20–40× too small for the walk anyway.
* **Also killed (2026-07-14)**: template *quality* as the cause of the template-swap
  dependence — noise level, smearing, subset size, bulk width, and band
  contamination are all measured and out. What remains of the swap effect: a
  minority-tail broadening of the bright-band *mean* template (~10% of the split;
  the median of the same rows is narrow) plus a fine-structure difference in the
  **early tail** (+3…+60 samples past the peak, ~1.7% of peak at ~+18) between
  dim- and bright-built templates. The SHAPE panel now also measures FWHM and the
  decay-side half-widths (notched) to keep this visible.
* **Also killed (2026-07-14, later)**: the early-tail structure as the walk's
  *driver* — band-stacked early-tail trends across 8 channels do not correlate
  with the shipped walk (Pearson −0.34, Spearman −0.26, inconsistent signs).
* **Open**: the *cause* of the real, small (−2.3…−3.7 samples over each channel's
  amplitude range) walk is unknown. Every bulk template property has been matched
  or refuted; the dim/bright template difference that moves the walk is real but
  unidentified. Known gap for the next attempt: the closure test certifies the
  estimator on *Gaussian* colored noise only — a real-noise (burst / pickup-phase)
  interaction with template fine structure would evade it. An opt-in
  `--template-trim` exists for the separately-measured mean-broadening artifact
  (A/B'd; see git history).
  The walk is harmless for amplitude reconstruction (well under the timing jitter,
  and the OF amplitude is evaluated at the trigger's own time), so no correction is
  applied in the pipeline — `timewalk_slope` is a QC number, not a calibration.

```bash
python timewalk_report.py --input run00270_ch0.h5 --save-plots --no-show
```

A channel's whole output here is ONE figure, so a run's channels share ONE folder —
`energy_reconstruction_results/timewalk/<dataset>_timewalk_results/`, holding
`<stem>_timewalk_report.png` per channel (`run00270_timewalk_results/` holds all eleven,
`run00270_ch7_clean` included). Any other dataset gets its own folder under its own name,
automatically. Unlike the compare folders this one is never `_N` versioned — it
accumulates, and re-running a channel replaces that channel's plot (`--overwrite` is
therefore a no-op for this program). It has no analysis mode, so it is grouped in its own
`timewalk/` subfolder rather than under a `<mode>_mode/` one (the same convention as
`preprocessing_results/triage/`).
