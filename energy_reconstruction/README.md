# energy_reconstruction

Amplitude ("energy") reconstruction for the scintillator-panel waveform data, built
for real colored noise and real pulses — no injected truth. One shared library
(`mv_pipeline.py`) plus three thin drivers and one standalone QC report.

Run everything with the miniconda python (the bare `python` on this machine has no
h5py): `C:\Users\remys\miniconda3\python.exe`.

## Files

| file | role |
|---|---|
| `mv_pipeline.py` | shared pipeline: loading, noise model, template, trigger, spectrum fits, cut, QC, plots. Library only — no `main`. |
| `optimal_filter.py` | driver: frequency-domain Wiener (optimal-filter) amplitude. |
| `boxcar.py` | driver: colored-noise-optimal top-hat integral amplitude. |
| `compare.py` | **the production driver**: runs both estimators on one preparation and compares them. |
| `timewalk_report.py` | standalone QC report on the residual time-walk (see below — retained deliberately). |
| `tests/test_energy_reconstruction.py` | synthetic end-to-end regression tests (no real data touched). |

## Pipeline stages, in order

Everything below happens inside `P.prepare()` + `analyze()`; the order is load-bearing
and enforced by the code (e.g. `noise_stop` raises if the window leaves no pulse-free
pre-pulse region).

1. **Good-time-interval gate** (`--exclude-hours`, off by default): drops events
   *before* anything else, so an excluded noisy period enters neither the noise model
   nor the spectrum.
2. **Model pass on a bounded prefix** (first 20k kept events): auto polarity
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
     no cut.
   Rail-clipped events are always excluded from the *fit*; `--exclude-pileup`
   optionally gates high-chi2 events with an amplitude-aware threshold.
   Two reliability guards (`min_mip_snr`, `min_mip_over_trigger`) refuse a cut when
   there is no trustworthy MIP line or no room below it for a gamma population.
6. **QC + plots**: resolution closure, per-event chi2 (+ the rising
   `c0 + (A/A0)^2` clean trend — expected, not a defect), time-walk slope,
   amplitude-vs-area band, saturation/linearity, noise stationarity, baseline tilt.

## Production commands

Results land in `energy_reconstruction_results/<mode>_mode/<stem>_compare_results/`
(`--overwrite` keeps one canonical folder per channel; without it a re-run gets a
fresh `_N` suffix).

```powershell
# gamma-muon mode: all raw single channels
foreach ($ch in @("run00270_ch0","run00270_ch1","run00270_ch2","run00270_ch3",
                  "run00270_ch4","run00270_ch5","run00270_ch6","run00270_ch7",
                  "run00270_ch9","run00270_ch10","caen_ch0")) {
  & C:\Users\remys\miniconda3\python.exe compare.py --input "$ch.h5" `
      --save-plots --no-show --overwrite
}
# muon mode: the triage-cleaned PMT exports
& C:\Users\remys\miniconda3\python.exe compare.py --mode muon --save-plots --no-show --overwrite `
    --input ..\preprocessing\preprocessing_results\run00270_ch9_triage_results_2\run00270_ch9_clean.h5
& C:\Users\remys\miniconda3\python.exe compare.py --mode muon --save-plots --no-show --overwrite `
    --input ..\preprocessing\preprocessing_results\run00270_ch10_triage_results\run00270_ch10_clean.h5
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
* **ch6's cuts are landmarks, not measurements**: both estimators' gamma models
  misfit (χ² ≈ 8–10) and re-triggering at 3σ recovers the sub-floor band without
  fixing the misfit — the low region is a genuinely non-Gaussian continuum. The
  OF/boxcar disagreement (0.238 vs 0.458) is that misfit meeting two resolutions;
  quote the OF number only as a landmark.

## Tests

```powershell
& C:\Users\remys\miniconda3\python.exe tests\test_energy_reconstruction.py
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
  amplitude-dependent pulse shape, a second pulse species.
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

```powershell
& C:\Users\remys\miniconda3\python.exe timewalk_report.py --input run00270_ch0.h5 --save-plots --no-show --overwrite
```

Its figures land in `energy_reconstruction_results/<stem>_timewalk_results/`.
