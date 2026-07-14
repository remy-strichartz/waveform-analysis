# timing_stability — the run's time axis, and everything only a clock can see

## Why this topic

Every analysis folder in this workspace treats the run as an unordered bag of
1024-sample windows:

| folder                  | what it covers                                            | time content |
|-------------------------|-----------------------------------------------------------|--------------|
| `file_manipulation`     | MIDAS/CAEN → HDF5, channel extraction, channel diagnostics | recovers the time axis at conversion (below) |
| `preprocessing`         | triage, pulse windows, hodoscope efficiency                | intra-event sample offsets; quotes the dead-time systematic |
| `energy_reconstruction` | OF/boxcar amplitudes, spectra (ADC units)                  | "noise stationarity" vs **event index**, not time; `--exclude-hours` gate |
| `sipm_characterization` | SPE gain, DCR, crosstalk, afterpulsing (bench)             | intra-record dark-pulse intervals only |

Nobody had ever decoded **when** the events happened.  Yet run00270 spans
**50 hours**, and three things that matter for the veto live exclusively on the
time axis:

1. **Livetime / dead time.** A muon arriving during dead time is an unvetoed
   muon; dead time is directly veto inefficiency, and no bound on it existed.
2. **Gain stability.** SiPM gain rides on temperature through the breakdown
   voltage (S13360: dV_br/dT ≈ 54 mV/K, ~2 %/K at 3 V overvoltage).  A two-day
   run in a lab with HVAC cycles is a free stability monitor — if you can put the
   events on a wall clock.
3. **Data integrity.** Rate steps, DAQ dropouts and noise bursts are invisible in
   a bag of windows and obvious on a time axis.

## The two-clock problem (`file_manipulation/clock_recovery.py`)

The raw material was already in the files, carried through by `midas_to_h5.py`
but never read:

* **CAEN trigger time tag (TTT)** — header word 39 of `/headers_DGH0`
  (auto-detected, not hard-coded): latched at each trigger with clock-cycle
  granularity, but a free-running counter modulo 2^30.  Run00270's mean event
  spacing (12.0 s) exceeds the rollover period (9.16 s), so the counter wraps an
  *unknown* number of times between consecutive events — TTT differences alone are
  un-unwrappable (43 % of raw differences are negative).
* **MIDAS wall clock** — absolute unix time per event, but truncated to whole
  seconds.

Neither clock suffices; together they are exact.  For each inter-event gap the
wall clock fixes the integer rollover count,

    k_i = round((Δwall_i · f − d_i) / M),      d_i = ΔTTT_i mod M,

unambiguous because the wall clock's ±1 s truncation is much smaller than half a
rollover period (4.6 s).  The TTT then restores fine time inside the second.

**No datasheet numbers enter.**  The modulus M comes from the data (smallest
bounding power of two, with a coverage check).  The frequency f is found by a
consistency scan over 1–600 MHz using short gaps only (wide basin), then refined
by iterating (resolve wraps → sigma-clipped fit of accumulated ticks vs wall
clock).  This measured **117.18659 MHz** — *not* a frequency anyone would have
guessed from a manual (it is 15/16 × 125 MHz, −7.8 ppm); an assumed 125 MHz would
have failed silently.

**The math lives in `file_manipulation/clock_recovery.py`, not here**, so both ends
of the chain share one implementation: `midas_to_h5.py` and `extract_channels.py`
call it at **conversion time** and write `/event_time_rel_s` into every waveform
file.  The time axis is therefore simply *present* — no downstream tool re-derives
the clock, and a separate times file can no longer fall out of row-alignment with
its waveforms.  (Older files without it are fixed by re-extracting the channel;
the backfilled axis is bit-identical to what `event_times.py` produces.)  A file
whose header bank holds no usable counter converts exactly as before, without the
dataset.

**Closure:** `event_times.py --selftest` runs the identical code path on synthetic
clocks with known truth (exponential arrivals, non-round 98.7654 MHz clock, a 2-h
DAQ dropout, latency jitter with 1.5-s stalls): recovered f to +0.17 ppm,
worst-case relative time 7 ms over 22 h, zero wrap failures.  Accuracy, stated
precisely: tick counts between events are *exact*; conversion to seconds is good to
the measured-f precision (~ppm — tens of ms across two days, microseconds across
minutes).

## Two programs, split by what the measurement is *of*

The rate, dead time and integrity numbers read **only the times**, so they are
properties of the **run** — identical for every channel of it.  Gain, baseline and
noise are properties of a **channel**.  The two are measured by different programs
and the run-level ones are measured **once**.

### `event_times.py` — run level

Recovers the axis, writes `<stem>_times.h5` + `<stem>_timing.json`, and measures
(on **all DAQ triggers** — the hardware trigger defines veto livetime, not an
offline pulse finder):

* **Rate & integrity**: binned rate with Poisson errors and χ²; arrival-time
  uniformity; a gap census against exponential order statistics (a dropout is a gap
  with expectation ≪ 1, not merely the biggest gap in the run).
* **Arrival statistics & dead time**: intervals vs the exponential law and lag-1
  rank correlation (bursts / retriggering).  KS p-values are **Monte-Carlo** (the
  rate is estimated from the same sample, so textbook tables would be
  anti-conservative) and floored at 1/n_sim rather than printed as an exact 0.
  Dead time removes short intervals entirely, so the smallest observed interval is
  a *hard* upper bound, quoted next to the run's sensitivity floor 3/(nλ).

Figures: `time_1_clock.png` (frequency-scan basin + zoom, fine-minus-wall residuals,
wrap margins) and `time_2_rate.png` (cumulative uniformity, binned rate, dead-time
search).

### `run_stability.py` — per channel

The pulse-height scale per time block from the house optimal-filter amplitude
(`mv_pipeline` + `optimal_filter` — the same estimator as the energy analyses),
tracked by **gain-equivariant quantiles** Q25/Q50/Q75 with bootstrap errors and no
event selection: a pure gain change scales every quantile identically, so common
motion = gain drift, divergence = a population/shape change — **no spectral model
assumed**, which matters because the paddle spectrum is broad (position-dependent
light collection), not a clean Landau line.  Plus per-block baseline, noise σ, and
the offline-trigger fraction.  Figure: `stab_1_gain.png`.

It also **defines the run's bad periods**: blocks whose noise σ exceeds
`--bad-noise-factor` (default 1.5) × the run median, merged into time windows,
written to the JSON as `bad_windows_h`, and printed as the literal `--exclude-hours`
line to paste into an energy analysis.

## Results for run00270

* **Clock**: 117.186590 MHz, modulus 2^30, 13,133 rollovers, **0 failed wraps**.
* **Livetime > 99.972 %** — dead time < 3.3 ms (hard bound; sensitivity floor
  2.4 ms).  Intervals are exponential (MC-KS p = 0.28), lag-1 ρ = +0.014: no bursts,
  no retriggering.  `hodoscope_efficiency` now quotes this: the dead-time systematic
  is < 0.028 % against a 0.63 % statistical error — **negligible, no correction
  warranted**, which is worth stating rather than leaving unexamined.
* **No DAQ dropouts**: largest gap 126 s (0.42 such gaps expected — an ordinary
  Poisson extreme).
* **The rate is not stationary** (uniformity p ≈ 0.003, χ² p = 0.006): a late-run
  excess, diffuse rather than a clean step.
* **A late-run disturbance, and what it does — measured, not assumed.**  On **ch10**
  the noise σ climbs from ~2.8 to 11.7 ADC over the last ~6.6 h (2.45× the run
  median), and the window is flagged automatically (43.35–49.98 h).  Inside it,
  **11.3 % of events carry no pulse at all vs 3.9 % outside** — the disturbance is
  adding junk triggers, not changing the muons.  ch0's noise wanders all run
  (11.4–20.6 ADC, max/median = 1.40) without a clean excursion, and ch9's moves by
  10 %; **neither is flagged, and neither should be** — there is no window to cut.

### The bias this actually causes (and the one it does not)

`mv_pipeline` builds **one** PSD, template and trigger threshold for the whole run.
A noisy window therefore inflates the response-noise scale that *every* event is
thresholded against.  Excluding ch10's bad window with `--exclude-hours 43.35 49.98`
lowers that threshold and **recovers 428 good-period pulses the contaminated model
had been rejecting** (12,350 → 12,778 triggered out of the same 12,853 events, +3.5 %),
preferentially at low amplitude — i.e. a real, quantified, spectrum-shape selection
bias.

It does **not**, however, fix ch10's resolution: the 16–84 spread stays at ~125–133 %
of the MPV either way.  **ch10's poor resolution is intrinsic (low light), not caused
by the disturbance** — a hypothesis worth testing and now settled in the negative,
which is the point of having the gate.

## What was cut, and why

The pipeline used to emit 7 figures per channel.  It emits 3 (one run-level clock QC,
one run-level rate/dead-time, one per-channel gain), because the rest were showing
nulls, duplicates, or confounds:

* **Rate + interval figures, per channel → run level.**  They read only the times, so
  three channels produced three *byte-identical* copies of one measurement, inviting
  them to be read as three confirmations of it.
* **The tag-structure figure** (raw tag vs event index, tag-phase uniformity KS,
  raw-difference histogram).  It illustrated *why* unwrapping is hard but could not
  catch a failure: a wrong TTT word does not survive the frequency-scan basin, which
  is the check with teeth.
* **The lag-1 interval scatter.**  ρ = +0.014 on every channel; 15k points of noise
  cannot show a null better than the number does (still quoted).
* **The interval histogram.**  Same distribution, same exponential overlay, as the
  dead-time cumulative that sits beside it and *also* carries the bound.
* **The correlations figure** (block gain vs block baseline; block gain vs block rate).
  Both were 15-point scatters of series already plotted against time, and "gain vs
  rate ρ = −0.52 (p = 0.04)" is confounded by construction: a disturbance that moves
  both makes them correlate.
* **The diurnal fold.**  ~2 cycles in a 50-h run cannot support it — and worse, a
  single localized disturbance necessarily folds onto particular hours-of-day and
  *manufactures* a diurnal signal.  A confidently-wrong estimator is the failure mode
  this workspace exists to avoid.
* **The Landau⊗Gauss MPV cross-check.**  Valid on every channel tried and merely
  reproduced the quantiles, at the cost of the heaviest fit path here (a guarded Landau
  fit per block per bootstrap).  The quantiles assume no spectral model, which is the
  stronger position on a broad spectrum; `mv_pipeline` still owns the Landau fit where
  it belongs, in the physics.

**Kept but re-reported:** baseline drift is now quoted as a *fraction of the pulse
scale* next to its p-value.  With ~10³ events per block the constant-fit p resolves
sub-ADC wobbles (ch0: p = 3e-7 — for a 0.55 ADC drift that is **0.037 % of the pulse
scale**, and `mv_pipeline` subtracts the baseline per event anyway).  A significant
p on a physically irrelevant drift must not read as a finding.

## Running it

```
python event_times.py --selftest --no-show                                   # closure test
python event_times.py --input run00270_ch9.h5 --save-plots                   # run level: once per run
python run_stability.py --input run00270_ch10.h5 --save-plots --polarity negative   # per channel
python run_stability.py --input run00270_ch0.h5 --save-plots --bad-noise-factor 1.35  # loosen the flag
```

`run_stability` recovers the time axis itself if `<stem>_times.h5` is missing (same
computation, minus the clock QC plots and the run-level analysis).  For older
conversions lacking `/event_time_unix`, pass `--mid run00270.mid`.  Both drivers take
`--overwrite` to refresh a channel's results folder in place; without it a re-run gets
a fresh `_N` folder beside the old one.

To act on a flagged window, in **any** energy_reconstruction driver:

```
python compare.py --input run00270_ch10.h5 --exclude-hours 43.35 49.98
```

Excluded events are dropped from the **noise model** (PSD, template, trigger threshold)
as well as the spectrum — the whole point is to keep a noisy window out of the model that
every other event is judged against.  The gate needs a time axis: `/event_time_rel_s` in
the input (there by default since the converter change), else `--times`.

Outputs: `time_1_clock.png`, `time_2_rate.png`, `<stem>_timing.json` in
`timing_stability_results/times/<stem>_times_results[_N]/`; `stab_1_gain.png`,
`<stem>_stability.json` (with `bad_windows_h`) in
`timing_stability_results/stability/<stem>_stability_results[_N]/`.
