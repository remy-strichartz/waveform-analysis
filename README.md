# waveform-analysis

The physics results for muon-veto scintillator panel R&D: **pulse amplitude to energy, and
whether the run held still while it was measured.** Optimal filter and colored-noise-optimal
boxcar compared head to head on one preparation, Landau⊗Gauss spectrum fits, the gamma/muon
cut — and the run-level time axis, livetime and gain stability.

Built against real detector noise and real pulses — no injected truth anywhere in the chain.
Analysis and code by **Remy Strichartz** (Yale).

The reference dataset is `run00270`: a 50-hour run of scintillator panels read out by SiPMs
and PMTs, with a hodoscope coincidence trigger. The panel under test is the CUPID muon-veto
prototype of [arXiv:2505.06129](https://arxiv.org/abs/2505.06129) ("Prototype 1":
100×50×2.5 cm³ EJ-200 with eight WLS-fiber mini-modules, one SiPM each — the ch0–ch7 analog
bank), sandwiched between small PMT trigger paddles (ch9 top, ch10 bottom; the trigger
footprint sits over the ch0 corner, which is why ch0 is bright and ch4–ch7 are dim).

## The three repos

```
   waveform-io          layout, ingestion, shared primitives
     ^        ^
     |        |
 waveform-qc  waveform-analysis
   triage,      <- HERE
   efficiency
```

This repo installs [waveform-io](https://github.com/remy-strichartz/waveform-io) and is
installed by nothing. It does not import `waveform-qc`: it re-runs the shared triage
classification in memory through `hodoscope_common.waveform_ops` — the same primitives that
repo uses — to drop NOISE/PILEUP/clipped events from its noise and template models.

| package | role |
|---|---|
| [`energy_reconstruction`](energy_reconstruction/README.md) | Amplitude reconstruction. Optimal filter and boxcar compared on one preparation; spectrum fits, the gamma/muon cut, the batch driver and the QC record. |
| [`timing_stability`](timing_stability/README.md) | Recovers *when* each event happened by combining the CAEN trigger time tag with the MIDAS wall clock — neither suffices alone. Livetime, gain stability, data integrity. |

`timing_stability` sits on top of `energy_reconstruction` (it reconstructs amplitudes in time
slices with the same pipeline the energy results use) and on `file_manipulation` from the base
repo (the MIDAS parser and TTT clock recovery). Those two travel together for that reason.

## Install

`waveform-io` supplies `hodoscope_common` **and** `file_manipulation`. It is not on PyPI, so
install it first:

```bash
# development — tracks your local edits
pip install -e ../waveform-io

# or reproducible — pinned tag.  waveform-io is public, so this needs no credentials.
pip install "waveform-io @ git+https://github.com/remy-strichartz/waveform-io.git@v0.1.0"
```

Then:

```bash
pip install -r requirements.txt
export WAVEFORM_FILES=/path/to/waveform_files      # required; see waveform-io's README
```

Run the drivers as scripts from the repo root, exactly as before:

```bash
python energy_reconstruction/compare.py --input run00270_ch9.h5
python energy_reconstruction/run_batch.py            # the canonical batch + QC record
```

## Environment

Python 3.11+. CI runs the suite on Linux against a plain pip install.

| package | floor | validated |
|---|---|---|
| numpy | | 2.4.1 |
| scipy | **>= 1.16** | 1.16.3 |
| h5py | | 3.16.0 |
| pandas | | 3.0.3 |
| matplotlib | | 3.10.8 |

**The scipy floor is not optional.** The MIP line is fit with `scipy.stats.landau`
(`mv_pipeline.py`), a recent scipy addition that older releases do not carry — and the
Landau⊗Gauss fit *is* the energy reconstruction, so an older scipy fails at import rather
than degrading gracefully. The "validated" column is the environment every number in the
results trees was produced under (`run_batch.py` records it in each batch's manifest).

## The figures are in this repo; the data is not

`energy_reconstruction_results/` and `timing_stability_results/` are tracked. Every plot the
pipelines produce — spectra, templates, stability series, time-walk reports — reads straight
from a clone, with no data and no re-run. `energy_reconstruction_results/logs/` carries the
batch manifest (the exact commands, exit codes, library versions, and the git commit the
numbers were produced at) next to `final_run_summary.json`.

Every `.h5` / `.h5.gz` / `.mid` file is gitignored wherever it lands.

## Tests

A synthetic end-to-end regression — it builds its own waveforms and touches no real data:

```bash
python energy_reconstruction/tests/test_energy_reconstruction.py
```

18 tests. Also collected by `pytest` from the repo root, and run on every push
([`.github/workflows/tests.yml`](.github/workflows/tests.yml), which checks out
`waveform-io` alongside this repo and installs it). `timing_stability` has no suite of its
own; `run_stability` is exercised through the pipeline tests above.
