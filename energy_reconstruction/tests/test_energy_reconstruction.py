#!/usr/bin/env python3
"""
Automated regression tests for the energy_reconstruction pipeline.

A SYNTHETIC dataset with known truth (template-shaped pulses, known amplitudes,
known timing, white noise, no time-walk by construction) is written to a temp
HDF5 file and pushed end-to-end through the PRODUCTION code -- prepare() ->
optimal-filter trigger -> OF / boxcar amplitudes -> spectrum fits -- and the
results are asserted against the truth.  No real dataset is read or written.

What is protected, and why:

  * TRIGGER      finds essentially every real pulse (no silent event loss).
  * ALIGNMENT    events["original_index"] maps reconstructed events back to the
                 right source rows, through both the plain streaming path and the
                 --exclude-hours GTI path (a sentinel-bright event is planted at a
                 known row and must come back at that row).
  * AMPLITUDE    the OF estimate is proportional to the true amplitude, with
                 measured resolution close to the predicted sigma_A
                 (optimal_filter_resolution's closure), and the boxcar agrees
                 with the OF on the shared template=1 scale.
  * TIME-WALK CLOSURE   on events with NO true walk the estimator must not
                 invent one.  This is the CLOSURE test from timewalk_report.py
                 distilled into a regression test: it is what proves the walk
                 seen on run00270's analog bank is in the DATA, not the
                 estimator.  If this ever fails, that conclusion is void and
                 timewalk_report.py must be re-run before trusting any
                 timewalk_slope QC number.
  * CHI2         per-event OF reduced chi2 is centred near 1 (PSD normalization).
  * SPECTRUM     fit_spectrum separates a known gamma+muon two-population
                 spectrum, places the cut BETWEEN the populations, and the
                 resulting classification agrees with truth; muon mode fits the
                 same MIP line (shared fit_muon_line) to the same MPV.
  * DETERMINISM  the same config on the same file gives bit-identical events,
                 template and fitted cut (all randomness is seeded).
  * LOUD FAILURE noise_stop raises on a window with no pulse-free region;
                 non-HDF5 inputs are rejected.

Run it with the project's python (the one with h5py/scipy):

    python tests/test_energy_reconstruction.py

Plain asserts, ASCII-only output; also importable by pytest (test_* functions).
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # never open a window from a test

import numpy as np
from scipy.stats import landau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from energy_reconstruction import boxcar as BX              # noqa: E402
from energy_reconstruction import mv_pipeline as P          # noqa: E402
from energy_reconstruction import optimal_filter as OF      # noqa: E402

# ---------------------------------------------------------------------------------
# Synthetic dataset (built once, shared by the tests)
# ---------------------------------------------------------------------------------

N_EVENTS = 4000
L_RECORD = 1024
CENTER = 450.0                  # true mean pulse-peak sample
JITTER = 2.0                    # true peak-time jitter (samples); UNCORRELATED with amplitude
A_MIP = 1200.0                  # muon Landau location, ADC peak units
GAMMA_MU, GAMMA_SIG = 0.32, 0.045   # gamma population, in units of A_MIP
GAMMA_FRAC = 0.15
NOISE_ADC = 12.0
BASELINE_ADC = 800.0
SENTINEL_ROW = 1234             # planted brightest event (alignment check)
SENTINEL_AMP = 6.0 * A_MIP
RUN_SECONDS = 7200.0            # uniform 2 h time axis for the GTI test
SEED = 987654321

_TAU_R, _TAU_F = 6.0, 45.0


def _pulse_shape(t: np.ndarray) -> np.ndarray:
    """Double-exponential pulse, peak-normalized to 1, causal (0 for t < 0)."""
    y = np.where(t >= 0.0, np.exp(-np.maximum(t, 0.0) / _TAU_F)
                 - np.exp(-np.maximum(t, 0.0) / _TAU_R), 0.0)
    tp = math.log(_TAU_F / _TAU_R) * _TAU_R * _TAU_F / (_TAU_F - _TAU_R)
    return y / (math.exp(-tp / _TAU_F) - math.exp(-tp / _TAU_R))


def _make_dataset(tmp: Path):
    """Write the synthetic file; return (path, truth dict)."""
    import h5py
    rng = np.random.default_rng(SEED)
    is_muon = rng.random(N_EVENTS) >= GAMMA_FRAC
    amp = np.empty(N_EVENTS)
    amp[is_muon] = A_MIP * np.clip(
        landau.rvs(loc=1.0, scale=0.07, size=int(is_muon.sum()), random_state=rng),
        None, 5.0)
    amp[~is_muon] = A_MIP * np.clip(
        rng.normal(GAMMA_MU, GAMMA_SIG, int((~is_muon).sum())), 0.18, 0.48)
    amp[SENTINEL_ROW] = SENTINEL_AMP
    is_muon[SENTINEL_ROW] = True
    t_peak = CENTER + rng.normal(0.0, JITTER, N_EVENTS)     # no amplitude correlation

    tp = math.log(_TAU_F / _TAU_R) * _TAU_R * _TAU_F / (_TAU_F - _TAU_R)
    grid = np.arange(L_RECORD, dtype=np.float64)[None, :]
    rows = _pulse_shape(grid - (t_peak[:, None] - tp)) * amp[:, None]
    wave = BASELINE_ADC + rows + rng.normal(0.0, NOISE_ADC, (N_EVENTS, L_RECORD))
    wave = np.clip(np.round(wave), 0, 65535).astype(np.uint16)

    path = tmp / "synthetic.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("waveforms", data=wave)
        f.create_dataset("event_time_rel_s",
                         data=np.arange(N_EVENTS, dtype=np.float64)
                         / N_EVENTS * RUN_SECONDS)
    return path, {"amp": amp, "is_muon": is_muon, "t_peak": t_peak}


_CTX: dict = {}


def _ctx() -> dict:
    """Build the dataset and run prepare() + both estimators ONCE; cache for all tests."""
    if _CTX:
        return _CTX
    tmp = Path(tempfile.mkdtemp(prefix="energy_reco_test_"))
    path, truth = _make_dataset(tmp)
    config = P.Config(input_path=path, output_dir=tmp / "out",
                      show_plots=False, save_plots=False, cut_bootstrap=0)
    prep = P.prepare(config)
    config = prep.config
    windows = OF.of_aligned_windows(prep, config)
    amp = OF.optimal_filter_amplitude(prep, prep.psd, config, windows)
    chi2 = OF.optimal_filter_chi2(prep, prep.psd, config, amp, windows)
    _CTX.update(tmp=tmp, path=path, truth=truth, config=config, prep=prep,
                windows=windows, amp=amp, chi2=chi2,
                oi=prep.events["original_index"].to_numpy(),
                fit=None)
    return _CTX


def _fit() -> dict:
    """The gamma-muon spectrum fit on the OF observable (cached: it is the slow one)."""
    c = _ctx()
    if c["fit"] is None:
        c["fit"] = P.fit_spectrum(c["amp"], c["config"])
    return c["fit"]


# ---------------------------------------------------------------------------------
# Unit tests (no dataset needed)
# ---------------------------------------------------------------------------------

def test_shift_roundtrip():
    """_shift by +d then -d must return the interior of a smooth pulse unchanged."""
    x = np.arange(200, dtype=float)
    row = np.exp(-0.5 * ((x - 100.0) / 12.0) ** 2)[None, :]
    d = 0.3
    back = P._shift(P._shift(row, np.array([d])), np.array([-d]))
    err = np.max(np.abs(back[0, 5:-5] - row[0, 5:-5]))
    assert err < 1e-3, f"cubic shift round-trip error {err:.2e} (interior)"


def test_equal_area_cut_balances():
    """The returned cut must actually balance gamma-above against muon-below."""
    def gfun(x):
        return 0.25 * np.exp(-0.5 * ((np.asarray(x) - 0.3) / 0.06) ** 2)

    def mfun(x):
        return 1.00 * np.exp(-0.5 * ((np.asarray(x) - 1.0) / 0.10) ** 2)

    centers = np.linspace(0.0, 2.0, 400)
    lo = int(np.searchsorted(centers, 0.3))
    hi = int(np.searchsorted(centers, 1.0))
    gamma_hi = 0.65
    cut = P._equal_area_cut(centers, gfun, mfun, lo, hi, gamma_hi=gamma_hi)
    assert cut is not None and 0.3 < cut < 1.0, f"cut {cut} not between the populations"
    xg = np.linspace(0.0, 2.0, 200001)
    dx = xg[1] - xg[0]
    g_above = float(np.sum(gfun(xg)[(xg >= cut) & (xg <= gamma_hi)]) * dx)
    m_below = float(np.sum(mfun(xg)[xg < cut]) * dx)
    imbalance = abs(g_above - m_below) / max(g_above, m_below, 1e-12)
    assert imbalance < 0.05, (f"areas not balanced at the cut: gamma-above {g_above:.4g} "
                              f"vs muon-below {m_below:.4g}")


def test_noise_stop_raises_on_bad_window():
    """A hand-set window with no pulse-free pre-pulse region must RAISE, not clamp."""
    cfg = P.Config(input_path=Path("x.h5"), auto_window=False, pulse_center=300,
                   search_pad=90, template_pre=100, template_post=250)
    try:
        P.noise_stop(1024, cfg)
    except ValueError:
        return
    raise AssertionError("noise_stop accepted a window with no pulse-free region")


def test_non_hdf5_rejected():
    try:
        P._require_h5py(Path("waveforms.csv"))
    except ValueError:
        return
    raise AssertionError("_require_h5py accepted a .csv input")


# ---------------------------------------------------------------------------------
# End-to-end tests on the synthetic dataset
# ---------------------------------------------------------------------------------

def test_trigger_finds_every_pulse():
    c = _ctx()
    frac = len(c["prep"].events) / c["prep"].n_events
    assert frac > 0.99, f"trigger found only {100 * frac:.1f}% of real pulses"
    sigma = c["prep"].noise_sigma
    assert abs(sigma - NOISE_ADC) / NOISE_ADC < 0.2, \
        f"noise sigma {sigma:.2f} vs true {NOISE_ADC}"


def test_event_alignment_sentinel():
    """The planted brightest event must come back at its own original_index."""
    c = _ctx()
    row_of_max = int(c["oi"][int(np.nanargmax(c["amp"]))])
    assert row_of_max == SENTINEL_ROW, \
        f"brightest event maps to source row {row_of_max}, planted at {SENTINEL_ROW}"


def test_of_amplitude_proportional_and_resolved():
    c = _ctx()
    truth = c["truth"]
    mu = truth["is_muon"][c["oi"]] & (truth["amp"][c["oi"]] < 3.0 * A_MIP)
    ratio = c["amp"][mu] / truth["amp"][c["oi"]][mu]
    k = float(np.median(ratio))
    assert k > 0, "OF amplitude scale is not positive"
    rel = 1.4826 * float(np.median(np.abs(ratio / k - 1.0)))
    assert rel < 0.02, f"OF amplitude scatters {100 * rel:.2f}% about proportionality"
    sig_pred, sig_meas = OF.optimal_filter_resolution(c["prep"], c["prep"].psd, c["config"])
    assert 0.7 < sig_meas / sig_pred < 1.4, \
        f"resolution closure meas/pred = {sig_meas / sig_pred:.2f}"


def test_timewalk_closure():
    """NO true walk was generated, so the estimator must not report one.

    This is timewalk_report.py's CLOSURE question as a regression test: it protects
    the conclusion that a nonzero timewalk_slope on real data is in the data."""
    c = _ctx()
    slope = P.timewalk_slope(c["amp"], c["prep"].events["peak_subsample"].to_numpy())
    assert np.isfinite(slope), "timewalk slope is NaN on a clean synthetic run"
    assert abs(slope) < 0.25, \
        f"estimator invented a time-walk of {slope:+.3f} samples/amp on walk-free events"


def test_chi2_centered_near_one():
    c = _ctx()
    med = float(np.median(c["chi2"][np.isfinite(c["chi2"])]))
    assert 0.7 < med < 1.4, f"median OF reduced chi2 = {med:.2f} (PSD normalization off)"


def test_spectrum_cut_separates_populations():
    c = _ctx()
    fit = _fit()
    assert fit["ok"], "fit_spectrum reported no usable cut on a clearly bimodal spectrum"
    truth = c["truth"]
    mu = truth["is_muon"][c["oi"]]
    g_med = float(np.median(c["amp"][~mu]))
    m_med = float(np.median(c["amp"][mu]))
    cut = float(fit["cut"])
    assert g_med < cut < m_med, \
        f"cut {cut:.3f} not between gamma ({g_med:.3f}) and muon ({m_med:.3f}) medians"
    muon_above = float(np.mean(c["amp"][mu] > cut))
    gamma_below = float(np.mean(c["amp"][~mu] < cut))
    assert muon_above > 0.97, f"only {100 * muon_above:.1f}% of true muons above the cut"
    assert gamma_below > 0.90, f"only {100 * gamma_below:.1f}% of true gammas below the cut"


def test_muon_mode_fits_the_same_line():
    """Both modes share fit_muon_line; on the same (muon-only) events the MPVs agree."""
    c = _ctx()
    fit = _fit()
    truth = c["truth"]
    mu = truth["is_muon"][c["oi"]]
    fit_m = P.fit_muon_spectrum(c["amp"][mu], c["config"])
    assert fit_m["ok"], "muon-mode line fit failed on clean muon-only data"
    d = abs(fit_m["mpv"] - fit["mpv"]) / fit["mpv"]
    assert d < 0.05, (f"muon-mode MPV {fit_m['mpv']:.4g} vs gamma-muon MPV "
                      f"{fit['mpv']:.4g} disagree by {100 * d:.1f}%")


def test_boxcar_agrees_with_of():
    c = _ctx()
    best, _, _ = BX.optimal_boxcar_window(c["prep"], c["config"])
    bx = BX.boxcar_amplitude(c["prep"], best, c["config"])
    truth = c["truth"]
    mu = truth["is_muon"][c["oi"]]
    r = bx[mu] / c["amp"][mu]
    r = r[np.isfinite(r)]
    med = float(np.median(r))
    assert abs(med - 1.0) < 0.05, \
        f"boxcar/OF median ratio = {med:.3f} (shared template=1 scale broken)"


def test_gti_gate_drops_the_right_rows():
    """--exclude-hours must drop exactly the configured window and keep row alignment."""
    c = _ctx()
    cfg = replace(c["config"], exclude_hours=((0.0, 1.0),),
                  explicit_window=("pulse_center", "search_pad",
                                   "template_pre", "template_post"))
    prep2 = P.prepare(cfg)
    half = N_EVENTS // 2                      # t < 1 h <=> row < N/2 (uniform 2 h axis)
    assert prep2.n_events == N_EVENTS - half, \
        f"GTI kept {prep2.n_events} events, expected {N_EVENTS - half}"
    oi2 = prep2.events["original_index"].to_numpy()
    assert int(oi2.min()) >= half, \
        f"GTI leaked row {int(oi2.min())} from the excluded first hour"


def test_determinism():
    """Same file + same config -> bit-identical events, template and cut."""
    c = _ctx()
    prep2 = P.prepare(c["config"])
    assert np.array_equal(prep2.events["original_index"].to_numpy(), c["oi"])
    assert np.array_equal(prep2.template, c["prep"].template), "template not reproducible"
    fit2 = P.fit_spectrum(c["amp"], c["config"])
    fit1 = _fit()
    assert fit2["cut"] == fit1["cut"] and fit2["mpv"] == fit1["mpv"], \
        f"fit not deterministic: cut {fit1['cut']} vs {fit2['cut']}"


# ---------------------------------------------------------------------------------

TESTS = [
    test_shift_roundtrip,
    test_equal_area_cut_balances,
    test_noise_stop_raises_on_bad_window,
    test_non_hdf5_rejected,
    test_trigger_finds_every_pulse,
    test_event_alignment_sentinel,
    test_of_amplitude_proportional_and_resolved,
    test_timewalk_closure,
    test_chi2_centered_near_one,
    test_spectrum_cut_separates_populations,
    test_muon_mode_fits_the_same_line,
    test_boxcar_agrees_with_of,
    test_gti_gate_drops_the_right_rows,
    test_determinism,
]


def main() -> int:
    import logging
    logging.basicConfig(level=logging.WARNING)
    failed = []
    for fn in TESTS:
        try:
            fn()
        except AssertionError as e:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                      # noqa: BLE001 - report, don't crash the runner
            failed.append(fn.__name__)
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"PASS  {fn.__name__}")
    if _CTX:
        shutil.rmtree(_CTX["tmp"], ignore_errors=True)
    print("-" * 60)
    print(f"{len(TESTS) - len(failed)} / {len(TESTS)} tests passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
