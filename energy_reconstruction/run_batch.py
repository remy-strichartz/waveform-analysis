"""Run the canonical energy_reconstruction batch and write its record.

This is the batch that produces the numbers of record: 10 gamma-muon compares, the three
muon-mode compares (hodoscope-tagged caen_ch0 and the two triage-cleaned PMT exports --
--mode is provenance, and theirs says the gamma population is already gone), and 11
timewalk reports.  It writes
`energy_reconstruction_results/logs/` (one log per job), `logs/manifest.txt` (the exact
commands, exit codes, wall times and library versions), `logs/COMPLETE.txt` (written
ONLY if every job exits 0 -- its absence means the batch did not finish), and
`final_run_summary.json` via `make_summary`.

It exists because every earlier batch was run ad hoc and the driver thrown away, which
let the plots and the QC record drift apart: after the 2026-07-14 closure fix the figures
were current while `final_run_summary.json` still quoted the pre-fix meas/pred.  Running
the sweep and rebuilding the summary from its own logs, in one command, is what keeps
that from recurring -- so re-run this, not a bare `compare.py`, after any change that
moves a printed QC number.

    C:\\Users\\remys\\miniconda3\\python.exe run_batch.py            # ~80 min
    C:\\Users\\remys\\miniconda3\\python.exe run_batch.py --dry-run  # list the jobs

Two paths have to be set, because since the three-repo split neither the data nor the
triage exports live beside this package:

    $WAVEFORM_FILES   the waveform_files/ data tree
    $TRIAGE_RESULTS   waveform-qc's preprocessing_results/triage/, which holds the
                      run00270_chN_clean.h5 exports (--triage-dir overrides it)

Both are .h5 and gitignored, so they exist only where they were produced.  `--dry-run`
checks that every clean export resolves before committing to the full ~80 minutes.

Console output is ASCII: redirected stdout falls back to cp1252 on this machine and a
stray non-ASCII character would kill the run after all the work is done.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ER = Path(__file__).resolve().parent
ROOT = ER.parent
LOGS = ER / "energy_reconstruction_results" / "logs"

# Where waveform-qc wrote its triage results.  The exports this batch needs are .h5 files,
# so they are gitignored and ship with NEITHER repo -- they exist only on a machine where
# triage has actually been run.  Since the three-repo split, preprocessing/ is not a sibling
# of this package either, so $TRIAGE_RESULTS (or --triage-dir) is how the two halves are
# joined.  The fallback is the old single-repo layout, which still resolves in a
# monorepo-style checkout.
def _triage_dir(override: str | None = None) -> Path:
    return Path(override or os.environ.get("TRIAGE_RESULTS")
                or ROOT / "preprocessing" / "preprocessing_results" / "triage")


# The triage-cleaned exports.  ch9/ch10 feed muon-mode compares (the gamma population is
# already gone upstream -- --mode is provenance, not shape); ch7 feeds a timewalk report.
def _clean_exports(triage: Path) -> dict[str, Path]:
    return {ch: triage / f"run00270_{ch}_triage_results" / f"run00270_{ch}_clean.h5"
            for ch in ("ch7", "ch9", "ch10")}


TRIAGE = _triage_dir()
CLEAN = _clean_exports(TRIAGE)

RAW = ["run00270_ch0", "run00270_ch1", "run00270_ch2", "run00270_ch3",
       "run00270_ch4", "run00270_ch5", "run00270_ch6", "run00270_ch7",
       "run00270_ch9", "run00270_ch10"]

PLOTS = ["--save-plots", "--no-show", "--overwrite"]


def jobs() -> list[tuple[str, list[str]]]:
    """(job name, argv) for the canonical batch, in run order.  The job name is also its
    log stem and its key in final_run_summary.json."""
    out = [(f"compare_{stem}", ["compare.py", "--input", f"{stem}.h5", *PLOTS])
           for stem in RAW]
    # caen_ch0 is hodoscope-TAGGED muon data -- the negative control of _gamma_evidence:
    # --mode is provenance, and its provenance says the gamma population was removed
    # upstream.  A gamma-muon run of it reports a cut at ~0.43 that the project itself
    # measured to be meaningless (mv_pipeline._gamma_evidence), so the RECORD fits it in
    # muon mode; rerun it in gamma-muon by hand only to reproduce that negative control.
    out += [("compare_caen_ch0",
             ["compare.py", "--mode", "muon", "--input", "caen_ch0.h5", *PLOTS])]
    out += [(f"compare_run00270_{ch}_clean",
             ["compare.py", "--mode", "muon", "--input", str(CLEAN[ch]), *PLOTS])
            for ch in ("ch9", "ch10")]
    # timewalk: the run00270 bank only -- the walk is a property of that DAQ's channels.
    out += [(f"timewalk_{stem}", ["timewalk_report.py", "--input", f"{stem}.h5", *PLOTS])
            for stem in RAW]
    out += [("timewalk_run00270_ch7_clean",
             ["timewalk_report.py", "--input", str(CLEAN["ch7"]), *PLOTS])]
    return out


def job_names() -> list[str]:
    return [name for name, _ in jobs()]


def _provenance() -> list[str]:
    out = [f"python {sys.version}"]
    for mod in ("numpy", "scipy", "h5py", "pandas", "matplotlib"):
        try:
            out.append(f"{mod} {__import__(mod).__version__}")
        except Exception as exc:                                    # noqa: BLE001
            out.append(f"{mod} UNAVAILABLE ({exc})")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=30).stdout.strip()
        out.append(f"git: {sha or 'unknown'}"
                   + (" (WORKING TREE DIRTY -- these numbers are not from a commit)"
                      if dirty else ""))
    except Exception as exc:                                        # noqa: BLE001
        out.append(f"git: unavailable ({exc})")
    return out


def _pretty(argv: list[str]) -> str:
    return "python " + " ".join(f'"{a}"' if " " in a or "\\" in a else a for a in argv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="list the jobs and check every input resolves; run nothing")
    ap.add_argument("--triage-dir", metavar="DIR",
                    help="waveform-qc's preprocessing_results/triage/, holding the "
                         "run00270_chN_clean.h5 exports. Defaults to $TRIAGE_RESULTS, "
                         "then to preprocessing/ beside this package (single-repo layout).")
    args = ap.parse_args()

    if args.triage_dir:
        global TRIAGE, CLEAN
        TRIAGE = _triage_dir(args.triage_dir)
        CLEAN = _clean_exports(TRIAGE)

    todo = jobs()

    if args.dry_run:
        print(f"{len(todo)} jobs:")
        for i, (name, argv) in enumerate(todo, 1):
            print(f"  [{i:2d}/{len(todo)}] {name:28s} {_pretty(argv)}")
        missing = [str(p) for p in CLEAN.values() if not p.exists()]
        for p in missing:
            print(f"  MISSING INPUT: {p}")
        print(f"\ntriage dir: {TRIAGE}")
        print("clean-export inputs: " + ("all present" if not missing
                                         else f"{len(missing)} MISSING -- set "
                                              "$TRIAGE_RESULTS or pass --triage-dir"))
        return 1 if missing else 0

    LOGS.mkdir(parents=True, exist_ok=True)

    # Provenance FIRST, before anything below touches the working tree.  Deleting
    # COMPLETE.txt is itself a modification, so reading the git state after it made
    # _provenance() report "WORKING TREE DIRTY" on every run once the results trees became
    # tracked -- the manifest disowned its own numbers, which is the exact opposite of the
    # guarantee it exists to give.
    started = datetime.datetime.now().astimezone()
    lines = ["Canonical run of the energy_reconstruction pipeline",
             f"date: {started.isoformat()}",
             *_provenance(),
             "seed: Config.seed = 20240601 (default; all pipeline randomness is seeded)",
             ""]

    complete = LOGS / "COMPLETE.txt"
    complete.unlink(missing_ok=True)      # absent until the batch actually finishes

    failures = 0
    for i, (name, argv) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name} ...", flush=True)
        t0 = time.time()
        log = LOGS / f"{name}.log"
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run([sys.executable, *argv], cwd=ER,
                                  stdout=fh, stderr=subprocess.STDOUT)
        dt = int(round(time.time() - t0))
        status = "exit 0" if proc.returncode == 0 else f"EXIT {proc.returncode}"
        failures += proc.returncode != 0
        print(f"    -> {status}, {dt}s", flush=True)
        lines += [f"command: {_pretty(argv)}", f"  -> {status}, {dt}s, log {log.name}"]

    (LOGS / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Rebuild the summary from the logs this batch just wrote, so the two cannot separate.
    sys.path.insert(0, str(ER.parent))                    # repo root (see README)
    from energy_reconstruction import make_summary
    summary = make_summary.write(job_names())
    print(f"\n{len(summary['runs'])} runs -> final_run_summary.json")

    if failures:
        print(f"BATCH FINISHED WITH {failures} FAILURE(S); COMPLETE.txt not written")
        return 1
    complete.write_text(
        f"BATCH COMPLETE {datetime.datetime.now().astimezone().isoformat()}\n",
        encoding="utf-8")
    print(f"BATCH COMPLETE: {len(todo)}/{len(todo)} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
