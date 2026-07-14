"""Rebuild `energy_reconstruction_results/final_run_summary.json` from the batch logs.

The summary is a DERIVED artifact: every number in it is parsed back out of the logs
`run_batch.py` writes, because the pipeline itself does not emit a machine-readable
result file.  That is a trap worth stating plainly -- re-running `compare.py` by hand
refreshes the plots and the log but leaves this file untouched, so the QC numbers of
record silently drift away from the figures beside them.  (After the 2026-07-14
resolution-closure fix the plots were current while the summary still reported ch0's
meas/pred as 1.51 -- the very bug that had just been fixed.)  `run_batch.py` calls this
at the end of every batch so the two cannot separate; run it standalone only to repair a
summary whose logs are already good.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ER = Path(__file__).resolve().parent
RESULTS = ER / "energy_reconstruction_results"
LOGS = RESULTS / "logs"
SUMMARY = RESULTS / "final_run_summary.json"

RE_TRIG = re.compile(r"Triggered\s+(\d+)\s*/\s*(\d+)\s+analyzed")
RE_MPV = re.compile(r"^ {2}(\S.*?)\s{2,}MPV=([\d.eE+-]+)\s+obs")
RE_CUT = re.compile(r"^ {2}(\S.*?)\s{2,}cut=([\d.eE+-]+)\s+\+/-\s+([\d.eE+-]+).*?:"
                    r"\s+([\d,]+)\s+muon-like,\s+([\d,]+)\s+gamma-like")
RE_NOCUT = re.compile(r"^ {2}(\S.*?)\s{2,}NO CUT DERIVED")
RE_RAIL = re.compile(r"\(fit excludes (\d+) rail-clipped events\)")
RE_RES = re.compile(r"predicted\s+([\d.eE+-]+),\s+measured\s+([\d.eE+-]+)")
RE_SLOPE = re.compile(r"Shipped slope \(n_bins=\d+\):\s+([+-]?[\d.]+)")
RE_CLOS = re.compile(r"^\s+(centered|jitter|offset \+3|offset -3)\s+([+-]?[\d.]+)\s*$")


def parse_log(name: str, text: str) -> dict:
    """One batch job's log -> its summary entry.  A field the log does not carry is
    OMITTED rather than defaulted, so a parser that silently stops matching shows up as
    a missing key instead of a plausible zero."""
    out: dict = {"log": f"{name}.log"}
    if (m := RE_TRIG.search(text)):
        out["n_triggered"], out["n_analyzed"] = int(m[1]), int(m[2])

    if name.startswith("timewalk_"):
        if (m := RE_SLOPE.search(text)):
            out["shipped_slope"] = float(m[1])
        closure = {m[1]: float(m[2]) for m in
                   (RE_CLOS.match(ln) for ln in text.splitlines()) if m}
        if closure:
            out["closure"] = closure
        return out

    mpv, cut, no_cut, classified = {}, {}, [], []
    for ln in text.splitlines():
        if (m := RE_MPV.match(ln)):
            mpv[m[1].strip()] = float(m[2])
        elif (m := RE_CUT.match(ln)):
            cut[m[1].strip()] = {"value": float(m[2]), "std": float(m[3])}
            classified.append({"muon_like": int(m[4].replace(",", "")),
                               "gamma_like": int(m[5].replace(",", ""))})
        elif (m := RE_NOCUT.match(ln)):
            no_cut.append(m[1].strip())
    out["mpv"], out["cut"], out["no_cut"], out["classified"] = mpv, cut, no_cut, classified
    out["n_rail_clipped_excluded"] = [int(x) for x in RE_RAIL.findall(text)]
    if (m := RE_RES.search(text)):
        out["resolution_pred"], out["resolution_meas"] = float(m[1]), float(m[2])
    return out


def build(job_names: list[str]) -> dict:
    runs, problems = {}, []
    for name in job_names:
        log = LOGS / f"{name}.log"
        if not log.exists():
            problems.append(f"{name}: log missing")
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        runs[name] = parse_log(name, text)
        if "Traceback" in text:
            problems.append(f"{name}: traceback in log")
    return {"runs": runs, "problems": problems}


def write(job_names: list[str]) -> dict:
    summary = build(job_names)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    from run_batch import job_names            # the batch defines the canonical set

    names = job_names()
    summary = write(names)
    runs, problems = summary["runs"], summary["problems"]
    print(f"{len(runs)}/{len(names)} runs summarized -> {SUMMARY.name}")
    for name in names:
        r = runs.get(name, {})
        if name.startswith("timewalk_"):
            continue
        pred, meas = r.get("resolution_pred"), r.get("resolution_meas")
        ratio = f"{meas / pred:.2f}" if pred and meas else "--"
        cuts = ", ".join(f"{k.split()[0]} {v['value']:.4f}"
                         for k, v in r.get("cut", {}).items())
        state = cuts or ("NO CUT" if r.get("no_cut") else "-")
        print(f"  {name:28s} meas/pred {ratio:>5s}   {state}")
    for p in problems:
        print("  PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
