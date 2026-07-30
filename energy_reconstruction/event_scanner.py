#!/usr/bin/env python3
"""Interactive per-event browser for the optimal-filter QC planes.

Runs the standard preparation (noise model, template, OF trigger) and then lets you
STEP THROUGH INDIVIDUAL EVENTS on one of two planes, seeing each event's actual
waveform with its fitted A*template overlaid -- the direct way to answer "what ARE
those points?":

  --plane chi2   (default)  per-event OF reduced chi2 vs reconstructed amplitude,
                            with the clean trend chi2(A) = c0 + (A/A0)^2 and the
                            amplitude-aware pileup gate drawn on it.
  --plane time              reconstructed peak time - expected center vs amplitude
                            (the time-walk check plane).

Which events are offered for scanning (--select):
  gated     events the amplitude-aware pileup gate FLAGS (chi2 above factor x the
            clean trend for their amplitude) -- the shape anomalies / pile-up.  [default]
  clean     the complement: events the gate passes.
  offtime   the VERTICAL population of the time-walk check: events whose reconstructed
            time sits far from the coincidence core (|t - median| > max(5 robust sigma,
            5 samples), the same rule plot_timing uses for its noise-trigger tail).
            These are typically near-threshold triggers where the filter latched onto
            noise rather than a pulse crest.
  all       every triggered event.
Optionally intersect with a rectangle: --amin/--amax (amplitude) and --ymin/--ymax
(the active plane's y: chi2 or time).

Interactive keys (a plot window must be open, i.e. do NOT pass --no-show):
  right / n     next event          left / p      previous event
  up / down     jump +/-10          r             random event
  s             save the current view as a PNG into the results folder
  q             quit
Clicking a highlighted point in the left panel jumps to that event.

Headless use: --export N saves the first N selected events as PNGs (sorted by
--order) into the results folder and exits; combine with --no-show.

    python event_scanner.py --input run00270_ch9_clean.h5 --mode muon
    python event_scanner.py --input run00270_ch10.h5 --plane time --select offtime
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (see README)

import matplotlib.pyplot as plt
import numpy as np

from energy_reconstruction import mv_pipeline as P       # noqa: E402
from energy_reconstruction import optimal_filter as OF   # noqa: E402

logger = logging.getLogger("event_scanner")


def _selection(args, obs, chi2, t_rel, gate_mask):
    """Boolean mask of the events offered for scanning (see module docstring)."""
    n = obs.size
    if args.select == "gated":
        sel = gate_mask.copy()
    elif args.select == "clean":
        sel = ~gate_mask
    elif args.select == "offtime":
        med = float(np.median(t_rel))
        rob = 1.4826 * float(np.median(np.abs(t_rel - med)))
        sel = np.abs(t_rel - med) > max(5.0 * rob, 5.0)
    else:
        sel = np.ones(n, dtype=bool)
    sel &= np.isfinite(obs) & np.isfinite(chi2)
    if args.amin is not None:
        sel &= obs >= args.amin
    if args.amax is not None:
        sel &= obs <= args.amax
    y = chi2 if args.plane == "chi2" else t_rel
    if args.ymin is not None:
        sel &= y >= args.ymin
    if args.ymax is not None:
        sel &= y <= args.ymax
    return sel


class Scanner:
    """Two-panel browser: the QC plane (left) and the current event's waveform with
    its A*template overlay (right)."""

    def __init__(self, prep, config, obs, chi2, t_rel, order, sel, plane,
                 trend, gate_factor):
        self.prep, self.config = prep, config
        self.obs, self.chi2, self.t_rel = obs, chi2, t_rel
        self.order, self.sel, self.plane = order, sel, plane
        self.c0, self.a0 = trend
        self.gate_factor = gate_factor
        self.pos = 0
        self.length = prep.corrected.shape[1]
        self.ws = prep.events["window_start"].to_numpy()
        self.orig = prep.events["original_index"].to_numpy()
        self.L = config.template_pre + config.template_post

        self.fig, (self.ax_plane, self.ax_wave) = plt.subplots(
            1, 2, figsize=(14, 5.5), layout="constrained")
        self._draw_plane()
        self.marker, = self.ax_plane.plot([], [], "*", color="red", ms=14, mec="k",
                                          zorder=6, label="current event")
        self.ax_plane.legend(loc="upper left", fontsize=8)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.show_event(0)

    # ------------------------------------------------------------------ plane
    def _y(self):
        return self.chi2 if self.plane == "chi2" else self.t_rel

    def _draw_plane(self):
        ax, a, y, sel = self.ax_plane, self.obs, self._y(), self.sel
        rng = np.random.default_rng(self.config.seed)
        rest = np.flatnonzero(~sel)
        if rest.size > 8000:
            rest = rng.choice(rest, 8000, replace=False)
        ax.scatter(a[rest], y[rest], s=3, alpha=0.15, color="0.6", label="other events")
        ax.scatter(a[sel], y[sel], s=8, alpha=0.6, color="C0",
                   label=f"selected (n={int(sel.sum())})")
        if a.min() > 0:
            ax.set_xscale("log")
            ax.set_xlim(float(a.min()) * 0.8, float(a.max()) * 1.25)
        if self.plane == "chi2":
            ax.set_yscale("log")
            if np.isfinite(self.a0) or np.isfinite(self.c0):
                xs = np.geomspace(*ax.get_xlim(), 200)
                ax.plot(xs, P.expected_chi2(xs, self.c0, self.a0), "-", color="C3",
                        lw=1.8, label="clean trend")
                ax.plot(xs, self.gate_factor * P.expected_chi2(xs, self.c0, self.a0),
                        "--", color="C3", lw=1.1, alpha=0.7,
                        label=rf"pileup gate ({self.gate_factor:g}$\times$)")
            ax.set(xlabel="Reconstructed amplitude", ylabel=r"Reduced $\chi^2$",
                   title=r"OF $\chi^2$ vs amplitude")
        else:
            ax.axhline(0, color="gray", lw=0.8)
            ax.set(xlabel="Reconstructed amplitude",
                   ylabel="Peak time - center (samples)", title="Time-walk plane")
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ event
    def show_event(self, pos: int) -> None:
        if not self.order.size:
            return
        self.pos = int(pos) % self.order.size
        i = int(self.order[self.pos])
        ax = self.ax_wave
        ax.clear()
        wave = np.asarray(self.prep.corrected[i], dtype=float)
        ax.plot(np.arange(self.length), wave, color="C0", lw=0.9, label="corrected waveform")
        xs = self.ws[i] + np.arange(self.L)
        model = self.obs[i] * self.prep.template
        ax.plot(xs, model, color="C3", lw=1.4, alpha=0.9, label=r"$A\cdot$template")
        # Residual around zero over the fitted window only (same units as the waveform).
        j0 = int(np.clip(np.round(self.ws[i]), 0, self.length - self.L))
        ax.plot(xs, wave[j0:j0 + self.L] - model, color="0.5", lw=0.8, alpha=0.8,
                label="residual (data - model)")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvspan(xs[0], xs[-1], color="C1", alpha=0.06)
        expect = P.expected_chi2(np.array([self.obs[i]]), self.c0, self.a0)[0]
        ax.set(xlabel="Sample", ylabel="Corrected ADC",
               title=f"event {self.pos + 1}/{self.order.size}  "
                     f"(row {i}, file row {self.orig[i]})")
        info = (rf"$A$ = {self.obs[i]:.3f}"
                + "\n" + rf"$\chi^2_\nu$ = {self.chi2[i]:.2f}"
                + f"  ({self.chi2[i] / expect:.1f}" + r"$\times$ clean trend)"
                + "\n" + rf"$t - t_0$ = {self.t_rel[i]:+.2f} samp")
        ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.85))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self.marker.set_data([self.obs[i]], [self._y()[i]])
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ events
    def _on_key(self, event) -> None:
        step = {"right": 1, "n": 1, "left": -1, "p": -1, "up": 10, "down": -10}
        if event.key in step:
            self.show_event(self.pos + step[event.key])
        elif event.key == "r":
            self.show_event(int(np.random.default_rng().integers(self.order.size)))
        elif event.key == "s":
            self.save_current()
        elif event.key == "q":
            plt.close(self.fig)

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax_plane or event.xdata is None:
            return
        a, y = self.obs[self.order], self._y()[self.order]
        # Nearest selected point in display (pixel) coordinates, so log axes work.
        pts = self.ax_plane.transData.transform(np.column_stack([a, y]))
        d2 = (pts[:, 0] - event.x) ** 2 + (pts[:, 1] - event.y) ** 2
        self.show_event(int(np.argmin(d2)))

    def save_current(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        i = int(self.order[self.pos])
        out = self.config.output_dir / f"event_{self.plane}_row{i}.png"
        self.fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out}")
        return out


def main() -> None:
    parser = P.build_arg_parser("Interactive per-event scanner for the OF chi2 / "
                                "time-walk planes.")
    parser.add_argument("--plane", choices=["chi2", "time"], default="chi2",
                        help="QC plane to scan on: OF reduced chi2 vs amplitude "
                             "(default) or peak time vs amplitude.")
    parser.add_argument("--select", choices=["gated", "clean", "offtime", "all"],
                        default="gated",
                        help="Which events to step through: 'gated' = flagged by the "
                             "amplitude-aware pileup gate (default), 'clean' = passed "
                             "it, 'offtime' = the vertical population of the time-walk "
                             "check (times far from the coincidence core), 'all'.")
    parser.add_argument("--amin", type=float, default=None, help="Min amplitude bound.")
    parser.add_argument("--amax", type=float, default=None, help="Max amplitude bound.")
    parser.add_argument("--ymin", type=float, default=None,
                        help="Min bound on the active plane's y (chi2 or time).")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Max bound on the active plane's y (chi2 or time).")
    parser.add_argument("--order", choices=["chi2", "amplitude", "time"], default=None,
                        help="Scan order (descending chi2 / amplitude / |time|). "
                             "Default: chi2 on the chi2 plane, |time| on the time plane.")
    parser.add_argument("--export", type=int, default=0, metavar="N",
                        help="Save the first N selected events as PNGs into the "
                             "results folder and exit (no interaction needed).")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    config = P.config_from_args(args, script_file=__file__, program="scan")

    prep = P.prepare(config)
    config = prep.config
    windows = OF.of_aligned_windows(prep, config)
    obs = OF.optimal_filter_amplitude(prep, prep.psd, config, windows)
    chi2 = OF.optimal_filter_chi2(prep, prep.psd, config, obs, windows)
    sat = P.saturation_qc(prep, config)
    clipped = sat["sat_mask"].astype(bool) if sat["clipped_pileup"] else None
    trend = P.chi2_amplitude_trend(obs, chi2, clipped)
    gate_mask = P.pileup_mask_from_chi2(chi2, config, amplitude=obs, clipped=clipped)
    center = P.pulse_center(prep.length, config)
    t_rel = prep.events["peak_subsample"].to_numpy() - center

    sel = _selection(args, obs, chi2, t_rel, gate_mask)
    n_sel = int(sel.sum())
    print(f"selection '{args.select}' on the {args.plane} plane: {n_sel} of "
          f"{obs.size} triggered events")
    if n_sel == 0:
        raise SystemExit("Nothing selected -- relax --select / the bounds.")

    order_key = args.order or ("chi2" if args.plane == "chi2" else "time")
    key = {"chi2": chi2, "amplitude": obs, "time": np.abs(t_rel)}[order_key]
    rows = np.flatnonzero(sel)
    order = rows[np.argsort(-key[rows], kind="stable")]

    scan = Scanner(prep, config, obs, chi2, t_rel, order, sel, args.plane,
                   trend, config.pileup_chi2_factor)
    if args.export > 0:
        for k in range(min(args.export, order.size)):
            scan.show_event(k)
            scan.save_current()
        plt.close(scan.fig)
        return
    if config.show_plots:
        print("keys: right/left step, up/down +/-10, r random, s save, q quit; "
              "click a point to jump to it")
        plt.show()
    else:
        print("Nothing to do: pass --export N for headless use, or drop --no-show "
              "for the interactive window.")


if __name__ == "__main__":
    main()
