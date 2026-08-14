"""Pulse amplitude -> energy: templates, optimal filtering and the muon-line fit.

Depends on `hodoscope_common` (from the waveform-io repo) and nothing else.  It re-runs the
shared triage classification in memory to drop NOISE/PILEUP/clipped events from its noise
and template models, but it does that through `hodoscope_common.waveform_ops` -- the same
primitives the preprocessing stage uses -- not by importing that stage.
"""
