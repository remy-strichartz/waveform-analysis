"""When events arrived, and whether the run's response drifted over time.

The top of the stack: depends on `hodoscope_common` and `file_manipulation` (the MIDAS parser
and TTT clock recovery), both from the waveform-io repo, and on `energy_reconstruction` in
this repo (run_stability reconstructs amplitudes in time slices with the same pipeline the
energy results use).
"""
