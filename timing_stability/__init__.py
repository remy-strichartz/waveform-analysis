"""When events arrived, and whether the run's response drifted over time.

The top of the stack: depends on `common`, on `file_manipulation` (the MIDAS parser and
TTT clock recovery) and on `energy_reconstruction` (run_stability reconstructs amplitudes
in time slices with the same pipeline the energy results use).
"""
