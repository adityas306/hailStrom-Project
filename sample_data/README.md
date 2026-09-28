# Sample / replay data

This folder intentionally contains no file pretending to be a live meteorological observation.

For model testing, place normalized arrays here, for example:

- radar_reflectivity.npy
- radar_velocity.npy
- insat_ir.npy
- insat_wv.npy
- lightning_density.npy

A production pipeline should timestamp every frame and preserve source provenance.
