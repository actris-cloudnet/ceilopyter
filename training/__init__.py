r"""Offline CHM15k overlap-model training (not shipped in the wheel).

Builds a temperature-dependent overlap model (Hervo et al. 2016) from raw
CHM15k files. Run from the repository root::

    python -m training.train --reference ref.cfg --range-from day.nc \\
        --output model.nc day1.nc day2.nc ...

These modules build on the shipped ``ceilopyter.overlap`` data model and apply
step; they are intentionally kept out of the installed package because they run
rarely and per-device. See the runtime side in ``ceilopyter/overlap/``.
"""
