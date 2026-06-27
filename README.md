# Ceilopyter

[![Run tests](https://github.com/actris-cloudnet/ceilopyter/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/ceilopyter/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/ceilopyter.svg)](https://badge.fury.io/py/ceilopyter)

Python package for reading ceilometer data and doing some post-processing like
screening background noise. This package is used in
[CloudnetPy](https://github.com/actris-cloudnet/cloudnetpy) for generating the
[Cloudnet lidar product](https://cloudnet.fmi.fi/product/lidar).

## Supported ceilometers

- Campbell Scientific CS135
- Lufft CHM 15k and 15k-x
- Vaisala CL31, CL51, CL61, CT25K and LD40

## CHM15k overlap correction

The CHM15k near-range signal is distorted by a temperature-dependent error in
the factory overlap function. Ceilopyter can correct this using the empirical
method of
[Hervo et al. (2016)](https://amt.copernicus.org/articles/9/2947/2016/): a
per-instrument model describes how the overlap deviates from a reference as a
function of the internal temperature.

### Applying a model

Given a trained model file, pass it to `read_chm15k`:

```python
from ceilopyter import read_chm15k

ceilo = read_chm15k("20240516_chm15k.nc", overlap_model="TUB120011_model.nc")
# ceilo.beta_raw / ceilo.beta are now overlap-corrected
```

`overlap_model` accepts a path or an `OverlapModel`. The correction is applied
before noise screening; without it, `read_chm15k` is unchanged. To correct an
already-read `Ceilo` (e.g. after averaging), use `correct_overlap(ceilo, model)`.

### Training a model

Training is a one-time, per-optical-module step and lives in the `training/`
package, which is **not installed** — run it from the repository root. It needs
the device's factory reference overlap (a `TUB…cfg` file; an example is bundled
in `ceilopyter/overlap_functions/`) and a multi-month set of raw files from the
same instrument spanning a wide internal-temperature range:

```bash
python -m training.train \
    --reference ceilopyter/overlap_functions/TUB120011_20121112_1024.cfg \
    --output TUB120011_model.nc \
    /data/chm15k/2024*.nc
```

The reference overlap must be the same factory overlap the firmware applied to
`beta_raw`, and each model is specific to one optical module.

`python -m training.example` runs the whole chain on real files — collect, fit,
write, read back, and apply to a CHM15k file (`--apply-to`) — as a worked example
of the workflow above.

## License

MIT
