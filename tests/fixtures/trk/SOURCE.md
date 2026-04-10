These vendored `.trk` fixtures were copied from nibabel's test data.

Upstream repository:
- `https://github.com/nipy/nibabel`

Upstream commit:
- `c0e5966774794bdbce4a067e1efe26bdc8c37115`

Upstream paths:
- `nibabel/tests/data/simple.trk`
- `nibabel/tests/data/complex.trk`
- `nibabel/tests/data/complex_big_endian.trk`
- `nibabel/tests/data/standard.trk`
- `nibabel/tests/data/standard.LPS.trk`

These files are vendored here so `trx-rs` tests remain self-contained in CI while
still tracking the original nibabel source of truth for TrackVis fixtures.
