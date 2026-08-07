# Applying ANTs transforms to tractograms

`trxrs transform` warps streamline vertex coordinates through an ITK Composite `.h5` (with embedded warp + affines), an Insight Transform File V1.0 (`.txt`, affine-only), or an ITK MATLAB v4 binary (`.mat`, affine-only — what ANTs writes for `*0GenericAffine.mat`).

## The "opposite-named h5" rule (cartoon BIDS)

Tractograms warp in the **opposite** spatial direction from images. Concretely, with paired BIDS h5 files for subject `sub-01`:

| You have | You want | Pass to `--transform` |
|----------|----------|----------------------|
| `sub-01_space-ACPC_tracts.trx` | tracts in `MNI152NLin2009cAsym` | `sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5` |
| `sub-01_space-MNI152NLin2009cAsym_tracts.trx` | tracts in `ACPC` | `sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5` |
| `sub-01_space-T1w_tracts.trx` | tracts in `MNI152NLin6Asym` | `sub-01_from-MNI152NLin6Asym_to-T1w_xfm.h5` |

If you are coming from `antsApplyTransforms` for images: pass the **same h5** you would use to warp an image of the destination space *into* the source space. (That's the same convention as `antsApplyTransformsToPoints`.)

## Why opposite-named?

Image warping with `antsApplyTransforms` is **pull-based**: the chain inside `from-X_to-Y_xfm.h5` (the file that warps an X-image onto a Y-grid, per BIDS) internally maps target Y voxels back to source X coordinates. Applied to a *point*, that same chain sends a Y-point to an X-point. So to warp a streamline FROM space A TO space B, you need a chain that maps A → B — which lives in the opposite-named file `from-B_to-A_xfm.h5`.

## Library usage

```rust
use trx_rs::{TrxFile, apply_transform};

let mut trx = TrxFile::<f32>::load("tractogram.trx")?;
let transform = load_transform_from_file("warp.h5")?;
apply_transform_in_place(&mut trx, &transform)?;
```

## CLI usage

```bash
trxrs transform \
    sub-01_space-ACPC_tracts.trx \
    sub-01_space-MNI152NLin2009cAsym_tracts.trx \
    --transform sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5 \
    --reference sub-01_space-MNI152NLin2009cAsym_T1w.nii.gz
```

The `--reference` flag is optional — if given, the output TRX header's `voxel_to_rasmm` and `dimensions` are taken from it. The streamline coordinates themselves are warped regardless.

## Affine-only chains

For an affine-only `.txt` or `.mat`, `--invert` numerically flips the chain so you can use a single file in either direction. Warps cannot be numerically inverted; for those, use the paired `from-Y_to-X_xfm.h5` instead.
