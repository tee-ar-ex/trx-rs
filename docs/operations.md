# Operations

## Subset

Extract a subset of streamlines by index:

```rust
use trx_rs::ops::subset::subset_streamlines;

let sub = subset_streamlines(&trx, &[0, 5, 10, 42])?;
```

## Merge (concatenate)

Merge multiple TRX files that share the same affine and dimensions:

```rust
use trx_rs::ops::merge::merge_trx_shards;

let merged = merge_trx_shards(&[&shard1, &shard2])?;
```

For more control over how groups are handled during concatenation, use `concatenate_any_trx` with `ConcatenateOptions`.

## Set operations on streamlines

```rust
use trx_rs::ops::streamline_ops::{difference, intersection, streamline_union};

let common = intersection(&trx_a, &trx_b)?;
let only_a = difference(&trx_a, &trx_b)?;
let all = streamline_union(&trx_a, &trx_b)?;
```

## Duplicate removal

```rust
use trx_rs::ops::streamline_ops::{remove_duplicates, DuplicateRemovalMode, DuplicateRemovalParams};

let params = DuplicateRemovalParams {
    mode: DuplicateRemovalMode::Near,
    tolerance_mm: 0.5,
    endpoint_tolerance_mm: 1.0,
    min_shared_voxel_fraction: 1.0,
};
let deduped = remove_duplicates(&trx, &params)?;
```

## Spatial queries

Find streamlines whose axis-aligned bounding box intersects a query region:

```rust
use trx_rs::ops::subset::{build_streamline_aabbs, query_aabb, query_aabb_cached};

// One-shot query (builds AABBs each time):
let hits = query_aabb(&trx, [-10.0, -10.0, -10.0], [10.0, 10.0, 10.0])?;

// Cached AABB query for repeated use:
let aabbs = build_streamline_aabbs(&trx);
let hits = query_aabb_cached(&aabbs, [-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
```

## Connectivity matrices

Compute group-to-group connectivity:

```rust
use trx_rs::ops::connectivity::{compute_group_connectivity, ConnectivityMeasure};

let matrix = compute_group_connectivity(&trx, ConnectivityMeasure::Count)?;
```

## Metadata operations

Copy DPS/DPV/group/DPG arrays between files:

```rust
use trx_rs::ops::copy_metadata::{copy_metadata, CopyMetadataOptions};

copy_metadata(&source, &dest, &CopyMetadataOptions {
    dps_names: Some(vec!["weights".into()]),
    dpv_names: None,  // copy all
    ..Default::default()
})?;
```
