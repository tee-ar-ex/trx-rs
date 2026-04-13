pub mod connectivity;
pub mod merge;
pub mod streamline_ops;
pub mod subset;

pub use merge::{concatenate_any_trx, merge_trx_shards, ConcatenateOptions};
pub use streamline_ops::{
    difference, difference_indices, intersection, intersection_indices, remove_duplicates,
    remove_duplicates_tractogram, retain_representative_indices,
    retain_tractogram_representative_indices, streamline_union, DuplicateRemovalMode,
    DuplicateRemovalParams,
};
pub use subset::{
    build_streamline_aabbs, build_streamline_aabbs_from_slices, query_aabb, query_aabb_cached,
    subset_streamlines, StreamlineAabb,
};
