pub mod connectivity;
pub mod merge;
pub mod streamline_ops;
pub mod subset;

pub use merge::merge_trx_shards;
pub use streamline_ops::{
    difference, difference_indices, intersection, intersection_indices, union,
};
pub use subset::{
    build_streamline_aabbs, build_streamline_aabbs_from_slices, query_aabb, query_aabb_cached,
    subset_streamlines, StreamlineAabb,
};
