use crate::dtype::TrxScalar;
use crate::error::{Result, TrxError};
use crate::trx_file::TrxFile;

/// How to measure connectivity between groups.
#[derive(Debug, Clone, Copy)]
pub enum ConnectivityMeasure {
    /// Count the number of streamlines connecting two groups.
    Count,
    /// Sum a DPS value for streamlines connecting two groups.
    WeightedSum,
}

/// Compute a pairwise connectivity matrix between groups.
///
/// Returns a packed upper-triangle vector of size `n*(n+1)/2` where
/// `n = group_names.len()`. Entry `(i,j)` with `i <= j` is at index
/// `i*n - i*(i+1)/2 + j`.
///
/// Each streamline is assigned to a group if its index appears in that
/// group's member list. A streamline connecting groups `i` and `j` increments
/// the `(min(i,j), max(i,j))` entry.
pub fn compute_group_connectivity<P: TrxScalar>(
    trx: &TrxFile<P>,
    group_names: &[&str],
    measure: ConnectivityMeasure,
    dps_weight_name: Option<&str>,
) -> Result<Vec<f64>> {
    let n = group_names.len();
    let matrix_size = n * (n + 1) / 2;
    let mut matrix = vec![0.0f64; matrix_size];

    // Build streamline → group membership
    let mut streamline_groups: Vec<Vec<usize>> = vec![Vec::new(); trx.nb_streamlines()];

    for (gi, &gname) in group_names.iter().enumerate() {
        let members = trx.group(gname)?;
        for &m in members {
            let idx = m as usize;
            if idx < streamline_groups.len() {
                streamline_groups[idx].push(gi);
            }
        }
    }

    // Optional DPS weights
    let weights: Option<Vec<f64>> = match (measure, dps_weight_name) {
        (ConnectivityMeasure::WeightedSum, Some(name)) => {
            let view = trx.dps::<f32>(name)?;
            Some(view.rows().map(|r| r[0] as f64).collect())
        }
        (ConnectivityMeasure::WeightedSum, None) => {
            return Err(TrxError::Argument(
                "WeightedSum requires a DPS weight name".into(),
            ));
        }
        _ => None,
    };

    // Accumulate
    for (si, groups) in streamline_groups.iter().enumerate() {
        let val = weights.as_ref().map_or(1.0, |w| w[si]);
        for &gi in groups {
            for &gj in groups {
                let (a, b) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let idx = a * n - a * (a + 1) / 2 + b;
                matrix[idx] += val;
            }
        }
    }

    Ok(matrix)
}
