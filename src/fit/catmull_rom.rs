//! Core Catmull-Rom fitting + sampling, free of any tractogram concerns so
//! the algorithm can be reused on raw polylines.

/// Endpoints are always kept, so the minimum output size is 2 — even when
/// the input has fewer points the function returns the input unchanged.
pub const MIN_KEPT_POINTS: usize = 2;

/// Number of evenly-spaced samples per segment used to tessellate the
/// candidate curve when measuring fit error. 32 keeps the polyline within
/// ~`segment_length / 32` of the true curve, which is well below typical
/// `epsilon_mm` values of 0.5–2 mm for brain tractography.
const SAMPLES_PER_SEGMENT: usize = 32;

/// Floor on chord length (mm) used when computing centripetal knots. Avoids
/// a divide-by-zero when two consecutive control points coincide.
const MIN_CHORD: f32 = 1e-4;

/// Fit a Catmull-Rom curve through a sparse subset of `points` such that the
/// curve stays within `epsilon_mm` of every input vertex. Returns the
/// retained input indices in ascending order.
///
/// `points` are 3D coordinates in any consistent unit; `epsilon_mm` must be
/// in the same unit. Inputs of fewer than 3 points or non-positive epsilons
/// fall through with every index retained.
///
/// Both endpoints are always retained.
pub fn fit_catmull_rom_indices(points: &[[f32; 3]], epsilon_mm: f32) -> Vec<usize> {
    let n = points.len();
    if n <= 2 || epsilon_mm <= 0.0 {
        return (0..n).collect();
    }
    let eps2 = epsilon_mm * epsilon_mm;

    let mut kept: Vec<usize> = Vec::with_capacity(16);
    kept.push(0);
    kept.push(n - 1);

    let mut tess_buf: Vec<[f32; 3]> = Vec::with_capacity(SAMPLES_PER_SEGMENT + 1);
    let mut additions: Vec<usize> = Vec::with_capacity(16);

    loop {
        additions.clear();

        for seg in 0..kept.len() - 1 {
            let start = kept[seg];
            let end = kept[seg + 1];
            if end <= start + 1 {
                continue; // no interior input vertices in this segment
            }

            tessellate_segment(points, &kept, seg, &mut tess_buf);

            let mut worst_idx = start + 1;
            let mut worst_d2 = 0.0_f32;
            for input_idx in (start + 1)..end {
                let d2 = point_to_polyline_dist2(points[input_idx], &tess_buf);
                if d2 > worst_d2 {
                    worst_d2 = d2;
                    worst_idx = input_idx;
                }
            }

            if worst_d2 > eps2 {
                additions.push(worst_idx);
            }
        }

        if additions.is_empty() {
            break;
        }

        // `additions` come from disjoint segments, so they're already in
        // ascending order. Insert into `kept` while maintaining sortedness.
        merge_sorted_into(&mut kept, &additions);
    }

    kept
}

/// Convenience: return the simplified streamline as owned positions.
pub fn simplify_streamline(points: &[[f32; 3]], epsilon_mm: f32) -> Vec<[f32; 3]> {
    let indices = fit_catmull_rom_indices(points, epsilon_mm);
    indices.iter().map(|&i| points[i]).collect()
}

/// Sample a Catmull-Rom curve through `cps` with `samples_per_segment`
/// evenly-spaced parameter steps per segment. The first CP appears once at
/// the start; each subsequent segment contributes `samples_per_segment`
/// points (so the final CP appears once at the end). Returns owned points.
pub fn sample_catmull_rom(cps: &[[f32; 3]], samples_per_segment: usize) -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    sample_catmull_rom_into(cps, samples_per_segment, &mut out);
    out
}

/// Sample a Catmull-Rom curve into a caller-provided buffer. The buffer is
/// cleared first; capacity is reused. Useful in tight loops to avoid
/// per-call allocations.
pub fn sample_catmull_rom_into(
    cps: &[[f32; 3]],
    samples_per_segment: usize,
    out: &mut Vec<[f32; 3]>,
) {
    out.clear();
    let n = cps.len();
    if n == 0 {
        return;
    }
    if n < 2 || samples_per_segment == 0 {
        out.extend_from_slice(cps);
        return;
    }
    out.reserve((n - 1) * samples_per_segment + 1);
    out.push(cps[0]);
    let inv = 1.0 / samples_per_segment as f32;
    for seg in 0..n - 1 {
        let cps4 = segment_control_points_from_dense(cps, seg);
        let knots = centripetal_knots(cps4);
        for k in 1..=samples_per_segment {
            let u = k as f32 * inv;
            out.push(catmull_rom_segment_with_knots(cps4, &knots, u));
        }
    }
}

// ─── internal helpers ────────────────────────────────────────────────────────

fn tessellate_segment(points: &[[f32; 3]], kept: &[usize], seg: usize, out: &mut Vec<[f32; 3]>) {
    out.clear();
    let cps = segment_control_points(points, kept, seg);
    let knots = centripetal_knots(cps);
    let inv = 1.0 / SAMPLES_PER_SEGMENT as f32;
    for k in 0..=SAMPLES_PER_SEGMENT {
        let u = k as f32 * inv;
        out.push(catmull_rom_segment_with_knots(cps, &knots, u));
    }
}

/// Pull out the four Catmull-Rom control points for the segment between
/// `kept[seg]` and `kept[seg + 1]`, using endpoint-reflection for `p0` /
/// `p3` at the curve's boundaries.
fn segment_control_points(points: &[[f32; 3]], kept: &[usize], seg: usize) -> [[f32; 3]; 4] {
    let n = kept.len();
    let p1 = points[kept[seg]];
    let p2 = points[kept[seg + 1]];
    let p0 = if seg == 0 {
        reflect(p1, p2)
    } else {
        points[kept[seg - 1]]
    };
    let p3 = if seg + 2 == n {
        reflect(p2, p1)
    } else {
        points[kept[seg + 2]]
    };
    [p0, p1, p2, p3]
}

/// Same as `segment_control_points` but indexes directly into `cps` (i.e.
/// every consecutive pair is a segment, no `kept` indirection).
fn segment_control_points_from_dense(cps: &[[f32; 3]], seg: usize) -> [[f32; 3]; 4] {
    let n = cps.len();
    let p1 = cps[seg];
    let p2 = cps[seg + 1];
    let p0 = if seg == 0 {
        reflect(p1, p2)
    } else {
        cps[seg - 1]
    };
    let p3 = if seg + 2 == n {
        reflect(p2, p1)
    } else {
        cps[seg + 2]
    };
    [p0, p1, p2, p3]
}

/// Reflect `b` across `a`: returns `a + (a - b) = 2a - b`.
#[inline]
fn reflect(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [2.0 * a[0] - b[0], 2.0 * a[1] - b[1], 2.0 * a[2] - b[2]]
}

/// Centripetal Catmull-Rom (α = 0.5).
///
/// Knot intervals are the chord lengths raised to the α = 0.5 power. This
/// is the parameterisation that's mathematically guaranteed not to form
/// loops or cusps near closely-spaced control points — see Yuksel,
/// Schaefer, Keyser, "Parameterization and Applications of Catmull-Rom
/// Curves" (CAD 2011). The plain uniform Catmull-Rom (α = 0) overshoots
/// when adjacent CPs are unevenly spaced, which is precisely the regime
/// the iterative fitter produces (it densifies CPs in high-curvature
/// regions).
///
/// Evaluated via Aitken-Neville so the formula stays compact and
/// numerically clean. The four knot values are precomputed once per
/// segment by `centripetal_knots` and threaded through here so the
/// per-sample work is just five lerps + a few divisions.
#[derive(Clone, Copy)]
struct CentripetalKnots {
    t0: f32,
    t1: f32,
    t2: f32,
    t3: f32,
}

#[inline]
fn centripetal_knots(cps: [[f32; 3]; 4]) -> CentripetalKnots {
    let [p0, p1, p2, p3] = cps;
    let t0 = 0.0;
    let t1 = t0 + chord_centripetal(p0, p1);
    let t2 = t1 + chord_centripetal(p1, p2);
    let t3 = t2 + chord_centripetal(p2, p3);
    CentripetalKnots { t0, t1, t2, t3 }
}

#[inline]
fn chord_centripetal(a: [f32; 3], b: [f32; 3]) -> f32 {
    // ||b - a||^0.5  =  (||b - a||²)^0.25  =  sqrt(sqrt(d²)).
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    let d2 = dx * dx + dy * dy + dz * dz;
    d2.sqrt().sqrt().max(MIN_CHORD)
}

#[inline]
fn catmull_rom_segment_with_knots(
    cps: [[f32; 3]; 4],
    knots: &CentripetalKnots,
    u: f32,
) -> [f32; 3] {
    let [p0, p1, p2, p3] = cps;
    let CentripetalKnots { t0, t1, t2, t3 } = *knots;
    // Map u ∈ [0, 1] to t ∈ [t1, t2] — the parameter interval that
    // produces the segment between p1 and p2.
    let t = t1 + u * (t2 - t1);

    // First-level Aitken interpolations.
    let a1 = lerp3(p0, p1, (t - t0) / (t1 - t0));
    let a2 = lerp3(p1, p2, (t - t1) / (t2 - t1));
    let a3 = lerp3(p2, p3, (t - t2) / (t3 - t2));

    // Second level.
    let b1 = lerp3(a1, a2, (t - t0) / (t2 - t0));
    let b2 = lerp3(a2, a3, (t - t1) / (t3 - t1));

    // Final blend.
    lerp3(b1, b2, (t - t1) / (t2 - t1))
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Minimum squared distance from `p` to the polyline formed by consecutive
/// pairs of `polyline`. `polyline.len()` must be ≥ 2.
#[inline]
fn point_to_polyline_dist2(p: [f32; 3], polyline: &[[f32; 3]]) -> f32 {
    let mut min_d2 = f32::INFINITY;
    for window in polyline.windows(2) {
        let d2 = point_to_segment_dist2(p, window[0], window[1]);
        if d2 < min_d2 {
            min_d2 = d2;
        }
    }
    min_d2
}

#[inline]
fn point_to_segment_dist2(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let abz = b[2] - a[2];
    let denom = abx * abx + aby * aby + abz * abz;
    if denom < f32::EPSILON {
        let dx = p[0] - a[0];
        let dy = p[1] - a[1];
        let dz = p[2] - a[2];
        return dx * dx + dy * dy + dz * dz;
    }
    let apx = p[0] - a[0];
    let apy = p[1] - a[1];
    let apz = p[2] - a[2];
    let mut t = (apx * abx + apy * aby + apz * abz) / denom;
    if t < 0.0 {
        t = 0.0;
    } else if t > 1.0 {
        t = 1.0;
    }
    let dx = apx - abx * t;
    let dy = apy - aby * t;
    let dz = apz - abz * t;
    dx * dx + dy * dy + dz * dz
}

/// Merge a *sorted* slice of new indices into a *sorted* `Vec`, preserving
/// sortedness and deduplicating. Single allocation, single pass.
fn merge_sorted_into(kept: &mut Vec<usize>, additions: &[usize]) {
    if additions.is_empty() {
        return;
    }
    let mut merged = Vec::with_capacity(kept.len() + additions.len());
    let (mut i, mut j) = (0, 0);
    while i < kept.len() && j < additions.len() {
        let a = kept[i];
        let b = additions[j];
        if a < b {
            merged.push(a);
            i += 1;
        } else if a > b {
            merged.push(b);
            j += 1;
        } else {
            merged.push(a);
            i += 1;
            j += 1;
        }
    }
    merged.extend_from_slice(&kept[i..]);
    merged.extend_from_slice(&additions[j..]);
    *kept = merged;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic_curve(steps: usize) -> Vec<[f32; 3]> {
        (0..=steps)
            .map(|i| {
                let t = i as f32 / steps as f32;
                let x = -25.0 + 50.0 * t;
                let y = 10.0 * (t * std::f32::consts::PI).sin();
                let z = 5.0 * t;
                [x, y, z]
            })
            .collect()
    }

    #[test]
    fn endpoints_always_retained() {
        let pts = cubic_curve(50);
        let kept = fit_catmull_rom_indices(&pts, 1.0);
        assert_eq!(kept.first(), Some(&0));
        assert_eq!(kept.last(), Some(&50));
    }

    #[test]
    fn straight_line_collapses_to_endpoints() {
        let pts: Vec<[f32; 3]> = (0..20).map(|i| [i as f32, 0.0, 0.0]).collect();
        let kept = fit_catmull_rom_indices(&pts, 0.01);
        assert_eq!(kept, vec![0, 19]);
    }

    #[test]
    fn fewer_than_three_points_pass_through() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        assert_eq!(fit_catmull_rom_indices(&pts, 1.0), vec![0, 1]);
        let pts = vec![[0.0, 0.0, 0.0]];
        assert_eq!(fit_catmull_rom_indices(&pts, 1.0), vec![0]);
        assert_eq!(fit_catmull_rom_indices(&[], 1.0), Vec::<usize>::new());
    }

    #[test]
    fn fit_within_tolerance_for_smooth_curve() {
        let pts = cubic_curve(200);
        let epsilon = 0.5;
        let kept = fit_catmull_rom_indices(&pts, epsilon);
        // Re-densify the kept set via Catmull-Rom and verify every input
        // vertex is within ε of the resulting polyline.
        let kept_pts: Vec<[f32; 3]> = kept.iter().map(|&i| pts[i]).collect();
        let dense = sample_catmull_rom(&kept_pts, 32);
        let max_d = pts
            .iter()
            .map(|&p| point_to_polyline_dist2(p, &dense).sqrt())
            .fold(0.0_f32, f32::max);
        assert!(
            max_d <= epsilon * 1.05,
            "max error {max_d} exceeded tolerance {epsilon}"
        );
        // Sanity: should be much smaller than the input.
        assert!(
            kept.len() < pts.len() / 4,
            "expected sparse output (got {} from {})",
            kept.len(),
            pts.len()
        );
    }

    #[test]
    fn smaller_epsilon_keeps_more_points() {
        let pts = cubic_curve(200);
        let coarse = fit_catmull_rom_indices(&pts, 2.0).len();
        let fine = fit_catmull_rom_indices(&pts, 0.1).len();
        assert!(fine > coarse, "fine {fine} should exceed coarse {coarse}");
    }

    #[test]
    fn sampler_passes_through_control_points() {
        let cps = vec![
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 3.0, 0.0],
        ];
        let dense = sample_catmull_rom(&cps, 10);
        assert_eq!(dense.first(), Some(&cps[0]));
        assert_eq!(dense.last(), Some(&cps[3]));
        for (i, cp) in cps.iter().enumerate() {
            let idx = i * 10;
            let d2 = (dense[idx][0] - cp[0]).powi(2)
                + (dense[idx][1] - cp[1]).powi(2)
                + (dense[idx][2] - cp[2]).powi(2);
            assert!(d2 < 1e-8, "CP {i} not hit at sample {idx}");
        }
    }

    /// Centripetal CR's defining property: the curve through any 4 CPs is
    /// guaranteed to stay within the convex hull of those CPs locally,
    /// even when CPs are clustered. We construct a curve with two CPs
    /// very close together and verify the densified samples between them
    /// don't bulge sideways (the failure mode of uniform CR).
    #[test]
    fn no_overshoot_with_clustered_cps() {
        // Polyline with a slight kink at the middle; close spacing between
        // CPs 1 and 2 is exactly the shape that makes uniform CR overshoot.
        let cps = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ];
        let dense = sample_catmull_rom(&cps, 64);
        // The cluster is colinear in y/z, so any non-zero excursion in y or z
        // is overshoot. Centripetal must keep that bounded by the chord.
        let max_y = dense.iter().map(|p| p[1].abs()).fold(0.0_f32, f32::max);
        let max_z = dense.iter().map(|p| p[2].abs()).fold(0.0_f32, f32::max);
        assert!(max_y < 1e-3, "centripetal CR overshot in y: {max_y}");
        assert!(max_z < 1e-3, "centripetal CR overshot in z: {max_z}");
    }

    #[test]
    fn merge_sorted_dedup_works() {
        let mut kept = vec![0, 5, 10];
        merge_sorted_into(&mut kept, &[3, 7]);
        assert_eq!(kept, vec![0, 3, 5, 7, 10]);

        let mut kept = vec![0, 10];
        merge_sorted_into(&mut kept, &[]);
        assert_eq!(kept, vec![0, 10]);

        let mut kept = vec![0, 5];
        merge_sorted_into(&mut kept, &[5, 7]); // dedup 5
        assert_eq!(kept, vec![0, 5, 7]);
    }
}
