use bytemuck::Pod;

/// A 2D typed view over a flat slice, providing row-based access.
///
/// This is a lightweight wrapper that doesn't own the data — it borrows
/// a `&[T]` and interprets it as `nrows × ncols`.
#[derive(Debug, Clone, Copy)]
pub struct TypedView2D<'a, T: Pod> {
    data: &'a [T],
    ncols: usize,
}

impl<'a, T: Pod> TypedView2D<'a, T> {
    /// Create a 2D view over `data` with `ncols` columns.
    ///
    /// Panics if `data.len()` is not divisible by `ncols`.
    pub fn new(data: &'a [T], ncols: usize) -> Self {
        assert!(
            ncols > 0 && data.len().is_multiple_of(ncols),
            "data length {} is not divisible by ncols {}",
            data.len(),
            ncols,
        );
        Self { data, ncols }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.data.len() / self.ncols
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Shape as `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols)
    }

    /// Access row `i` as a slice of `ncols` elements.
    pub fn row(&self, i: usize) -> &'a [T] {
        let start = i * self.ncols;
        &self.data[start..start + self.ncols]
    }

    /// The underlying flat slice.
    pub fn as_flat_slice(&self) -> &'a [T] {
        self.data
    }

    /// Iterate over rows.
    pub fn rows(&self) -> impl Iterator<Item = &'a [T]> {
        self.data.chunks_exact(self.ncols)
    }
}

/// Mutable 2D typed view.
#[derive(Debug)]
pub struct TypedView2DMut<'a, T: Pod> {
    data: &'a mut [T],
    ncols: usize,
}

impl<'a, T: Pod> TypedView2DMut<'a, T> {
    pub fn new(data: &'a mut [T], ncols: usize) -> Self {
        assert!(
            ncols > 0 && data.len().is_multiple_of(ncols),
            "data length {} is not divisible by ncols {}",
            data.len(),
            ncols,
        );
        Self { data, ncols }
    }

    pub fn nrows(&self) -> usize {
        self.data.len() / self.ncols
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.ncols;
        let ncols = self.ncols;
        &mut self.data[start..start + ncols]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_view_2d_basics() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TypedView2D::new(&data, 3);

        assert_eq!(view.shape(), (2, 3));
        assert_eq!(view.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(view.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn typed_view_2d_rows_iter() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TypedView2D::new(&data, 2);
        let rows: Vec<_> = view.rows().collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2], &[5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn typed_view_2d_bad_shape() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        TypedView2D::new(&data, 3);
    }
}
