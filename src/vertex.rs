use bytemuck::{Pod, Zeroable};

/// A 3D position stored as `[T; 3]`, `#[repr(C)]` for zero-copy GPU upload.
///
/// For `T = f32`, this is 12 bytes with no padding and maps directly to
/// `wgpu::VertexFormat::Float32x3`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Position3<T: Copy> {
    pub coords: [T; 3],
}

// SAFETY: [f32; 3] is Pod+Zeroable, and Position3 is #[repr(C)] with no padding.
unsafe impl Zeroable for Position3<f32> {}
unsafe impl Pod for Position3<f32> {}

unsafe impl Zeroable for Position3<f64> {}
unsafe impl Pod for Position3<f64> {}

unsafe impl Zeroable for Position3<half::f16> {}
unsafe impl Pod for Position3<half::f16> {}

impl<T: Copy> Position3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { coords: [x, y, z] }
    }
}

impl<T: Copy> From<[T; 3]> for Position3<T> {
    fn from(coords: [T; 3]) -> Self {
        Self { coords }
    }
}

impl<T: Copy> From<Position3<T>> for [T; 3] {
    fn from(p: Position3<T>) -> Self {
        p.coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position3_size_and_alignment() {
        assert_eq!(std::mem::size_of::<Position3<f32>>(), 12);
        assert_eq!(std::mem::align_of::<Position3<f32>>(), 4);

        assert_eq!(std::mem::size_of::<Position3<f64>>(), 24);

        assert_eq!(std::mem::size_of::<Position3<half::f16>>(), 6);
    }

    #[test]
    fn position3_bytemuck_cast() {
        let positions = vec![
            Position3::new(1.0f32, 2.0, 3.0),
            Position3::new(4.0f32, 5.0, 6.0),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&positions);
        assert_eq!(bytes.len(), 24);

        let back: &[Position3<f32>] = bytemuck::cast_slice(bytes);
        assert_eq!(back[0].coords, [1.0, 2.0, 3.0]);
        assert_eq!(back[1].coords, [4.0, 5.0, 6.0]);
    }
}
