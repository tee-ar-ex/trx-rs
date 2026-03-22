use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::Result;

/// TRX file header (stored as JSON in `header.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    /// 4x4 affine matrix mapping voxel coordinates to RAS+mm space.
    /// Stored row-major as `[[f64; 4]; 4]`.
    #[serde(rename = "VOXEL_TO_RASMM")]
    pub voxel_to_rasmm: [[f64; 4]; 4],

    /// Volume dimensions `[x, y, z]`.
    #[serde(rename = "DIMENSIONS")]
    pub dimensions: [u64; 3],

    /// Total number of streamlines.
    #[serde(rename = "NB_STREAMLINES")]
    pub nb_streamlines: u64,

    /// Total number of vertices (points) across all streamlines.
    #[serde(rename = "NB_VERTICES")]
    pub nb_vertices: u64,

    /// Any extra fields not covered above.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Header {
    /// Read header from a `header.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let header: Header = serde_json::from_str(&data)?;
        Ok(header)
    }

    /// Serialize header to JSON string.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Write header to a file.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Identity affine (no transform).
    pub fn identity_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_serde_round_trip() {
        let header = Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [256, 256, 256],
            nb_streamlines: 100,
            nb_vertices: 5000,
            extra: HashMap::new(),
        };

        let json = header.to_json().unwrap();
        let parsed: Header = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.nb_streamlines, 100);
        assert_eq!(parsed.nb_vertices, 5000);
        assert_eq!(parsed.dimensions, [256, 256, 256]);
        assert_eq!(parsed.voxel_to_rasmm[0][0], 1.0);
        assert_eq!(parsed.voxel_to_rasmm[3][3], 1.0);
    }

    #[test]
    fn header_with_extra_fields() {
        let json = r#"{
            "VOXEL_TO_RASMM": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            "DIMENSIONS": [100, 100, 100],
            "NB_STREAMLINES": 42,
            "NB_VERTICES": 420,
            "CUSTOM_FIELD": "hello"
        }"#;

        let header: Header = serde_json::from_str(json).unwrap();
        assert_eq!(header.nb_streamlines, 42);
        assert_eq!(
            header.extra.get("CUSTOM_FIELD").unwrap(),
            &serde_json::Value::String("hello".into())
        );
    }
}
