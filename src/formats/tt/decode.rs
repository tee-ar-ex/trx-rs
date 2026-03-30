use std::collections::HashMap;

use crate::error::{Result, TrxError};

use super::mat::MatRecord;

pub(super) struct TinyTrackData {
    pub dimensions: [u32; 3],
    pub affine: [[f64; 4]; 4],
    pub streamlines_vox: Vec<Vec<[f32; 3]>>,
    pub cluster_ids: Vec<u16>,
    pub colors: Vec<u32>,
    pub report: Option<String>,
    pub parameter_id: Option<String>,
}

pub(super) fn decode_tiny_track(records: &HashMap<String, MatRecord>) -> Result<TinyTrackData> {
    let dimensions = parse_u32_triplet(required_record(records, "dimension")?)?;
    let _voxel_size = parse_f32_triplet(required_record(records, "voxel_size")?)?;
    let affine_flat = parse_f32_array(required_record(records, "trans_to_mni")?, 16)?;
    let affine = affine_from_flat(&affine_flat);

    let cluster_ids = records
        .get("cluster")
        .map(parse_u16_vec)
        .transpose()?
        .unwrap_or_default();
    let colors = records
        .get("color")
        .map(parse_u32_vec)
        .transpose()?
        .unwrap_or_default();
    let report = records.get("report").map(parse_text).transpose()?;
    let parameter_id = records.get("parameter_id").map(parse_text).transpose()?;

    let mut streamlines_vox = Vec::new();
    for (_, payload) in track_records(records)? {
        streamlines_vox.extend(decode_track_block(payload)?);
    }

    if !cluster_ids.is_empty() && cluster_ids.len() != streamlines_vox.len() {
        return Err(TrxError::Format(format!(
            "TT cluster count {} does not match streamline count {}",
            cluster_ids.len(),
            streamlines_vox.len()
        )));
    }

    Ok(TinyTrackData {
        dimensions,
        affine,
        streamlines_vox,
        cluster_ids,
        colors,
        report,
        parameter_id,
    })
}

pub(super) fn apply_affine(affine: [[f64; 4]; 4], point: [f32; 3]) -> [f32; 3] {
    [
        (affine[0][0] * point[0] as f64
            + affine[0][1] * point[1] as f64
            + affine[0][2] * point[2] as f64
            + affine[0][3]) as f32,
        (affine[1][0] * point[0] as f64
            + affine[1][1] * point[1] as f64
            + affine[1][2] * point[2] as f64
            + affine[1][3]) as f32,
        (affine[2][0] * point[0] as f64
            + affine[2][1] * point[1] as f64
            + affine[2][2] * point[2] as f64
            + affine[2][3]) as f32,
    ]
}

fn required_record<'a>(
    records: &'a HashMap<String, MatRecord>,
    name: &str,
) -> Result<&'a MatRecord> {
    records
        .get(name)
        .ok_or_else(|| TrxError::Format(format!("missing TT record '{name}'")))
}

fn parse_u32_triplet(record: &MatRecord) -> Result<[u32; 3]> {
    if record.rows * record.cols != 3 || record.ty != 20 {
        return Err(TrxError::Format("invalid TT dimension record".into()));
    }
    let mut out = [0u32; 3];
    for (index, value) in out.iter_mut().enumerate() {
        let start = index * 4;
        *value = u32::from_le_bytes(record.payload[start..start + 4].try_into().unwrap());
    }
    Ok(out)
}

fn parse_f32_triplet(record: &MatRecord) -> Result<[f32; 3]> {
    let values = parse_f32_array(record, 3)?;
    Ok([values[0], values[1], values[2]])
}

fn parse_f32_array(record: &MatRecord, len: usize) -> Result<Vec<f32>> {
    if record.rows * record.cols != len || record.ty != 10 {
        return Err(TrxError::Format("invalid TT float record".into()));
    }
    Ok(record
        .payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_u16_vec(record: &MatRecord) -> Result<Vec<u16>> {
    if record.ty != 40 {
        return Err(TrxError::Format("invalid TT cluster record dtype".into()));
    }
    Ok(record
        .payload
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_u32_vec(record: &MatRecord) -> Result<Vec<u32>> {
    if record.ty != 20 {
        return Err(TrxError::Format("invalid TT uint32 record dtype".into()));
    }
    Ok(record
        .payload
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_text(record: &MatRecord) -> Result<String> {
    if record.ty != 50 {
        return Err(TrxError::Format("invalid TT text record dtype".into()));
    }
    let text = record
        .payload
        .split(|byte| *byte == 0)
        .next()
        .unwrap_or_default();
    Ok(std::str::from_utf8(text)
        .map_err(|_| TrxError::Format("TT text record is not valid UTF-8".into()))?
        .to_string())
}

fn track_records(records: &HashMap<String, MatRecord>) -> Result<Vec<(usize, &[u8])>> {
    let mut tracks = Vec::new();
    for (name, record) in records {
        if name == "track" {
            if record.ty != 50 {
                return Err(TrxError::Format("TT track block must be uint8".into()));
            }
            tracks.push((0, record.payload.as_slice()));
            continue;
        }
        if let Some(index) = name.strip_prefix("track") {
            let block = index
                .parse::<usize>()
                .map_err(|_| TrxError::Format(format!("invalid TT track block name '{name}'")))?;
            if record.ty != 50 {
                return Err(TrxError::Format("TT track block must be uint8".into()));
            }
            tracks.push((block, record.payload.as_slice()));
        }
    }
    if tracks.is_empty() {
        return Err(TrxError::Format("missing TT track payload".into()));
    }
    tracks.sort_by_key(|(index, _)| *index);
    Ok(tracks)
}

fn decode_track_block(payload: &[u8]) -> Result<Vec<Vec<[f32; 3]>>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < payload.len() {
        if cursor + 16 > payload.len() {
            return Err(TrxError::Format("truncated TT tract header".into()));
        }
        let count = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        let x0 = i32::from_le_bytes(payload[cursor + 4..cursor + 8].try_into().unwrap());
        let y0 = i32::from_le_bytes(payload[cursor + 8..cursor + 12].try_into().unwrap());
        let z0 = i32::from_le_bytes(payload[cursor + 12..cursor + 16].try_into().unwrap());
        cursor += 16;

        if count < 3 {
            return Err(TrxError::Format(format!(
                "invalid TT tract coordinate count {count}"
            )));
        }

        let delta_count = count - 3;
        if cursor + delta_count > payload.len() {
            return Err(TrxError::Format("truncated TT tract delta payload".into()));
        }

        let mut values = Vec::with_capacity(count);
        values.extend([x0, y0, z0]);
        for delta in &payload[cursor..cursor + delta_count] {
            let previous = values[values.len() - 3];
            values.push(previous + i32::from(*delta as i8));
        }
        cursor += delta_count;

        let streamline = values
            .chunks_exact(3)
            .map(|chunk| {
                [
                    chunk[0] as f32 / 32.0,
                    chunk[1] as f32 / 32.0,
                    chunk[2] as f32 / 32.0,
                ]
            })
            .collect();
        out.push(streamline);
    }
    Ok(out)
}

fn affine_from_flat(values: &[f32]) -> [[f64; 4]; 4] {
    [
        [
            values[0] as f64,
            values[1] as f64,
            values[2] as f64,
            values[3] as f64,
        ],
        [
            values[4] as f64,
            values[5] as f64,
            values[6] as f64,
            values[7] as f64,
        ],
        [
            values[8] as f64,
            values[9] as f64,
            values[10] as f64,
            values[11] as f64,
        ],
        [
            values[12] as f64,
            values[13] as f64,
            values[14] as f64,
            values[15] as f64,
        ],
    ]
}
