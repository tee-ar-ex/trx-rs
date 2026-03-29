use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;
use serde_json::json;

use crate::dtype::DType;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::mmap_backing::vec_to_bytes;
use crate::tractogram::Tractogram;
use crate::trx_file::DataArray;

pub fn read_tt(path: &Path) -> Result<Tractogram> {
    if !path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".tt.gz"))
    {
        return Err(TrxError::Format(
            "Tiny Track import currently supports .tt.gz inputs only".into(),
        ));
    }

    let bytes = read_gzip(path)?;
    let records = parse_mat_records(&bytes)?;

    let dimensions = parse_u32_triplet(required_record(&records, "dimension")?)?;
    let _voxel_size = parse_f32_triplet(required_record(&records, "voxel_size")?)?;
    let affine_flat = parse_f32_array(required_record(&records, "trans_to_mni")?, 16)?;
    let affine = affine_from_flat(&affine_flat);

    let sidecar_labels = read_labels_sidecar(path)?;
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
    let report = records
        .get("report")
        .map(parse_text)
        .transpose()?
        .unwrap_or_default();
    let parameter_id = records
        .get("parameter_id")
        .map(parse_text)
        .transpose()?
        .unwrap_or_default();

    let mut streamlines_vox = Vec::new();
    for (_, payload) in track_records(&records)? {
        streamlines_vox.extend(decode_track_block(payload)?);
    }

    if !cluster_ids.is_empty() && cluster_ids.len() != streamlines_vox.len() {
        return Err(TrxError::Format(format!(
            "TT cluster count {} does not match streamline count {}",
            cluster_ids.len(),
            streamlines_vox.len()
        )));
    }

    let mut header = Header {
        voxel_to_rasmm: affine,
        dimensions: [
            dimensions[0] as u64,
            dimensions[1] as u64,
            dimensions[2] as u64,
        ],
        nb_streamlines: 0,
        nb_vertices: 0,
        extra: Default::default(),
    };
    if !report.is_empty() {
        header
            .extra
            .insert("tt_report".into(), serde_json::Value::String(report));
    }
    if !parameter_id.is_empty() {
        header.extra.insert(
            "tt_parameter_id".into(),
            serde_json::Value::String(parameter_id),
        );
    }

    let mut tractogram = Tractogram::with_header(header);
    let mut cluster_members: BTreeMap<u16, Vec<u32>> = BTreeMap::new();

    for (index, streamline_vox) in streamlines_vox.iter().enumerate() {
        let streamline_world: Vec<[f32; 3]> = streamline_vox
            .iter()
            .map(|point| apply_affine(affine, *point))
            .collect();
        tractogram.push_streamline(&streamline_world)?;
        if let Some(&cluster_id) = cluster_ids.get(index) {
            cluster_members
                .entry(cluster_id)
                .or_default()
                .push(index as u32);
        }
    }

    let group_names = resolve_group_names(&cluster_members, &sidecar_labels);
    for (cluster_id, members) in &cluster_members {
        let name = group_names
            .get(cluster_id)
            .ok_or_else(|| TrxError::Format("missing resolved TT group name".into()))?;
        tractogram.insert_group(name.clone(), members.clone());
        if let Some(&packed) = colors.get(*cluster_id as usize) {
            tractogram.insert_dpg(
                name.clone(),
                "color",
                DataArray::owned_bytes(vec_to_bytes(vec![packed_color_to_rgb(packed)]), 3, DType::UInt8),
            );
        }
    }

    if !colors.is_empty() {
        tractogram.extra_mut().insert(
            "tt_raw_colors".into(),
            json!(colors
                .iter()
                .map(|color| format!("0x{color:08x}"))
                .collect::<Vec<_>>()),
        );
    }

    Ok(tractogram)
}

fn read_gzip(path: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut decoder = MultiGzDecoder::new(&mut file);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[derive(Debug)]
struct MatRecord {
    rows: usize,
    cols: usize,
    ty: u32,
    payload: Vec<u8>,
}

fn parse_mat_records(bytes: &[u8]) -> Result<HashMap<String, MatRecord>> {
    let mut records = HashMap::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        if cursor + 20 > bytes.len() {
            return Err(TrxError::Format("truncated TT MAT record header".into()));
        }
        let ty = read_u32(bytes, &mut cursor)?;
        let rows = read_u32(bytes, &mut cursor)? as usize;
        let cols = read_u32(bytes, &mut cursor)? as usize;
        let _imagf = read_u32(bytes, &mut cursor)?;
        let namelen = read_u32(bytes, &mut cursor)? as usize;

        let elem_size = mat_elem_size(ty)?;
        if namelen == 0 {
            return Err(TrxError::Format(
                "TT MAT record has zero-length name".into(),
            ));
        }
        if cursor + namelen > bytes.len() {
            return Err(TrxError::Format("truncated TT MAT record name".into()));
        }
        let name_bytes = &bytes[cursor..cursor + namelen];
        cursor += namelen;
        let name = name_bytes
            .split(|byte| *byte == 0)
            .next()
            .unwrap_or_default();
        let name = std::str::from_utf8(name)
            .map_err(|_| TrxError::Format("TT MAT record name is not valid UTF-8".into()))?
            .to_string();

        let payload_len = rows
            .checked_mul(cols)
            .and_then(|size| size.checked_mul(elem_size))
            .ok_or_else(|| TrxError::Format(format!("TT MAT record '{name}' is too large")))?;
        if cursor + payload_len > bytes.len() {
            return Err(TrxError::Format(format!(
                "truncated TT MAT payload for '{name}'"
            )));
        }
        let payload = bytes[cursor..cursor + payload_len].to_vec();
        cursor += payload_len;
        records.insert(
            name,
            MatRecord {
                rows,
                cols,
                ty,
                payload,
            },
        );
    }
    Ok(records)
}

fn mat_elem_size(ty: u32) -> Result<usize> {
    match ty {
        0 => Ok(8),
        10 => Ok(4),
        20 => Ok(4),
        30 | 40 => Ok(2),
        50 => Ok(1),
        60 => Ok(8),
        other => Err(TrxError::Format(format!(
            "unsupported TT MAT type code {other}"
        ))),
    }
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

fn apply_affine(affine: [[f64; 4]; 4], point: [f32; 3]) -> [f32; 3] {
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

fn read_labels_sidecar(path: &Path) -> Result<Vec<String>> {
    let mut sidecar = PathBuf::from(path);
    let file_name = sidecar
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| TrxError::Argument(format!("invalid TT path {}", path.display())))?;
    sidecar.set_file_name(format!("{file_name}.txt"));
    if !sidecar.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(sidecar)?;
    Ok(text.lines().map(|line| line.trim().to_string()).collect())
}

fn resolve_group_names(
    cluster_members: &BTreeMap<u16, Vec<u32>>,
    labels: &[String],
) -> HashMap<u16, String> {
    let mut used = HashSet::new();
    let mut resolved = HashMap::new();
    for cluster_id in cluster_members.keys() {
        let base = labels
            .get(*cluster_id as usize)
            .map(|name| sanitize_group_name(name))
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| format!("cluster_{cluster_id}"));
        let mut candidate = base.clone();
        let mut suffix = 1usize;
        while !used.insert(candidate.clone()) {
            candidate = format!("{base}_{suffix}");
            suffix += 1;
        }
        resolved.insert(*cluster_id, candidate);
    }
    resolved
}

fn sanitize_group_name(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '\0' => '_',
            _ => ch,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

fn packed_color_to_rgb(color: u32) -> [u8; 3] {
    [
        ((color >> 16) & 0xff) as u8,
        ((color >> 8) & 0xff) as u8,
        (color & 0xff) as u8,
    ]
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = *cursor + 4;
    let chunk = bytes
        .get(*cursor..end)
        .ok_or_else(|| TrxError::Format("truncated TT MAT integer".into()))?;
    *cursor = end;
    Ok(u32::from_le_bytes(chunk.try_into().unwrap()))
}
