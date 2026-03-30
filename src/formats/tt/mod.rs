mod decode;
mod mat;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::json;

use crate::dtype::DType;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::mmap_backing::vec_to_bytes;
use crate::tractogram::Tractogram;
use crate::trx_file::DataArray;

use self::decode::{apply_affine, decode_tiny_track, TinyTrackData};
use self::mat::read_tt_mat_records;

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

    let records = read_tt_mat_records(path)?;
    let tt = decode_tiny_track(&records)?;
    let sidecar_labels = read_labels_sidecar(path)?;
    build_tractogram(tt, &sidecar_labels)
}

fn build_tractogram(tt: TinyTrackData, sidecar_labels: &[String]) -> Result<Tractogram> {
    let mut header = Header {
        voxel_to_rasmm: tt.affine,
        dimensions: tt.dimensions.map(u64::from),
        nb_streamlines: 0,
        nb_vertices: 0,
        extra: Default::default(),
    };
    if let Some(report) = tt.report {
        header
            .extra
            .insert("tt_report".into(), serde_json::Value::String(report));
    }
    if let Some(parameter_id) = tt.parameter_id {
        header.extra.insert(
            "tt_parameter_id".into(),
            serde_json::Value::String(parameter_id),
        );
    }

    let mut tractogram = Tractogram::with_header(header);
    let mut cluster_members: BTreeMap<u16, Vec<u32>> = BTreeMap::new();

    for (index, streamline_vox) in tt.streamlines_vox.iter().enumerate() {
        let streamline_world: Vec<[f32; 3]> = streamline_vox
            .iter()
            .map(|point| apply_affine(tt.affine, *point))
            .collect();
        tractogram.push_streamline(&streamline_world)?;
        if let Some(cluster_id) = tt.cluster_ids.get(index).copied() {
            cluster_members
                .entry(cluster_id)
                .or_default()
                .push(index as u32);
        }
    }

    let group_names = resolve_group_names(&cluster_members, sidecar_labels);
    for (cluster_id, members) in &cluster_members {
        let name = group_names
            .get(cluster_id)
            .ok_or_else(|| TrxError::Format("missing resolved TT group name".into()))?;
        tractogram.insert_group(name.clone(), members.clone());
        if let Some(&packed) = tt.colors.get(*cluster_id as usize) {
            tractogram.insert_dpg(
                name.clone(),
                "color",
                DataArray::owned_bytes(
                    vec_to_bytes(vec![packed_color_to_rgb(packed)]),
                    3,
                    DType::UInt8,
                ),
            );
        }
    }

    if !tt.colors.is_empty() {
        tractogram.extra_mut().insert(
            "tt_raw_colors".into(),
            json!(tt
                .colors
                .iter()
                .map(|color| format!("0x{color:08x}"))
                .collect::<Vec<_>>()),
        );
    }

    Ok(tractogram)
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
