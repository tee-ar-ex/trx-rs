use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use flate2::read::MultiGzDecoder;

use crate::any_trx_file::AnyTrxFile;
use crate::dtype::DType;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::tractogram::Tractogram;
use crate::trx_file::{DataArray, TrxFile, TrxParts};

const TRK_HEADER_SIZE: usize = 1000;
const MAX_NAMED_SCALARS_PER_POINT: usize = 10;
const MAX_NAMED_PROPERTIES_PER_STREAMLINE: usize = 10;

pub fn read_trk(path: &Path, _header_override: Option<Header>) -> Result<Tractogram> {
    let parsed = parse_trk(path)?;
    if !parsed.dpv.is_empty() || !parsed.dps.is_empty() {
        return Err(TrxError::Format(
            "this .trk file contains TrackVis scalars/properties; convert it to .trx first to preserve metadata"
                .into(),
        ));
    }

    let mut tractogram = Tractogram::with_header(parsed.header);
    for streamline in parsed.streamlines {
        tractogram.push_streamline(&streamline)?;
    }
    Ok(tractogram)
}

pub fn convert_trk_to_trx(
    input: &Path,
    output: &Path,
    options: &crate::formats::ConversionOptions,
) -> Result<()> {
    let parsed = parse_trk(input)?;
    let positions: Vec<[f32; 3]> = parsed.streamlines.iter().flatten().copied().collect();
    let offsets = build_offsets(&parsed.streamlines)?;

    let file = TrxFile::from_parts(TrxParts {
        header: Header {
            nb_streamlines: parsed.streamlines.len() as u64,
            nb_vertices: positions.len() as u64,
            ..parsed.header
        },
        positions_backing: MmapBacking::Owned(vec_to_bytes(positions)),
        offsets_backing: MmapBacking::Owned(vec_to_bytes(offsets)),
        dps: parsed.dps,
        dpv: parsed.dpv,
        groups: HashMap::new(),
        dpg: HashMap::new(),
        tempdir: None,
    });

    let any = AnyTrxFile::F32(file);
    let any = if options.trx_positions_dtype == DType::Float32 {
        any
    } else {
        any.convert_positions_dtype(options.trx_positions_dtype)?
    };
    any.save(output)
}

struct ParsedTrk {
    header: Header,
    streamlines: Vec<Vec<[f32; 3]>>,
    dpv: HashMap<String, DataArray>,
    dps: HashMap<String, DataArray>,
}

#[derive(Clone, Copy)]
enum Endianness {
    Little,
    Big,
}

impl Endianness {
    fn read_i16(self, bytes: &[u8]) -> i16 {
        match self {
            Endianness::Little => i16::from_le_bytes(bytes.try_into().unwrap()),
            Endianness::Big => i16::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_i32(self, bytes: &[u8]) -> i32 {
        match self {
            Endianness::Little => i32::from_le_bytes(bytes.try_into().unwrap()),
            Endianness::Big => i32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_f32(self, bytes: &[u8]) -> f32 {
        match self {
            Endianness::Little => f32::from_le_bytes(bytes.try_into().unwrap()),
            Endianness::Big => f32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }
}

#[derive(Clone)]
struct SliceSpec {
    name: String,
    start: usize,
    len: usize,
}

fn parse_trk(path: &Path) -> Result<ParsedTrk> {
    let bytes = read_maybe_gzip(path)?;
    if bytes.len() < TRK_HEADER_SIZE {
        return Err(TrxError::Format(
            "TRK file is smaller than the 1000-byte header".into(),
        ));
    }
    if &bytes[..5] != b"TRACK" {
        return Err(TrxError::Format(
            "file does not start with TrackVis magic".into(),
        ));
    }

    let header_bytes = &bytes[..TRK_HEADER_SIZE];
    let endian = detect_endianness(header_bytes)?;
    let version = endian.read_i32(&header_bytes[992..996]);
    if version != 2 {
        return Err(TrxError::Format(format!(
            "unsupported TrackVis version {version}; only v2 is supported"
        )));
    }

    let dimensions = [
        parse_positive_i16(endian.read_i16(&header_bytes[6..8]), "dim[0]")? as usize,
        parse_positive_i16(endian.read_i16(&header_bytes[8..10]), "dim[1]")? as usize,
        parse_positive_i16(endian.read_i16(&header_bytes[10..12]), "dim[2]")? as usize,
    ];
    let voxel_sizes = [
        parse_positive_f32(endian.read_f32(&header_bytes[12..16]), "voxel_size[0]")?,
        parse_positive_f32(endian.read_f32(&header_bytes[16..20]), "voxel_size[1]")?,
        parse_positive_f32(endian.read_f32(&header_bytes[20..24]), "voxel_size[2]")?,
    ];

    let n_scalars = parse_nonnegative_i16(endian.read_i16(&header_bytes[36..38]), "n_scalars")?;
    let n_properties =
        parse_nonnegative_i16(endian.read_i16(&header_bytes[238..240]), "n_properties")?;

    let scalar_specs = parse_name_specs(
        &header_bytes[38..238],
        n_scalars,
        MAX_NAMED_SCALARS_PER_POINT,
        "scalars",
    )?;
    let property_specs = parse_name_specs(
        &header_bytes[240..440],
        n_properties,
        MAX_NAMED_PROPERTIES_PER_STREAMLINE,
        "properties",
    )?;

    let voxel_to_rasmm_f32 = parse_affine_f32(endian, &header_bytes[440..504])?;
    if affine_is_all_zero(&voxel_to_rasmm_f32) {
        return Err(TrxError::Format(
            "TRK vox_to_ras is missing or zero; convert it with a more permissive tool first"
                .into(),
        ));
    }
    let affine_codes = affine_to_axcodes(&voxel_to_rasmm_f32)?;
    let header_codes = parse_voxel_order(&header_bytes[948..952])?;

    let declared_count = endian.read_i32(&header_bytes[988..992]);
    let mut cursor = TRK_HEADER_SIZE;
    let mut count = 0usize;
    let mut streamlines = Vec::new();
    let mut dpv_buffers = allocate_field_buffers(&scalar_specs);
    let mut dps_buffers = allocate_field_buffers(&property_specs);

    while cursor < bytes.len() {
        if declared_count > 0 && count >= declared_count as usize {
            break;
        }

        if cursor + 4 > bytes.len() {
            return Err(TrxError::Format(
                "TRK payload ended while reading streamline length".into(),
            ));
        }
        let len = endian.read_i32(&bytes[cursor..cursor + 4]);
        cursor += 4;
        let len = usize::try_from(len)
            .map_err(|_| TrxError::Format("TRK streamline length cannot be negative".into()))?;

        let mut streamline = Vec::with_capacity(len);
        for _ in 0..len {
            if cursor + 12 > bytes.len() {
                return Err(TrxError::Format(
                    "TRK payload ended while reading streamline points".into(),
                ));
            }
            let point_voxmm = [
                endian.read_f32(&bytes[cursor..cursor + 4]),
                endian.read_f32(&bytes[cursor + 4..cursor + 8]),
                endian.read_f32(&bytes[cursor + 8..cursor + 12]),
            ];
            cursor += 12;
            let point_world = trackvis_to_rasmm(
                point_voxmm,
                voxel_sizes,
                dimensions,
                header_codes,
                affine_codes,
                &voxel_to_rasmm_f32,
            )?;
            streamline.push(point_world);

            if n_scalars > 0 {
                let scalar_values =
                    read_f32_row(&bytes, &mut cursor, n_scalars, endian, "scalars")?;
                append_slices(&mut dpv_buffers, &scalar_specs, &scalar_values);
            }
        }

        if n_properties > 0 {
            let property_values =
                read_f32_row(&bytes, &mut cursor, n_properties, endian, "properties")?;
            append_slices(&mut dps_buffers, &property_specs, &property_values);
        }

        streamlines.push(streamline);
        count += 1;
    }

    if declared_count > 0 && count != declared_count as usize {
        return Err(TrxError::Format(format!(
            "TRK header declares {declared_count} streamlines but parsed {count}"
        )));
    }

    let header = Header {
        voxel_to_rasmm: voxel_to_rasmm_f32.map(|row| row.map(f64::from)),
        dimensions: dimensions.map(|value| value as u64),
        nb_streamlines: streamlines.len() as u64,
        nb_vertices: streamlines.iter().map(Vec::len).sum::<usize>() as u64,
        extra: Default::default(),
    };

    Ok(ParsedTrk {
        header,
        streamlines,
        dpv: finalize_field_buffers(dpv_buffers),
        dps: finalize_field_buffers(dps_buffers),
    })
}

fn detect_endianness(header: &[u8]) -> Result<Endianness> {
    let le = i32::from_le_bytes(header[996..1000].try_into().unwrap());
    if le == TRK_HEADER_SIZE as i32 {
        return Ok(Endianness::Little);
    }
    let be = i32::from_be_bytes(header[996..1000].try_into().unwrap());
    if be == TRK_HEADER_SIZE as i32 {
        return Ok(Endianness::Big);
    }
    Err(TrxError::Format("TRK header size is invalid".into()))
}

fn parse_positive_i16(value: i16, label: &str) -> Result<i16> {
    if value <= 0 {
        return Err(TrxError::Format(format!(
            "TRK {label} must be positive, got {value}"
        )));
    }
    Ok(value)
}

fn parse_nonnegative_i16(value: i16, label: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| TrxError::Format(format!("TRK {label} cannot be negative, got {value}")))
}

fn parse_positive_f32(value: f32, label: &str) -> Result<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TrxError::Format(format!(
            "TRK {label} must be positive, got {value}"
        )));
    }
    Ok(value)
}

fn parse_affine_f32(endian: Endianness, bytes: &[u8]) -> Result<[[f32; 4]; 4]> {
    let mut affine = [[0.0f32; 4]; 4];
    for (row, row_values) in affine.iter_mut().enumerate() {
        for (col, value) in row_values.iter_mut().enumerate() {
            let start = (row * 4 + col) * 4;
            *value = endian.read_f32(&bytes[start..start + 4]);
        }
    }
    if !affine.iter().flatten().all(|value| value.is_finite()) {
        return Err(TrxError::Format(
            "TRK vox_to_ras contains non-finite values".into(),
        ));
    }
    Ok(affine)
}

fn affine_is_all_zero(affine: &[[f32; 4]; 4]) -> bool {
    affine
        .iter()
        .flatten()
        .all(|value| value.abs() <= f32::EPSILON)
}

fn parse_voxel_order(bytes: &[u8]) -> Result<[char; 3]> {
    let text = bytes
        .iter()
        .copied()
        .take_while(|byte| *byte != 0)
        .map(char::from)
        .collect::<String>()
        .trim()
        .to_ascii_uppercase();
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 3 {
        return Err(TrxError::Format(
            "TRK voxel order is missing or incomplete; convert it with another tool first".into(),
        ));
    }
    Ok([chars[0], chars[1], chars[2]])
}

fn affine_to_axcodes(affine: &[[f32; 4]; 4]) -> Result<[char; 3]> {
    let mut used_world_axes = [false; 3];
    let mut out = ['R'; 3];
    for (voxel_axis, slot) in out.iter_mut().enumerate() {
        let mut best_axis = 0usize;
        let mut best_value = 0.0f32;
        for (world_axis, row) in affine.iter().enumerate().take(3) {
            let value = row[voxel_axis].abs();
            if value > best_value {
                best_value = value;
                best_axis = world_axis;
            }
        }
        if best_value <= f32::EPSILON || used_world_axes[best_axis] {
            return Err(TrxError::Format(
                "TRK vox_to_ras affine is ambiguous or singular".into(),
            ));
        }
        used_world_axes[best_axis] = true;
        let sign = affine[best_axis][voxel_axis].signum();
        *slot = match (best_axis, sign >= 0.0) {
            (0, true) => 'R',
            (0, false) => 'L',
            (1, true) => 'A',
            (1, false) => 'P',
            (2, true) => 'S',
            (2, false) => 'I',
            _ => unreachable!(),
        };
    }
    Ok(out)
}

fn parse_name_specs(
    bytes: &[u8],
    total: usize,
    max_entries: usize,
    fallback_name: &str,
) -> Result<Vec<SliceSpec>> {
    let mut specs = Vec::new();
    let mut cursor = 0usize;
    for entry in 0..max_entries {
        let start = entry * 20;
        let end = start + 20;
        let (name, count) = decode_name_field(&bytes[start..end])?;
        if count == 0 {
            continue;
        }
        specs.push(SliceSpec {
            name,
            start: cursor,
            len: count,
        });
        cursor += count;
    }

    if cursor < total {
        specs.push(SliceSpec {
            name: fallback_name.to_string(),
            start: cursor,
            len: total - cursor,
        });
        cursor = total;
    }
    if cursor != total {
        return Err(TrxError::Format(format!(
            "TrackVis named field layout is inconsistent with declared column count {total}"
        )));
    }
    Ok(specs)
}

fn decode_name_field(bytes: &[u8]) -> Result<(String, usize)> {
    let decoded = bytes.iter().map(|&byte| byte as char).collect::<String>();
    let trimmed = decoded.trim_end_matches('\0');
    if trimmed.is_empty() {
        return Ok((String::new(), 0));
    }

    let mut parts = trimmed.split('\0');
    let name = parts.next().unwrap().to_string();
    let count = match parts.next() {
        Some(count) => count
            .parse::<usize>()
            .map_err(|_| TrxError::Format(format!("invalid TrackVis name encoding '{trimmed}'")))?,
        None => 1,
    };
    if parts.next().is_some() {
        return Err(TrxError::Format(format!(
            "invalid TrackVis name encoding '{trimmed}'"
        )));
    }
    Ok((name, count))
}

fn read_f32_row(
    bytes: &[u8],
    cursor: &mut usize,
    count: usize,
    endian: Endianness,
    label: &str,
) -> Result<Vec<f32>> {
    let byte_count = count
        .checked_mul(4)
        .ok_or_else(|| TrxError::Format(format!("TRK {label} row is too large")))?;
    if *cursor + byte_count > bytes.len() {
        return Err(TrxError::Format(format!(
            "TRK payload ended while reading {label}"
        )));
    }
    let row = (0..count)
        .map(|index| {
            let start = *cursor + index * 4;
            endian.read_f32(&bytes[start..start + 4])
        })
        .collect();
    *cursor += byte_count;
    Ok(row)
}

fn allocate_field_buffers(specs: &[SliceSpec]) -> HashMap<String, (usize, Vec<f32>)> {
    specs
        .iter()
        .map(|spec| (spec.name.clone(), (spec.len, Vec::new())))
        .collect()
}

fn append_slices(
    buffers: &mut HashMap<String, (usize, Vec<f32>)>,
    specs: &[SliceSpec],
    row: &[f32],
) {
    for spec in specs {
        if let Some((_, values)) = buffers.get_mut(&spec.name) {
            values.extend_from_slice(&row[spec.start..spec.start + spec.len]);
        }
    }
}

fn finalize_field_buffers(
    buffers: HashMap<String, (usize, Vec<f32>)>,
) -> HashMap<String, DataArray> {
    buffers
        .into_iter()
        .map(|(name, (ncols, values))| {
            (
                name,
                DataArray::owned_bytes(vec_to_bytes(values), ncols, DType::Float32),
            )
        })
        .collect()
}

fn trackvis_to_rasmm(
    point_voxmm: [f32; 3],
    voxel_sizes: [f32; 3],
    dimensions: [usize; 3],
    header_codes: [char; 3],
    affine_codes: [char; 3],
    voxel_to_rasmm: &[[f32; 4]; 4],
) -> Result<[f32; 3]> {
    let voxel_center = [
        point_voxmm[0] / voxel_sizes[0] - 0.5,
        point_voxmm[1] / voxel_sizes[1] - 0.5,
        point_voxmm[2] / voxel_sizes[2] - 0.5,
    ];
    let oriented = reorient_voxel_coords(voxel_center, dimensions, header_codes, affine_codes)?;
    Ok(apply_affine(voxel_to_rasmm, oriented))
}

fn reorient_voxel_coords(
    point: [f32; 3],
    dimensions: [usize; 3],
    from: [char; 3],
    to: [char; 3],
) -> Result<[f32; 3]> {
    let mut out = [0.0f32; 3];
    for (dst_axis, dst_code) in to.iter().enumerate() {
        let dst_family = axis_family(*dst_code)?;
        let dst_sign = axis_sign(*dst_code)?;
        let src_axis = from
            .iter()
            .position(|code| axis_family(*code).ok() == Some(dst_family))
            .ok_or_else(|| {
                TrxError::Format("TRK voxel order does not match affine orientation".into())
            })?;
        let src_sign = axis_sign(from[src_axis])?;
        out[dst_axis] = if src_sign == dst_sign {
            point[src_axis]
        } else {
            (dimensions[src_axis] as f32 - 1.0) - point[src_axis]
        };
    }
    Ok(out)
}

fn axis_family(code: char) -> Result<usize> {
    match code {
        'L' | 'R' => Ok(0),
        'P' | 'A' => Ok(1),
        'I' | 'S' => Ok(2),
        other => Err(TrxError::Format(format!(
            "unsupported anatomical axis code '{other}'"
        ))),
    }
}

fn axis_sign(code: char) -> Result<i8> {
    match code {
        'R' | 'A' | 'S' => Ok(1),
        'L' | 'P' | 'I' => Ok(-1),
        other => Err(TrxError::Format(format!(
            "unsupported anatomical axis code '{other}'"
        ))),
    }
}

fn apply_affine(affine: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
    [
        affine[0][0] * point[0] + affine[0][1] * point[1] + affine[0][2] * point[2] + affine[0][3],
        affine[1][0] * point[0] + affine[1][1] * point[1] + affine[1][2] * point[2] + affine[1][3],
        affine[2][0] * point[0] + affine[2][1] * point[1] + affine[2][2] * point[2] + affine[2][3],
    ]
}

fn build_offsets(streamlines: &[Vec<[f32; 3]>]) -> Result<Vec<u32>> {
    let mut offsets = Vec::with_capacity(streamlines.len() + 1);
    offsets.push(0);
    let mut total = 0usize;
    for streamline in streamlines {
        total += streamline.len();
        offsets
            .push(u32::try_from(total).map_err(|_| {
                TrxError::Format("TRK file has more than u32::MAX vertices".into())
            })?);
    }
    Ok(offsets)
}

fn read_maybe_gzip(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let file = File::open(path)?;
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".gz"))
    {
        let mut decoder = MultiGzDecoder::new(file);
        decoder.read_to_end(&mut bytes)?;
    } else {
        let mut file = file;
        file.read_to_end(&mut bytes)?;
    }
    Ok(bytes)
}
