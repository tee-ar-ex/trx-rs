use std::path::Path;

use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::tractogram::Tractogram;

pub fn read_vtk(path: &Path, header_override: Option<Header>) -> Result<Tractogram> {
    let bytes = std::fs::read(path)?;
    let parsed = parse_vtk_bytes(&bytes)?;

    let mut tractogram = Tractogram::with_header(header_override.unwrap_or(Header {
        voxel_to_rasmm: Header::identity_affine(),
        dimensions: [1, 1, 1],
        nb_streamlines: 0,
        nb_vertices: 0,
        extra: Default::default(),
    }));

    for streamline in parsed {
        tractogram.push_streamline(&streamline)?;
    }

    Ok(tractogram)
}

pub fn write_vtk(path: &Path, tractogram: &Tractogram) -> Result<()> {
    let mut text = String::new();
    text.push_str("# vtk DataFile Version 4.2\n");
    text.push_str("trx-rs tractogram\n");
    text.push_str("ASCII\n");
    text.push_str("DATASET POLYDATA\n");
    text.push_str(&format!("POINTS {} float\n", tractogram.nb_vertices()));
    for point in tractogram.positions() {
        let vtk_point = ras_to_vtk_world(*point);
        text.push_str(&format!(
            "{} {} {}\n",
            vtk_point[0], vtk_point[1], vtk_point[2]
        ));
    }

    let total_line_entries: usize = tractogram
        .streamlines()
        .map(|streamline| streamline.len() + 1)
        .sum();
    text.push_str(&format!(
        "LINES {} {}\n",
        tractogram.nb_streamlines(),
        total_line_entries
    ));

    let mut vertex_index = 0u32;
    for streamline in tractogram.streamlines() {
        text.push_str(&streamline.len().to_string());
        for _ in streamline {
            text.push(' ');
            text.push_str(&vertex_index.to_string());
            vertex_index += 1;
        }
        text.push('\n');
    }

    std::fs::write(path, text)?;
    Ok(())
}

fn parse_vtk_bytes(bytes: &[u8]) -> Result<Vec<Vec<[f32; 3]>>> {
    let mut cursor = 0usize;
    let version = read_line(bytes, &mut cursor)?;
    if !version.starts_with("# vtk DataFile Version") {
        return Err(TrxError::Format("not a legacy VTK file".into()));
    }

    let _comment = read_line(bytes, &mut cursor)?;
    let format = read_line(bytes, &mut cursor)?;
    let dataset = read_line(bytes, &mut cursor)?;
    if dataset.trim() != "DATASET POLYDATA" {
        return Err(TrxError::Format(
            "only VTK POLYDATA streamline files are supported".into(),
        ));
    }

    let points_header = read_line(bytes, &mut cursor)?;
    let mut header_parts = points_header.split_whitespace();
    if header_parts.next() != Some("POINTS") {
        return Err(TrxError::Format(
            "VTK file is missing POINTS section".into(),
        ));
    }
    let point_count = header_parts
        .next()
        .ok_or_else(|| TrxError::Format("missing VTK point count".into()))?
        .parse::<usize>()
        .map_err(|_| TrxError::Format("invalid VTK point count".into()))?;
    let point_type = header_parts
        .next()
        .ok_or_else(|| TrxError::Format("missing VTK points datatype".into()))?;

    let is_binary = format.trim().eq_ignore_ascii_case("BINARY");
    let points = if is_binary {
        parse_binary_points(bytes, &mut cursor, point_count, point_type)?
    } else if format.trim().eq_ignore_ascii_case("ASCII") {
        parse_ascii_points_and_lines(bytes, cursor, point_count)?
    } else {
        return Err(TrxError::Format(format!(
            "unsupported VTK encoding '{}'",
            format.trim()
        )));
    };

    Ok(points
        .into_iter()
        .map(|streamline| streamline.into_iter().map(vtk_world_to_ras).collect())
        .collect())
}

fn parse_ascii_points_and_lines(
    bytes: &[u8],
    cursor: usize,
    point_count: usize,
) -> Result<Vec<Vec<[f32; 3]>>> {
    let text = std::str::from_utf8(&bytes[cursor..])
        .map_err(|_| TrxError::Format("ASCII VTK body is not valid UTF-8".into()))?;
    let mut tokens = text.split_whitespace();

    let mut points = Vec::with_capacity(point_count);
    for _ in 0..point_count {
        let x = next_token(&mut tokens, "point x")?
            .parse::<f32>()
            .map_err(|_| TrxError::Format("invalid VTK point coordinate".into()))?;
        let y = next_token(&mut tokens, "point y")?
            .parse::<f32>()
            .map_err(|_| TrxError::Format("invalid VTK point coordinate".into()))?;
        let z = next_token(&mut tokens, "point z")?
            .parse::<f32>()
            .map_err(|_| TrxError::Format("invalid VTK point coordinate".into()))?;
        points.push([x, y, z]);
    }

    if next_token(&mut tokens, "LINES keyword")? != "LINES" {
        return Err(TrxError::Format("VTK file is missing LINES section".into()));
    }
    let line_count = next_token(&mut tokens, "line count")?
        .parse::<usize>()
        .map_err(|_| TrxError::Format("invalid VTK line count".into()))?;
    let _line_size = next_token(&mut tokens, "line size")?
        .parse::<usize>()
        .map_err(|_| TrxError::Format("invalid VTK line size".into()))?;

    build_streamlines_from_tokens(points, line_count, &mut tokens)
}

fn parse_binary_points(
    bytes: &[u8],
    cursor: &mut usize,
    point_count: usize,
    point_type: &str,
) -> Result<Vec<Vec<[f32; 3]>>> {
    let element_size = match point_type {
        "float" => 4,
        "double" => 8,
        other => {
            return Err(TrxError::Format(format!(
                "unsupported binary VTK point datatype '{other}'"
            )))
        }
    };
    let points_bytes = point_count
        .checked_mul(3)
        .and_then(|count| count.checked_mul(element_size))
        .ok_or_else(|| TrxError::Format("VTK points section is too large".into()))?;
    let end = cursor
        .checked_add(points_bytes)
        .ok_or_else(|| TrxError::Format("VTK points section overflow".into()))?;
    let data = bytes
        .get(*cursor..end)
        .ok_or_else(|| TrxError::Format("VTK points section is truncated".into()))?;

    let mut points = Vec::with_capacity(point_count);
    match point_type {
        "float" => {
            for chunk in data.chunks_exact(12) {
                points.push([
                    f32::from_be_bytes(chunk[0..4].try_into().unwrap()),
                    f32::from_be_bytes(chunk[4..8].try_into().unwrap()),
                    f32::from_be_bytes(chunk[8..12].try_into().unwrap()),
                ]);
            }
        }
        "double" => {
            for chunk in data.chunks_exact(24) {
                points.push([
                    f64::from_be_bytes(chunk[0..8].try_into().unwrap()) as f32,
                    f64::from_be_bytes(chunk[8..16].try_into().unwrap()) as f32,
                    f64::from_be_bytes(chunk[16..24].try_into().unwrap()) as f32,
                ]);
            }
        }
        _ => unreachable!(),
    }
    *cursor = end;
    skip_newlines(bytes, cursor);

    let lines_header = read_line(bytes, cursor)?;
    let mut parts = lines_header.split_whitespace();
    if parts.next() != Some("LINES") {
        return Err(TrxError::Format("VTK file is missing LINES section".into()));
    }
    let line_count = parts
        .next()
        .ok_or_else(|| TrxError::Format("missing VTK line count".into()))?
        .parse::<usize>()
        .map_err(|_| TrxError::Format("invalid VTK line count".into()))?;
    let total_size = parts
        .next()
        .ok_or_else(|| TrxError::Format("missing VTK total line size".into()))?
        .parse::<usize>()
        .map_err(|_| TrxError::Format("invalid VTK total line size".into()))?;

    let line_bytes = total_size
        .checked_mul(4)
        .ok_or_else(|| TrxError::Format("VTK line section is too large".into()))?;
    let end = cursor
        .checked_add(line_bytes)
        .ok_or_else(|| TrxError::Format("VTK line section overflow".into()))?;
    let data = bytes
        .get(*cursor..end)
        .ok_or_else(|| TrxError::Format("VTK line section is truncated".into()))?;

    let mut ints = Vec::with_capacity(total_size);
    for chunk in data.chunks_exact(4) {
        ints.push(i32::from_be_bytes(chunk.try_into().unwrap()));
    }

    build_streamlines_from_ints(points, line_count, ints)
}

fn build_streamlines_from_ints(
    points: Vec<[f32; 3]>,
    line_count: usize,
    ints: Vec<i32>,
) -> Result<Vec<Vec<[f32; 3]>>> {
    let mut cursor = 0usize;
    let mut streamlines = Vec::with_capacity(line_count);
    for _ in 0..line_count {
        let length = *ints
            .get(cursor)
            .ok_or_else(|| TrxError::Format("VTK LINES section ended unexpectedly".into()))?;
        cursor += 1;
        let length = usize::try_from(length)
            .map_err(|_| TrxError::Format("VTK streamline length cannot be negative".into()))?;
        let mut streamline = Vec::with_capacity(length);
        for _ in 0..length {
            let point_index = *ints.get(cursor).ok_or_else(|| {
                TrxError::Format("VTK LINES point index section ended unexpectedly".into())
            })?;
            cursor += 1;
            let point_index = usize::try_from(point_index)
                .map_err(|_| TrxError::Format("VTK point index cannot be negative".into()))?;
            let point = *points.get(point_index).ok_or_else(|| {
                TrxError::Format(format!("VTK point index {point_index} is out of bounds"))
            })?;
            streamline.push(point);
        }
        streamlines.push(streamline);
    }
    Ok(streamlines)
}

fn build_streamlines_from_tokens<'a>(
    points: Vec<[f32; 3]>,
    line_count: usize,
    tokens: &mut impl Iterator<Item = &'a str>,
) -> Result<Vec<Vec<[f32; 3]>>> {
    let mut streamlines = Vec::with_capacity(line_count);
    for _ in 0..line_count {
        let length = next_token(tokens, "streamline length")?
            .parse::<usize>()
            .map_err(|_| TrxError::Format("invalid VTK streamline length".into()))?;
        let mut streamline = Vec::with_capacity(length);
        for _ in 0..length {
            let point_index = next_token(tokens, "streamline point index")?
                .parse::<usize>()
                .map_err(|_| TrxError::Format("invalid VTK point index".into()))?;
            let point = *points.get(point_index).ok_or_else(|| {
                TrxError::Format(format!("VTK point index {point_index} is out of bounds"))
            })?;
            streamline.push(point);
        }
        streamlines.push(streamline);
    }
    Ok(streamlines)
}

fn read_line<'a>(bytes: &'a [u8], cursor: &mut usize) -> Result<&'a str> {
    let start = *cursor;
    while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
        *cursor += 1;
    }
    let end = *cursor;
    if *cursor < bytes.len() && bytes[*cursor] == b'\n' {
        *cursor += 1;
    }
    std::str::from_utf8(&bytes[start..end])
        .map(|line| line.trim_end_matches('\r'))
        .map_err(|_| TrxError::Format("VTK header is not valid UTF-8".into()))
}

fn skip_newlines(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() && matches!(bytes[*cursor], b'\n' | b'\r') {
        *cursor += 1;
    }
}

fn next_token<'a>(tokens: &mut impl Iterator<Item = &'a str>, label: &str) -> Result<&'a str> {
    tokens
        .next()
        .ok_or_else(|| TrxError::Format(format!("missing VTK token for {label}")))
}

fn vtk_world_to_ras(point: [f32; 3]) -> [f32; 3] {
    [-point[0], -point[1], point[2]]
}

fn ras_to_vtk_world(point: [f32; 3]) -> [f32; 3] {
    [-point[0], -point[1], point[2]]
}
