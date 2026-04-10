use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::tractogram::Tractogram;

const TCK_MAGIC: &str = "mrtrix tracks";

pub fn read_tck(path: &Path, header_override: Option<Header>) -> Result<Tractogram> {
    let bytes = read_maybe_gzip(path)?;
    let parsed = parse_tck_bytes(&bytes)?;

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

pub fn write_tck(path: &Path, tractogram: &Tractogram) -> Result<()> {
    let bytes = build_tck_bytes(tractogram);
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".gz"))
    {
        let file = File::create(path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(&bytes)?;
        encoder.finish()?;
    } else {
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

fn read_maybe_gzip(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut file = File::open(path)?;
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".gz"))
    {
        let mut decoder = MultiGzDecoder::new(file);
        decoder.read_to_end(&mut bytes)?;
    } else {
        file.read_to_end(&mut bytes)?;
    }
    Ok(bytes)
}

fn parse_tck_bytes(bytes: &[u8]) -> Result<Vec<Vec<[f32; 3]>>> {
    let (data_offset, declared_count) = parse_header(bytes)?;
    let payload = bytes
        .get(data_offset..)
        .ok_or_else(|| TrxError::Format("TCK file payload offset is beyond end of file".into()))?;

    if payload.len() % 12 != 0 {
        return Err(TrxError::Format(
            "TCK payload length is not divisible by 12 bytes".into(),
        ));
    }

    let mut streamlines = Vec::new();
    let mut current = Vec::new();
    let mut saw_eof = false;

    for chunk in payload.chunks_exact(12) {
        let point = [
            f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
            f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
        ];

        if point.iter().all(|value| value.is_infinite()) {
            saw_eof = true;
            break;
        }
        if point.iter().all(|value| value.is_nan()) {
            streamlines.push(std::mem::take(&mut current));
            continue;
        }

        current.push(point);
    }

    if !current.is_empty() {
        streamlines.push(current);
    }

    if !saw_eof {
        return Err(TrxError::Format("TCK file is missing EOF delimiter".into()));
    }

    if let Some(count) = declared_count {
        let actual = u64::try_from(streamlines.len()).unwrap_or(u64::MAX);
        if count != actual {
            return Err(TrxError::Format(format!(
                "TCK header count ({count}) does not match parsed streamline count ({actual})"
            )));
        }
    }

    Ok(streamlines)
}

fn parse_header(bytes: &[u8]) -> Result<(usize, Option<u64>)> {
    let mut cursor = 0usize;
    let mut lines = Vec::new();
    while cursor < bytes.len() {
        let line_start = cursor;
        while cursor < bytes.len() && bytes[cursor] != b'\n' {
            cursor += 1;
        }
        let line_end = cursor;
        if cursor < bytes.len() && bytes[cursor] == b'\n' {
            cursor += 1;
        }
        let line = std::str::from_utf8(&bytes[line_start..line_end])
            .map_err(|_| TrxError::Format("TCK header is not valid UTF-8".into()))?;
        // Some generators leave trailing spaces on header lines, and a UTF-8 BOM can
        // appear at the beginning of the file. Accept those variants.
        let line = line.trim_start_matches('\u{feff}').trim();
        if line == "END" {
            break;
        }
        lines.push(line.to_owned());
    }

    if lines.is_empty() || lines[0] != TCK_MAGIC {
        return Err(TrxError::Format(
            "file does not start with 'mrtrix tracks'".into(),
        ));
    }

    let mut data_offset = cursor;
    let mut declared_count = None;

    for line in lines.iter().skip(1) {
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim();
            let value = value.trim();
            match key {
                "file" => {
                    let mut tokens = value.split_whitespace();
                    let locator = tokens.next().unwrap_or_default();
                    let offset = tokens
                        .next()
                        .ok_or_else(|| TrxError::Format("invalid TCK file offset".into()))?;
                    if locator != "." {
                        return Err(TrxError::Format(
                            "only inline TCK payloads ('file: . <offset>') are supported".into(),
                        ));
                    }
                    data_offset = offset.parse::<usize>().map_err(|_| {
                        TrxError::Format(format!("invalid TCK payload offset '{offset}'"))
                    })?;
                }
                "count" => {
                    declared_count = Some(value.parse::<u64>().map_err(|_| {
                        TrxError::Format(format!("invalid TCK streamline count '{value}'"))
                    })?);
                }
                "datatype" => {
                    if value != "Float32LE" {
                        return Err(TrxError::Format(format!(
                            "unsupported TCK datatype '{value}', expected Float32LE"
                        )));
                    }
                }
                _ => {}
            }
        }
    }

    Ok((data_offset, declared_count))
}

fn build_tck_bytes(tractogram: &Tractogram) -> Vec<u8> {
    let header = build_header(tractogram.nb_streamlines());
    let mut bytes = header.into_bytes();

    for streamline in tractogram.streamlines() {
        for point in streamline {
            bytes.extend_from_slice(&point[0].to_le_bytes());
            bytes.extend_from_slice(&point[1].to_le_bytes());
            bytes.extend_from_slice(&point[2].to_le_bytes());
        }
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
    }

    bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());
    bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());
    bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());
    bytes
}

fn build_header(streamline_count: usize) -> String {
    let mut offset = 0usize;
    loop {
        let header = format!(
            "{TCK_MAGIC}\ncount: {streamline_count:010}\ndatatype: Float32LE\nfile: . {offset}\nEND\n"
        );
        let next_offset = header.len();
        if next_offset == offset {
            return header;
        }
        offset = next_offset;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_builder_converges() {
        let header = build_header(13);
        let (_, count) = parse_header(header.as_bytes()).unwrap();
        assert_eq!(count, Some(13));
    }

    #[test]
    fn parser_accepts_padded_magic_line() {
        let mut offset = 0usize;
        let header = loop {
            let header = format!(
                "mrtrix tracks    \ncount: 0000000001\ndatatype: Float32LE\nfile: . {offset}\nEND\n"
            );
            let next_offset = header.len();
            if next_offset == offset {
                break header;
            }
            offset = next_offset;
        };

        let mut bytes = header.into_bytes();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());
        bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());
        bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());

        let streamlines = parse_tck_bytes(&bytes).unwrap();
        assert_eq!(streamlines, vec![vec![[1.0, 2.0, 3.0]]]);
    }
}
