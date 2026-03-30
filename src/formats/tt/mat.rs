use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use flate2::read::MultiGzDecoder;

use crate::error::{Result, TrxError};

#[derive(Debug)]
pub(super) struct MatRecord {
    pub rows: usize,
    pub cols: usize,
    pub ty: u32,
    pub payload: Vec<u8>,
}

pub(super) fn read_tt_mat_records(path: &Path) -> Result<HashMap<String, MatRecord>> {
    let mut file = File::open(path)?;
    let mut decoder = MultiGzDecoder::new(&mut file);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    parse_mat_records(&bytes)
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

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = *cursor + 4;
    let chunk = bytes
        .get(*cursor..end)
        .ok_or_else(|| TrxError::Format("truncated TT MAT integer".into()))?;
    *cursor = end;
    Ok(u32::from_le_bytes(chunk.try_into().unwrap()))
}
