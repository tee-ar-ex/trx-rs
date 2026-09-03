use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::Tractogram;

pub fn load_trk(path: &Path) -> Result<Tractogram, Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;

    if buffer.len() < 1000 {
        return Err("File too small".into());
    }

    let n_scalars = i16::from_le_bytes(buffer[36..38].try_into().unwrap());
    let n_properties = i16::from_le_bytes(buffer[238..240].try_into().unwrap());

    let mut voxel_sizes = [
        f32::from_le_bytes(buffer[12..16].try_into().unwrap()),
        f32::from_le_bytes(buffer[16..20].try_into().unwrap()),
        f32::from_le_bytes(buffer[20..24].try_into().unwrap()),
    ];

    // Protect against division by zero for corrupted headers
    if voxel_sizes[0] == 0.0 {
        voxel_sizes[0] = 1.0;
    }
    if voxel_sizes[1] == 0.0 {
        voxel_sizes[1] = 1.0;
    }
    if voxel_sizes[2] == 0.0 {
        voxel_sizes[2] = 1.0;
    }

    let mut vox_to_ras = nalgebra::Matrix4::zeros();
    let mut mat_offset = 440;
    for r in 0..4 {
        for c in 0..4 {
            vox_to_ras[(r, c)] =
                f32::from_le_bytes(buffer[mat_offset..mat_offset + 4].try_into().unwrap());
            mat_offset += 4;
        }
    }

    let mut tr = Tractogram::new();

    let dims = [
        i16::from_le_bytes(buffer[6..8].try_into().unwrap()) as u64,
        i16::from_le_bytes(buffer[8..10].try_into().unwrap()) as u64,
        i16::from_le_bytes(buffer[10..12].try_into().unwrap()) as u64,
    ];

    let mut vox_to_ras_f64 = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            vox_to_ras_f64[r][c] = vox_to_ras[(r, c)] as f64;
        }
    }
    tr.set_spatial_metadata(vox_to_ras_f64, dims);

    let mut offset = 1000;

    while offset + 4 <= buffer.len() {
        let n_points = i32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap());
        offset += 4;

        if n_points < 0 {
            return Err("Negative number of points in streamline".into());
        }

        let required_bytes = (n_points as usize) * (3 + n_scalars as usize) * 4;
        if offset + required_bytes > buffer.len() {
            return Err("Unexpected EOF reading streamline points".into());
        }

        let mut streamline = Vec::with_capacity(n_points as usize);
        for _ in 0..n_points {
            let raw_x = f32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap());
            let raw_y = f32::from_le_bytes(buffer[offset + 4..offset + 8].try_into().unwrap());
            let raw_z = f32::from_le_bytes(buffer[offset + 8..offset + 12].try_into().unwrap());

            let cx = (raw_x / voxel_sizes[0]) - 0.5;
            let cy = (raw_y / voxel_sizes[1]) - 0.5;
            let cz = (raw_z / voxel_sizes[2]) - 0.5;

            let p_vox = nalgebra::Point3::new(cx, cy, cz);
            let p_ras = vox_to_ras.transform_point(&p_vox);

            streamline.push([p_ras.x, p_ras.y, p_ras.z]);
            offset += (3 + n_scalars as usize) * 4;
        }
        tr.push_streamline(&streamline)?;
        offset += (n_properties as usize) * 4;
    }

    Ok(tr)
}

pub fn load_vtk(path: &Path) -> Result<Tractogram, Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;

    let header_str = String::from_utf8_lossy(&buffer[0..std::cmp::min(1024, buffer.len())]);
    let points_idx = header_str.find("POINTS ").ok_or("No POINTS")?;

    let points_str = header_str[points_idx..]
        .split_whitespace()
        .nth(1)
        .ok_or("No POINTS count")?;
    let num_points: usize = points_str.parse()?;

    let mut is_double = false;
    if let Some(type_str) = header_str[points_idx..].split_whitespace().nth(2) {
        if type_str == "double" {
            is_double = true;
        }
    }

    let header_end = header_str[points_idx..]
        .find('\n')
        .ok_or("No newline after POINTS")?
        + points_idx
        + 1;
    let mut pts = Vec::with_capacity(num_points * 3);

    let mut offset = header_end;
    for _ in 0..num_points * 3 {
        if is_double {
            let chunk = buffer
                .get(offset..offset + 8)
                .ok_or("Unexpected EOF reading points")?;
            let val = f64::from_be_bytes(chunk.try_into().unwrap());
            pts.push(val as f32);
            offset += 8;
        } else {
            let chunk = buffer
                .get(offset..offset + 4)
                .ok_or("Unexpected EOF reading points")?;
            let val = f32::from_be_bytes(chunk.try_into().unwrap());
            pts.push(val);
            offset += 4;
        }
    }

    let search_window = std::cmp::min(offset + 1024, buffer.len());
    let lines_str_chunk = String::from_utf8_lossy(&buffer[offset..search_window]);

    let lines_idx_in_chunk = lines_str_chunk.find("LINES ").ok_or("No LINES")?;
    let lines_idx = offset + lines_idx_in_chunk;

    let lines_str = lines_str_chunk[lines_idx_in_chunk..]
        .split_whitespace()
        .nth(1)
        .ok_or("No LINES count")?;
    let num_lines: usize = lines_str.parse()?;

    let lines_header_end = lines_str_chunk[lines_idx_in_chunk..]
        .find('\n')
        .ok_or("No newline after LINES")?
        + lines_idx
        + 1;
    offset = lines_header_end;

    let mut tr = Tractogram::new();

    if buffer
        .get(offset..)
        .is_some_and(|b| b.starts_with(b"OFFSETS"))
    {
        let offsets_header_end = buffer[offset..]
            .iter()
            .position(|&c| c == b'\n')
            .ok_or("No newline after OFFSETS")?
            + offset
            + 1;

        let header_str = String::from_utf8_lossy(&buffer[offset..offsets_header_end]);
        let tokens: Vec<&str> = header_str.split_whitespace().collect();

        let num_offsets = if tokens.len() >= 3 {
            tokens[2].parse().unwrap_or(num_lines)
        } else {
            num_lines
        };

        let is_int64 = header_str.contains("int64");
        offset = offsets_header_end;

        let mut offsets_vec = Vec::with_capacity(num_offsets);
        for _ in 0..num_offsets {
            if is_int64 {
                let chunk = buffer
                    .get(offset..offset + 8)
                    .ok_or("Unexpected EOF reading offsets")?;
                let val = u64::from_be_bytes(chunk.try_into().unwrap());
                offsets_vec.push(val as usize);
                offset += 8;
            } else {
                let chunk = buffer
                    .get(offset..offset + 4)
                    .ok_or("Unexpected EOF reading offsets")?;
                let val = u32::from_be_bytes(chunk.try_into().unwrap());
                offsets_vec.push(val as usize);
                offset += 4;
            }
        }

        let actual_num_lines = num_offsets.saturating_sub(1);
        for i in 0..actual_num_lines {
            let start = offsets_vec[i];
            let end = offsets_vec[i + 1];

            if end > pts.len() / 3 {
                return Err("Offset points out of bounds".into());
            }
            let mut streamline = Vec::with_capacity(end.saturating_sub(start));
            for pt_idx in start..end {
                streamline.push([pts[pt_idx * 3], pts[pt_idx * 3 + 1], pts[pt_idx * 3 + 2]]);
            }
            tr.push_streamline(&streamline)?;
        }
        return Ok(tr);
    }

    let mut pt_idx = 0;
    for _ in 0..num_lines {
        if offset + 4 > buffer.len() {
            break;
        }
        let n_pts = i32::from_be_bytes(buffer[offset..offset + 4].try_into().unwrap());
        offset += 4;

        if n_pts <= 0 {
            continue;
        }
        if pt_idx + (n_pts as usize) > num_points {
            break;
        }

        let mut streamline = Vec::with_capacity(n_pts as usize);
        for _ in 0..n_pts {
            offset += 4;
            streamline.push([pts[pt_idx * 3], pts[pt_idx * 3 + 1], pts[pt_idx * 3 + 2]]);
            pt_idx += 1;
        }
        tr.push_streamline(&streamline)?;
    }

    Ok(tr)
}

pub fn load_nifti_header(path: &Path) -> Result<crate::header::Header, Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;

    if buffer.len() < 348 {
        return Err("NIfTI file too small".into());
    }

    let mut sizeof_hdr_bytes = [0u8; 4];
    sizeof_hdr_bytes.copy_from_slice(&buffer[0..4]);
    let sizeof_hdr = i32::from_le_bytes(sizeof_hdr_bytes);

    let (is_nifti2, swap_endian) = if sizeof_hdr == 348 {
        (false, false)
    } else if sizeof_hdr == 348i32.swap_bytes() {
        (false, true)
    } else if sizeof_hdr == 540 {
        (true, false)
    } else if sizeof_hdr == 540i32.swap_bytes() {
        (true, true)
    } else {
        return Err(format!("Unsupported NIfTI sizeof_hdr: {}", sizeof_hdr).into());
    };

    if is_nifti2 && buffer.len() < 540 {
        return Err("NIfTI-2 file too small".into());
    }

    let read_i16 = |offset: usize| -> Result<i16, Box<dyn std::error::Error>> {
        let bytes = buffer.get(offset..offset + 2).ok_or("Buffer too small")?;
        let mut arr = [0u8; 2];
        arr.copy_from_slice(bytes);
        let val = if swap_endian {
            i16::from_be_bytes(arr)
        } else {
            i16::from_le_bytes(arr)
        };
        Ok(val)
    };

    let read_i32 = |offset: usize| -> Result<i32, Box<dyn std::error::Error>> {
        let bytes = buffer.get(offset..offset + 4).ok_or("Buffer too small")?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        let val = if swap_endian {
            i32::from_be_bytes(arr)
        } else {
            i32::from_le_bytes(arr)
        };
        Ok(val)
    };

    let read_i64 = |offset: usize| -> Result<i64, Box<dyn std::error::Error>> {
        let bytes = buffer.get(offset..offset + 8).ok_or("Buffer too small")?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        let val = if swap_endian {
            i64::from_be_bytes(arr)
        } else {
            i64::from_le_bytes(arr)
        };
        Ok(val)
    };

    let read_f32 = |offset: usize| -> Result<f32, Box<dyn std::error::Error>> {
        let bytes = buffer.get(offset..offset + 4).ok_or("Buffer too small")?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        let val = if swap_endian {
            f32::from_be_bytes(arr)
        } else {
            f32::from_le_bytes(arr)
        };
        Ok(val)
    };

    let read_f64 = |offset: usize| -> Result<f64, Box<dyn std::error::Error>> {
        let bytes = buffer.get(offset..offset + 8).ok_or("Buffer too small")?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        let val = if swap_endian {
            f64::from_be_bytes(arr)
        } else {
            f64::from_le_bytes(arr)
        };
        Ok(val)
    };

    let mut dimensions = [1, 1, 1];
    let qform_code;
    let sform_code;

    let mut pixdim = [1.0; 8];
    let mut srow_x = [0.0; 4];
    let mut srow_y = [0.0; 4];
    let mut srow_z = [0.0; 4];
    let quatern_b;
    let quatern_c;
    let quatern_d;
    let qoffset_x;
    let qoffset_y;
    let qoffset_z;

    if is_nifti2 {
        for i in 1..=3 {
            dimensions[i - 1] = read_i64(16 + i * 8)? as u64;
        }
        for (i, px) in pixdim.iter_mut().enumerate() {
            *px = read_f64(80 + i * 8)?;
        }
        qform_code = read_i32(344)?;
        sform_code = read_i32(348)?;
        quatern_b = read_f64(352)?;
        quatern_c = read_f64(360)?;
        quatern_d = read_f64(368)?;
        qoffset_x = read_f64(376)?;
        qoffset_y = read_f64(384)?;
        qoffset_z = read_f64(392)?;
        for i in 0..4 {
            srow_x[i] = read_f64(400 + i * 8)?;
            srow_y[i] = read_f64(432 + i * 8)?;
            srow_z[i] = read_f64(464 + i * 8)?;
        }
    } else {
        for i in 1..=3 {
            dimensions[i - 1] = read_i16(40 + i * 2)? as u64;
        }
        for (i, px) in pixdim.iter_mut().enumerate() {
            *px = read_f32(76 + i * 4)? as f64;
        }
        qform_code = read_i16(252)? as i32;
        sform_code = read_i16(254)? as i32;
        quatern_b = read_f32(256)? as f64;
        quatern_c = read_f32(260)? as f64;
        quatern_d = read_f32(264)? as f64;
        qoffset_x = read_f32(268)? as f64;
        qoffset_y = read_f32(272)? as f64;
        qoffset_z = read_f32(276)? as f64;
        for i in 0..4 {
            srow_x[i] = read_f32(280 + i * 4)? as f64;
            srow_y[i] = read_f32(296 + i * 4)? as f64;
            srow_z[i] = read_f32(312 + i * 4)? as f64;
        }
    }

    let mut voxel_to_rasmm = crate::header::Header::identity_affine();

    if sform_code > 0 {
        voxel_to_rasmm[0] = srow_x;
        voxel_to_rasmm[1] = srow_y;
        voxel_to_rasmm[2] = srow_z;
        voxel_to_rasmm[3] = [0.0, 0.0, 0.0, 1.0];
    } else if qform_code > 0 {
        let b = quatern_b;
        let c = quatern_c;
        let d = quatern_d;
        let a = (1.0 - b * b - c * c - d * d).max(0.0).sqrt();
        let qfac = if pixdim[0] == 0.0 { 1.0 } else { pixdim[0] };
        let dx = pixdim[1];
        let dy = pixdim[2];
        let dz = pixdim[3];

        let r00 = a * a + b * b - c * c - d * d;
        let r01 = 2.0 * (b * c - a * d);
        let r02 = 2.0 * (b * d + a * c);

        let r10 = 2.0 * (b * c + a * d);
        let r11 = a * a + c * c - b * b - d * d;
        let r12 = 2.0 * (c * d - a * b);

        let r20 = 2.0 * (b * d - a * c);
        let r21 = 2.0 * (c * d + a * b);
        let r22 = a * a + d * d - c * c - b * b;

        voxel_to_rasmm[0] = [r00 * dx, r01 * dy, r02 * qfac * dz, qoffset_x];
        voxel_to_rasmm[1] = [r10 * dx, r11 * dy, r12 * qfac * dz, qoffset_y];
        voxel_to_rasmm[2] = [r20 * dx, r21 * dy, r22 * qfac * dz, qoffset_z];
        voxel_to_rasmm[3] = [0.0, 0.0, 0.0, 1.0];
    } else {
        return Err("NIfTI file has no valid spatial transform".into());
    }

    let header = crate::header::Header {
        voxel_to_rasmm,
        dimensions,
        nb_streamlines: 0,
        nb_vertices: 0,
        extra: Default::default(),
    };
    Ok(header)
}

pub fn write_trx(
    path: &Path,
    tractogram: &Tractogram,
    ref_nifti: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tractogram = tractogram.clone();
    let header_empty = tractogram.header().voxel_to_rasmm
        == crate::header::Header::identity_affine()
        && tractogram.header().dimensions == [1, 1, 1];

    if header_empty {
        if let Some(p) = ref_nifti {
            let hdr = load_nifti_header(p)?;
            tractogram.set_header(hdr);
        } else {
            return Err("TCK -> TRX requires a reference NIfTI file".into());
        }
    }

    let any_trx = tractogram.to_trx(crate::dtype::DType::Float32)?;
    any_trx.save(path)?;
    Ok(())
}

/// Derive the 3-byte voxel_order field from a 4×4 affine matrix,
/// replicating nibabel's `io_orientation` polar-decomposition approach:
///   1. Normalize columns of the 3×3 block by their L2 norm (removes zoom/scale).
///   2. SVD of the normalized matrix → R = U * V^T (closest pure rotation).
///   3. For each input axis (column of R), pick the dominant output axis
///      (argmax of abs values) with axis-exclusion to handle oblique cases.
fn axcodes_from_affine(aff: &[[f64; 4]; 4]) -> [u8; 3] {
    use nalgebra::{Matrix3, SVD};
    const POS: [u8; 3] = *b"RAS";
    const NEG: [u8; 3] = *b"LPI";

    // Step 1: build column-normalized 3×3 matrix
    let mut rs = Matrix3::<f64>::zeros();
    for col in 0..3 {
        let norm = (0..3).map(|r| aff[r][col].powi(2)).sum::<f64>().sqrt();
        let norm = if norm == 0.0 { 1.0 } else { norm };
        for row in 0..3 {
            rs[(row, col)] = aff[row][col] / norm;
        }
    }

    // Step 2: SVD → R = U * V^T (polar factor, closest orthonormal matrix)
    let svd = SVD::new(rs, true, true);
    let u = svd.u.expect("SVD U not computed");
    let v_t = svd.v_t.expect("SVD V^T not computed");
    let r = u * v_t;

    // Step 3: per-column argmax with axis exclusion (mirrors nibabel exactly)
    let mut used = [false; 3];
    let mut codes = [b'?'; 3];
    for col in 0..3 {
        let mut best_row = 0usize;
        let mut best_val = -1.0f64;
        for row in 0..3 {
            if !used[row] && r[(row, col)].abs() > best_val {
                best_val = r[(row, col)].abs();
                best_row = row;
            }
        }
        used[best_row] = true;
        codes[col] = if r[(best_row, col)] >= 0.0 {
            POS[best_row]
        } else {
            NEG[best_row]
        };
    }
    codes
}

pub fn write_trk(
    path: &Path,
    tractogram: &Tractogram,
    ref_nifti: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    let mut header_bytes = vec![0u8; 1000];

    header_bytes[0..5].copy_from_slice(b"TRACK");

    let mut header = tractogram.header().clone();
    let header_empty = header.voxel_to_rasmm == crate::header::Header::identity_affine()
        && header.dimensions == [1, 1, 1];
    if header_empty {
        if let Some(p) = ref_nifti {
            header = load_nifti_header(p)?;
        } else {
            return Err("TCK -> TRK requires a reference NIfTI file".into());
        }
    }
    let dims = [
        header.dimensions[0] as i16,
        header.dimensions[1] as i16,
        header.dimensions[2] as i16,
    ];
    header_bytes[6..8].copy_from_slice(&dims[0].to_le_bytes());
    header_bytes[8..10].copy_from_slice(&dims[1].to_le_bytes());
    header_bytes[10..12].copy_from_slice(&dims[2].to_le_bytes());

    let vox_to_ras = header.voxel_to_rasmm;
    let voxel_sizes = [
        ((vox_to_ras[0][0].powi(2) + vox_to_ras[1][0].powi(2) + vox_to_ras[2][0].powi(2)).sqrt())
            as f32,
        ((vox_to_ras[0][1].powi(2) + vox_to_ras[1][1].powi(2) + vox_to_ras[2][1].powi(2)).sqrt())
            as f32,
        ((vox_to_ras[0][2].powi(2) + vox_to_ras[1][2].powi(2) + vox_to_ras[2][2].powi(2)).sqrt())
            as f32,
    ];
    header_bytes[12..16].copy_from_slice(&voxel_sizes[0].to_le_bytes());
    header_bytes[16..20].copy_from_slice(&voxel_sizes[1].to_le_bytes());
    header_bytes[20..24].copy_from_slice(&voxel_sizes[2].to_le_bytes());

    let mut offset = 440;
    for row in &vox_to_ras {
        for &elem in row {
            let val = elem as f32;
            header_bytes[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
            offset += 4;
        }
    }

    let axcodes = axcodes_from_affine(&vox_to_ras);
    header_bytes[948..951].copy_from_slice(&axcodes);
    header_bytes[951] = 0;

    let nb_streamlines = tractogram.nb_streamlines() as i32;
    header_bytes[988..992].copy_from_slice(&nb_streamlines.to_le_bytes());

    header_bytes[992..996].copy_from_slice(&2i32.to_le_bytes());

    header_bytes[996..1000].copy_from_slice(&1000i32.to_le_bytes());

    file.write_all(&header_bytes)?;

    let mut mat = nalgebra::Matrix4::zeros();
    for r in 0..4 {
        for c in 0..4 {
            mat[(r, c)] = vox_to_ras[r][c] as f32;
        }
    }
    let inv_mat = mat.try_inverse().unwrap_or(nalgebra::Matrix4::identity());

    let offsets = tractogram.offsets();
    let positions = tractogram.positions();
    let mut chunk = Vec::with_capacity(4 * 1024 * 1024);

    for i in 0..tractogram.nb_streamlines() {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let n_points = (end - start) as i32;

        chunk.extend_from_slice(&n_points.to_le_bytes());
        for &pt in &positions[start..end] {
            let p_ras = nalgebra::Point3::new(pt[0], pt[1], pt[2]);
            let p_center = inv_mat.transform_point(&p_ras);

            let vox_x = (p_center.x + 0.5) * voxel_sizes[0];
            let vox_y = (p_center.y + 0.5) * voxel_sizes[1];
            let vox_z = (p_center.z + 0.5) * voxel_sizes[2];

            chunk.extend_from_slice(&vox_x.to_le_bytes());
            chunk.extend_from_slice(&vox_y.to_le_bytes());
            chunk.extend_from_slice(&vox_z.to_le_bytes());
        }

        if chunk.len() >= 4_000_000 {
            file.write_all(&chunk)?;
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        file.write_all(&chunk)?;
    }

    Ok(())
}
