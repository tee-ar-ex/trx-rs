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

    let mut tr = Tractogram::new();
    let mut offset = 1000;

    while offset + 4 <= buffer.len() {
        let n_points = i32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap());
        offset += 4;

        let mut streamline = Vec::with_capacity(n_points as usize);
        for _ in 0..n_points {
            let x = f32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap());
            let y = f32::from_le_bytes(buffer[offset+4..offset+8].try_into().unwrap());
            let z = f32::from_le_bytes(buffer[offset+8..offset+12].try_into().unwrap());
            streamline.push([x, y, z]);
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
    
    let points_str = header_str[points_idx..].split_whitespace().nth(1).ok_or("No POINTS count")?;
    let num_points: usize = points_str.parse()?;

    let mut is_double = false;
    if let Some(type_str) = header_str[points_idx..].split_whitespace().nth(2) {
        if type_str == "double" {
            is_double = true;
        }
    }

    let header_end = header_str[points_idx..].find('\n').unwrap() + points_idx + 1;
    let mut pts = Vec::with_capacity(num_points * 3);
    
    let mut offset = header_end;
    for _ in 0..num_points * 3 {
        if is_double {
            let val = f64::from_be_bytes(buffer[offset..offset+8].try_into().unwrap());
            pts.push(val as f32);
            offset += 8;
        } else {
            let val = f32::from_be_bytes(buffer[offset..offset+4].try_into().unwrap());
            pts.push(val);
            offset += 4;
        }
    }

    let search_window = std::cmp::min(offset + 1024, buffer.len());
    let lines_str_chunk = String::from_utf8_lossy(&buffer[offset..search_window]);
    
    let lines_idx_in_chunk = lines_str_chunk.find("LINES ").ok_or("No LINES")?;
    let lines_idx = offset + lines_idx_in_chunk;
    
    let lines_str = lines_str_chunk[lines_idx_in_chunk..].split_whitespace().nth(1).ok_or("No LINES count")?;
    let num_lines: usize = lines_str.parse()?;

    let lines_header_end = lines_str_chunk[lines_idx_in_chunk..].find('\n').unwrap() + lines_idx + 1;
    offset = lines_header_end;

    let mut tr = Tractogram::new();
    
    if buffer[offset..].starts_with(b"OFFSETS") {
        let offsets_header_end = buffer[offset..].iter().position(|&c| c == b'\n').unwrap() + offset + 1;
        let is_int64 = buffer[offset..offsets_header_end].windows(5).any(|w| w == b"int64");
        offset = offsets_header_end;
        
        let mut offsets_vec = Vec::with_capacity(num_lines);
        for _ in 0..num_lines {
            if is_int64 {
                let val = u64::from_be_bytes(buffer[offset..offset+8].try_into().unwrap());
                offsets_vec.push(val as usize);
                offset += 8;
            } else {
                let val = u32::from_be_bytes(buffer[offset..offset+4].try_into().unwrap());
                offsets_vec.push(val as usize);
                offset += 4;
            }
        }
        
        for i in 0..num_lines-1 {
            let start = offsets_vec[i];
            let end = offsets_vec[i+1];
            let mut streamline = Vec::with_capacity(end - start);
            for pt_idx in start..end {
                streamline.push([pts[pt_idx*3], pts[pt_idx*3+1], pts[pt_idx*3+2]]);
            }
            tr.push_streamline(&streamline)?;
        }
        return Ok(tr);
    }

    let mut pt_idx = 0;
    for _ in 0..num_lines {
        if offset + 4 > buffer.len() { break; }
        let n_pts = i32::from_be_bytes(buffer[offset..offset+4].try_into().unwrap());
        offset += 4;
        
        if n_pts <= 0 { continue; }
        if pt_idx + (n_pts as usize) > num_points { break; }
        
        let mut streamline = Vec::with_capacity(n_pts as usize);
        for _ in 0..n_pts {
            offset += 4;
            streamline.push([pts[pt_idx*3], pts[pt_idx*3+1], pts[pt_idx*3+2]]);
            pt_idx += 1;
        }
        tr.push_streamline(&streamline)?;
    }

    Ok(tr)
}

pub fn write_trk(path: &Path, tractogram: &Tractogram) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    let mut header_bytes = vec![0u8; 1000];

    header_bytes[0..5].copy_from_slice(b"TRACK");

    let header = tractogram.header();
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
        ((vox_to_ras[0][0].powi(2) + vox_to_ras[1][0].powi(2) + vox_to_ras[2][0].powi(2)).sqrt()) as f32,
        ((vox_to_ras[0][1].powi(2) + vox_to_ras[1][1].powi(2) + vox_to_ras[2][1].powi(2)).sqrt()) as f32,
        ((vox_to_ras[0][2].powi(2) + vox_to_ras[1][2].powi(2) + vox_to_ras[2][2].powi(2)).sqrt()) as f32,
    ];
    header_bytes[12..16].copy_from_slice(&voxel_sizes[0].to_le_bytes());
    header_bytes[16..20].copy_from_slice(&voxel_sizes[1].to_le_bytes());
    header_bytes[20..24].copy_from_slice(&voxel_sizes[2].to_le_bytes());

    let mut offset = 440;
    for r in 0..4 {
        for c in 0..4 {
            let val = vox_to_ras[r][c] as f32;
            header_bytes[offset..offset+4].copy_from_slice(&val.to_le_bytes());
            offset += 4;
        }
    }

    header_bytes[948..952].copy_from_slice(b"RAS\0");

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
        let end = offsets[i+1] as usize;
        let n_points = (end - start) as i32;

        chunk.extend_from_slice(&n_points.to_le_bytes());
        for j in start..end {
            let pt = positions[j];
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
