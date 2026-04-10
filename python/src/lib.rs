use std::path::PathBuf;

use bytemuck::cast_slice;
use half::f16;
use ndarray::{ArrayView1, ArrayView2};
use numpy::{Element, PyArray1, PyArray2};
use pyo3::exceptions::{PyFileNotFoundError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use serde_json::Value;
use trx_rs::{
    convert as rs_convert, header_from_reference, read_tractogram, write_tractogram, AnyTrxFile,
    ConversionOptions, DType, DataArray, Header, PositionsRef, Tractogram as RsTractogram,
    TrxError,
};

#[pyclass(module = "trxrs._core", name = "TrxFile")]
struct PyTrxFile {
    inner: AnyTrxFile,
}

#[pyclass(module = "trxrs._core", name = "Tractogram")]
struct PyTractogram {
    inner: RsTractogram,
}

#[pyfunction]
fn load(path: PathBuf) -> PyResult<PyTrxFile> {
    Ok(PyTrxFile {
        inner: AnyTrxFile::load(&path).map_err(map_trx_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (path, reference = None))]
fn load_tractogram(path: PathBuf, reference: Option<PathBuf>) -> PyResult<PyTractogram> {
    Ok(PyTractogram {
        inner: read_tractogram(
            &path,
            &ConversionOptions {
                header: load_reference_header(reference.as_deref())?,
                ..Default::default()
            },
        )
        .map_err(map_trx_error)?,
    })
}

#[pyfunction(signature = (input, output, reference = None, positions_dtype = "float32"))]
fn convert(
    input: PathBuf,
    output: PathBuf,
    reference: Option<PathBuf>,
    positions_dtype: &str,
) -> PyResult<()> {
    let dtype = parse_positions_dtype(positions_dtype)?;
    rs_convert(
        &input,
        &output,
        &ConversionOptions {
            header: load_reference_header(reference.as_deref())?,
            trx_positions_dtype: dtype,
            ..Default::default()
        },
    )
    .map_err(map_trx_error)
}

#[pymethods]
impl PyTrxFile {
    fn __len__(&self) -> usize {
        self.inner.nb_streamlines()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrxFile(dtype='{}', nb_streamlines={}, nb_vertices={})",
            self.inner.dtype().name(),
            self.inner.nb_streamlines(),
            self.inner.nb_vertices()
        )
    }

    #[getter]
    fn nb_streamlines(&self) -> usize {
        self.inner.nb_streamlines()
    }

    #[getter]
    fn nb_vertices(&self) -> usize {
        self.inner.nb_vertices()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }

    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        header_to_pydict(py, self.inner.header(), &dict)?;
        Ok(dict)
    }

    fn positions<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        match borrow.inner.positions_ref() {
            PositionsRef::F16(data) => array2_from_rows_f16(owner, data),
            PositionsRef::F32(data) => array2_from_rows(owner, data),
            PositionsRef::F64(data) => array2_from_rows(owner, data),
        }
    }

    fn offsets<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        match &borrow.inner {
            AnyTrxFile::F16(trx) => array1_from_slice(owner, without_sentinel(trx.offsets())),
            AnyTrxFile::F32(trx) => array1_from_slice(owner, without_sentinel(trx.offsets())),
            AnyTrxFile::F64(trx) => array1_from_slice(owner, without_sentinel(trx.offsets())),
        }
    }

    fn streamline<'py>(slf: Bound<'py, Self>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        let count = borrow.inner.nb_streamlines();
        if index >= count {
            return Err(PyKeyError::new_err(format!(
                "streamline index {index} out of range for {count} streamlines"
            )));
        }

        match &borrow.inner {
            AnyTrxFile::F16(trx) => array2_from_rows_f16(owner, trx.streamline(index)),
            AnyTrxFile::F32(trx) => array2_from_rows(owner, trx.streamline(index)),
            AnyTrxFile::F64(trx) => array2_from_rows(owner, trx.streamline(index)),
        }
    }

    fn get_dps<'py>(slf: Bound<'py, Self>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        dps_numpy(slf, name)
    }

    fn get_dpv<'py>(slf: Bound<'py, Self>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        dpv_numpy(slf, name)
    }

    fn get_group<'py>(slf: Bound<'py, Self>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        group_numpy(slf, name)
    }

    fn get_dpg<'py>(slf: Bound<'py, Self>, group: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
        dpg_numpy(slf, group, name)
    }

    fn dps_keys(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .inner
            .dps_entries()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        names.sort();
        names
    }

    fn dpv_keys(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .inner
            .dpv_entries()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        names.sort();
        names
    }

    fn group_keys(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .inner
            .groups_owned()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        names.sort();
        names
    }

    fn dpg_keys(&self) -> std::collections::HashMap<String, Vec<String>> {
        let mut groups = std::collections::HashMap::new();
        for (group, entries) in self.inner.dpg_group_entries() {
            let mut names: Vec<String> = entries.into_iter().map(|(name, _)| name).collect();
            names.sort();
            groups.insert(group, names);
        }
        groups
    }

    #[getter]
    fn data_per_streamline<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let names = slf.borrow().dps_keys();
        for name in names {
            let value = dps_numpy(slf.clone(), &name)?;
            dict.set_item(name, value)?;
        }
        Ok(dict)
    }

    #[getter]
    fn data_per_vertex<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let names = slf.borrow().dpv_keys();
        for name in names {
            let value = dpv_numpy(slf.clone(), &name)?;
            dict.set_item(name, value)?;
        }
        Ok(dict)
    }

    #[getter]
    fn groups<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let names = slf.borrow().group_keys();
        for name in names {
            let value = group_numpy(slf.clone(), &name)?;
            dict.set_item(name, value)?;
        }
        Ok(dict)
    }

    #[getter]
    fn data_per_group<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let groups = slf.borrow().dpg_keys();
        for (group, names) in groups {
            let inner = PyDict::new(py);
            for name in names {
                let value = dpg_numpy(slf.clone(), &group, &name)?;
                inner.set_item(name, value)?;
            }
            dict.set_item(group, inner)?;
        }
        Ok(dict)
    }
}

#[pymethods]
impl PyTractogram {
    fn __len__(&self) -> usize {
        self.inner.nb_streamlines()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tractogram(nb_streamlines={}, nb_vertices={})",
            self.inner.nb_streamlines(),
            self.inner.nb_vertices()
        )
    }

    #[getter]
    fn nb_streamlines(&self) -> usize {
        self.inner.nb_streamlines()
    }

    #[getter]
    fn nb_vertices(&self) -> usize {
        self.inner.nb_vertices()
    }

    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        header_to_pydict(py, self.inner.header(), &dict)?;
        Ok(dict)
    }

    fn positions<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        array2_from_rows(owner, borrow.inner.positions())
    }

    fn offsets<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        array1_from_slice(owner, without_sentinel(borrow.inner.offsets()))
    }

    fn streamline<'py>(slf: Bound<'py, Self>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        let count = borrow.inner.nb_streamlines();
        if index >= count {
            return Err(PyKeyError::new_err(format!(
                "streamline index {index} out of range for {count} streamlines"
            )));
        }
        array2_from_rows(owner, borrow.inner.streamline(index))
    }

    fn group_keys(&self) -> Vec<String> {
        let mut names: Vec<String> = self.inner.group_names().map(ToString::to_string).collect();
        names.sort();
        names
    }

    fn dpg_keys(&self) -> std::collections::HashMap<String, Vec<String>> {
        let mut groups = std::collections::HashMap::new();
        for (group, entries) in self.inner.dpg() {
            let mut names: Vec<String> = entries.keys().cloned().collect();
            names.sort();
            groups.insert(group.clone(), names);
        }
        groups
    }

    fn get_group<'py>(slf: Bound<'py, Self>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        let members = borrow
            .inner
            .group(name)
            .ok_or_else(|| PyKeyError::new_err(format!("no group named '{name}'")))?;
        array1_from_slice(owner, members)
    }

    fn get_dpg<'py>(slf: Bound<'py, Self>, group: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.clone().into_any();
        let borrow = slf.borrow();
        let entries = borrow
            .inner
            .dpg()
            .get(group)
            .ok_or_else(|| PyKeyError::new_err(format!("no DPG group named '{group}'")))?;
        let arr = entries.get(name).ok_or_else(|| {
            PyKeyError::new_err(format!("no DPG named '{name}' in group '{group}'"))
        })?;
        data_array_to_numpy(owner, arr, false)
    }

    #[getter]
    fn groups<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let names = slf.borrow().group_keys();
        for name in names {
            let value = tractogram_group_numpy(slf.clone(), &name)?;
            dict.set_item(name, value)?;
        }
        Ok(dict)
    }

    #[getter]
    fn data_per_group<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let groups = slf.borrow().dpg_keys();
        for (group, names) in groups {
            let inner = PyDict::new(py);
            for name in names {
                let value = tractogram_dpg_numpy(slf.clone(), &group, &name)?;
                inner.set_item(name, value)?;
            }
            dict.set_item(group, inner)?;
        }
        Ok(dict)
    }

    #[pyo3(signature = (path, positions_dtype = "float32"))]
    fn save(&self, path: PathBuf, positions_dtype: &str) -> PyResult<()> {
        let dtype = parse_positions_dtype(positions_dtype)?;
        write_tractogram(
            &path,
            &self.inner,
            &ConversionOptions {
                header: None,
                trx_positions_dtype: dtype,
                ..Default::default()
            },
        )
        .map_err(map_trx_error)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrxFile>()?;
    m.add_class::<PyTractogram>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_tractogram, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    Ok(())
}

fn map_trx_error(err: TrxError) -> PyErr {
    match err {
        TrxError::FileNotFound(path) => PyFileNotFoundError::new_err(path.display().to_string()),
        TrxError::Argument(message) | TrxError::Format(message) | TrxError::DType(message) => {
            PyValueError::new_err(message)
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

fn map_lookup_error(err: TrxError) -> PyErr {
    match err {
        TrxError::Argument(message) => PyKeyError::new_err(message),
        other => map_trx_error(other),
    }
}

fn parse_positions_dtype(value: &str) -> PyResult<DType> {
    let dtype = match value {
        "f16" => DType::Float16,
        "f32" => DType::Float32,
        "f64" => DType::Float64,
        other => DType::parse(other).map_err(map_trx_error)?,
    };
    if !matches!(dtype, DType::Float16 | DType::Float32 | DType::Float64) {
        return Err(PyValueError::new_err(format!(
            "positions_dtype must be one of float16, float32, float64, f16, f32, or f64; got {value}"
        )));
    }
    Ok(dtype)
}

fn load_reference_header(reference: Option<&std::path::Path>) -> PyResult<Option<Header>> {
    reference
        .map(header_from_reference)
        .transpose()
        .map_err(map_trx_error)
}

fn header_to_pydict<'py>(
    py: Python<'py>,
    header: &Header,
    dict: &Bound<'py, PyDict>,
) -> PyResult<()> {
    dict.set_item(
        "VOXEL_TO_RASMM",
        json_value_to_py(py, &serde_json::to_value(header.voxel_to_rasmm).unwrap())?,
    )?;
    dict.set_item(
        "DIMENSIONS",
        json_value_to_py(py, &serde_json::to_value(header.dimensions).unwrap())?,
    )?;
    dict.set_item("NB_STREAMLINES", header.nb_streamlines)?;
    dict.set_item("NB_VERTICES", header.nb_vertices)?;
    for (key, value) in &header.extra {
        dict.set_item(key, json_value_to_py(py, value)?)?;
    }
    Ok(())
}

fn json_value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(v) => Ok(PyBool::new(py, *v).to_owned().unbind().into()),
        Value::Number(v) => {
            if let Some(i) = v.as_i64() {
                Ok(i.into_pyobject(py)?.unbind().into())
            } else if let Some(u) = v.as_u64() {
                Ok(u.into_pyobject(py)?.unbind().into())
            } else {
                Ok(v.as_f64().unwrap().into_pyobject(py)?.unbind().into())
            }
        }
        Value::String(v) => Ok(v.into_pyobject(py)?.unbind().into()),
        Value::Array(values) => {
            let list = PyList::empty(py);
            for item in values {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.unbind().into())
        }
        Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, item) in values {
                dict.set_item(key, json_value_to_py(py, item)?)?;
            }
            Ok(dict.unbind().into())
        }
    }
}

fn data_array_to_numpy<'py>(
    owner: Bound<'py, PyAny>,
    arr: &DataArray,
    flatten_single_col: bool,
) -> PyResult<Bound<'py, PyAny>> {
    match arr.dtype() {
        DType::Float16 => data_array_f16(owner, arr, flatten_single_col),
        DType::Float32 => data_array_typed::<f32>(owner, arr, flatten_single_col),
        DType::Float64 => data_array_typed::<f64>(owner, arr, flatten_single_col),
        DType::Int8 => data_array_typed::<i8>(owner, arr, flatten_single_col),
        DType::Int16 => data_array_typed::<i16>(owner, arr, flatten_single_col),
        DType::Int32 => data_array_typed::<i32>(owner, arr, flatten_single_col),
        DType::Int64 => data_array_typed::<i64>(owner, arr, flatten_single_col),
        DType::UInt8 => data_array_typed::<u8>(owner, arr, flatten_single_col),
        DType::UInt16 => data_array_typed::<u16>(owner, arr, flatten_single_col),
        DType::UInt32 => data_array_typed::<u32>(owner, arr, flatten_single_col),
        DType::UInt64 => data_array_typed::<u64>(owner, arr, flatten_single_col),
    }
}

fn dps_numpy<'py>(slf: Bound<'py, PyTrxFile>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let owner = slf.clone().into_any();
    let borrow = slf.borrow();
    match &borrow.inner {
        AnyTrxFile::F16(trx) => {
            data_array_to_numpy(owner, trx.dps_array(name).map_err(map_lookup_error)?, false)
        }
        AnyTrxFile::F32(trx) => {
            data_array_to_numpy(owner, trx.dps_array(name).map_err(map_lookup_error)?, false)
        }
        AnyTrxFile::F64(trx) => {
            data_array_to_numpy(owner, trx.dps_array(name).map_err(map_lookup_error)?, false)
        }
    }
}

fn dpv_numpy<'py>(slf: Bound<'py, PyTrxFile>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let owner = slf.clone().into_any();
    let borrow = slf.borrow();
    match &borrow.inner {
        AnyTrxFile::F16(trx) => {
            data_array_to_numpy(owner, trx.dpv_array(name).map_err(map_lookup_error)?, false)
        }
        AnyTrxFile::F32(trx) => {
            data_array_to_numpy(owner, trx.dpv_array(name).map_err(map_lookup_error)?, false)
        }
        AnyTrxFile::F64(trx) => {
            data_array_to_numpy(owner, trx.dpv_array(name).map_err(map_lookup_error)?, false)
        }
    }
}

fn group_numpy<'py>(slf: Bound<'py, PyTrxFile>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let owner = slf.clone().into_any();
    let borrow = slf.borrow();
    match &borrow.inner {
        AnyTrxFile::F16(trx) => data_array_to_numpy(
            owner,
            trx.group_array(name).map_err(map_lookup_error)?,
            true,
        ),
        AnyTrxFile::F32(trx) => data_array_to_numpy(
            owner,
            trx.group_array(name).map_err(map_lookup_error)?,
            true,
        ),
        AnyTrxFile::F64(trx) => data_array_to_numpy(
            owner,
            trx.group_array(name).map_err(map_lookup_error)?,
            true,
        ),
    }
}

fn dpg_numpy<'py>(
    slf: Bound<'py, PyTrxFile>,
    group: &str,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let owner = slf.clone().into_any();
    let borrow = slf.borrow();
    match &borrow.inner {
        AnyTrxFile::F16(trx) => data_array_to_numpy(
            owner,
            trx.dpg_array(group, name).map_err(map_lookup_error)?,
            false,
        ),
        AnyTrxFile::F32(trx) => data_array_to_numpy(
            owner,
            trx.dpg_array(group, name).map_err(map_lookup_error)?,
            false,
        ),
        AnyTrxFile::F64(trx) => data_array_to_numpy(
            owner,
            trx.dpg_array(group, name).map_err(map_lookup_error)?,
            false,
        ),
    }
}

fn tractogram_group_numpy<'py>(
    slf: Bound<'py, PyTractogram>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let owner = slf.clone().into_any();
    let borrow = slf.borrow();
    let members = borrow
        .inner
        .group(name)
        .ok_or_else(|| PyKeyError::new_err(format!("no group named '{name}'")))?;
    array1_from_slice(owner, members)
}

fn tractogram_dpg_numpy<'py>(
    slf: Bound<'py, PyTractogram>,
    group: &str,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let owner = slf.clone().into_any();
    let borrow = slf.borrow();
    let entries = borrow
        .inner
        .dpg()
        .get(group)
        .ok_or_else(|| PyKeyError::new_err(format!("no DPG group named '{group}'")))?;
    let arr = entries
        .get(name)
        .ok_or_else(|| PyKeyError::new_err(format!("no DPG named '{name}' in group '{group}'")))?;
    data_array_to_numpy(owner, arr, false)
}

fn data_array_typed<'py, T>(
    owner: Bound<'py, PyAny>,
    arr: &DataArray,
    flatten_single_col: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Element + bytemuck::Pod,
{
    let slice: &[T] = cast_slice(arr.as_bytes());
    if arr.ncols() == 1 && flatten_single_col {
        array1_from_slice(owner, slice)
    } else {
        let view = ArrayView2::from_shape((arr.nrows(), arr.ncols()), slice)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let array = unsafe { PyArray2::borrow_from_array(&view, owner) };
        freeze_array(array.into_any())
    }
}

fn data_array_f16<'py>(
    owner: Bound<'py, PyAny>,
    arr: &DataArray,
    flatten_single_col: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let words: &[u16] = cast_slice(arr.as_bytes());
    if arr.ncols() == 1 && flatten_single_col {
        array1_f16_from_words(owner, words)
    } else {
        array2_f16_from_words(owner, words, arr.nrows(), arr.ncols())
    }
}

fn array2_from_rows<'py, T>(
    owner: Bound<'py, PyAny>,
    data: &[[T; 3]],
) -> PyResult<Bound<'py, PyAny>>
where
    T: Element + bytemuck::Pod,
{
    let flat: &[T] = cast_slice(data);
    let view = ArrayView2::from_shape((data.len(), 3), flat)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let array = unsafe { PyArray2::borrow_from_array(&view, owner) };
    freeze_array(array.into_any())
}

fn array2_from_rows_f16<'py>(
    owner: Bound<'py, PyAny>,
    data: &[[f16; 3]],
) -> PyResult<Bound<'py, PyAny>> {
    let words: &[u16] = cast_slice(data);
    array2_f16_from_words(owner, words, data.len(), 3)
}

fn array1_from_slice<'py, T>(owner: Bound<'py, PyAny>, data: &[T]) -> PyResult<Bound<'py, PyAny>>
where
    T: Element,
{
    let view = ArrayView1::from(data);
    let array = unsafe { PyArray1::borrow_from_array(&view, owner) };
    freeze_array(array.into_any())
}

fn without_sentinel(offsets: &[u32]) -> &[u32] {
    offsets.split_last().map_or(offsets, |(_, rest)| rest)
}

fn array1_f16_from_words<'py>(
    owner: Bound<'py, PyAny>,
    data: &[u16],
) -> PyResult<Bound<'py, PyAny>> {
    let raw = array1_from_slice(owner, data)?;
    reinterpret_as_f16(raw)
}

fn array2_f16_from_words<'py>(
    owner: Bound<'py, PyAny>,
    data: &[u16],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let view = ArrayView2::from_shape((rows, cols), data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let raw = unsafe { PyArray2::borrow_from_array(&view, owner) };
    reinterpret_as_f16(raw.into_any())
}

fn reinterpret_as_f16<'py>(array: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let viewed = array.call_method1("view", ("float16",))?;
    freeze_array(viewed)
}

fn freeze_array<'py>(array: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let flags = array.getattr("flags")?;
    flags.setattr("writeable", false)?;
    Ok(array)
}
