use numpy::prelude::*;
use numpy::Element;
use pyo3::prelude::*;

const WIDTH: u16 = 1280;
const HEIGHT: u16 = 720;
const SPATIAL_DOWNSAMPLING: u16 = 4;
const SIGN_CHECK_RADIUS: u16 = 1;
const ACTIVITY_TAU: u64 = 20000; // µs
const TIMELINE_LENGTH: usize = 256;
const SAMPLING_FREQUENCY: f64 = 10.0; // Hz
const MOST_ACTIVE_TIMELINES_COUNT: usize = 40;

const DOWNSAMPLED_WIDTH: u16 = WIDTH / SPATIAL_DOWNSAMPLING;
const DOWNSAMPLED_HEIGHT: u16 = HEIGHT / SPATIAL_DOWNSAMPLING;
const ACTIVITY_MU: f64 = -1.0 / (ACTIVITY_TAU as f64);
const FFT_FREQUENCY: f64 = 1000.0; // Hz
const FFT_SAMPLES: usize = 1000; // samples

#[derive(Clone, Copy)]
struct Timeline {
    timestamps: [u64; TIMELINE_LENGTH],
    timestamps_index: usize,
    activity: f64,
    activity_t: u64,
}

impl Timeline {
    fn push(&mut self, t: u64) {
        self.timestamps[self.timestamps_index] = t;
        self.timestamps_index = (self.timestamps_index + 1) % TIMELINE_LENGTH;
        self.activity = (self.activity * ((t - self.activity_t) as f64 * ACTIVITY_MU).exp()) + 1.0;
        self.activity_t = t;
    }

    fn fill(&self, fft_samples: &mut Vec<rustfft::num_complex::Complex32>, t: u64) {
        fft_samples.fill(rustfft::num_complex::Complex32::default());
        let mut index = self.timestamps_index;
        loop {
            let timestamp = self.timestamps[index];
            if timestamp != u64::MAX {
                let fft_reverse_index =
                    ((t - timestamp) as f64 * (FFT_FREQUENCY / 1e6)).round() as usize;
                if fft_reverse_index < FFT_SAMPLES {
                    fft_samples[FFT_SAMPLES - 1 - fft_reverse_index].re = 1.0;
                }
            }
            index = (index + 1) % TIMELINE_LENGTH;
            if index == self.timestamps_index {
                break;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sign {
    None,
    Negative,
    Positive,
}

#[pyclass]
pub struct RpmCalculator {
    signed_timestamps: Vec<f64>,
    timelines: Vec<Timeline>,
    signs: Vec<Sign>,
    sample_index: usize,
    next_sample_t: u64,
    rpms: Vec<f64>,
    timelines_activities_and_indices: Vec<(f64, usize)>,
    fft_sum: Vec<f32>,
    fft_samples: Vec<rustfft::num_complex::Complex32>,
    fft_scratch: Vec<rustfft::num_complex::Complex32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

#[pymethods]
impl RpmCalculator {
    #[new]
    fn new() -> PyResult<Self> {
        let downsampled_length = DOWNSAMPLED_WIDTH as usize * DOWNSAMPLED_HEIGHT as usize;
        Ok(Self {
            signed_timestamps: vec![0.0; downsampled_length],
            timelines: vec![
                Timeline {
                    timestamps: [u64::MAX; TIMELINE_LENGTH],
                    timestamps_index: 0,
                    activity: 0.0,
                    activity_t: 0,
                };
                downsampled_length
            ],
            signs: vec![Sign::None; downsampled_length],
            sample_index: 0,
            next_sample_t: (1e6 / SAMPLING_FREQUENCY).round() as u64,
            rpms: Vec::new(),
            timelines_activities_and_indices: vec![(0.0, 0); downsampled_length],
            fft_sum: vec![0.0; FFT_SAMPLES],
            fft_samples: vec![rustfft::num_complex::Complex32::default(); FFT_SAMPLES],
            fft_scratch: vec![rustfft::num_complex::Complex32::default(); FFT_SAMPLES],
            fft: rustfft::FftPlanner::new().plan_fft_forward(FFT_SAMPLES),
        })
    }

    pub fn process(
        &mut self,
        events: &pyo3::Bound<'_, pyo3::types::PyAny>,
        spectrum: &pyo3::Bound<'_, numpy::PyArray1<f32>>,
    ) -> PyResult<Option<Vec<f64>>> {
        Python::with_gil(|python| -> PyResult<Option<Vec<f64>>> {
            let (array, length) = check_array(python, ArrayType::Dvs, events)?;
            self.rpms.clear();
            if length > 0 {
                for index in 0..length {
                    let (t, x, y, polarity) = unsafe {
                        let event_cell: *mut neuromorphic_types::DvsEvent<u64, u16, u16> =
                            array_at(python, array, index);
                        (
                            (*event_cell).t,
                            (*event_cell).x,
                            (*event_cell).y,
                            (*event_cell).polarity,
                        )
                    };
                    while t > self.next_sample_t {
                        for (index, timeline) in self.timelines.iter().enumerate() {
                            self.timelines_activities_and_indices[index] = (
                                timeline.activity
                                    * ((t - timeline.activity_t) as f64 * ACTIVITY_MU).exp(),
                                index,
                            );
                        }
                        self.timelines_activities_and_indices.sort_by(|a, b| {
                            if a.0 < b.0 {
                                std::cmp::Ordering::Greater
                            } else if a.0 > b.0 {
                                std::cmp::Ordering::Less
                            } else {
                                a.1.cmp(&b.1)
                            }
                        });
                        self.fft_sum.fill(0.0);
                        for (_, index) in self
                            .timelines_activities_and_indices
                            .iter()
                            .take(MOST_ACTIVE_TIMELINES_COUNT)
                        {
                            self.timelines[*index].fill(&mut self.fft_samples, t);
                            self.fft
                                .process_with_scratch(&mut self.fft_samples, &mut self.fft_scratch);
                            for (sample_index, sample) in self.fft_samples.iter().enumerate() {
                                self.fft_sum[sample_index] += sample.re.abs();
                            }
                        }

                        self.rpms.push(0.0); // @DEV

                        self.sample_index += 1;
                        self.next_sample_t =
                            (self.sample_index as f64 * (1e6 / SAMPLING_FREQUENCY)).round() as u64;
                    }
                    let x = x / SPATIAL_DOWNSAMPLING;
                    let y = y / SPATIAL_DOWNSAMPLING;
                    let downsampled_index = x as usize + (y as usize * DOWNSAMPLED_WIDTH as usize);
                    self.signed_timestamps[downsampled_index] = match polarity {
                        neuromorphic_types::DvsPolarity::Off => -(t as f64),
                        neuromorphic_types::DvsPolarity::On => t as f64,
                    };
                    if x >= SIGN_CHECK_RADIUS
                        && x < DOWNSAMPLED_WIDTH - SIGN_CHECK_RADIUS
                        && y >= SIGN_CHECK_RADIUS
                        && y < DOWNSAMPLED_HEIGHT - SIGN_CHECK_RADIUS
                    {
                        let mut sign = Sign::None;
                        'outer: for window_y in y - SIGN_CHECK_RADIUS..=y + SIGN_CHECK_RADIUS {
                            for window_x in x - SIGN_CHECK_RADIUS..=x + SIGN_CHECK_RADIUS {
                                let window_t = self.signed_timestamps[window_x as usize
                                    + (window_y as usize * DOWNSAMPLED_WIDTH as usize)];
                                if window_t == 0.0 {
                                    sign = Sign::None;
                                    break 'outer;
                                }
                                if window_t < 0.0 {
                                    match sign {
                                        Sign::None => {
                                            sign = Sign::Negative;
                                        }
                                        Sign::Negative => {}
                                        Sign::Positive => {
                                            sign = Sign::None;
                                            break 'outer;
                                        }
                                    }
                                } else {
                                    match sign {
                                        Sign::None => {
                                            sign = Sign::Positive;
                                        }
                                        Sign::Negative => {
                                            sign = Sign::None;
                                            break 'outer;
                                        }
                                        Sign::Positive => {}
                                    }
                                }
                            }
                        }
                        if !matches!(sign, Sign::None) {
                            let previous_sign = self.signs[downsampled_index];
                            if !matches!(previous_sign, Sign::None) {
                                if sign != previous_sign {
                                    self.timelines[downsampled_index].push(t);
                                }
                            }
                            self.signs[downsampled_index] = sign;
                        }
                    }
                }
            }
            {
                let mut array = unsafe { spectrum.as_array_mut() };
                if array.len() != FFT_SAMPLES {
                    return Err(pyo3::exceptions::PyException::new_err(format!(
                        "spectrum must have {} elements (got {})",
                        FFT_SAMPLES,
                        array.len()
                    )));
                }
                let slice = array.as_slice_mut().expect("spectrum is contiguous");
                for (index, value) in self.fft_sum.iter().enumerate() {
                    slice[index] = *value;
                }
            }
            if self.rpms.is_empty() {
                Ok(None)
            } else {
                Ok(Some(self.rpms.clone()))
            }
        })
    }
}

#[pymodule]
#[pyo3(name = "extension")]
fn figet_spinner(
    python: Python<'_>,
    module: &pyo3::Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_class::<RpmCalculator>()?;
    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub enum CheckArrayError {
    #[error("the object is not a numpy array")]
    PyArrayCheck,

    #[error("expected a one-dimensional array (got a {0} array)")]
    Dimensions(String),

    #[error("the array is not structured (https://numpy.org/doc/stable/user/basics.rec.html)")]
    NotStructured,

    #[error("the array must have a field \"{0}\"")]
    MissingField(String),

    #[error("the field \"{name}\" must have the type \"{expected_type}\" (got \"{actual_type}\")")]
    Field {
        name: String,
        expected_type: String,
        actual_type: String,
    },

    #[error(
        "the field \"{name}\" must have the offset \"{expected_offset}\" (got \"{actual_offset}\")"
    )]
    FieldOffset {
        name: String,
        expected_offset: core::ffi::c_long,
        actual_offset: core::ffi::c_long,
    },

    #[error("the array has extra fields (expected {expected}, got {actual})")]
    ExtraFields { expected: String, actual: String },
}

impl Into<PyErr> for CheckArrayError {
    fn into(self) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(self.to_string())
    }
}

pub fn check_array(
    python: Python,
    array_type: ArrayType,
    object: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> PyResult<(*mut numpy::npyffi::PyArrayObject, numpy::npyffi::npy_intp)> {
    if unsafe { numpy::npyffi::array::PyArray_Check(python, object.as_ptr()) } == 0 {
        return Err(CheckArrayError::PyArrayCheck.into());
    }
    let array = object.as_ptr() as *mut numpy::npyffi::PyArrayObject;
    let dimensions_length = unsafe { (*array).nd };
    if dimensions_length != 1 {
        let mut dimensions = String::new();
        for dimension in 0..dimensions_length {
            use std::fmt::Write;
            write!(
                dimensions,
                "{}{}",
                unsafe { *((*array).dimensions.offset(dimension as isize)) },
                if dimension < dimensions_length - 1 {
                    "x"
                } else {
                    ""
                }
            )
            .expect("write! did not fail");
        }
        return Err(CheckArrayError::Dimensions(dimensions).into());
    }
    let fields = unsafe { numpy::npyffi::PyDataType_FIELDS(python, (*array).descr) };
    if unsafe { pyo3::ffi::PyMapping_Check(fields) } == 0 {
        return Err(CheckArrayError::NotStructured.into());
    }
    let expected_fields = array_type.fields();
    let mut expected_offset = 0;
    for expected_field in expected_fields.iter() {
        let actual_field = unsafe {
            pyo3::ffi::PyMapping_GetItemString(
                fields,
                expected_field.null_terminated_name.as_ptr() as *const core::ffi::c_char,
            )
        };
        if actual_field.is_null() {
            return Err(CheckArrayError::MissingField(expected_field.name()).into());
        }
        let actual_description = unsafe { pyo3::ffi::PyTuple_GetItem(actual_field, 0) }
            as *mut numpy::npyffi::PyArray_Descr;
        let expected_description = expected_field.dtype(python);
        unsafe {
            (*expected_description).byteorder = b'<' as core::ffi::c_char;
        }
        if unsafe {
            numpy::PY_ARRAY_API.PyArray_EquivTypes(python, expected_description, actual_description)
        } == 0
            || unsafe { (*expected_description).byteorder != (*actual_description).byteorder }
        {
            let error = CheckArrayError::Field {
                name: expected_field.name(),
                expected_type: simple_description_to_string(python, expected_description),
                actual_type: simple_description_to_string(python, actual_description),
            };
            unsafe { pyo3::ffi::Py_DECREF(actual_field) };
            return Err(error.into());
        }
        let actual_offset =
            unsafe { pyo3::ffi::PyLong_AsLong(pyo3::ffi::PyTuple_GetItem(actual_field, 1)) };
        if actual_offset != expected_offset {
            unsafe { pyo3::ffi::Py_DECREF(actual_field) };
            return Err(CheckArrayError::FieldOffset {
                name: expected_field.name(),
                actual_offset,
                expected_offset,
            }
            .into());
        }
        expected_offset += expected_field.size() as core::ffi::c_long;
        unsafe { pyo3::ffi::Py_DECREF(actual_field) };
    }
    let expected_fields_length = expected_fields.len();
    let actual_names = unsafe { numpy::npyffi::PyDataType_NAMES(python, (*array).descr) };
    let actual_names_length = unsafe { pyo3::ffi::PyTuple_GET_SIZE(actual_names) };
    if actual_names_length != expected_fields_length as pyo3::ffi::Py_ssize_t {
        use std::fmt::Write;
        let mut expected = "[".to_owned();
        for (index, expected_field) in expected_fields.iter().enumerate() {
            write!(
                &mut expected,
                "\"{}\"{}",
                &expected_field.null_terminated_name
                    [0..expected_field.null_terminated_name.len() - 1],
                if index == expected_fields_length - 1 {
                    ""
                } else {
                    ", "
                }
            )
            .unwrap();
        }
        write!(&mut expected, "]").unwrap();
        let mut actual = "[".to_owned();
        for index in 0..actual_names_length {
            let mut length: pyo3::ffi::Py_ssize_t = 0;
            let data = unsafe {
                pyo3::ffi::PyUnicode_AsUTF8AndSize(
                    pyo3::ffi::PyTuple_GET_ITEM(actual_names, index),
                    &mut length as *mut pyo3::ffi::Py_ssize_t,
                )
            } as *const u8;
            write!(
                &mut actual,
                "\"{}\"{}",
                std::str::from_utf8(unsafe { std::slice::from_raw_parts(data, length as usize) })
                    .expect("pyo3::ffi::PyUnicode_AsUTF8AndSize returned valid UTF8 bytes"),
                if index == actual_names_length - 1 {
                    ""
                } else {
                    ", "
                }
            )
            .unwrap();
        }
        write!(&mut actual, "]").unwrap();
        return Err(CheckArrayError::ExtraFields { expected, actual }.into());
    }
    Ok((array, unsafe { *((*array).dimensions) }))
}

fn simple_description_to_string(
    python: Python,
    description: *mut numpy::npyffi::PyArray_Descr,
) -> String {
    format!(
        "{}{}{}",
        unsafe { (*description).byteorder } as u8 as char,
        unsafe { (*description).type_ } as u8 as char,
        unsafe { numpy::npyffi::PyDataType_ELSIZE(python, description) }
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayType {
    Dvs,
    AedatImu,
    AedatTrigger,
    Dat,
    EsGeneric,
    EsAtis,
    EsColor,
    EvtTrigger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Empty,
    Bool,
    F32,
    U8,
    U16,
    U64,
    Object,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field {
    pub null_terminated_name: &'static str,
    pub title: Option<&'static str>,
    pub field_type: FieldType,
}

impl Field {
    pub const fn new(
        null_terminated_name: &'static str,
        title: Option<&'static str>,
        field_type: FieldType,
    ) -> Self {
        Self {
            null_terminated_name,
            title,
            field_type,
        }
    }

    pub const fn size(&self) -> usize {
        match self.field_type {
            FieldType::Empty => 0,
            FieldType::Bool => 1,
            FieldType::F32 => 4,
            FieldType::U8 => 1,
            FieldType::U16 => 2,
            FieldType::U64 => 8,
            FieldType::Object => std::mem::size_of::<usize>(),
        }
    }

    pub fn name(&self) -> String {
        self.null_terminated_name[0..self.null_terminated_name.len() - 1].to_owned()
    }

    pub fn num(&self, python: Python) -> core::ffi::c_int {
        match self.field_type {
            FieldType::Empty => panic!("Field::num called on an empty field"),
            FieldType::Bool => bool::get_dtype(python).num(),
            FieldType::F32 => f32::get_dtype(python).num(),
            FieldType::U8 => u8::get_dtype(python).num(),
            FieldType::U16 => u16::get_dtype(python).num(),
            FieldType::U64 => u64::get_dtype(python).num(),
            FieldType::Object => numpy::PyArrayDescr::object(python).num(),
        }
    }

    pub fn dtype(&self, python: Python) -> *mut numpy::npyffi::PyArray_Descr {
        let dtype = unsafe { numpy::PY_ARRAY_API.PyArray_DescrFromType(python, self.num(python)) };
        if dtype.is_null() {
            panic!("PyArray_DescrFromType failed");
        }
        dtype
    }
}

const EMPTY: Field = Field {
    null_terminated_name: "\0",
    title: None,
    field_type: FieldType::Empty,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fields([Field; 11]);

impl ArrayType {
    pub const fn fields(self) -> Fields {
        Fields(match self {
            ArrayType::Dvs => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("on\0", Some("p"), FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::AedatImu => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("temperature\0", None, FieldType::F32),
                Field::new("accelerometer_x\0", None, FieldType::F32),
                Field::new("accelerometer_y\0", None, FieldType::F32),
                Field::new("accelerometer_z\0", None, FieldType::F32),
                Field::new("gyroscope_x\0", None, FieldType::F32),
                Field::new("gyroscope_y\0", None, FieldType::F32),
                Field::new("gyroscope_z\0", None, FieldType::F32),
                Field::new("magnetometer_x\0", None, FieldType::F32),
                Field::new("magnetometer_y\0", None, FieldType::F32),
                Field::new("magnetometer_z\0", None, FieldType::F32),
            ],
            ArrayType::AedatTrigger => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("source\0", None, FieldType::U8),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::Dat => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("payload\0", None, FieldType::U8),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EsGeneric => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("bytes\0", None, FieldType::Object),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EsAtis => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("exposure\0", Some("e"), FieldType::Bool),
                Field::new("polarity\0", Some("p"), FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EsColor => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("r\0", None, FieldType::Bool),
                Field::new("g\0", None, FieldType::Bool),
                Field::new("b\0", None, FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EvtTrigger => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("source\0", None, FieldType::U8),
                Field::new("rising\0", None, FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
        })
    }

    #[allow(unused)]
    pub fn dtype(self, python: Python) -> *mut numpy::npyffi::PyArray_Descr {
        self.fields().dtype(python)
    }

    pub fn new_array(
        self,
        python: Python,
        length: numpy::npyffi::npy_intp,
    ) -> *mut numpy::npyffi::PyArrayObject {
        self.fields().new_array(python, length)
    }
}

pub struct FieldIterator<'a> {
    fields: &'a Fields,
    index: usize,
    length: usize,
}

impl<'a> Iterator for FieldIterator<'a> {
    type Item = Field;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.length {
            self.index += 1;
            Some(self.fields.0[self.index - 1])
        } else {
            None
        }
    }
}

impl Fields {
    pub const fn len(&self) -> usize {
        let mut index = 0;
        while index < self.0.len() {
            if matches!(self.0[index].field_type, FieldType::Empty) {
                return index;
            }
            index += 1;
        }
        index
    }

    pub fn iter(&self) -> FieldIterator {
        FieldIterator {
            fields: self,
            index: 0,
            length: self.len(),
        }
    }

    pub fn dtype(&self, python: Python) -> *mut numpy::npyffi::PyArray_Descr {
        unsafe {
            let dtype_as_list = pyo3::ffi::PyList_New(self.len() as pyo3::ffi::Py_ssize_t);
            for (index, field) in self.iter().enumerate() {
                set_dtype_as_list_field(
                    python,
                    dtype_as_list,
                    index,
                    field.null_terminated_name,
                    field.title,
                    field.num(python),
                );
            }
            let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
            if numpy::PY_ARRAY_API.PyArray_DescrConverter(python, dtype_as_list, &mut dtype) < 0 {
                panic!("PyArray_DescrConverter failed");
            }
            pyo3::ffi::Py_DECREF(dtype_as_list);
            dtype
        }
    }

    pub fn new_array(
        &self,
        python: Python,
        mut length: numpy::npyffi::npy_intp,
    ) -> *mut numpy::npyffi::PyArrayObject {
        let dtype = self.dtype(python);
        unsafe {
            numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                python,
                numpy::PY_ARRAY_API
                    .get_type_object(python, numpy::npyffi::array::NpyTypes::PyArray_Type),
                dtype,
                1_i32,
                &mut length,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0_i32,
                std::ptr::null_mut(),
            ) as *mut numpy::npyffi::PyArrayObject
        }
    }
}

unsafe fn set_dtype_as_list_field(
    python: pyo3::Python,
    list: *mut pyo3::ffi::PyObject,
    index: usize,
    null_terminated_name: &str,
    title: Option<&str>,
    numpy_type: core::ffi::c_int,
) {
    let tuple = pyo3::ffi::PyTuple_New(2);
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        0 as pyo3::ffi::Py_ssize_t,
        match title {
            Some(title) => {
                let tuple = pyo3::ffi::PyTuple_New(2);
                if pyo3::ffi::PyTuple_SetItem(
                    tuple,
                    0 as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyUnicode_FromStringAndSize(
                        title.as_ptr() as *const core::ffi::c_char,
                        title.len() as pyo3::ffi::Py_ssize_t,
                    ),
                ) < 0
                {
                    panic!("PyTuple_SetItem 1 failed");
                }
                if pyo3::ffi::PyTuple_SetItem(
                    tuple,
                    1 as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyUnicode_FromStringAndSize(
                        null_terminated_name.as_ptr() as *const core::ffi::c_char,
                        (null_terminated_name.len() - 1) as pyo3::ffi::Py_ssize_t,
                    ),
                ) < 0
                {
                    panic!("PyTuple_SetItem 0 failed");
                }
                tuple
            }
            None => pyo3::ffi::PyUnicode_FromStringAndSize(
                null_terminated_name.as_ptr() as *const core::ffi::c_char,
                (null_terminated_name.len() - 1) as pyo3::ffi::Py_ssize_t,
            ),
        },
    ) < 0
    {
        panic!("PyTuple_SetItem 0 failed");
    }
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        1 as pyo3::ffi::Py_ssize_t,
        numpy::PY_ARRAY_API.PyArray_TypeObjectFromType(python, numpy_type),
    ) < 0
    {
        panic!("PyTuple_SetItem 1 failed");
    }
    if pyo3::ffi::PyList_SetItem(list, index as pyo3::ffi::Py_ssize_t, tuple) < 0 {
        panic!("PyList_SetItem failed");
    }
}

#[inline(always)]
pub unsafe fn array_at<T>(
    python: pyo3::Python,
    array: *mut numpy::npyffi::PyArrayObject,
    mut index: numpy::npyffi::npy_intp,
) -> *mut T {
    numpy::PY_ARRAY_API.PyArray_GetPtr(python, array, &mut index as *mut numpy::npyffi::npy_intp)
        as *mut T
}
