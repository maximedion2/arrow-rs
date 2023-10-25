use arrow_data::ArrayData;
use itertools::Itertools;

use crate::zarr_chunk::chunk::{
    ArrayParams, CompressorParams, LocalZarrFile, ZarrChunkReader, ZarrError, ZarrRead,
};
use arrow_array::*;
use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use arrow_schema::{DataType, TimeUnit};
use arrow_schema::{Field, FieldRef, Schema};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

//**********************
// a struct to read a zarr store with multiple variables
//**********************
pub struct ZarrStoreReader<Z: ZarrRead> {
    compressor_params: HashMap<String, Option<CompressorParams>>,
    array_params: HashMap<String, ArrayParams>,
    zarr_chunks: HashMap<String, Vec<Z>>,
    fill_values: HashMap<String, String>,
    types: HashMap<String, String>,
    combine_chunks: bool,
    chunk_it: Option<usize>,
    vars: Vec<String>,
    total_size: usize,
    n_chunks: usize,
}

impl<Z: ZarrRead> ZarrStoreReader<Z> {
    fn read_chunk_into_buffer(
        &self,
        chnk: &Z,
        var: &str,
        buf: &mut [u8],
    ) -> Result<usize, ZarrError> {
        let t = &self.types[var];
        let bytes: usize;

        macro_rules! read_with_data_type {
            ($data_type: ty) => {{
                let mut chnk_reader: ZarrChunkReader<$data_type, Z> =
                    ZarrChunkReader::new(
                        chnk,
                        &self.array_params[var],
                        self.compressor_params[var].as_ref(),
                        <$data_type>::from_str(&self.fill_values[var]).unwrap(),
                    );
                bytes = chnk_reader.read(buf).unwrap();
            }};
        }

        if t.contains("u1") {
            read_with_data_type!(u8);
            Ok(bytes)
        } else if t.contains("u2") {
            read_with_data_type!(u16);
            Ok(bytes)
        } else if t.contains("u4") {
            read_with_data_type!(u32);
            Ok(bytes)
        } else if t.contains("u8") {
            read_with_data_type!(u64);
            Ok(bytes)
        } else if t.contains("i1") {
            read_with_data_type!(i8);
            Ok(bytes)
        } else if t.contains("i2") {
            read_with_data_type!(i16);
            Ok(bytes)
        } else if t.contains("i4") {
            read_with_data_type!(i32);
            Ok(bytes)
        } else if t.contains("i8") {
            read_with_data_type!(i64);
            Ok(bytes)
        } else if t.contains("f4") {
            read_with_data_type!(f32);
            Ok(bytes)
        } else if t.contains("f8") {
            read_with_data_type!(f64);
            Ok(bytes)
        } else if t.contains("M8[s]") {
            read_with_data_type!(i64);
            Ok(bytes)
        } else if t.contains("M8[ms]") {
            read_with_data_type!(i64);
            Ok(bytes)
        } else if t.contains("M8[us]") {
            read_with_data_type!(i64);
            Ok(bytes)
        } else if t.contains("M8[ns]") {
            read_with_data_type!(i64);
            Ok(bytes)
        } else {
            return Err(ZarrError::InvalidArrayParams(format!(
                "could not determine data type from {}",
                t
            )));
        }
    }

    fn read_all_chunks(&mut self, buffers: &mut HashMap<String, Vec<u8>>) -> Option<()> {
        for var in self.vars.iter() {
            let mut curr_pos: usize = 0;
            let buf = buffers.get_mut(var).unwrap();
            for zarr_chunk in self.zarr_chunks[var].iter() {
                curr_pos += self
                    .read_chunk_into_buffer(&zarr_chunk, var, &mut buf[curr_pos..])
                    .unwrap();
            }
        }
        self.chunk_it = None;
        Some(())
    }

    fn read_one_chunk(&mut self, buffers: &mut HashMap<String, Vec<u8>>) -> Option<()> {
        for var in self.vars.iter() {
            let buf = buffers.get_mut(var).unwrap();
            let zarr_chunk = &self.zarr_chunks[var][self.chunk_it.unwrap()];
            self.read_chunk_into_buffer(zarr_chunk, var, buf).unwrap();
        }
        if self.chunk_it.unwrap() == self.n_chunks - 1 {
            self.chunk_it = None
        } else {
            self.chunk_it = Some(self.chunk_it.unwrap() + 1);
        }
        Some(())
    }
}

//**********************
// zarr store reader specialization for local chunk files
//**********************
type ZarrLocalStoreReader = ZarrStoreReader<LocalZarrFile>;

impl ZarrLocalStoreReader {
    fn new(
        store_path: &str,
        vars: Vec<String>,
        grid_positions: Vec<Vec<usize>>,
        compressor_params: HashMap<String, Option<CompressorParams>>,
        array_params: HashMap<String, ArrayParams>,
        fill_values: HashMap<String, String>,
        types: HashMap<String, String>,
        chunks: &Vec<usize>,
        shape: &Vec<usize>,
        combine_chunks: bool,
    ) -> Result<Self, ZarrError> {
        if chunks.len() != shape.len() {
            return Err(ZarrError::InvalidArrayParams(format!(
                "chunk and shape dimensions must match, found {} and {}",
                chunks.len(),
                shape.len()
            )));
        }

        let mut zarr_chunks: HashMap<String, Vec<LocalZarrFile>> = HashMap::new();
        for var in vars.iter() {
            zarr_chunks.insert(
                var.to_string(),
                grid_positions
                    .iter()
                    .map(|pos| {
                        let chunk_id = pos.iter().map(|i| i.to_string()).join(".");
                        LocalZarrFile::new(
                            Path::new(store_path)
                                .join(var)
                                .join(chunk_id)
                                .to_str()
                                .unwrap()
                                .to_string(),
                            pos.to_vec(),
                            shape,
                            chunks,
                        )
                    })
                    .collect(),
            );
        }

        // which variable we pick here is arbitrary, they should all have the
        // same chunks and shape (if they didn't this would have been caught
        // in the builder before getting here).
        let total_size = zarr_chunks[&vars[0]]
            .iter()
            .fold(0, |sum, zarr_chnk| sum + zarr_chnk.get_chunk_size());

        Ok(ZarrStoreReader {
            compressor_params: compressor_params,
            array_params: array_params,
            zarr_chunks: zarr_chunks,
            fill_values: fill_values,
            types: types,
            combine_chunks: combine_chunks,
            chunk_it: Some(0),
            vars: vars,
            total_size: total_size,
            n_chunks: grid_positions.len(),
        })
    }
}

//**********************
// implement the Iterator trait for a zarr store reader
//**********************
fn build_array(buf: Vec<u8>, dtype: &str) -> Result<ArrayRef, ArrowError> {
    let arr: ArrayRef;

    macro_rules! build_arr2 {
        ($arr_type: ty, $data_type: path, $native_type: ty) => {{
            let data = ArrayData::builder($data_type)
                .len(buf.len() / std::mem::size_of::<$native_type>())
                .add_buffer(Buffer::from(buf))
                .build()
                .unwrap();
            arr = Arc::new(<$arr_type>::from(data));
        }};
    }

    macro_rules! build_ts_arr {
        ($arr_type: ty, $time_unit: path) => {{
            let data = ArrayData::builder(DataType::Timestamp($time_unit, None))
                .len(buf.len() / std::mem::size_of::<i64>())
                .add_buffer(Buffer::from(buf))
                .build()
                .unwrap();
            arr = Arc::new(<$arr_type>::from(data));
        }};
    }

    if dtype.contains("u1") {
        build_arr2!(UInt8Array, DataType::UInt8, u8);
    } else if dtype.contains("u2") {
        build_arr2!(UInt16Array, DataType::UInt16, u16);
    } else if dtype.contains("u4") {
        build_arr2!(UInt32Array, DataType::UInt32, u32);
    } else if dtype.contains("u8") {
        build_arr2!(UInt64Array, DataType::UInt64, u64);
    } else if dtype.contains("i1") {
        build_arr2!(Int8Array, DataType::Int8, i8);
    } else if dtype.contains("i2") {
        build_arr2!(Int16Array, DataType::Int16, i16);
    } else if dtype.contains("i4") {
        build_arr2!(Int32Array, DataType::Int32, i32);
    } else if dtype.contains("i8") {
        build_arr2!(Int64Array, DataType::Int64, i64);
    } else if dtype.contains("f4") {
        build_arr2!(Float32Array, DataType::Float32, f32);
    } else if dtype.contains("f8") {
        build_arr2!(Float64Array, DataType::Float64, f64);
    } else if dtype.contains("M8[s]") {
        build_ts_arr!(TimestampSecondArray, TimeUnit::Second);
    } else if dtype.contains("M8[ms]") {
        build_ts_arr!(TimestampMillisecondArray, TimeUnit::Millisecond);
    } else if dtype.contains("M8[us]") {
        build_ts_arr!(TimestampMicrosecondArray, TimeUnit::Microsecond);
    } else if dtype.contains("M8[ns]") {
        build_ts_arr!(TimestampNanosecondArray, TimeUnit::Nanosecond);
    } else {
        return Err(ArrowError::ParseError(format!(
            "Unsupported data type {} when creating array",
            dtype
        )));
    }

    Ok(arr)
}

fn build_field(var_name: &str, dtype: &str) -> (usize, Result<FieldRef, ArrowError>) {
    let field: FieldRef;

    let mut size: usize = 0;
    if dtype.contains("u1") {
        size = 1;
        field = Arc::new(Field::new(var_name, DataType::UInt8, false));
    } else if dtype.contains("u2") {
        size = 2;
        field = Arc::new(Field::new(var_name, DataType::UInt16, false));
    } else if dtype.contains("u4") {
        size = 4;
        field = Arc::new(Field::new(var_name, DataType::UInt32, false));
    } else if dtype.contains("u8") {
        size = 8;
        field = Arc::new(Field::new(var_name, DataType::UInt64, false));
    } else if dtype.contains("i1") {
        size = 1;
        field = Arc::new(Field::new(var_name, DataType::Int8, false));
    } else if dtype.contains("i2") {
        size = 2;
        field = Arc::new(Field::new(var_name, DataType::Int16, false));
    } else if dtype.contains("i4") {
        size = 4;
        field = Arc::new(Field::new(var_name, DataType::Int32, false));
    } else if dtype.contains("i8") {
        size = 8;
        field = Arc::new(Field::new(var_name, DataType::Int64, false));
    } else if dtype.contains("f4") {
        size = 4;
        field = Arc::new(Field::new(var_name, DataType::Float32, false));
    } else if dtype.contains("f8") {
        size = 8;
        field = Arc::new(Field::new(var_name, DataType::Float64, false));
    } else if dtype.contains("M8[s]") {
        size = 8;
        field = Arc::new(Field::new(
            var_name,
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ));
    } else if dtype.contains("M8[ms]") {
        size = 8;
        field = Arc::new(Field::new(
            var_name,
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ));
    } else if dtype.contains("M8[us]") {
        field = Arc::new(Field::new(
            var_name,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ));
    } else if dtype.contains("M8[ns]") {
        size = 8;
        field = Arc::new(Field::new(
            var_name,
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ));
    } else {
        return (
            size,
            Err(ArrowError::SchemaError(format!(
                "Unsupported data type {} when creating field",
                dtype
            ))),
        );
    }

    (size, Ok(field))
}

impl<Z: ZarrRead> Iterator for ZarrStoreReader<Z> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk_it.is_none() {
            return None;
        }

        let mut fields: Vec<FieldRef> = Vec::with_capacity(self.vars.len());
        let mut buffers: HashMap<String, Vec<u8>> = HashMap::new();
        for var in self.vars.iter() {
            let (type_size, field) = build_field(var, &self.types[var]);
            fields.push(field.unwrap());
            let buf_size = if !self.combine_chunks {
                self.zarr_chunks[var][self.chunk_it.unwrap()].get_chunk_size()
            } else {
                self.total_size
            };
            buffers.insert(var.to_string(), vec![0; buf_size * type_size]);
        }

        if !self.combine_chunks {
            self.read_one_chunk(&mut buffers);
        } else {
            self.read_all_chunks(&mut buffers);
        }

        let mut arrs: Vec<ArrayRef> = Vec::with_capacity(self.vars.len());
        for var in self.vars.iter() {
            arrs.push(
                build_array(buffers.remove(var).unwrap(), &self.types[var]).unwrap(),
            );
        }

        Some(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs))
    }
}

//**********************
// a builder for zarr store readers
//**********************
pub struct ZarrStoreReaderBuilder<Z: ZarrRead> {
    store_path: String,
    grid_positions: Vec<Vec<usize>>,
    vars: Vec<String>,
    compressor_params: HashMap<String, Option<CompressorParams>>,
    array_params: HashMap<String, ArrayParams>,
    fill_values: HashMap<String, String>,
    types: HashMap<String, String>,
    chunks: Vec<usize>,
    shape: Vec<usize>,
    combine_chunks: bool,

    marker: std::marker::PhantomData<Z>,
}

impl<Z: ZarrRead> ZarrStoreReaderBuilder<Z> {
    pub fn new(store_path: String, combine_chunks: bool) -> Result<Self, ZarrError> {
        let mut chunks: Option<Vec<usize>> = None;
        let mut shape: Option<Vec<usize>> = None;
        let mut c_params: HashMap<String, Option<CompressorParams>> = HashMap::new();
        let mut a_params: HashMap<String, ArrayParams> = HashMap::new();
        let mut fill_values: HashMap<String, String> = HashMap::new();
        let mut types: HashMap<String, String> = HashMap::new();

        // regex expressions to extract types
        let integer_re = Regex::new(r"([\|><][ui][1248])").unwrap();
        let float_re = Regex::new(r"([\|><]f[48])").unwrap();
        let ts_re = Regex::new(r"([\|><]M8(\[s\]|\[ms\]|\[us\]|\[ns\]))").unwrap();

        let mut vars: Vec<String> = fs::read_dir(&store_path)
            .unwrap()
            .into_iter()
            .filter(|r| r.is_ok())
            .map(|r| r.unwrap().path())
            .filter(|r| r.is_dir())
            .map(|p| {
                p.as_path()
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        vars.sort();

        // go through the variables and make sure that chunks and shapes are
        // valid and consistent across variables. also extract fill values
        // and data types.
        for var in vars.iter() {
            let zarray_path = Path::new(&store_path)
                .join(var)
                .join(".zarray")
                .to_str()
                .unwrap()
                .to_string();
            a_params.insert(var.to_string(), ArrayParams::new(&zarray_path).unwrap());
            c_params.insert(
                var.to_string(),
                CompressorParams::new(&zarray_path).unwrap(),
            );
            if let (Some(c), Some(s)) = (&chunks, &shape) {
                if c != a_params[var].get_chunks() {
                    return Err(ZarrError::InvalidArrayParams(
                        "all chunks must be the same across varibles in zarr store"
                            .to_string(),
                    ));
                }
                if s != a_params[var].get_shape() {
                    return Err(ZarrError::InvalidArrayParams(
                        "all shapes must be the same across varibles in zarr store"
                            .to_string(),
                    ));
                }
            } else {
                chunks = Some(a_params[var].get_chunks().to_vec());
                shape = Some(a_params[var].get_shape().to_vec());
            }

            let metadata = fs::read_to_string(&zarray_path).unwrap();
            let j: serde_json::Value = serde_json::from_str(&metadata).unwrap();
            fill_values.insert(var.to_string(), j["fill_value"].to_string());
            let t = j["dtype"].to_string();

            let mut matched = false;
            for re in [&integer_re, &float_re, &ts_re] {
                if let Some(capt) = re.captures(&t) {
                    types.insert(
                        var.to_string(),
                        capt.get(0).unwrap().as_str().to_string(),
                    );
                    matched = true;
                    break;
                }
            }

            if !matched {
                return Err(ZarrError::InvalidArrayParams(format!(
                    "could not extract support type from {}",
                    t
                )));
            }
        }
        let chunks = chunks.unwrap();
        let shape = shape.unwrap();

        // build the chunk position vector
        let grid_positions: Vec<Vec<usize>>;
        let n_chunks: Vec<usize> = shape
            .iter()
            .zip(&chunks)
            .map(|(&shp, &chnk)| (shp as f64 / chnk as f64).ceil() as usize)
            .collect();
        if n_chunks.len() == 1 {
            grid_positions = (0..n_chunks[0]).map(|x| vec![x; 1]).collect();
        } else if n_chunks.len() == 2 {
            grid_positions = (0..n_chunks[0])
                .cartesian_product(0..n_chunks[1])
                .map(|(a, b)| vec![a, b])
                .collect();
        } else if n_chunks.len() == 3 {
            grid_positions = (0..n_chunks[0])
                .cartesian_product(0..n_chunks[1])
                .cartesian_product(0..n_chunks[2])
                .map(|((a, b), c)| vec![a, b, c])
                .collect();
        } else {
            return Err(ZarrError::ChunkError(
                "only up to 3 dimensions are supported".to_string(),
            ));
        }

        Ok(ZarrStoreReaderBuilder {
            store_path: store_path,
            grid_positions: grid_positions,
            vars: vars,
            compressor_params: c_params,
            array_params: a_params,
            fill_values: fill_values,
            types: types,
            chunks: chunks,
            shape: shape,
            combine_chunks: combine_chunks,
            marker: std::marker::PhantomData,
        })
    }

    pub fn as_projection(mut self, vars: Vec<String>) -> Self {
        self.vars = vars;
        self
    }
}

//**********************
// implement build methods for a zarr store builder on local files
//**********************
impl ZarrStoreReaderBuilder<LocalZarrFile> {
    pub fn build(self) -> ZarrLocalStoreReader {
        ZarrStoreReader::new(
            &self.store_path,
            self.vars,
            self.grid_positions,
            self.compressor_params,
            self.array_params,
            self.fill_values,
            self.types,
            &self.chunks,
            &self.shape,
            self.combine_chunks,
        )
        .unwrap()
    }

    pub fn build_multiple(self, n_readers: usize) -> Vec<ZarrLocalStoreReader> {
        let n_chunks =
            (self.grid_positions.len() as f64 / n_readers as f64).ceil() as usize;
        // the to_vec and clone calls are because of the closure that's run
        // multiple times, calling the build_multiple consumes the builder but
        // we can't have the readers take ownership since there's more than one.
        self.grid_positions
            .chunks(n_chunks)
            .map(|grid_pos| {
                ZarrLocalStoreReader::new(
                    &self.store_path,
                    self.vars.to_vec(),
                    grid_pos.to_vec(),
                    self.compressor_params.clone(),
                    self.array_params.clone(),
                    self.fill_values.clone(),
                    self.types.clone(),
                    &self.chunks,
                    &self.shape,
                    self.combine_chunks,
                )
                .unwrap()
            })
            .collect()
    }
}

#[cfg(test)]
mod zarr_store_reader_tests {
    use arrow_array::{
        cast::AsArray,
        types::{Float64Type, Int64Type, TimestampMillisecondType, UInt32Type},
    };
    use std::collections::HashSet;

    use super::*;
    const _TEST_DATA_PATH: &str =
        "/home/maxime/Documents/repos/arrow-rs/testing/data/zarr/";

    #[test]
    fn single_reader_all_chunks_at_once() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("float_data_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), true).unwrap();
        let mut reader = builder.build();
        let record_batch = reader.next().unwrap().unwrap();

        assert_eq!("lat", record_batch.schema().fields[0].name());
        assert_eq!(
            &DataType::Float64,
            record_batch.schema().fields[0].data_type()
        );

        let target_lats = [
            32.0, 33.0, 32.0, 33.0, 34.0, 35.0, 34.0, 35.0, 36.0, 36.0, 32.0, 33.0, 32.0,
            33.0, 34.0, 35.0, 34.0, 35.0, 36.0, 36.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        ];
        assert_eq!(
            record_batch
                .column(0)
                .as_primitive::<Float64Type>()
                .values(),
            &target_lats
        );

        assert_eq!("lon", record_batch.schema().fields[1].name());
        assert_eq!(
            &DataType::Float64,
            record_batch.schema().fields[1].data_type()
        );

        let target_lons = [
            -119.0, -119.0, -118.0, -118.0, -119.0, -119.0, -118.0, -118.0, -119.0,
            -118.0, -117.0, -117.0, -116.0, -116.0, -117.0, -117.0, -116.0, -116.0,
            -117.0, -116.0, -115.0, -115.0, -115.0, -115.0, -115.0,
        ];
        assert_eq!(
            record_batch
                .column(1)
                .as_primitive::<Float64Type>()
                .values(),
            &target_lons
        );

        assert_eq!("temperature", record_batch.schema().fields[2].name());
        assert_eq!(
            &DataType::Float64,
            record_batch.schema().fields[2].data_type()
        );

        let target_temps = [
            -2.13, 16.46, 28.71, -0.08, 13.73, 15.75, 23.78, 19.56, 1.91, 4.27, -3.79,
            24.02, 14.81, -3.57, 26.69, 7.46, -2.71, 18.91, -3.34, 4.25, 25.54, -4.23,
            19.93, 15.33, -9.15,
        ];
        assert_eq!(
            record_batch
                .column(2)
                .as_primitive::<Float64Type>()
                .values(),
            &target_temps
        );
    }

    #[test]
    fn read_with_projection() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("float_data_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let mut builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), true).unwrap();
        let vars = vec!["lat".to_string(), "lon".to_string()];
        builder = builder.as_projection(vars.clone());

        let mut reader = builder.build();
        let record_batch = reader.next().unwrap().unwrap();

        let fields: Vec<String> = record_batch
            .schema()
            .fields
            .iter()
            .map(|f| f.name().to_string())
            .collect();

        assert_eq!(fields.len(), 2);
        let target_fields: HashSet<String> = vars.into_iter().collect();
        assert_eq!(HashSet::from_iter(fields), target_fields);
    }

    #[test]
    fn single_reader_chunks_one_at_a_time() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("float_data_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), false).unwrap();
        let reader = builder.build();

        let mut n_reads = 0;
        for record_batch in reader.into_iter() {
            let record_batch = record_batch.unwrap();
            if [0, 1, 3, 4].contains(&n_reads) {
                assert_eq!(record_batch.num_rows(), 4);
            } else if [2, 5, 6, 7].contains(&n_reads) {
                assert_eq!(record_batch.num_rows(), 2);
            } else if n_reads == 8 {
                assert_eq!(record_batch.num_rows(), 1);
            }

            if n_reads == 2 {
                assert_eq!(record_batch.schema().fields[0].name(), "lat");
                assert_eq!(
                    record_batch
                        .column(0)
                        .as_primitive::<Float64Type>()
                        .values(),
                    &[36.0, 36.0],
                )
            }

            n_reads += 1;
        }
    }

    #[test]
    fn multiple_readers_all_chunks_at_once() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("float_data_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), true).unwrap();
        let mut readers = builder.build_multiple(2);
        let record_batch1 = readers[0].next().unwrap().unwrap();
        let record_batch2 = readers[1].next().unwrap().unwrap();

        assert_eq!(
            record_batch1
                .column(0)
                .as_primitive::<Float64Type>()
                .values(),
            &[
                32.0, 33.0, 32.0, 33.0, 34.0, 35.0, 34.0, 35.0, 36.0, 36.0, 32.0, 33.0,
                32.0, 33.0, 34.0, 35.0, 34.0, 35.0
            ]
        );
        assert_eq!(
            record_batch2
                .column(0)
                .as_primitive::<Float64Type>()
                .values(),
            &[36.0, 36.0, 32.0, 33.0, 34.0, 35.0, 36.0]
        );
    }

    #[test]
    fn readers_with_threads() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("float_data_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), true).unwrap();
        let mut readers = builder.build_multiple(2);
        let mut reader1 = readers.remove(0);
        let mut reader2 = readers.remove(0);

        let handle1 = std::thread::spawn(move || reader1.next().unwrap().unwrap());
        let handle2 = std::thread::spawn(move || reader2.next().unwrap().unwrap());

        let record_batch1 = handle1.join().unwrap();
        let record_batch2 = handle2.join().unwrap();

        assert_eq!(
            record_batch1
                .column(0)
                .as_primitive::<Float64Type>()
                .values(),
            &[
                32.0, 33.0, 32.0, 33.0, 34.0, 35.0, 34.0, 35.0, 36.0, 36.0, 32.0, 33.0,
                32.0, 33.0, 34.0, 35.0, 34.0, 35.0
            ]
        );
        assert_eq!(
            record_batch2
                .column(0)
                .as_primitive::<Float64Type>()
                .values(),
            &[36.0, 36.0, 32.0, 33.0, 34.0, 35.0, 36.0]
        );
    }

    #[test]
    fn multiple_readers_chunks_one_at_a_time() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("float_data_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), false).unwrap();
        let mut readers = builder.build_multiple(2);

        let mut n_reads = 0;
        for record_batch in readers.remove(0).into_iter() {
            let record_batch = record_batch.unwrap();
            if [0, 1, 3, 4].contains(&n_reads) {
                assert_eq!(record_batch.num_rows(), 4);
            } else if n_reads == 2 {
                assert_eq!(record_batch.num_rows(), 2);
            }

            if n_reads == 2 {
                assert_eq!(
                    record_batch
                        .column(1)
                        .as_primitive::<Float64Type>()
                        .values(),
                    &[-119.0, -118.0],
                )
            }
            n_reads += 1;
        }

        n_reads = 0;
        for record_batch in readers.remove(0).into_iter() {
            let record_batch = record_batch.unwrap();
            if [0, 1, 2].contains(&n_reads) {
                assert_eq!(record_batch.num_rows(), 2);
            } else if n_reads == 3 {
                assert_eq!(record_batch.num_rows(), 1);
            }

            if n_reads == 1 {
                assert_eq!(
                    record_batch
                        .column(1)
                        .as_primitive::<Float64Type>()
                        .values(),
                    &[-115.0, -115.0],
                )
            }
            n_reads += 1;
        }
    }

    #[test]
    fn timestamps_and_unsigned_ints() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("timestamp_data_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), false).unwrap();
        let mut reader = builder.build();

        let mut record_batch = reader.next().unwrap().unwrap();
        assert_eq!(
            record_batch.column(0).as_primitive::<UInt32Type>().values(),
            &[100, 101, 105, 106]
        );

        record_batch = reader.next().unwrap().unwrap();
        assert_eq!(
            record_batch
                .column(1)
                .as_primitive::<TimestampMillisecondType>()
                .values(),
            &[1685750400000, 1685836800000, 1686182400000, 1686268800000]
        );
    }

    #[test]
    fn one_d_array() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("one_d_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), true).unwrap();
        let mut reader = builder.build();

        let record_batch = reader.next().unwrap().unwrap();
        assert_eq!(
            record_batch.column(0).as_primitive::<Int64Type>().values(),
            &[500, 501, 502, 503, 504, 505, 506, 507, 508, 509]
        );
    }

    #[test]
    fn three_d_array() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("three_d_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), false).unwrap();
        let mut reader = builder.build();

        let mut record_batch = reader.next().unwrap().unwrap();
        assert_eq!(
            record_batch.column(0).as_primitive::<Int64Type>().values(),
            &[500, 501, 503, 504, 509, 510, 512, 513]
        );

        record_batch = reader.next().unwrap().unwrap();
        assert_eq!(
            record_batch.column(0).as_primitive::<Int64Type>().values(),
            &[502, 505, 511, 514]
        );
    }

    #[test]
    fn missing_chunks() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("missing_chunk_reader_test.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), true).unwrap();
        let mut reader = builder.build();

        let record_batch = reader.next().unwrap().unwrap();
        assert_eq!(
            record_batch
                .column(0)
                .as_primitive::<Float64Type>()
                .values(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.01, 2.31, 6.69, 4.82, 0.0, 0.0,
                0.0, 0.0
            ],
        );
        assert_eq!(
            record_batch
                .column(1)
                .as_primitive::<Float64Type>()
                .values(),
            &[
                15.0, 15.0, 15.0, 15.0, 18.6, 17.63, 11.86, -4.54, 15.0, 15.0, 15.0,
                15.0, 15.0, 15.0, 15.0, 15.0
            ],
        );
    }

    #[test]
    fn large_dataset() {
        let store_path = Path::new(_TEST_DATA_PATH)
            .join("large_dataset.zarr")
            .to_str()
            .unwrap()
            .to_string();
        let vars = vec![
            "lat".to_string(),
            "lon".to_string(),
            "temperature".to_string(),
        ];
        let mut builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), true).unwrap();
        builder = builder.as_projection(vars);
        //let mut reader = builder.build();
        let mut readers = builder.build_multiple(4);

        let t1 = std::time::Instant::now();
        //let record_batch = reader.next().unwrap().unwrap();
        let mut reader1 = readers.remove(0);
        let mut reader2 = readers.remove(0);
        let mut reader3 = readers.remove(0);
        let mut reader4 = readers.remove(0);

        let handle1 = std::thread::spawn(move || reader1.next().unwrap().unwrap());
        let handle2 = std::thread::spawn(move || reader2.next().unwrap().unwrap());
        let handle3 = std::thread::spawn(move || reader3.next().unwrap().unwrap());
        let handle4 = std::thread::spawn(move || reader4.next().unwrap().unwrap());

        let record_batch1 = handle1.join().unwrap();
        let record_batch2 = handle2.join().unwrap();
        let record_batch3 = handle3.join().unwrap();
        let record_batch4 = handle4.join().unwrap();

        println!("{:?} secs", t1.elapsed());
        println!("{:?}", record_batch1);
        println!("{:?}", record_batch2);
        println!("{:?}", record_batch3);
        println!("{:?}", record_batch4);
    }
}
