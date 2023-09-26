use itertools::Itertools;

use crate::zarr_chunk::chunk::{
    ArrayParams, CompressorParams, LocalZarrFile, ZarrChunkReader, ZarrError, ZarrRead,
};
use arrow_array::*;
use arrow_schema::ArrowError;
use arrow_schema::{DataType, TimeUnit};
use arrow_schema::{Field, FieldRef, Schema};
use bytemuck::cast_slice;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
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
    read_chunks_one_at_a_time: bool,
    chunk_it: usize,
    vars: Vec<String>,
    chunk_size: usize,
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
        let bytes;

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
        } else {
            return Err(ZarrError::InvalidArrayParams(format!(
                "could not determine data type from {}",
                t
            )));
        }
    }

    fn read_all_chunks(
        &mut self,
        buffers: &mut HashMap<String, Vec<u8>>,
    ) -> Result<(), ZarrError> {
        if self.chunk_it > 0 {
            return Err(ZarrError::ChunkError(
                "zarr chunks have already been read once".to_string(),
            ));
        }
        for var in self.vars.iter() {
            let mut curr_pos: usize = 0;
            let buf = buffers.get_mut(var).unwrap();
            for zarr_chunk in self.zarr_chunks[var].iter() {
                curr_pos += self
                    .read_chunk_into_buffer(&zarr_chunk, var, &mut buf[curr_pos..])
                    .unwrap();
            }
        }
        self.chunk_it += 1;
        Ok(())
    }

    fn read_one_chunk(
        &mut self,
        buffers: &mut HashMap<String, Vec<u8>>,
    ) -> Result<usize, ZarrError> {
        let mut bytes_read: usize = 0;
        for var in self.vars.iter() {
            if self.chunk_it >= self.zarr_chunks[var].len() {
                return Err(ZarrError::ChunkError(
                    "trying to read passed last zarr chunk".to_string(),
                ));
            }
            let buf = buffers.get_mut(var).unwrap();
            let zarr_chunk = &self.zarr_chunks[var][self.chunk_it];
            bytes_read = self.read_chunk_into_buffer(zarr_chunk, var, buf).unwrap();
        }
        self.chunk_it += 1;
        Ok(bytes_read)
    }
}

//**********************
// a builder for zarr store readers
//**********************
struct ZarrStoreReaderBuilder<Z: ZarrRead> {
    store_path: String,
    grid_positions: Vec<Vec<usize>>,
    vars: Vec<String>,
    read_chunks_one_at_a_time: bool,

    marker: std::marker::PhantomData<Z>,
}

impl ZarrStoreReaderBuilder<LocalZarrFile> {
    pub fn build(self) -> ZarrLocalStoreReader {
        ZarrStoreReader::new(
            self.store_path,
            self.vars,
            self.grid_positions,
            self.read_chunks_one_at_a_time,
        )
        .unwrap()
    }

    pub fn build_multiple(self, n_readers: usize) -> Vec<ZarrLocalStoreReader> {
        let n_chunks =
            (self.grid_positions.len() as f64 / n_readers as f64).ceil() as usize;
        self.grid_positions
            .chunks(n_chunks)
            .map(|grid_pos| {
                ZarrLocalStoreReader::new(
                    self.store_path.to_string(),
                    self.vars.to_vec(),
                    grid_pos.to_vec(),
                    self.read_chunks_one_at_a_time,
                )
                .unwrap()
            })
            .collect()
    }
}

impl ZarrStoreReaderBuilder<LocalZarrFile> {
    pub fn new(
        store_path: String,
        vars: Vec<String>,
        read_chunks_one_at_a_time: bool,
    ) -> Result<Self, ZarrError> {
        let mut chunks: Vec<usize> = vec![0; 1];
        let mut shape: Vec<usize> = vec![0; 1];
        for var in vars.iter() {
            let zarray_path = format!("{}{}/.zarray", store_path, var);
            let array_params = ArrayParams::new(&zarray_path).unwrap();
            chunks = array_params.get_chunks().to_vec();
            shape = array_params.get_shape().to_vec();
        }

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
            read_chunks_one_at_a_time: read_chunks_one_at_a_time,
            marker: std::marker::PhantomData,
        })
    }
}

//**********************
// zarr store reader specialization for local chunk files
//**********************
type ZarrLocalStoreReader = ZarrStoreReader<LocalZarrFile>;

impl ZarrLocalStoreReader {
    pub fn new(
        store_path: String,
        vars: Vec<String>,
        grid_positions: Vec<Vec<usize>>,
        read_one_at_a_time: bool,
    ) -> Result<Self, ZarrError> {
        let mut compressor_params: HashMap<String, Option<CompressorParams>> =
            HashMap::new();
        let mut array_params: HashMap<String, ArrayParams> = HashMap::new();
        let mut zarr_chunks: HashMap<String, Vec<LocalZarrFile>> = HashMap::new();
        let mut fill_values: HashMap<String, String> = HashMap::new();
        let mut types: HashMap<String, String> = HashMap::new();

        let mut chunks: Option<Vec<usize>> = None;
        let mut shape: Option<Vec<usize>> = None;
        for var in vars.iter() {
            let zarray_path = format!("{}{}/.zarray", store_path, var);
            array_params.insert(var.to_string(), ArrayParams::new(&zarray_path).unwrap());
            compressor_params.insert(
                var.to_string(),
                CompressorParams::new(&zarray_path).unwrap(),
            );

            if let Some(c) = chunks {
                if &c != array_params[var].get_chunks() {
                    return Err(ZarrError::InvalidArrayParams(
                        "all variables in the store need to have the same chunks"
                            .to_string(),
                    ));
                }
            }
            chunks = Some(array_params[var].get_chunks().to_vec());
            if let Some(s) = shape {
                if &s != array_params[var].get_shape() {
                    return Err(ZarrError::InvalidArrayParams(
                        "all variables in the store need to have the same shape"
                            .to_string(),
                    ));
                }
            }
            shape = Some(array_params[var].get_shape().to_vec());

            zarr_chunks.insert(
                var.to_string(),
                grid_positions
                    .iter()
                    .map(|pos| {
                        let chunk_id = pos.iter().map(|i| i.to_string()).join(".");
                        LocalZarrFile::new(
                            format!("{}/{}/{}", store_path, var, chunk_id),
                            pos.to_vec(),
                            array_params[var].get_shape(),
                            array_params[var].get_chunks(),
                        )
                    })
                    .collect(),
            );

            let metadata =
                fs::read_to_string(&zarray_path).expect("Unable to read .zarray");
            let j: serde_json::Value = serde_json::from_str(&metadata).unwrap();
            fill_values.insert(var.to_string(), j["fill_value"].to_string());
            types.insert(var.to_string(), (j["dtype"].to_string())[1..4].to_string());
        }

        Ok(ZarrStoreReader {
            compressor_params: compressor_params,
            array_params: array_params,
            zarr_chunks: zarr_chunks,
            fill_values: fill_values,
            types: types,
            read_chunks_one_at_a_time: read_one_at_a_time,
            chunk_it: 0,
            vars: vars,
            chunk_size: chunks.unwrap().iter().fold(1, |mult, x| mult * x),
            total_size: shape.unwrap().iter().fold(1, |mult, x| mult * x),
            n_chunks: grid_positions.len(),
        })
    }
}

//**********************
// implement the Iterator trait for a zarr store reader
//**********************
fn build_array(buf: Vec<u8>, dtype: &str) -> Result<ArrayRef, ArrowError> {
    let arr: ArrayRef;
    macro_rules! build_arr {
        ($data_type: ty, $arrow_type: ty) => {{
            let v: Vec<Option<$data_type>> =
                cast_slice(&buf).to_vec().into_iter().map(Some).collect();
            arr = Arc::new(<$arrow_type>::from(v));
        }};
    }

    if dtype.contains("u1") {
        build_arr!(u8, UInt8Array);
    } else if dtype.contains("u2") {
        build_arr!(u16, UInt16Array);
    } else if dtype.contains("u4") {
        build_arr!(u32, UInt32Array);
    } else if dtype.contains("u8") {
        build_arr!(u64, UInt64Array);
    } else if dtype.contains("i1") {
        build_arr!(i8, Int8Array);
    } else if dtype.contains("i2") {
        build_arr!(i16, Int16Array);
    } else if dtype.contains("i4") {
        build_arr!(i32, Int32Array);
    } else if dtype.contains("i8") {
        build_arr!(i64, Int64Array);
    } else if dtype.contains("f4") {
        build_arr!(f32, Float32Array);
    } else if dtype.contains("f8") {
        build_arr!(f64, Float64Array);
    } else if dtype.contains("M8[s]") {
        build_arr!(i64, TimestampSecondArray);
    } else if dtype.contains("M8[ms]") {
        build_arr!(i64, TimestampMillisecondArray);
    } else if dtype.contains("M8[us]") {
        build_arr!(i64, TimestampMicrosecondArray);
    } else if dtype.contains("M8[ns]") {
        build_arr!(i64, TimestampNanosecondArray);
    } else {
        return Err(ArrowError::ParseError(format!(
            "Unsupported data type {} when creating array",
            dtype
        )));
    }

    Ok(arr)
}

fn build_field(var_name: &str, dtype: &str) -> Result<FieldRef, ArrowError> {
    let field: FieldRef;

    if dtype.contains("u1") {
        field = Arc::new(Field::new(var_name, DataType::UInt8, false));
    } else if dtype.contains("u2") {
        field = Arc::new(Field::new(var_name, DataType::UInt16, false));
    } else if dtype.contains("u4") {
        field = Arc::new(Field::new(var_name, DataType::UInt32, false));
    } else if dtype.contains("u8") {
        field = Arc::new(Field::new(var_name, DataType::UInt64, false));
    } else if dtype.contains("i1") {
        field = Arc::new(Field::new(var_name, DataType::Int8, false));
    } else if dtype.contains("i2") {
        field = Arc::new(Field::new(var_name, DataType::Int16, false));
    } else if dtype.contains("i4") {
        field = Arc::new(Field::new(var_name, DataType::Int32, false));
    } else if dtype.contains("i8") {
        field = Arc::new(Field::new(var_name, DataType::Int64, false));
    } else if dtype.contains("f4") {
        field = Arc::new(Field::new(var_name, DataType::Float32, false));
    } else if dtype.contains("f8") {
        field = Arc::new(Field::new(var_name, DataType::Float64, false));
    } else if dtype.contains("M8[s]") {
        field = Arc::new(Field::new(
            var_name,
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ));
    } else if dtype.contains("M8[ms]") {
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
        field = Arc::new(Field::new(
            var_name,
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ));
    } else {
        return Err(ArrowError::SchemaError(format!(
            "Unsupported data type {} when creating field",
            dtype
        )));
    }

    Ok(field)
}

impl<Z: ZarrRead> Iterator for ZarrStoreReader<Z> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.read_chunks_one_at_a_time {
            if self.chunk_it == self.n_chunks {
                return None;
            }
        } else {
            if self.chunk_it != 0 {
                return None;
            }
        }

        let mut buffers: HashMap<String, Vec<u8>> = HashMap::new();
        for var in self.vars.iter() {
            let buf_size = if self.read_chunks_one_at_a_time {
                self.zarr_chunks[var][self.chunk_it].get_chunk_size()
            } else {
                self.total_size
            };
            buffers.insert(
                var.to_string(),
                vec![
                    0;
                    buf_size
                        * self.types[var]
                            .chars()
                            .last()
                            .unwrap()
                            .to_digit(10)
                            .unwrap() as usize
                ],
            );
        }

        if self.read_chunks_one_at_a_time {
            self.read_one_chunk(&mut buffers).unwrap();
        } else {
            self.read_all_chunks(&mut buffers).unwrap();
        }

        let mut arrs: Vec<ArrayRef> = Vec::with_capacity(self.vars.len());
        let mut fields: Vec<FieldRef> = Vec::with_capacity(self.vars.len());
        for var in self.vars.iter() {
            arrs.push(
                build_array(buffers.get_mut(var).unwrap().to_vec(), &self.types[var])
                    .unwrap(),
            );
            fields.push(build_field(var, &self.types[var]).unwrap());
        }

        Some(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs))
    }
}

#[cfg(test)]
mod zarr_store_reader_tests {
    use arrow_array::{cast::AsArray, types::Float64Type};

    use super::*;

    #[test]
    fn single_reader_all_chunks_at_once() {
        let store_path =
            "/home/maxime/Documents/repos/arrow-rs/testing/data/zarr/example6.zarr/";
        let vars = vec![
            "lat".to_string(),
            "lon".to_string(),
            "temperature".to_string(),
        ];
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), vars, false).unwrap();
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
    fn single_reader_chunks_one_at_a_time() {
        let store_path =
            "/home/maxime/Documents/repos/arrow-rs/testing/data/zarr/example6.zarr/";
        let vars = vec![
            "lat".to_string(),
            "lon".to_string(),
            "temperature".to_string(),
        ];
        let builder: ZarrStoreReaderBuilder<LocalZarrFile> =
            ZarrStoreReaderBuilder::new(store_path.to_string(), vars, true).unwrap();
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
}
