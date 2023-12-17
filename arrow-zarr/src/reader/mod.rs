use zarr_read::{ColumnProjection, ZarrRead, ZarrInMemoryChunk, ZarrInMemoryArray};
use arrow_schema::{Field, FieldRef, Schema, DataType, TimeUnit};
use metadata::{ZarrStoreMetadata, ZarrArrayMetadata, ZarrDataType,  CompressorType, Endianness, MatrixOrder};
use errors::{ZarrResult, ZarrError};
use arrow_array::*;
use std::sync::Arc;
use arrow_data::ArrayData;
use arrow_buffer::Buffer;
use arrow_buffer::ToByteSlice;
use std::io::Read;
use itertools::Itertools;

mod metadata;
pub mod zarr_read;
pub mod errors;


pub(crate) struct ZarrStore<T: ZarrRead> {
    meta: ZarrStoreMetadata,
    chunk_positions: Vec<Vec<usize>>,
    zarr_reader: T,
    projection: Option<ColumnProjection>,
    curr_chunk: usize,
}

impl<T: ZarrRead> ZarrStore<T> {
    pub(crate) fn new(
        zarr_reader: T,
        chunk_positions: Vec<Vec<usize>>,
        projection: Option<ColumnProjection>,
    ) -> ZarrResult<Self> {
        Ok(Self {
            meta: zarr_reader.get_zarr_metadata()?,
            chunk_positions,
            zarr_reader,
            projection,
            curr_chunk: 0,
        })
    }

    pub(crate) fn get_metadata(&self) -> &ZarrStoreMetadata {
        &self.meta
    }
}

impl<'a, T: ZarrRead> Iterator for ZarrStore<T> {
    type Item = ZarrResult<ZarrInMemoryChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_chunk == self.chunk_positions.len() {
            return None;
        }

        let pos = &self.chunk_positions[self.curr_chunk];
        let chnk = self.zarr_reader.get_zarr_chunk(
            pos,
            self.meta.get_columns().clone(),
            self.meta.get_real_dims(pos),
            self.projection.as_ref(),
        )
        ;
        self.curr_chunk += 1;
        Some(chnk)
    }

}

fn build_field(t: &ZarrDataType, col_name: String) -> ZarrResult<(usize, FieldRef)> {
    match t {
        ZarrDataType::Bool => {
            return Ok((1, Arc::new(Field::new(col_name, DataType::Boolean, false))))
        },
        ZarrDataType::UInt(s) => {
            match s {
                1 => {
                    return Ok((1, Arc::new(Field::new(col_name, DataType::UInt8, false))))
                },
                2 => {
                    return Ok((2, Arc::new(Field::new(col_name, DataType::UInt16, false))))
                },
                4 => {
                    return Ok((4, Arc::new(Field::new(col_name, DataType::UInt32, false))))
                },
                8 => {
                    return Ok((8, Arc::new(Field::new(col_name, DataType::UInt64, false))))
                },
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        ZarrDataType::Int(s) => {
            match s {
                1 => {
                    return Ok((1, Arc::new(Field::new(col_name, DataType::Int8, false))))
                },
                2 => {
                    return Ok((2, Arc::new(Field::new(col_name, DataType::Int16, false))))
                },
                4 => {
                    return Ok((4, Arc::new(Field::new(col_name, DataType::Int32, false))))
                },
                8 => {
                    return Ok((8, Arc::new(Field::new(col_name, DataType::Int64, false))))
                },
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        ZarrDataType::Float(s) => {
            match s {
                4 => {
                    return Ok((4, Arc::new(Field::new(col_name, DataType::Float32, false))))
                },
                8 => {
                    return Ok((8, Arc::new(Field::new(col_name, DataType::Float64, false))))
                },
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        ZarrDataType::FixedLengthString(s) => {
            return Ok((*s, Arc::new(Field::new(col_name, DataType::Utf8, false))))
        },
        ZarrDataType::TimeStamp(8, u) => {
            match u.as_str() {
                "s" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Second, None),
                            false,
                        )
                    )))
                },
                "ms" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Millisecond, None),
                            false,
                        )
                    )))
                },
                "us" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Microsecond, None),
                            false,
                        )
                    )))
                },
                "ns" => {
                    return Ok((
                        8, 
                        Arc::new(Field::new(
                            col_name,
                            DataType::Timestamp(TimeUnit::Nanosecond, None),
                            false,
                        )
                    )))
                }
                _ => {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
            }
        },
        _ =>  {return Err(ZarrError::InvalidMetadata("Invalid data type".to_string()))}
    }
}

fn build_array(buf: Vec<u8>, t: &DataType, s: usize) -> ZarrResult<ArrayRef> {
    let data = match t {
        DataType::Utf8 => {
            ArrayData::builder(t.clone())
            .len(buf.len() / s)
            .add_buffer(
                Buffer::from(
                    (0..=buf.len()).step_by(s).map(|x| x as i32).collect::<Vec<i32>>().to_byte_slice()
                )
            )
            .add_buffer(Buffer::from(buf))
            .build()?
        },
        _ => {
            ArrayData::builder(t.clone())
            .len(buf.len() / s)
            .add_buffer(Buffer::from(buf))
            .build()?
        }
    };

    match t {
        DataType::Boolean => return Ok(Arc::new(BooleanArray::from(data))),
        DataType::UInt8 => return Ok(Arc::new(UInt8Array::from(data))),
        DataType::UInt16 => return Ok(Arc::new(UInt16Array::from(data))),
        DataType::UInt32 => return Ok(Arc::new(UInt32Array::from(data))),
        DataType::UInt64 => return Ok(Arc::new(UInt64Array::from(data))),
        DataType::Int8 => return Ok(Arc::new(Int8Array::from(data))),
        DataType::Int16 => return Ok(Arc::new(Int16Array::from(data))),
        DataType::Int32 => return Ok(Arc::new(Int32Array::from(data))),
        DataType::Int64 => return Ok(Arc::new(Int64Array::from(data))),
        DataType::Float32 => return Ok(Arc::new(Float32Array::from(data))),
        DataType::Float64 => return Ok(Arc::new(Float64Array::from(data))),
        DataType::Utf8 => return Ok(Arc::new(StringArray::from(data))),
        DataType::Timestamp(TimeUnit::Second, None) => {
            return Ok(Arc::new(TimestampSecondArray::from(data)))
        },
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            return Ok(Arc::new(TimestampMillisecondArray::from(data)))
        },
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            return Ok(Arc::new(TimestampMicrosecondArray::from(data)))
        },
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            return Ok(Arc::new(TimestampNanosecondArray::from(data)))
        },
        _ => Err(ZarrError::InvalidMetadata("Invalid zarr datatype".to_string()))
    }
}

fn decompress_blosc(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    output.copy_from_slice(unsafe { &blosc::decompress_bytes(chunk_data).unwrap() });
    Ok(())
}

fn decompress_zlib(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    let mut z = flate2::read::ZlibDecoder::new(chunk_data);
    z.read(output).unwrap();
    Ok(())
}

fn decompress_bz2(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    bzip2::Decompress::new(false)
        .decompress(chunk_data, output)
        .unwrap();
    Ok(())
}

fn decompress_lzma(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    let decomp_data = lzma::decompress(chunk_data).unwrap();
    output.copy_from_slice(&decomp_data[..]);
    Ok(())
}

fn decompress_array(
    raw_data: Vec<u8>, uncompressed_size: usize, compressor_params: Option<&CompressorType>,
) -> Vec<u8> {
    if let Some(comp) = compressor_params {
        let mut output: Vec<u8> = vec![0; uncompressed_size];
        match comp {
            CompressorType::Zlib => {
                decompress_zlib(&raw_data, &mut output).unwrap();
            }
            CompressorType::Bz2 => {
                decompress_bz2(&raw_data, &mut output).unwrap();
            }
            CompressorType::Lzma => {
                decompress_lzma(&raw_data, &mut output).unwrap();
            }
            CompressorType::Blosc => {
                decompress_blosc(&raw_data, &mut output).unwrap();
            }
        }
        return output;
    }
    else {
        return raw_data;
    }
}

fn get_2d_dim_order(order: &MatrixOrder) -> [usize; 2] {
    match order {
        MatrixOrder::RowMajor => [0, 1],
        MatrixOrder::ColumnMajor => [1, 0],
    }
}

fn get_3d_dim_order(order: &MatrixOrder) -> [usize; 3] {
    match order {
        MatrixOrder::RowMajor => [0, 1, 2],
        MatrixOrder::ColumnMajor => [2, 1, 0],
    }
}

fn process_edge_chunk(
    buf: &mut [u8],
    chunk_dims: &Vec<usize>,
    real_dims: &Vec<usize>,
    data_size: usize,
    order: &MatrixOrder,
) {
    let chunk_size = real_dims.iter().fold(1, |mult, x| mult * x);
    let mut indices_to_keep: Vec<usize> = vec![0; chunk_size];

    let n_dims = chunk_dims.len();
    indices_to_keep = match n_dims {
        1 => {(0..real_dims[0]).collect()},
        2 => {
            let [first_dim, second_dim] = get_2d_dim_order(order);
            (0..real_dims[first_dim])
                .cartesian_product(0..real_dims[second_dim])
                .map(|t| t.0 * chunk_dims[1] + t.1)
                .collect()
        },
        3 => {
            let [first_dim, second_dim, third_dim] = get_3d_dim_order(order);
            (0..real_dims[first_dim])
            .cartesian_product(0..real_dims[second_dim])
            .cartesian_product(0..real_dims[third_dim])
            .map(|t| {
                t.0 .0 * chunk_dims[1] * chunk_dims[2]
                    + t.0 .1 * chunk_dims[2]
                    + t.1
            })
            .collect()
        },
        _ => {panic!("Edge chunk with more than 3 domensions, 3 is the limit")}
    };

    let mut output_idx = 0;
    for data_idx in indices_to_keep {
        buf.copy_within(
            data_idx * data_size..(data_idx + 1) * data_size,
            output_idx * data_size
        );
        output_idx += 1;
    }
}

pub struct ZarrRecordBatchReader<T> 
where 
    T: Iterator<Item = ZarrResult<ZarrInMemoryChunk>>
{
    meta: ZarrStoreMetadata,
    zarr_store: T 
}

impl<T> ZarrRecordBatchReader<T>
where 
    T: Iterator<Item = ZarrResult<ZarrInMemoryChunk>>
{
    pub(crate) fn new(meta: ZarrStoreMetadata, zarr_store: T) -> Self {
        Self {meta, zarr_store}
    }

    pub(crate) fn unpack_chunk(&self, mut chunk: ZarrInMemoryChunk) -> ZarrResult<RecordBatch> {
        let mut arrs: Vec<ArrayRef> = Vec::with_capacity(self.meta.get_num_columns());
        let mut fields: Vec<FieldRef> = Vec::with_capacity(self.meta.get_num_columns());
        for col in self.meta.get_columns() {
            let data = chunk.get_array_data(col)?;
            let (arr, field) = self.unpack_array_chunk(
                col.to_string(), data, chunk.get_real_dims(), self.meta.get_chunk_dims()
            )?;
            arrs.push(arr);
            fields.push(field);
        }

        Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs)?)
    }

    fn unpack_array_chunk(
        &self,
        col_name: String,
        arr_chnk: ZarrInMemoryArray,
        real_dims: &Vec<usize>,
        chunk_dims: &Vec<usize>,
    ) -> ZarrResult<(ArrayRef, FieldRef)> {
            // get the metadata for the array
            let meta = self.meta.get_array_meta(&col_name)?;

            // get the field, data size and the data raw data from the array
            let (data_size, field) = build_field(meta.get_type(), col_name)?;
            let mut data = arr_chnk.take_data();

            // uncompress the data
            let chunk_size = chunk_dims.iter().fold(1, |mult, x| mult * x);
            data = decompress_array(data, chunk_size * data_size, meta.get_compressor().as_ref());

            // handle big endianness by converting to little endianness
            if meta.get_endianness() == &Endianness::Big {
                for idx in (0..chunk_size) {
                    data[idx*data_size..(idx+1)*data_size].reverse();
                }
            }

            // handle edge chunks
            if chunk_dims != real_dims {
                process_edge_chunk(&mut data, chunk_dims, real_dims, data_size, meta.get_order());
            }

            // create the array
            let real_size = real_dims.iter().fold(1, |mult, x| mult * x) * data_size;
            data.resize(real_size, 0);
            let arr = build_array(data, field.data_type(), data_size)?;

            Ok((arr, field))
    }
}

impl<T> Iterator for ZarrRecordBatchReader<T>
where
T: Iterator<Item = ZarrResult<ZarrInMemoryChunk>>
{
    type Item = ZarrResult<RecordBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        let next_batch = self.zarr_store.next();
        if next_batch.is_none() {
            return None
        }

        let next_batch = next_batch.unwrap();
        if let Err(err) = next_batch {
            return Some(Err(err));
        }

        let next_batch = next_batch.unwrap();
        return Some(self.unpack_chunk(next_batch));
    }
}

pub struct ZarrRecordBatchReaderBuilder<T: ZarrRead + Clone> 
{
    zarr_reader: T,
    projection: Option<ColumnProjection>,
}

impl<T: ZarrRead + Clone> ZarrRecordBatchReaderBuilder<T> {
    pub fn new(zarr_reader: T, projection: Option<ColumnProjection>,) -> Self {
        Self{zarr_reader, projection}
    }

    pub fn build(self) -> ZarrResult<ZarrRecordBatchReader<ZarrStore<T>>> {
        let meta = self.zarr_reader.get_zarr_metadata()?;
        let chunk_pos: Vec<Vec<usize>> = meta.get_chunk_positions();
        let zarr_store = ZarrStore::new(self.zarr_reader, chunk_pos, self.projection)?;
        
        Ok(ZarrRecordBatchReader::new(meta, zarr_store))
    }

    pub fn build_multiple(self, n_readers: usize) -> ZarrResult<Vec<ZarrRecordBatchReader<ZarrStore<T>>>> {
        let meta = self.zarr_reader.get_zarr_metadata()?;
        let chunk_pos: Vec<Vec<usize>> = meta.get_chunk_positions();
        let n_chunks = (chunk_pos.len() as f64 / n_readers as f64).ceil() as usize;
        let readers: Vec<ZarrRecordBatchReader<ZarrStore<T>>> = chunk_pos.chunks(n_chunks).map(
            |chnk_positions| {
                let store = ZarrStore::new(
                    self.zarr_reader.clone(), chnk_positions.to_vec(), self.projection.clone()
                ).unwrap();
                ZarrRecordBatchReader::new(meta.clone(), store)
            }
        ).collect();

        Ok(readers)
    }
}

#[cfg(test)]
mod zarr_reader_tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use itertools::enumerate;

    use super::*;
    use std::{path::PathBuf, collections::HashMap};

    fn get_test_data_path(zarr_store: String) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testing/data/zarr").join(zarr_store)
    }

    fn validate_names_and_types(targets: HashMap<String, DataType>, rec: &RecordBatch) {
        let mut target_cols: Vec<&String> = targets.keys().collect();
        let schema = rec.schema();
        let mut from_rec: Vec<&String> = schema.fields.iter().map(|f| f.name()).collect();

        from_rec.sort();
        target_cols.sort();
        assert_eq!(from_rec, target_cols);

        for field in schema.fields.iter() {
            assert_eq!(
                field.data_type(),
                targets.get(field.name()).unwrap()
            );
        }
    }

    #[test]
    fn compression_tests() {
        let p = get_test_data_path("compression_example.zarr".to_string());
        let reader = ZarrRecordBatchReaderBuilder::new(p, None).build().unwrap();
        let records: Vec<RecordBatch> = reader.map(|x| x.unwrap()).collect();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("uint_data".to_string(), DataType::UInt64),
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        // center chunk
        let rec = &records[4];
        validate_names_and_types(target_types, rec);
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name() == "uint_data" {
                assert_eq!(
                    rec.column(idx).as_primitive::<UInt64Type>().values(),
                    &[27, 28, 29, 35, 36, 37, 43, 44, 45]
                )
            }
        }

        println!("{:?}", records[4]);
    }
}