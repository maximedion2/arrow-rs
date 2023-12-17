use crate::reader::metadata::{ZarrStoreMetadata, ZarrArrayMetadata};
use std::collections::HashMap;
use std::fs::{read_to_string, read};
use std::path::PathBuf;

use crate::reader::errors::ZarrError;

use super::errors::ZarrResult;

#[derive(Debug)]
pub(crate) struct ZarrInMemoryArray {
    data: Vec<u8>,
}

impl ZarrInMemoryArray {
    pub(crate) fn new(data: Vec<u8>) -> Self {
        Self {data}
    }

    pub(crate) fn take_data(self) -> Vec<u8> {
        self.data
    }
}


#[derive(Debug)]
pub(crate) struct ZarrInMemoryChunk {
    data: HashMap<String, ZarrInMemoryArray>,
    real_dims: Vec<usize>,
}

impl ZarrInMemoryChunk {
    fn new(real_dims: Vec<usize>) -> Self {
        Self {
            data: HashMap::new(),
            real_dims: real_dims,
        }
    }

    fn add_array(
        &mut self,
        col_name: String,
        data: Vec<u8>,
    ) {
        self.data.insert(col_name, ZarrInMemoryArray::new(data));
    }

    pub(crate) fn get_real_dims(&self) -> &Vec<usize> {
        &self.real_dims
    }

    pub(crate) fn get_array_data(&mut self, col: &str) -> ZarrResult<ZarrInMemoryArray> {
        self.data.remove(col).ok_or(ZarrError::MissingArray(col.to_string()))
    }
}

#[derive(Clone)]
pub struct ColumnProjection {
    skipping: bool,
    col_names: Vec<String>,
}

impl ColumnProjection {
    pub(crate) fn new(skipping: bool, col_names: Vec<String>) -> Self {
        Self {skipping, col_names}
    }
}

pub trait ZarrRead {
    fn get_zarr_metadata(&self) -> Result<ZarrStoreMetadata, ZarrError>;
    fn get_zarr_chunk(
        &self,
        position: &Vec<usize>,
        cols: Vec<String>,
        real_dims: Vec<usize>,
        col_proj: Option<&ColumnProjection>
    ) -> Result<ZarrInMemoryChunk, ZarrError>;
}

impl ZarrRead for PathBuf {
    fn get_zarr_metadata(&self) -> Result<ZarrStoreMetadata, ZarrError> {
        let mut meta = ZarrStoreMetadata::new();
        let dir = self.read_dir().unwrap();

        for dir_entry in dir {
            let dir_entry = dir_entry?;
            let p = dir_entry.path().join(".zarray");
            if p.exists() {
                let meta_str = read_to_string(p)?;
                meta.add_column(
                    dir_entry.path().file_name().unwrap().to_str().unwrap().to_string(),
                    &meta_str
                )?;
            }
        }

        if meta.get_num_columns() == 0 {
            return Err(ZarrError::InvalidMetadata("Could not find valid metadata in zarr store".to_string()))
        }
        Ok(meta)
    }

    fn get_zarr_chunk(
        &self,
        position: &Vec<usize>,
        mut cols: Vec<String>,
        real_dims: Vec<usize>,
        col_proj: Option<&ColumnProjection>,
    ) -> Result<ZarrInMemoryChunk, ZarrError> {
        if let Some(proj) = col_proj {
            if proj.skipping {
                for col in &proj.col_names {
                    cols.remove(
                        cols.iter()
                                   .position(|x| x == col)
                                   .ok_or(ZarrError::InvalidMetadata("Columnn projection not found".to_string()))?
                    );
                }
            } else {
                cols = proj.col_names.clone();
            }
        }

        let mut chunk = ZarrInMemoryChunk::new(real_dims);
        for var in &cols {
            let s: Vec<String> = position.into_iter().map(|i| i.to_string()).collect();
            let s = s.join(".");
            let path = self.join(var).join(s);

            if !path.exists() {
                return Err(ZarrError::MissingChunk(position.clone()))
            }
            let data = read(path)?;
            chunk.add_array(var.to_string(), data);
        }

        Ok(chunk)
    }
}

pub struct ZarrStore<T: ZarrRead> {
    meta: ZarrStoreMetadata,
    chunk_positions: Vec<String>,
    zarr_reader: T,
    curr_chunk: usize,
}

impl<T: ZarrRead> ZarrStore<T> {
    pub fn new(meta: ZarrStoreMetadata, zarr_reader: T, chunk_positions: Vec<String>) -> Self {
        Self {
            meta,
            chunk_positions,
            zarr_reader,
            curr_chunk: 0,
        }
    }
}


#[cfg(test)]
mod zarr_read_tests {
    use super::*;
    use crate::reader::metadata::{ZarrDataType, MatrixOrder, Endianness};
    use std::path::PathBuf;
    use std::collections::HashSet;

    fn get_test_data_path(zarr_store: String) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testing/data/zarr").join(zarr_store)
    }

    #[test]
    fn read_metadata() {
        let p = get_test_data_path("raw_bytes_example.zarr".to_string());
        let meta = p.get_zarr_metadata().unwrap();

        assert_eq!(meta.get_columns(), &vec!["byte_data", "float_data"]);
        assert_eq!(
            meta.get_array_meta("byte_data").unwrap(),
            &ZarrArrayMetadata::new (
                2,
                ZarrDataType::UInt(1),
                None,
                MatrixOrder::RowMajor,
                Endianness::Little,
            )
        );
        assert_eq!(
            meta.get_array_meta("float_data").unwrap(),
            &ZarrArrayMetadata::new (
                2,
                ZarrDataType::Float(8),
                None,
                MatrixOrder::RowMajor,
                Endianness::Little,
            )
        );
    }

    #[test]
    fn read_raw_chunks() {
        let p = get_test_data_path("raw_bytes_example.zarr".to_string());
        let meta = p.get_zarr_metadata().unwrap();

        // test read from an array where the data is just raw bytes
        let pos = vec![1, 2];
        let chunk = p.get_zarr_chunk(
            &pos, meta.get_columns().clone(), meta.get_real_dims(&pos), None
        ).unwrap();
        assert_eq!(
            chunk.data.keys().collect::<HashSet<&String>>(),
            HashSet::from([&"float_data".to_string(), &"byte_data".to_string()])
        );
        assert_eq!(
            chunk.data.get("byte_data").unwrap().data,
            vec![33, 34, 35, 42, 43, 44, 51, 52, 53],
        );

        // test selecting only one of the 2 columns
        let col_proj = ColumnProjection::new(true, vec!["float_data".to_string()]);
        let chunk = p.get_zarr_chunk(
            &pos, meta.get_columns().clone(), meta.get_real_dims(&pos), Some(&col_proj)
        ).unwrap();
        assert_eq!(chunk.data.keys().collect::<Vec<&String>>(), vec!["byte_data"]);

        // same as above, but specify columsn to keep instead of to skip
        let col_proj = ColumnProjection::new(false, vec!["float_data".to_string()]);
        let chunk = p.get_zarr_chunk(
            &pos, meta.get_columns().clone(), meta.get_real_dims(&pos), Some(&col_proj)
        ).unwrap();
        assert_eq!(chunk.data.keys().collect::<Vec<&String>>(), vec!["float_data"]);
    }
}