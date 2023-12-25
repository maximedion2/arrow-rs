use itertools::Itertools;
use crate::reader::metadata::ZarrStoreMetadata;
use std::collections::{HashMap, HashSet};
use std::fs::{read_to_string, read};
use std::path::PathBuf;
use crate::reader::errors::ZarrError;
use super::errors::ZarrResult;

/// An in-memory representation of the data contained in one chunk
/// of one zarr array. 
#[derive(Debug, Clone)]
pub struct ZarrInMemoryArray {
    data: Vec<u8>,
}

impl ZarrInMemoryArray {
    pub fn new(data: Vec<u8>) -> Self {
        Self {data}
    }

    pub(crate) fn take_data(self) -> Vec<u8> {
        self.data
    }
}

/// An in-memory representation of the data contained in one chunk
/// of one whole zarr store with one or more zarr arrays. 
#[derive(Debug)]
pub struct ZarrInMemoryChunk {
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

    pub(crate) fn add_array(
        &mut self,
        col_name: String,
        data: Vec<u8>,
    ) {
        self.data.insert(col_name, ZarrInMemoryArray::new(data));
    }

    pub(crate) fn get_cols_in_chunk(&self) -> Vec<String> {
        self.data.keys().map(|s| s.to_string()).collect_vec()
    }

    pub(crate) fn get_real_dims(&self) -> &Vec<usize> {
        &self.real_dims
    }

    pub(crate) fn take_array(&mut self, col: &str) -> ZarrResult<ZarrInMemoryArray> {
        self.data.remove(col).ok_or(ZarrError::MissingArray(col.to_string()))
    }

    pub(crate) fn copy_array(&self, col: &str) -> ZarrResult<ZarrInMemoryArray> {
        Ok(
            self.data.get(col).ok_or(ZarrError::MissingArray(col.to_string()))?.clone()
        )
    }
}

#[derive(Clone, PartialEq)]
pub(crate) enum ProjectionType {
    Select,
    Skip,
    Null, 
}

#[derive(Clone)]
pub struct ZarrProjection {
    projection_type: ProjectionType,
    col_names: Option<Vec<String>>,
}

impl ZarrProjection {
    pub fn all() -> Self {
        Self{projection_type: ProjectionType::Null, col_names: None}
    }

    pub fn skip(col_names: Vec<String>) -> Self {
        Self{projection_type: ProjectionType::Skip, col_names: Some(col_names)}
    }

    pub fn keep(col_names: Vec<String>) -> Self {
        Self { projection_type: ProjectionType::Select, col_names: Some(col_names)}
    }

    pub(crate) fn apply_selection(&self, all_cols: &Vec<String>) -> ZarrResult<Vec<String>> {
        match self.projection_type {
            ProjectionType::Null => {return Ok(all_cols.clone())},
            ProjectionType::Skip => {
                let col_names = self.col_names.as_ref().unwrap();
                return Ok(
                    all_cols.iter().filter(|x| !col_names.contains(x)).map(|x| x.to_string()).collect()
                )
            },
            ProjectionType::Select => {
                let col_names = self.col_names.as_ref().unwrap();
                for col in all_cols {
                    if !col_names.contains(&col) {
                        return Err(ZarrError::MissingArray("Column in selection missing columns to select from".to_string()))
                    }
                }
                return Ok(col_names.clone())
            }
        }
    }

    pub(crate) fn update(&mut self, other_proj: ZarrProjection) {
        if other_proj.projection_type == ProjectionType::Null {
            return
        }

        if self.projection_type == ProjectionType::Null {
            self.projection_type = other_proj.projection_type;
            self.col_names = other_proj.col_names;
            return
        }

        let col_names = self.col_names.as_mut().unwrap();
        if other_proj.projection_type == self.projection_type {
            let mut s: HashSet<String> = HashSet::from_iter(col_names.clone().into_iter());

            let other_cols = other_proj.col_names.unwrap();
            s.extend::<HashSet<String>>(HashSet::from_iter(other_cols.into_iter()));
            self.col_names = Some(s.into_iter().collect_vec());
        } else {
            for col in other_proj.col_names.as_ref().unwrap() {
                if let Some(index) = col_names.iter().position(|value| value == col) {
                    col_names.remove(index);
                }
            }
        }
    }

    pub(crate) fn into_updated(mut self, other_proj: ZarrProjection) -> Self {
        self.update(other_proj);
        self
    }
}

#[derive(Clone)]
pub struct ColumnProjection {
    skipping: bool,
    col_names: Vec<String>,
}

impl ColumnProjection {
    pub fn new(skipping: bool, col_names: Vec<String>) -> Self {
        Self {skipping, col_names}
    }

    pub(crate) fn get_cols_to_read(&self, all_cols: &Vec<String>) -> Vec<String> {
        if self.skipping {
            return all_cols.iter()
                           .filter(|x| !self.col_names
                           .contains(x))
                           .map(|x| x.to_string())
                           .collect();
        } else {
            return self.col_names.clone();
        }
    }

    pub(crate) fn contains(&self, to_check: &str) -> bool {
        self.col_names.contains(&to_check.to_string())
    }

    pub(crate) fn update(mut self, other_proj: ColumnProjection) -> ColumnProjection{
        if other_proj.skipping == self.skipping {
            let mut s: HashSet<String> = HashSet::from_iter(self.col_names.clone().into_iter());
            s.extend::<HashSet<String>>(HashSet::from_iter(other_proj.col_names.into_iter()));
            self.col_names = s.into_iter().collect_vec();
        } else {
            for col in other_proj.col_names {
                if let Some(index) = self.col_names.iter().position(|value| *value == col) {
                    self.col_names.remove(index);
                }
            }
        }

        self
    }

    pub(crate) fn to_skip_projection(mut self) -> ColumnProjection {
        self.skipping = true;
        self
    }
}

pub trait ZarrRead {
    fn get_zarr_metadata(&self) -> Result<ZarrStoreMetadata, ZarrError>;
    fn get_zarr_chunk(
        &self,
        position: &Vec<usize>,
        cols: &Vec<String>,
        real_dims: Vec<usize>,
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
        cols: &Vec<String>,
        real_dims: Vec<usize>,
    ) -> Result<ZarrInMemoryChunk, ZarrError> {
        let mut chunk = ZarrInMemoryChunk::new(real_dims);
        for var in cols {
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

#[cfg(test)]
mod zarr_read_tests {
    use super::*;
    use crate::reader::metadata::{ZarrDataType, MatrixOrder, Endianness, ZarrArrayMetadata};
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
            &pos, meta.get_columns(), meta.get_real_dims(&pos)
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
        let cols = col_proj.get_cols_to_read(meta.get_columns());
        let chunk = p.get_zarr_chunk(&pos, &cols, meta.get_real_dims(&pos)).unwrap();
        assert_eq!(chunk.data.keys().collect::<Vec<&String>>(), vec!["byte_data"]);

        // same as above, but specify columsn to keep instead of to skip
        let col_proj = ColumnProjection::new(false, vec!["float_data".to_string()]);
        let cols = col_proj.get_cols_to_read(meta.get_columns());
        let chunk = p.get_zarr_chunk(
            &pos, &cols, meta.get_real_dims(&pos)).unwrap();
        assert_eq!(chunk.data.keys().collect::<Vec<&String>>(), vec!["float_data"]);
    }
}