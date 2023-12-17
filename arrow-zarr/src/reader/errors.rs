use std::error::Error;
use std::io;
use std::result::Result;
use arrow_schema::ArrowError;

#[derive(Debug)]
pub enum ZarrError {
    InvalidMetadata(String),
    MissingChunk(Vec<usize>),
    MissingArray(String),
    Io(Box<dyn Error + Send + Sync>),
    Arrow(Box<dyn Error + Send + Sync>)
}

impl std::fmt::Display for ZarrError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self {
            ZarrError::InvalidMetadata(msg) => write!(fmt, "Invalid zarr metadata: {msg}"),
            ZarrError::MissingChunk(pos) => {
                let s: Vec<String> = pos.into_iter().map(|i| i.to_string()).collect();
                let s = s.join(".");
                write!(fmt, "Missing zarr chunk file: {s}")
            },
            ZarrError::MissingArray(arr_name) => write!(fmt, "Missing zarr chunk file: {arr_name}"),
            ZarrError::Io(e) => write!(fmt, "IO error: {e}"),
            ZarrError::Arrow(e) => write!(fmt, "Arrow error: {e}"),
        }
    }
}

impl From<io::Error> for ZarrError {
    fn from(e: io::Error) -> ZarrError {
        ZarrError::Io(Box::new(e))
    }
}

impl From<ArrowError> for ZarrError {
    fn from(e: ArrowError) -> ZarrError {
        ZarrError::Arrow(Box::new(e))
    }
}

pub type ZarrResult<T, E = ZarrError> = Result<T, E>;