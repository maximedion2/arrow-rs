use bytemuck::{cast_slice, Pod};
use bzip2::Decompress;
use flate2::read::ZlibDecoder;
use itertools::Itertools;
use serde::Deserialize;
use std::fs;
use std::io::Read;
use std::path::Path;

//**********************
// custom zarr errors
//**********************
#[derive(Debug)]
pub enum ZarrError {
    InvalidCompressorOptions(String),
    InvalidArrayParams(String),
    ChunkError(String),
    InvalidOperation(String),
}

//**********************
// compressor parameters and utils
//**********************
#[derive(PartialEq, Debug, Clone)]
enum ZarrCompressor {
    Blosc,
    Zlib,
    Bz2,
    Lzma,
}

#[derive(Deserialize)]
struct RawBloscParams {
    id: String,
    clevel: u8,
    cname: String,
    shuffle: u8,
}

#[derive(Deserialize)]
struct RawBloscOuterParams {
    compressor: RawBloscParams,
}

#[derive(Deserialize)]
struct RawLzmaParams {
    id: String,
    preset: u8,
}

#[derive(Deserialize)]
struct RawLzmaOuterParams {
    compressor: RawLzmaParams,
}

#[derive(Deserialize)]
struct RawOtherParams {
    id: String,
    level: u8,
}

#[derive(Deserialize)]
struct RawOtherOuterParams {
    compressor: RawOtherParams,
}

#[derive(Debug, Clone)]
pub(crate) struct CompressorParams {
    compressor: ZarrCompressor,
    _level: u8,
    _cname: Option<String>,
    _shuffle: Option<u8>,
}

impl CompressorParams {
    pub(crate) fn new(path_to_metadata: &str) -> Result<Option<Self>, ZarrError> {
        let metadata = fs::read_to_string(&path_to_metadata)
            .expect(&format!("Unable to read .zarray at {}", path_to_metadata));

        // check for Blosc params
        let j: Result<RawBloscOuterParams, serde_json::Error> =
            serde_json::from_str(&metadata);
        if let Ok(raw_params) = j {
            if raw_params.compressor.id != "blosc" {
                return Err(ZarrError::InvalidCompressorOptions(
                    "expect compressor id to be blosc given parameters".to_string(),
                ));
            }
            return Ok(Some(CompressorParams {
                compressor: ZarrCompressor::Blosc,
                _level: raw_params.compressor.clevel,
                _cname: Some(raw_params.compressor.cname),
                _shuffle: Some(raw_params.compressor.shuffle),
            }));
        }

        // check for Lzma params
        let j: Result<RawLzmaOuterParams, serde_json::Error> =
            serde_json::from_str(&metadata);
        if let Ok(raw_params) = j {
            if raw_params.compressor.id != "lzma" {
                return Err(ZarrError::InvalidCompressorOptions(
                    "expect compressor id to be lzma given parameters".to_string(),
                ));
            }
            return Ok(Some(CompressorParams {
                compressor: ZarrCompressor::Lzma,
                _level: raw_params.compressor.preset,
                _cname: None,
                _shuffle: None,
            }));
        }

        // check for Zlib or Bz2 params
        let j: Result<RawOtherOuterParams, serde_json::Error> =
            serde_json::from_str(&metadata);
        if let Ok(raw_params) = j {
            if raw_params.compressor.id != "zlib" && raw_params.compressor.id != "bz2" {
                return Err(ZarrError::InvalidCompressorOptions(
                    "expect compressor id to be zlib or bz2 given parameters".to_string(),
                ));
            }
            let comp = if raw_params.compressor.id == "zlib" {
                ZarrCompressor::Zlib
            } else {
                ZarrCompressor::Bz2
            };
            return Ok(Some(CompressorParams {
                compressor: comp,
                _level: raw_params.compressor.level,
                _cname: None,
                _shuffle: None,
            }));
        }

        // check if the compressor is null
        let j: serde_json::Value = serde_json::from_str(&metadata).unwrap();
        if j["compressor"].is_null() {
            return Ok(None);
        }

        Err(ZarrError::InvalidCompressorOptions(
            "could not parse compressor parameters".to_string(),
        ))
    }
}

fn decompress_blosc(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    output.copy_from_slice(unsafe { &blosc::decompress_bytes(chunk_data).unwrap() });
    Ok(())
}

fn decompress_zlib(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    let mut z = ZlibDecoder::new(chunk_data);
    z.read(output).unwrap();
    Ok(())
}

fn decompress_bz2(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    Decompress::new(false)
        .decompress(chunk_data, output)
        .unwrap();
    Ok(())
}

fn decompress_lzma(chunk_data: &[u8], output: &mut [u8]) -> Result<(), ZarrError> {
    let decomp_data = lzma::decompress(chunk_data).unwrap();
    output.copy_from_slice(&decomp_data[..]);
    Ok(())
}

fn decompress_chunk(
    chunk_data: &[u8],
    output: &mut [u8],
    compressor_params: Option<&CompressorParams>,
) {
    if let Some(comp) = compressor_params {
        match comp.compressor {
            ZarrCompressor::Zlib => {
                decompress_zlib(chunk_data, output).unwrap();
            }
            ZarrCompressor::Bz2 => {
                decompress_bz2(chunk_data, output).unwrap();
            }
            ZarrCompressor::Lzma => {
                decompress_lzma(chunk_data, output).unwrap();
            }
            ZarrCompressor::Blosc => {
                decompress_blosc(chunk_data, output).unwrap();
            }
        }
        return;
    }

    output.copy_from_slice(chunk_data);
}

//**********************
// array parameters
//**********************
#[derive(Debug, Clone)]
enum Endianness {
    Little,
    Big,
}

#[derive(Debug, Clone)]
enum MatrixOrder {
    RowMajor,
    ColumnMajor,
}

#[derive(Deserialize, Debug)]
struct RawArrayParams {
    zarr_format: u8,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    dtype: String,
    order: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ArrayParams {
    n_dims: u8,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    order: MatrixOrder,
    _endianness: Endianness,
}

impl ArrayParams {
    pub(crate) fn new(path_to_metadata: &str) -> Result<Self, ZarrError> {
        let metadata =
            fs::read_to_string(&path_to_metadata).expect("Unable to read .zarray");
        let j: Result<RawArrayParams, serde_json::Error> =
            serde_json::from_str(&metadata);
        if let Ok(raw_params) = j {
            if raw_params.zarr_format != 2 {
                return Err(ZarrError::InvalidArrayParams(
                    "only zarr format 2 is currently supported".to_string(),
                ));
            }
            if raw_params.chunks.len() != raw_params.shape.len() {
                return Err(ZarrError::InvalidArrayParams(format!(
                    "dimenstion mismatch between shape {:?} and chunks {:?}",
                    raw_params.shape, raw_params.chunks
                )));
            }
            if raw_params.shape.len() > 3 {
                return Err(ZarrError::InvalidArrayParams(
                    "only up to 3 dimension chunks are supported".to_string(),
                ));
            }

            let order = match raw_params.order.as_str() {
                "C" => Some(MatrixOrder::RowMajor),
                "F" => Some(MatrixOrder::ColumnMajor),
                _ => None,
            };
            if order.is_none() {
                return Err(ZarrError::InvalidArrayParams(format!(
                    "invalid matrix order {} in array params",
                    raw_params.order.as_str()
                )));
            }

            let endianness = match raw_params.dtype.chars().next().unwrap() {
                '<' | '|' => Some(Endianness::Little),
                '>' => Some(Endianness::Big),
                _ => None,
            };
            if endianness.is_none() {
                return Err(ZarrError::InvalidArrayParams(
                    "invalid endianness in dtype in array params".to_string(),
                ));
            }
            return Ok(Self {
                n_dims: raw_params.shape.len() as u8,
                shape: raw_params.shape,
                chunks: raw_params.chunks,
                order: order.unwrap(),
                _endianness: endianness.unwrap(),
            });
        }

        Err(ZarrError::InvalidArrayParams(format!(
            "could not parse array params at {}",
            path_to_metadata
        )))
    }

    pub(crate) fn get_chunks(&self) -> &Vec<usize> {
        &self.chunks
    }

    pub(crate) fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }
}

//**********************
// a trait for any struct that retrieves data from a zarr chunk
//**********************
pub trait ZarrRead {
    fn exists(&self) -> bool;
    fn read_into_buf(&self, output_buf: &mut [u8]) -> Result<(), ZarrError>;
    fn get_buf(&self) -> Result<Vec<u8>, ZarrError>;
    fn get_chunk_size(&self) -> usize;
    fn get_chunk_dims(&self) -> Vec<usize>;
    fn check_if_edge_chunk(&self) -> bool;
}

//**********************
// a interface for a local zarr chunk file
//**********************
pub struct LocalZarrFile {
    path: String,
    chunk_dims: Vec<usize>,
    is_edge_chunk: bool,
}

impl LocalZarrFile {
    pub fn new(
        path: String,
        position: Vec<usize>,
        shape: &Vec<usize>,
        chunks: &Vec<usize>,
    ) -> Self {
        let mut is_edge_chunk = false;
        let n_chunks: Vec<usize> = chunks
            .iter()
            .zip(shape)
            .map(|(&chnk, &shp)| (shp as f64 / chnk as f64).ceil() as usize)
            .collect();
        let real_chunks_dims: Vec<usize> = chunks
            .iter()
            .zip(shape)
            .zip(n_chunks)
            .zip(position.iter())
            .map(|(((chnk, shp), n_chnk), pos)| {
                if *pos == n_chnk - 1 {
                    is_edge_chunk = true;
                    shp - pos * chnk
                } else {
                    *chnk
                }
            })
            .collect();
        LocalZarrFile {
            path: path,
            chunk_dims: real_chunks_dims,
            is_edge_chunk: is_edge_chunk,
        }
    }
}

impl ZarrRead for LocalZarrFile {
    fn exists(&self) -> bool {
        Path::new(&self.path).exists()
    }

    fn read_into_buf(&self, output_buf: &mut [u8]) -> Result<(), ZarrError> {
        if !self.exists() {
            return Err(ZarrError::ChunkError(format!(
                "could not find chunk file {}",
                self.path
            )));
        }
        let mut f = fs::File::open(&self.path).unwrap();
        f.read(&mut output_buf[..]).unwrap();
        Ok(())
    }

    fn get_buf(&self) -> Result<Vec<u8>, ZarrError> {
        if !self.exists() {
            return Err(ZarrError::ChunkError(format!(
                "could not find chunk file {}",
                self.path
            )));
        }
        let buf = fs::read(&self.path).unwrap();
        Ok(buf.to_vec())
    }

    fn get_chunk_size(&self) -> usize {
        self.chunk_dims.iter().fold(1, |mult, x| mult * x)
    }

    fn check_if_edge_chunk(&self) -> bool {
        self.is_edge_chunk
    }

    fn get_chunk_dims(&self) -> Vec<usize> {
        self.chunk_dims.to_vec()
    }
}

//**********************
// the zarr chunk reader
//**********************
pub struct ZarrChunkReader<'a, T: Copy, Z: ZarrRead> {
    chunk_file: &'a Z,
    params: &'a ArrayParams,
    compressor_params: Option<&'a CompressorParams>,
    fill_value: T,
    full_chunk_size: usize,
}

impl<'a, T: Copy, Z: ZarrRead> ZarrChunkReader<'a, T, Z> {
    pub(crate) fn new(
        chunk_file: &'a Z,
        params: &'a ArrayParams,
        compressor_params: Option<&'a CompressorParams>,
        fill_value: T,
    ) -> Self {
        let full_chunk_size: usize =
            params.chunks.iter().fold(1, |mult, &x| mult * x as usize);
        ZarrChunkReader {
            chunk_file: chunk_file,
            params: params,
            compressor_params: compressor_params,
            fill_value: fill_value,
            full_chunk_size: full_chunk_size,
        }
    }

    fn get_2d_dim_order(&self) -> [usize; 2] {
        match self.params.order {
            MatrixOrder::RowMajor => [0, 1],
            MatrixOrder::ColumnMajor => [1, 0],
        }
    }

    fn get_3d_dim_order(&self) -> [usize; 3] {
        match self.params.order {
            MatrixOrder::RowMajor => [0, 1, 2],
            MatrixOrder::ColumnMajor => [2, 1, 0],
        }
    }

    fn read_edge_chunk(&self, buf: &[u8], output_buf: &mut [u8]) {
        let type_size = std::mem::size_of::<T>();
        let mut indices_to_keep: Vec<usize> = vec![0; self.chunk_file.get_chunk_size()];
        let real_chunk_dims = self.chunk_file.get_chunk_dims();
        if self.params.n_dims == 1 {
            indices_to_keep = (0..real_chunk_dims[0]).collect();
        } else if self.params.n_dims == 2 {
            let [first_dim, second_dim] = self.get_2d_dim_order();
            indices_to_keep = (0..real_chunk_dims[first_dim])
                .cartesian_product(0..real_chunk_dims[second_dim])
                .map(|t| t.0 * self.params.chunks[1] + t.1)
                .collect();
        } else if self.params.n_dims == 3 {
            let [first_dim, second_dim, third_dim] = self.get_3d_dim_order();
            indices_to_keep = (0..real_chunk_dims[first_dim])
                .cartesian_product(0..real_chunk_dims[second_dim])
                .cartesian_product(0..real_chunk_dims[third_dim])
                .map(|t| {
                    t.0 .0 * self.params.chunks[1] * self.params.chunks[2]
                        + t.0 .1 * self.params.chunks[2]
                        + t.1
                })
                .collect();
        }

        let mut output_idx = 0;
        for data_idx in indices_to_keep {
            output_buf[output_idx * type_size..(output_idx + 1) * type_size]
                .copy_from_slice(&buf[data_idx * type_size..(data_idx + 1) * type_size]);
            output_idx += 1;
        }
    }
}

//**********************
// implement the Read trait for the chunk reader
//**********************
impl<'a, T: Copy + Pod, Z: ZarrRead> Read for ZarrChunkReader<'a, T, Z> {
    fn read(&mut self, output_buf: &mut [u8]) -> Result<usize, std::io::Error> {
        // determine the amount of bytes to write
        let write_size = self.chunk_file.get_chunk_size() * std::mem::size_of::<T>();

        // handle the (simpler) case where it's not an edge chunk
        if !self.chunk_file.check_if_edge_chunk() {
            if self.chunk_file.exists() {
                if let Some(_comp) = self.compressor_params {
                    let buf = self.chunk_file.get_buf().unwrap();
                    decompress_chunk(
                        &buf[..],
                        &mut output_buf[..write_size],
                        self.compressor_params,
                    );
                } else {
                    self.chunk_file.read_into_buf(output_buf).unwrap();
                }
            } else {
                let typed_data = vec![self.fill_value; self.full_chunk_size];
                output_buf[..write_size].copy_from_slice(&cast_slice(&typed_data));
            }
            return Ok(write_size);
        }

        // handle the edge chunk case
        let mut data: Vec<u8>;
        if self.chunk_file.exists() {
            if let Some(_comp) = self.compressor_params {
                let buf = self.chunk_file.get_buf().unwrap();
                data = vec![0u8; self.full_chunk_size * std::mem::size_of::<T>()];
                decompress_chunk(&buf[..], &mut data[..], self.compressor_params);
            } else {
                data = self.chunk_file.get_buf().unwrap();
            }
        } else {
            let typed_data = vec![self.fill_value; self.full_chunk_size];
            data = vec![0u8; self.full_chunk_size * std::mem::size_of::<T>()];
            data.copy_from_slice(&cast_slice(&typed_data));
        }

        self.read_edge_chunk(&data[..], output_buf);
        Ok(write_size)
    }
}

#[cfg(test)]
mod zarr_chunk_tests {
    use super::*;

    fn read_chunk_and_assert<T: Copy + std::fmt::Debug + Pod + PartialEq>(
        store_name: &str,
        expected_n_bytes: usize,
        expected_res: &[T],
        pos: Vec<usize>,
        chnk_str: &str,
        fill_value: T,
    ) {
        // small hack on the fill value here, just manually passed in because
        // it's simpler for testing purposes, and that's basically how it will
        // be handled in the real code as well, so not even really a hack.

        let zarray_path = format!(
            "/home/maxime/Documents/repos/arrow-rs/testing/data/zarr/{}/.zarray",
            store_name
        );
        let chunk_path = format!(
            "/home/maxime/Documents/repos/arrow-rs/testing/data/zarr/{}/{}",
            store_name, chnk_str
        );

        let arr_params: ArrayParams = ArrayParams::new(&zarray_path).unwrap();
        let comp_params = CompressorParams::new(&zarray_path).unwrap();

        let chunk_file =
            LocalZarrFile::new(chunk_path, pos, &arr_params.shape, &arr_params.chunks);
        let mut chunk: ZarrChunkReader<T, LocalZarrFile> = ZarrChunkReader::new(
            &chunk_file,
            &arr_params,
            comp_params.as_ref(),
            fill_value,
        );

        let mut outupt = vec![0u8; expected_n_bytes];
        let bytes = chunk.read(&mut outupt).unwrap();
        let typed_data: &[T] = cast_slice(&outupt);

        assert_eq!(bytes, expected_n_bytes);
        assert_eq!(typed_data, expected_res);
    }

    #[test]
    fn data_with_no_compression() {
        let store_name = "example1.zarr";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }

    #[test]
    fn data_with_zlib_compression() {
        // level = 1
        let store_name = "example2.zarr/zlib_level1";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );

        // level = 2
        let store_name = "example2.zarr/zlib_level2";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }

    #[test]
    fn data_with_bz2_compression() {
        // level = 1
        let store_name = "example2.zarr/bz2_level1";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );

        // level = 2
        let store_name = "example2.zarr/bz2_level2";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }

    #[test]
    fn data_with_lzma_compression() {
        // preset = 0
        let store_name = "example2.zarr/lzma_preset0";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );

        // preset = 1
        let store_name = "example2.zarr/lzma_preset1";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }

    #[test]
    fn data_with_zlib_through_blosc_compression() {
        // level = 1, shuffle = 0
        let store_name = "example2.zarr/zlib_w_blosc_shuffle0_level1";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );

        // level = 2, shuffle = 1
        let store_name = "example2.zarr/zlib_w_blosc_shuffle1_level2";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }

    #[test]
    fn data_with_lz4_through_blosc_compression() {
        // level = 1, shuffle = 0
        let store_name = "example2.zarr/lz4_w_blosc_shuffle0_level1";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );

        // level = 2, shuffle = 1
        let store_name = "example2.zarr/lz4_w_blosc_shuffle1_level2";
        read_chunk_and_assert::<i16>(
            &store_name,
            18,
            &[1033, 1034, 1035, 1042, 1043, 1044, 1051, 1052, 1053],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }

    #[test]
    fn data_with_edge_chunks() {
        let store_name = "example3.zarr";
        read_chunk_and_assert::<i16>(
            &store_name,
            12,
            &[1030, 1031, 1038, 1039, 1046, 1047],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }

    #[test]
    fn column_major_data_with_edge_chunks_and() {
        let store_name = "example5.zarr";
        read_chunk_and_assert::<i16>(
            &store_name,
            12,
            &[1030, 1038, 1046, 1031, 1039, 1047],
            vec![1, 2],
            "1.2",
            0 as i16,
        );
    }
}
