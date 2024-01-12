use crate::async_reader::zarr_read_async::ZarrReadAsync;
use crate::reader::zarr_read::{ZarrProjection, ZarrInMemoryChunk};
use crate::reader::errors::{ZarrResult, ZarrError};
use crate::reader::metadata::ZarrStoreMetadata;
use crate::reader::filters::ZarrChunkFilter;
use crate::reader::{unwrap_or_return, ZarrRecordBatchReader, ZarrIterator};
use std::pin::Pin;
use std::task::{Context, Poll};
use futures::{ready, FutureExt};
use futures::stream::Stream;
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use arrow_array::{RecordBatch, BooleanArray};

pub mod zarr_read_async;

/// A zarr store that holds an async reader for all the zarr data.
pub struct ZarrStoreAsync<T: ZarrReadAsync> {
    meta: ZarrStoreMetadata,
    chunk_positions: Vec<Vec<usize>>,
    zarr_reader: T,
    projection: ZarrProjection,
    curr_chunk: usize,
}

impl<T: ZarrReadAsync> ZarrStoreAsync<T> {
    async fn new(
        zarr_reader: T,
        chunk_positions: Vec<Vec<usize>>,
        projection: ZarrProjection,
    ) -> ZarrResult<Self> {
        let meta = zarr_reader.get_zarr_metadata().await?;
        Ok(Self {
            meta: meta,
            chunk_positions,
            zarr_reader,
            projection,
            curr_chunk: 0,
        })
    }
}

#[async_trait]
pub trait ZarrStream {
    async fn poll_next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>>;
    fn skip_next_chunk(&mut self);
}

#[async_trait]
impl<T> ZarrStream for ZarrStoreAsync<T> 
where
    T: ZarrReadAsync + Unpin + Send + 'static,
{
    async fn poll_next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>> {
        if self.curr_chunk == self.chunk_positions.len() {
            return None;
        }

        let pos = &self.chunk_positions[self.curr_chunk];
        let cols = self.projection.apply_selection(self.meta.get_columns());
        let cols = unwrap_or_return!(cols);

        let chnk = self.zarr_reader.get_zarr_chunk(
            pos, &cols, self.meta.get_real_dims(pos),
        ).await;

        self.curr_chunk += 1;
        Some(chnk)
    }

    fn skip_next_chunk(&mut self) {
        if self.curr_chunk < self.chunk_positions.len() {
            self.curr_chunk += 1;
        }
    }
}

// a simple struct to expose the zarr iterator trait for a single, 
// preprocessed in memory chunk.
struct ZarrInMemoryChunkContainer {
    data: ZarrInMemoryChunk,
    done: bool
}

impl ZarrInMemoryChunkContainer {
    fn new(data: ZarrInMemoryChunk) -> Self {
        Self{data, done: false}
    }
}

impl ZarrIterator for ZarrInMemoryChunkContainer {
    fn next_chunk(&mut self) -> Option<ZarrResult<ZarrInMemoryChunk>> {
        if self.done {
            return None;
        }
        self.done = true;
        Some(Ok(std::mem::take(&mut self.data)))
    }

    fn skip_chunk(&mut self) {
        self.done = true;
    }
}

// struct to bundle the store and the chunk data it returns together
// in a future so that that future's lifetime is static.
struct ZarrStoreWrapper<T: ZarrStream> {
    store: T
}

impl<T: ZarrStream> ZarrStoreWrapper<T> {
    fn new(store: T) -> Self {
        Self{store}
    }

    async fn get_next(mut self) -> (Self, Option<ZarrResult<ZarrInMemoryChunk>>) {
        let next = self.store.poll_next_chunk().await;
        return (self, next);
    }
}
type StoreReadResults<T> = (ZarrStoreWrapper<T>, Option<ZarrResult<ZarrInMemoryChunk>>);

enum ZarrStreamState<T: ZarrStream> {
    Init,
    ReadingPredicateData(BoxFuture<'static, StoreReadResults<T>>),
    ProcessingPredicate(ZarrRecordBatchReader<ZarrInMemoryChunkContainer>),
    Reading(BoxFuture<'static, StoreReadResults<T>>),
    Decoding(ZarrRecordBatchReader<ZarrInMemoryChunkContainer>),
    Error,
}

pub struct ZarrRecordBatchStream<T: ZarrStream> 
{
    meta: ZarrStoreMetadata,
    filter: Option<ZarrChunkFilter>,
    state: ZarrStreamState<T>,
    mask: Option<BooleanArray>,

    // an option so that we can "take" the wrapper and bundle it
    // in a future when polling the stream.
    store_wrapper: Option<ZarrStoreWrapper<T>>,
    
    // this one is an option because it may or may not be present, not
    // just so that we can take it later (but it's useful for that too)
    predicate_store_wrapper: Option<ZarrStoreWrapper<T>>,
}

impl<T: ZarrStream> ZarrRecordBatchStream<T> {
    fn new(
        meta: ZarrStoreMetadata,
        zarr_store: T,
        filter: Option<ZarrChunkFilter>,
        mut predicate_store: Option<T>
    ) -> Self {
        let mut predicate_store_wrapper = None;
        if predicate_store.is_some() {
            predicate_store_wrapper = Some(ZarrStoreWrapper::new(predicate_store.take().unwrap()));
        }
        Self {
            meta,
            filter,
            predicate_store_wrapper,
            store_wrapper: Some(ZarrStoreWrapper::new(zarr_store)),
            state: ZarrStreamState::Init,
            mask: None,
        }
    }
}

const LOST_STORE_ERR: &str = "unexpectedly lost store wrapper in zarr record batch stream";
impl<T> Stream for ZarrRecordBatchStream<T> 
where
    T: ZarrStream + Unpin + Send + 'static,
{
    type Item = ZarrResult<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                ZarrStreamState::Init => {
                    if self.predicate_store_wrapper.is_none() {
                        let wrapper = self.store_wrapper.take().expect(LOST_STORE_ERR);
                        let fut = wrapper.get_next().boxed();
                        self.state = ZarrStreamState::Reading(fut);

                    } else {
                        let wrapper = self.predicate_store_wrapper.take().unwrap();
                        let fut = wrapper.get_next().boxed();
                        self.state = ZarrStreamState::ReadingPredicateData(fut);
                    }
                },
                ZarrStreamState::ReadingPredicateData(f) => {
                    let (wrapper, chunk) = ready!(f.poll_unpin(cx));
                    self.predicate_store_wrapper = Some(wrapper);

                    // if the predicate store returns none, it's the end and it's
                    // time to return
                    if chunk.is_none() {
                        self.state = ZarrStreamState::Init;
                        return Poll::Ready(None);
                    }

                    let chunk = chunk.unwrap();
                    if let Err(e) = chunk {
                        self.state = ZarrStreamState::Error;
                        return Poll::Ready(Some(Err(e)));
                    } 

                    let chunk = chunk.unwrap();
                    let container = ZarrInMemoryChunkContainer::new(chunk);

                    if self.filter.is_none() {
                        self.state = ZarrStreamState::Error;
                        return Poll::Ready(
                            Some(Err(ZarrError::InvalidMetadata(
                                "predicate store provided with no filter in zarr record batch stream".to_string()
                            )))
                        );
                    }
                    let zarr_reader = ZarrRecordBatchReader::new(
                        self.meta.clone(), None, self.filter.as_ref().cloned(), Some(container)
                    );
                    self.state = ZarrStreamState::ProcessingPredicate(zarr_reader);
                },
                ZarrStreamState::ProcessingPredicate(reader) => {
                    // this call should always return something, we should never get a None because
                    // if we're here it means we provided a filter and some predicate data to evaluate.
                    let mask = reader.next().expect("could not get mask in zarr record batch stream");
                    if let Err(e) = mask {
                        self.state = ZarrStreamState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }

                    // here we know that mask will have a single boolean array column because of the
                    // way the reader was created in the previous state.
                    let mask = mask.unwrap()
                                   .column(0)
                                   .as_any()
                                   .downcast_ref::<BooleanArray>()
                                   .expect("could not cast mask to boolean array in zarr record batch stream")
                                   .clone();
                    if mask.true_count() == 0 {
                        self.store_wrapper.as_mut()
                                          .expect(LOST_STORE_ERR)
                                          .store
                                          .skip_next_chunk();
                        self.state = ZarrStreamState::Init;
                    } else {
                        self.mask = Some(mask);
                        let wrapper = self.store_wrapper.take().expect(LOST_STORE_ERR);
                        let fut = wrapper.get_next().boxed();
                        self.state = ZarrStreamState::Reading(fut);
                    }

                },
                ZarrStreamState::Reading(f) => {
                    let (wrapper, chunk) = ready!(f.poll_unpin(cx));
                    self.store_wrapper = Some(wrapper);

                    // if store returns none, it's the end and it's time to return
                    if chunk.is_none() {
                        self.state = ZarrStreamState::Init;
                        return Poll::Ready(None);
                    }

                    let chunk = chunk.unwrap();
                    if let Err(e) = chunk {
                        self.state = ZarrStreamState::Error;
                        return Poll::Ready(Some(Err(e)));
                    } 

                    let chunk = chunk.unwrap();
                    let container = ZarrInMemoryChunkContainer::new(chunk);
                    let mut zarr_reader = ZarrRecordBatchReader::new(
                        self.meta.clone(), Some(container), None, None
                    );
                    
                    if self.mask.is_some() {
                        zarr_reader = zarr_reader.with_row_mask(self.mask.take().unwrap());
                    }
                    self.state = ZarrStreamState::Decoding(zarr_reader);
                },
                ZarrStreamState::Decoding(reader) => {
                    // this call should always return something, we should never get a None because
                    // if we're here it means we provided store with a zarr in memory chunk to the reader
                    let rec_batch = reader.next().expect("could not get record batch in zarr record batch stream");

                    if let Err(e) = rec_batch {
                        self.state = ZarrStreamState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }

                    self.state = ZarrStreamState::Init;
                    return Poll::Ready(Some(rec_batch));
                },
                ZarrStreamState::Error => return Poll::Ready(None),
            }
        }
    }
}

pub struct ZarrRecordBatchStreamBuilder<T: ZarrReadAsync + Clone + Unpin + Send> 
{
    zarr_reader_async: T,
    projection: ZarrProjection,
    filter: Option<ZarrChunkFilter>,
}

impl<T: ZarrReadAsync + Clone + Unpin + Send + 'static> ZarrRecordBatchStreamBuilder<T> {
    pub fn new(zarr_reader_async: T) -> Self {
        Self{zarr_reader_async, projection: ZarrProjection::all(), filter: None}
    }

    pub fn with_projection(self, projection: ZarrProjection) -> Self {
        Self {projection: projection, ..self}
    }

    pub fn with_filter(self, filter: ZarrChunkFilter) -> Self {
        Self {filter: Some(filter), ..self}
    }

    pub async fn build_partial_reader(
        self, chunk_range: Option<(usize, usize)>
    ) -> ZarrResult<ZarrRecordBatchStream<ZarrStoreAsync<T>>> {
        let meta = self.zarr_reader_async.get_zarr_metadata().await?;
        let mut chunk_pos: Vec<Vec<usize>> = meta.get_chunk_positions();
        if let Some(chunk_range) = chunk_range {
            if (chunk_range.0 > chunk_range.1) | (chunk_range.1 > chunk_pos.len()) {
                return Err(ZarrError::InvalidChunkRange(chunk_range.0, chunk_range.1, chunk_pos.len()))
            }
            chunk_pos = chunk_pos[chunk_range.0..chunk_range.1].to_vec();
        }

        let mut predicate_stream: Option<ZarrStoreAsync<T>> = None;
        if let Some(filter) = &self.filter {
            let predicate_proj = filter.get_all_projections();
            predicate_stream = Some(
                ZarrStoreAsync::new(
                    self.zarr_reader_async.clone(), chunk_pos.clone(), predicate_proj.clone()
                ).await?
            );
        }

        let zarr_stream = ZarrStoreAsync::new(
            self.zarr_reader_async, chunk_pos, self.projection.clone()
        ).await?;
        Ok(ZarrRecordBatchStream::new(meta, zarr_stream, self.filter, predicate_stream))
    }

    pub async fn build(self) -> ZarrResult<ZarrRecordBatchStream<ZarrStoreAsync<T>>> {
        self.build_partial_reader(None).await
    }
}

#[cfg(test)]
mod zarr_async_reader_tests {
    use super::*;
    use arrow_array::types::*;
    use arrow_array::*;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use crate::async_reader::zarr_read_async::ZarrPath;
    use futures_util::TryStreamExt;
    use object_store::path::Path;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use itertools::enumerate;
    use std::{path::PathBuf, collections::HashMap, fmt::Debug};
    use crate::reader::filters::{ZarrArrowPredicateFn, ZarrArrowPredicate};
    use arrow::compute::kernels::cmp::{gt_eq, lt};

    fn get_test_data_path(zarr_store: String) -> ZarrPath {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .unwrap()
                        .join("testing/data/zarr")
                        .join(zarr_store);
        ZarrPath::new(
            Arc::new(LocalFileSystem::new()),
            Path::from_absolute_path(p).unwrap()
        )
    }

    fn validate_names_and_types(targets: &HashMap<String, DataType>, rec: &RecordBatch) {
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

    fn validate_bool_column(col_name: &str, rec: &RecordBatch, targets: &[bool]) {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(
                    rec.column(idx).as_boolean(),
                    &BooleanArray::from(targets.to_vec()),
                );
                matched = true;
            }
        }
        assert!(matched);
    }

    fn validate_primitive_column<T, U>(col_name: &str, rec: &RecordBatch, targets: &[U]) 
    where
        T: ArrowPrimitiveType,
        [U]: AsRef<[<T as arrow_array::ArrowPrimitiveType>::Native]>,
        U: Debug
    {
        let mut matched = false;
        for (idx, col) in enumerate(rec.schema().fields.iter()) {
            if col.name().as_str() == col_name {
                assert_eq!(
                    rec.column(idx).as_primitive::<T>().values(),
                    targets,
                );
                matched = true;
            }
        }
        assert!(matched);
    }

    #[tokio::test]
    async fn projection_tests() {
        let zp = get_test_data_path("compression_example.zarr".to_string());
        let proj = ZarrProjection::keep(vec!["bool_data".to_string(), "int_data".to_string()]);
        let stream_builder = ZarrRecordBatchStreamBuilder::new(zp).with_projection(proj);

        let stream = stream_builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("int_data".to_string(), DataType::Int64),
        ]);

        // center chunk
        let rec = &records[4];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[false, true, false, false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[-4, -3, -2, 4, 5, 6, 12, 13, 14]);
    }

    #[tokio::test]
    async fn filters_tests() {
        // set the filters to select part of the raster, based on lat and
        // lon coordinates.
        let mut filters: Vec<Box<dyn ZarrArrowPredicate>> = Vec::new();
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lat".to_string()]),
            move |batch| (
                gt_eq(batch.column_by_name("lat").unwrap(), &Scalar::new(&Float64Array::from(vec![38.6])))
            ),
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| (
                gt_eq(batch.column_by_name("lon").unwrap(), &Scalar::new(&Float64Array::from(vec![-109.7])))
            ),
        );
        filters.push(Box::new(f));
        let f = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["lon".to_string()]),
            move |batch| (
               lt(batch.column_by_name("lon").unwrap(), &Scalar::new(&Float64Array::from(vec![-109.2])))
            ),
        );
        filters.push(Box::new(f));

        let zp = get_test_data_path("lat_lon_example.zarr".to_string());
        let stream_builder = ZarrRecordBatchStreamBuilder::new(zp).with_filter(ZarrChunkFilter::new(filters));
        let stream = stream_builder.build().await.unwrap();
        let records: Vec<_> = stream.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("lat".to_string(), DataType::Float64),
            ("lon".to_string(), DataType::Float64),
            ("float_data".to_string(), DataType::Float64),
        ]);

        let rec = &records[1];
        validate_names_and_types(&target_types, rec);
        validate_primitive_column::<Float64Type, f64>(&"lat", rec, &[38.8, 38.9, 39.0]);
        validate_primitive_column::<Float64Type, f64>(&"lon", rec, &[-109.7, -109.7, -109.7]);
        validate_primitive_column::<Float64Type, f64>(&"float_data", rec, &[1042.0, 1043.0, 1044.0]);
    }

    #[tokio::test]
    async fn multiple_readers_tests() {
        let zp = get_test_data_path("compression_example.zarr".to_string());
        let stream1 = ZarrRecordBatchStreamBuilder::new(zp.clone()).build_partial_reader(Some((0, 5))).await.unwrap();
        let stream2 = ZarrRecordBatchStreamBuilder::new(zp).build_partial_reader(Some((5, 9))).await.unwrap();

        let records1: Vec<_> = stream1.try_collect().await.unwrap();
        let records2: Vec<_> = stream2.try_collect().await.unwrap();

        let target_types = HashMap::from([
            ("bool_data".to_string(), DataType::Boolean),
            ("uint_data".to_string(), DataType::UInt64),
            ("int_data".to_string(), DataType::Int64),
            ("float_data".to_string(), DataType::Float64),
            ("float_data_no_comp".to_string(), DataType::Float64),
        ]);

        // center chunk
        let rec = &records1[4];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[false, true, false, false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[-4, -3, -2, 4, 5, 6, 12, 13, 14]);
        validate_primitive_column::<UInt64Type, u64>(&"uint_data", rec, &[27, 28, 29, 35, 36, 37, 43, 44, 45]);
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[127., 128., 129., 135., 136., 137., 143., 144., 145.]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data_no_comp", rec, &[227., 228., 229., 235., 236., 237., 243., 244., 245.]
        );
 

        // bottom edge chunk
        let rec = &records2[2];
        validate_names_and_types(&target_types, rec);
        validate_bool_column(&"bool_data", rec, &[false, true, false, false, true, false]);
        validate_primitive_column::<Int64Type, i64>(&"int_data", rec, &[20, 21, 22, 28, 29, 30]);
        validate_primitive_column::<UInt64Type, u64>(&"uint_data", rec, &[51, 52, 53, 59, 60, 61]);
        validate_primitive_column::<Float64Type, f64>(
            &"float_data", rec, &[151.0, 152.0, 153.0, 159.0, 160.0, 161.0]
        );
        validate_primitive_column::<Float64Type, f64>(
            &"float_data_no_comp", rec, &[251.0, 252.0, 253.0, 259.0, 260.0, 261.0]
        );
    }
}