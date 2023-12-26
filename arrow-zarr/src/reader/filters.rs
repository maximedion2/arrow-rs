use crate::reader::zarr_read::ZarrProjection;
use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::ArrowError;


/// A predicate operating on [`RecordBatch`]
pub trait ZarrArrowPredicate: Send + 'static {
    /// Returns the [`ZarrProjecction`] that describes the columns required
    /// to evaluate this predicate. Those must be present in record batches
    /// that aare passed into the [`evaluate`] method.
    fn projection(&self) -> &ZarrProjection;

    /// Evaluate this predicate for the given [`RecordBatch`] containing the columns
    /// identified by [`projection`]. Rows that are `true` in the returned [`BooleanArray`]
    /// satisfy the predicate condition, whereas those that are `false` or do not.
    /// The method should not return any `Null` values.
    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError>;
}


/// A [`ZarrArrowPredicate`] created from an [`FnMut`]
pub struct ZarrArrowPredicateFn<F> {
    f: F,
    projection: ZarrProjection,
}

impl<F> ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    pub fn new(projection: ZarrProjection, f: F) -> Self {
        Self { f, projection }
    }
}

impl<F> ZarrArrowPredicate for ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    fn projection(&self) -> &ZarrProjection {
        &self.projection
    }

    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        (self.f)(batch)
    }
}


/// A collection of one or more objects that implement [`ZarrArrowPredicate`].
pub struct ZarrChunkFilter {
    /// A list of [`ZarrArrowPredicate`]
    pub(crate) predicates: Vec<Box<dyn ZarrArrowPredicate>>,
}

impl ZarrChunkFilter {
    /// Create a new [`ZarrChunkFilter`] from an array of [`ZarrArrowPredicate`]
    pub fn new(predicates: Vec<Box<dyn ZarrArrowPredicate>>) -> Self {
        Self { predicates }
    }

    /// Get the combined projections for all the predicates in the filter. 
    pub fn get_all_projections(&self) -> ZarrProjection {
        let mut proj = ZarrProjection::all();
        for pred in self.predicates.iter() {
            proj.update(pred.projection().clone());
        }

        proj
    }
}


#[cfg(test)]
mod zarr_predicate_tests {
    use super::*;
    use arrow_array::RecordBatch;
    use std::sync::Arc;
    use arrow_schema::{Schema, Field, DataType};
    use arrow_array::{BooleanArray, Float64Array, ArrayRef};
    use crate::reader::zarr_read::ZarrProjection;
    use arrow::compute::kernels::cmp::eq;

    fn generate_rec_batch() -> RecordBatch {
        let fields = vec![
            Arc::new(Field::new("var1".to_string(), DataType::Float64, false)),
            Arc::new(Field::new("var2".to_string(), DataType::Float64, false)),
            Arc::new(Field::new("var3".to_string(), DataType::Float64, false)),
        ];
        let arrs = vec![
            Arc::new(Float64Array::from(vec![38.0, 39.0, 40.0, 41.0, 42.0, 43.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![39.0, 38.0, 40.0, 41.0, 52.0, 53.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![38.0, 1.0, 2.0, 3.0, 4.0, 5.0])) as ArrayRef,
        ];

        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs).unwrap()
    }

    #[test]
    fn apply_predicate_tests() {
        let rec = generate_rec_batch();
        let mut filter = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["var1".to_string(), "var2".to_string()]),
            move |batch| eq(batch.column_by_name("var1").unwrap(), batch.column_by_name("var2").unwrap()),
        );
        let mask = filter.evaluate(&rec).unwrap();
        assert_eq!(
            mask,
            BooleanArray::from(vec![false, false, true, true, false, false]),
        );

        let rec = generate_rec_batch();
        let mut filter = ZarrArrowPredicateFn::new(
            ZarrProjection::keep(vec!["var1".to_string(), "var3".to_string()]),
            move |batch| eq(batch.column_by_name("var1").unwrap(), batch.column_by_name("var3").unwrap()),
        );
        let mask = filter.evaluate(&rec).unwrap();
        assert_eq!(
            mask,
            BooleanArray::from(vec![true, false, false, false, false, false]),
        );
    }
}