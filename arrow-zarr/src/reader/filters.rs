use crate::reader::zarr_read::ColumnProjection;
use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::ArrowError;


pub trait ZarrArrowPredicate: Send + 'static {
    fn projection(&self) -> &ColumnProjection;
    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError>;
}


pub struct ZarrArrowPredicateFn<F> {
    f: F,
    projection: ColumnProjection,
}

impl<F> ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    pub fn new(projection: ColumnProjection, f: F) -> Self {
        Self { f, projection }
    }
}

impl<F> ZarrArrowPredicate for ZarrArrowPredicateFn<F>
where
    F: FnMut(&RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    fn projection(&self) -> &ColumnProjection {
        &self.projection
    }

    fn evaluate(&mut self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        (self.f)(batch)
    }
}

pub struct ZarrChunkFilter {
    /// A list of [`ArrowPredicate`]
    pub(crate) predicates: Vec<Box<dyn ZarrArrowPredicate>>,
}

impl ZarrChunkFilter {
    /// Create a new [`RowFilter`] from an array of [`ArrowPredicate`]
    pub fn new(predicates: Vec<Box<dyn ZarrArrowPredicate>>) -> Self {
        Self { predicates }
    }

    pub fn get_all_projections(&self) -> ColumnProjection {
        let mut proj = ColumnProjection::new(false, Vec::new());
        for pred in self.predicates.iter() {
            proj = proj.update(pred.projection().clone());
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
    use crate::reader::zarr_read::ColumnProjection;
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
            ColumnProjection::new(false, vec!["var1".to_string(), "var2".to_string()]),
            move |batch| eq(batch.column_by_name("var1").unwrap(), batch.column_by_name("var2").unwrap()),
        );
        let mask = filter.evaluate(&rec).unwrap();
        assert_eq!(
            mask,
            BooleanArray::from(vec![false, false, true, true, false, false]),
        );

        let rec = generate_rec_batch();
        let mut filter = ZarrArrowPredicateFn::new(
            ColumnProjection::new(false, vec!["var1".to_string(), "var3".to_string()]),
            move |batch| eq(batch.column_by_name("var1").unwrap(), batch.column_by_name("var3").unwrap()),
        );
        let mask = filter.evaluate(&rec).unwrap();
        assert_eq!(
            mask,
            BooleanArray::from(vec![true, false, false, false, false, false]),
        );
    }
}