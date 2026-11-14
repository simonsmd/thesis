use super::{
    ByteVector, Classifier, Mask, StaticSimd, structural_bytes, structural_bytes_in_field,
    structural_bytes_in_quoted_field,
};
use crate::config::Config;

/// Performs one comparison per structural byte and combines the masks with or.
#[derive(Debug, Clone, Copy)]
pub struct CmpOr<V> {
    structural: [V; 5],
    in_field: [V; 5],
    in_quoted_field: [V; 2],
}

/// The lanes of `v` equal to one of `needles`.
#[inline(always)]
fn any_eq<V: ByteVector, const N: usize>(v: V, needles: [V; N]) -> Mask<V> {
    needles.map(|needle| v.simd_eq(needle)).into_iter().reduce(|a, b| a | b).unwrap()
}

impl<V: ByteVector> Classifier for CmpOr<V> {
    type Vector = V;

    fn new(config: &Config) -> Self {
        let splat = |byte| V::splat(V::Simd::static_token(), byte);
        Self {
            structural: structural_bytes(config).map(splat),
            in_field: structural_bytes_in_field(config).map(splat),
            in_quoted_field: structural_bytes_in_quoted_field(config).map(splat),
        }
    }

    #[inline(always)]
    fn structural(&self, v: V) -> Mask<V> {
        any_eq(v, self.structural)
    }

    #[inline(always)]
    fn in_field(&self, v: V) -> Mask<V> {
        any_eq(v, self.in_field)
    }

    #[inline(always)]
    fn in_quoted_field(&self, v: V) -> Mask<V> {
        any_eq(v, self.in_quoted_field)
    }
}
