use super::{
    ByteVector, Classifier, Mask, StaticSimd, structural_bytes, structural_bytes_in_field,
    structural_bytes_in_quoted_field,
};
use crate::config::Config;

/// Classifies every byte using two table lookups, one by its low and one by its high nibble.
///
/// The low and high nibble of each byte gets mapped to a bitmask (one byte). Each structural byte
/// gets one bit in that bitmask that is set when the nibble matches. The low and high nibble masks
/// then get combined with and so the final bitmask per byte only contains a set bit when the input
/// byte matches one of the structrual characters exactly.
///
/// The structural bytes have to be ASCII, so that bytes of 0x80 and above never match.
#[derive(Debug, Clone, Copy)]
pub struct NibbleMap<V> {
    low_table: V,
    high_table: V,
    in_field_bits: V,
    in_quoted_field_bits: V,
}

impl<V: ByteVector> NibbleMap<V> {
    /// Perform the nibble map operation which maps each byte to a bitmask where each bit
    /// corresponds to one structural character and is set when the input byte matches that
    /// structural character
    #[inline(always)]
    fn bits(&self, v: V) -> V {
        // some architectures (e.g. ARM NEON) use the entire byte as the index and return 0 for
        // indices above 15, so the low nibble has to be masked.
        // On x86 the pshufb instruction uses the lower nibble for the index and only sets the
        // result to zero if the most significant bit is set. This case never happens because the
        // structural bytes are limited to ASCII so the most significant bit is never set. If the
        // mask includes the most significant bit as well, LLVM eliminates the mask on x86 because
        // it is redundant.
        let low = self.low_table.swizzle_dyn_within_blocks(v & 0x8F);
        // the high nibble is always in range
        let high = self.high_table.swizzle_dyn_within_blocks(v >> 4);
        low & high
    }
}

impl<V: ByteVector> Classifier for NibbleMap<V> {
    type Vector = V;

    fn new(config: &Config) -> Self {
        let simd = V::Simd::static_token();
        let structural = structural_bytes(config);

        // one bit per structural byte, the array has a fixed length of 5
        let mut low_table = [0; 16];
        let mut high_table = [0; 16];
        for (i, byte) in structural.into_iter().enumerate() {
            low_table[(byte & 0x0F) as usize] |= 1 << i;
            high_table[(byte >> 4) as usize] |= 1 << i;
        }

        // the state specific sets are subsets of the structural bytes, so the bitmasks for the
        // state specific sets can be computed from the state agnostic bitmask by masking away all
        // bits corresponding to structural bytes not in that set using and.
        let bits = |set: &[u8]| {
            let bits = structural
                .iter()
                .enumerate()
                .filter(|(_, structural_byte)| set.contains(structural_byte))
                .fold(0, |bits, (i, _)| bits | 1 << i);
            V::splat(simd, bits)
        };

        Self {
            low_table: V::from_fn(simd, |i| low_table[i % 16]),
            high_table: V::from_fn(simd, |i| high_table[i % 16]),
            in_field_bits: bits(&structural_bytes_in_field(config)),
            in_quoted_field_bits: bits(&structural_bytes_in_quoted_field(config)),
        }
    }

    #[inline(always)]
    fn structural(&self, v: V) -> Mask<V> {
        self.bits(v).simd_ne(0)
    }

    #[inline(always)]
    fn in_field(&self, v: V) -> Mask<V> {
        (self.bits(v) & self.in_field_bits).simd_ne(0)
    }

    #[inline(always)]
    fn in_quoted_field(&self, v: V) -> Mask<V> {
        (self.bits(v) & self.in_quoted_field_bits).simd_ne(0)
    }

    #[inline(always)]
    fn per_state(&self, v: V) -> [Mask<V>; 2] {
        let bits = self.bits(v);
        [
            (bits & self.in_field_bits).simd_ne(0),
            (bits & self.in_quoted_field_bits).simd_ne(0),
        ]
    }
}
