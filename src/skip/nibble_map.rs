use fearless_simd::{Simd, prelude::*, u8x16};

use super::{BLOCK, Skip, StaticSimd, structural_bytes};
use crate::config::Config;

/// Classifies every byte by two table lookups, one by its low and one by its high nibble.
///
/// Each of the [`structural_bytes`] gets its own bit, set in the low table at its low nibble and
/// in the high table at its high nibble. A byte is structural if the two lookups share a bit.
/// The structural bytes have to be ASCII, so that bytes of 0x80 and above never match.
///
/// Uses the native vector width of `S`: 16 bytes on SSE and NEON, 32 on AVX2, 64 on AVX-512.
#[derive(Debug, Clone, Copy)]
pub struct NibbleMap<S: Simd> {
    simd: S,
    low_table: S::u8s,
    high_table: S::u8s,
}

impl<S: StaticSimd> Skip for NibbleMap<S> {
    fn new(config: &Config) -> Self {
        let simd = S::static_token();

        // one bit per structural byte, the array has a fixed length of 5
        let mut low_table = [0; 16];
        let mut high_table = [0; 16];
        for (i, byte) in structural_bytes(config).into_iter().enumerate() {
            low_table[(byte & 0x0F) as usize] |= 1 << i;
            high_table[(byte >> 4) as usize] |= 1 << i;
        }

        Self {
            simd,
            low_table: S::u8s::block_splat(u8x16::from_slice(simd, &low_table)),
            high_table: S::u8s::block_splat(u8x16::from_slice(simd, &high_table)),
        }
    }

    #[inline(always)]
    fn structural_mask(&self, block: &[u8; BLOCK]) -> u64 {
        let nibble_mask = S::u8s::splat(self.simd, 0x0F);
        let mut mask = 0;
        // TODO: on NEON every to_bitmask is a horizontal add (vaddvq), combining the four
        // compare results with pairwise adds first needs only one
        for (i, chunk) in block.chunks_exact(S::u8s::LEN).enumerate() {
            let v = S::u8s::from_slice(self.simd, chunk);
            // NEON tbl returns 0 for indices above 15, so the low nibble has to be masked
            let low_index = v & nibble_mask;
            let low = self.low_table.swizzle_dyn_within_blocks(low_index);
            // a u8 shift, the high nibble is always in range
            let high = self.high_table.swizzle_dyn_within_blocks(v >> 4);
            mask |= (low & high).simd_ne(0).to_bitmask() << (i * S::u8s::LEN);
        }
        mask
    }
}
