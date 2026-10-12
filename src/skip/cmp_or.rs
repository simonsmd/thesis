use fearless_simd::{Simd, prelude::*};

use super::{BLOCK, Skip, StaticSimd, structural_bytes};
use crate::config::Config;

/// Compares every byte against each of the [`structural_bytes`] and ORs the results.
///
/// Uses the native vector width of `S`: 16 bytes on SSE and NEON, 32 on AVX2, 64 on AVX-512.
#[derive(Debug, Clone, Copy)]
pub struct CmpOr<S: Simd> {
    simd: S,
    needles: [S::u8s; 5],
}

impl<S: StaticSimd> Skip for CmpOr<S> {
    fn new(config: &Config) -> Self {
        let simd = S::static_token();
        Self {
            simd,
            needles: structural_bytes(config).map(|byte| S::u8s::splat(simd, byte)),
        }
    }

    #[inline(always)]
    fn structural_mask(&self, block: &[u8; BLOCK]) -> u64 {
        let [a, b, c, d, e] = self.needles;
        let mut mask = 0;
        // TODO: on NEON every to_bitmask is a horizontal add (vaddvq), combining the four
        // compare results with pairwise adds first needs only one
        for (i, chunk) in block.chunks_exact(S::u8s::LEN).enumerate() {
            let v = S::u8s::from_slice(self.simd, chunk);
            let found = v.simd_eq(a) | v.simd_eq(b) | v.simd_eq(c) | v.simd_eq(d) | v.simd_eq(e);
            mask |= found.to_bitmask() << (i * S::u8s::LEN);
        }
        mask
    }
}
