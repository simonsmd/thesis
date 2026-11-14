#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, _SIDD_CMP_EQUAL_ANY, _SIDD_UBYTE_OPS, _SIDD_UNIT_MASK, _mm_cmpestrm, _mm_cmpistrm,
    _mm_cmpistrz,
};
use std::fmt::Debug;
use std::marker::PhantomData;

use fearless_simd::{
    Avx2, Avx512, SimdFrom, SimdSplit, Sse4_2, mask8x16, mask8x32, mask8x64, prelude::*, u8x16,
    u8x32, u8x64,
};

use super::{
    ByteVector, Classifier, Mask, StaticSimd, structural_bytes, structural_bytes_in_field,
    structural_bytes_in_quoted_field,
};
use crate::config::Config;

/// Flags for the imm8 bit field argument to the string compare instructions.
/// The input consists of unsigned bytes and serach for any match in the structural bytes.
const EQUAL_ANY: i32 = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY;

/// The fearless_simd levels that have the SSE 4.2 string instructions.
pub trait StringInstructions: StaticSimd {
    /// Get the SSE 4.2 token.
    fn sse4_2(self) -> Sse4_2;
}

impl StringInstructions for Sse4_2 {
    #[inline(always)]
    fn sse4_2(self) -> Sse4_2 {
        self
    }
}

impl StringInstructions for Avx2 {
    #[inline(always)]
    fn sse4_2(self) -> Sse4_2 {
        self.level().as_sse4_2().expect("AVX2 includes SSE 4.2")
    }
}

impl StringInstructions for Avx512 {
    #[inline(always)]
    fn sse4_2(self) -> Sse4_2 {
        self.level().as_sse4_2().expect("AVX-512 includes SSE 4.2")
    }
}

fearless_simd::kernel!(
    #[inline(always)]
    fn mask_implicit(_level: Sse4_2, needles: __m128i, data: __m128i) -> (__m128i, bool) {
        const IMM: i32 = EQUAL_ANY | _SIDD_UNIT_MASK;
        let mask = _mm_cmpistrm::<IMM>(needles, data);
        // The _mm_cmpistrm and _mm_cmpistrz intrinsics get merged into one pcmpistrm instruction
        (mask, _mm_cmpistrz::<IMM>(needles, data) != 0)
    }
);

fearless_simd::kernel!(
    #[inline(always)]
    fn mask_explicit(_level: Sse4_2, needles: __m128i, neeldes_len: i32, data: __m128i) -> __m128i {
        _mm_cmpestrm::<{ EQUAL_ANY | _SIDD_UNIT_MASK }>(needles, neeldes_len, data, 16)
    }
);

/// One set of needles: its bytes padded with zeros to 16, and how many there are.
///
/// Zero needles come after the others, so that the implicit length form sees all the nonzero ones.
#[derive(Debug, Clone, Copy)]
pub struct Needles {
    bytes: __m128i,
    len: i32,
}

/// How the string instructions find the length of their operands: [`Implicit`] or [`Explicit`].
pub trait Lengths: Debug + Copy {
    /// 0xFF in every byte of `data` in `needles`, 0 in the others.
    fn mask(simd: Sse4_2, needles: Needles, data: __m128i) -> __m128i;
}

#[derive(Debug, Clone, Copy)]
pub struct Implicit;

impl Lengths for Implicit {
    #[inline(always)]
    fn mask(simd: Sse4_2, needles: Needles, data: __m128i) -> __m128i {
        // pcmpistrm marks any bytes after the first null byte as invalid and does not compare
        // them. The UTF-8 input can contain null bytes so without a separate check for null bytes,
        // structural bytes after a null byte or structural bytes that are the null byte get
        // skipped. pcmpistrm sets the zero flag when the input contains null bytes, in which case
        // we run the slower pcmpestrm to get the correct result.
        match mask_implicit(simd, needles.bytes, data) {
            (_, true) => Explicit::mask(simd, needles, data),
            (mask, false) => mask,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Explicit;

impl Lengths for Explicit {
    #[inline(always)]
    fn mask(simd: Sse4_2, needles: Needles, data: __m128i) -> __m128i {
        mask_explicit(simd, needles.bytes, needles.len, data)
    }
}

/// Finds the structural bytes with the SSE 4.2 string instructions, which comparea 16 bytes
/// against a set of up to 16 needles at once and return a mask of the matches.
///
/// Wider vectors run one instruction per 16 bytes. `L` selects the implicit (pcmpistrm) or the
/// explicit length form (pcmpestrm).
#[derive(Debug, Clone, Copy)]
pub struct StringCompare<V, L> {
    structural: Needles,
    in_field: Needles,
    in_quoted_field: Needles,
    marker: PhantomData<(V, L)>,
}

/// [`StringCompare`] with `pcmpistrm`, with fallback to `pcmpestrm` inputs that contain a zero
/// byte.
pub type Pcmpistr<V> = StringCompare<V, Implicit>;

/// [`StringCompare`] with `pcmpestrm`.
pub type Pcmpestr<V> = StringCompare<V, Explicit>;

impl<V, L> StringCompare<V, L> {
    fn build<S: Simd>(simd: S, config: &Config) -> Self {
        let needles = |set: &[u8]| {
            // with pcmpistrm a zero byte marks the end of the needles array, so the zero needles have
            // to be moved to the end. for pcmpestrm this does not matter because it uses an
            // explicit length so it can also search for null bytes in the input.
            let mut bytes = [0; 16];
            for (slot, &byte) in bytes.iter_mut().zip(set.iter().filter(|&&byte| byte != 0)) {
                *slot = byte;
            }
            Needles {
                bytes: u8x16::from_slice(simd, &bytes).into(),
                len: set.len() as i32,
            }
        };
        Self {
            structural: needles(&structural_bytes(config)),
            in_field: needles(&structural_bytes_in_field(config)),
            in_quoted_field: needles(&structural_bytes_in_quoted_field(config)),
            marker: PhantomData,
        }
    }
}

/// A [`ByteVector`] of a level with the string instructions, classified 16 bytes at a time.
///
/// The widths differ in how they split into 16 bytes, so this needs an impl per width, but the
/// [`Classifier`] implementation of [`StringCompare`] does not.
pub trait StringVector: ByteVector<Simd: StringInstructions> {
    fn mask<L: Lengths>(needles: Needles, v: Self) -> Mask<Self>;
}

impl<S: StringInstructions> StringVector for u8x16<S> {
    #[inline(always)]
    fn mask<L: Lengths>(needles: Needles, v: Self) -> mask8x16<S> {
        let simd = S::static_token();
        mask8x16::simd_from(simd, L::mask(simd.sse4_2(), needles, v.into()))
    }
}

impl<S: StringInstructions> StringVector for u8x32<S> {
    #[inline(always)]
    fn mask<L: Lengths>(needles: Needles, v: Self) -> mask8x32<S> {
        let (low, high) = v.split();
        S::static_token().combine_mask8x16(
            StringVector::mask::<L>(needles, low),
            StringVector::mask::<L>(needles, high),
        )
    }
}

impl<S: StringInstructions> StringVector for u8x64<S> {
    #[inline(always)]
    fn mask<L: Lengths>(needles: Needles, v: Self) -> mask8x64<S> {
        let (low, high) = v.split();
        S::static_token().combine_mask8x32(
            StringVector::mask::<L>(needles, low),
            StringVector::mask::<L>(needles, high),
        )
    }
}

impl<V: StringVector, L: Lengths> Classifier for StringCompare<V, L> {
    type Vector = V;

    fn new(config: &Config) -> Self {
        Self::build(V::Simd::static_token(), config)
    }

    #[inline(always)]
    fn structural(&self, v: V) -> Mask<V> {
        V::mask::<L>(self.structural, v)
    }

    #[inline(always)]
    fn in_field(&self, v: V) -> Mask<V> {
        V::mask::<L>(self.in_field, v)
    }

    #[inline(always)]
    fn in_quoted_field(&self, v: V) -> Mask<V> {
        V::mask::<L>(self.in_quoted_field, v)
    }
}
