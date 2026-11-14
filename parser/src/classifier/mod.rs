//! This module implements different methods using SIMD instructions to find (structural) bytes in
//! a chunk of input bytes.
//!
//! Usually, most of the input consists of field data. When inside a field, the DFA stays in the
//! same state (InField or InQuotedFieldOutput) and outputs all bytes unless it encounters a
//! structural byte. Processing these bytes without running the DFA can speed up parsing
//! significantly.
//!
//! The SIMD implementations use `fearless_simd` without runtime dispatch: they are generic over a
//! [`ByteVector`] of a [`StaticSimd`] level whose instructions are enabled at compile time, usually
//! [`Native`]. The vector type is the number of bytes classified at once: [`u8x16`], [`u8x32`] or
//! [`u8x64`], which `fearless_simd` splits into several vectors if the level is narrower.

mod cmp_or;
mod nibble_map;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod string_compare;

pub use cmp_or::CmpOr;
pub use fearless_simd::{u8x16, u8x32, u8x64};
pub use nibble_map::NibbleMap;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use string_compare::{
    Explicit, Implicit, Lengths, Pcmpestr, Pcmpistr, StringCompare, StringInstructions,
    StringVector,
};

use fearless_simd::{Level, Simd, SimdBase, SimdInt};

use crate::config::{Config, QuoteStyle, RecordTerminator};

/// A SIMD level whose instructions are enabled at compile time, so its token can be created
/// without runtime feature detection.
pub trait StaticSimd: Simd {
    /// The token for this level.
    ///
    /// The check against [`Level::baseline`] is resolved at compile time. It panics if the level
    /// is not enabled for the compilation target.
    fn static_token() -> Self;
}

macro_rules! static_simd {
    ($arch:meta, $token:ident, $as_token:ident) => {
        #[cfg($arch)]
        impl StaticSimd for fearless_simd::$token {
            #[inline(always)]
            fn static_token() -> Self {
                Level::baseline().$as_token().expect(concat!(
                    stringify!($token),
                    " is not enabled at compile time, build with -C target-cpu=native",
                ))
            }
        }
    };
}

static_simd!(any(target_arch = "x86", target_arch = "x86_64"), Sse2, as_sse2);
static_simd!(any(target_arch = "x86", target_arch = "x86_64"), Sse4_2, as_sse4_2);
static_simd!(any(target_arch = "x86", target_arch = "x86_64"), Avx2, as_avx2);
static_simd!(any(target_arch = "x86", target_arch = "x86_64"), Avx512, as_avx512);
static_simd!(target_arch = "aarch64", Neon, as_neon);

/// Select the best fearless_simd level for the target architecture at compile time
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type Native = <baseline::Baseline<{ baseline::level() }> as baseline::Pick>::Simd;
#[cfg(target_arch = "aarch64")]
pub type Native = fearless_simd::Neon;

/// Turns [`Level::baseline`] into a type
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod baseline {
    use fearless_simd::{Avx2, Avx512, Level, Sse2, Sse4_2};

    use super::StaticSimd;

    /// [`Level::baseline`] as a number, so that it can select a type through [`Baseline`].
    pub const fn level() -> u8 {
        match Level::baseline() {
            Level::Avx512(_) => 3,
            Level::Avx2(_) => 2,
            Level::Sse4_2(_) => 1,
            _ => 0,
        }
    }

    pub struct Baseline<const L: u8>;

    /// Maps a [`Baseline`] to its level.
    pub trait Pick {
        type Simd: StaticSimd;
    }

    impl Pick for Baseline<3> {
        type Simd = Avx512;
    }

    impl Pick for Baseline<2> {
        type Simd = Avx2;
    }

    impl Pick for Baseline<1> {
        type Simd = Sse4_2;
    }

    impl Pick for Baseline<0> {
        type Simd = Sse2;
    }
}

/// A `fearless_simd` byte vector, [`u8x16`], [`u8x32`] or [`u8x64`], together with its SIMD level.
///
/// `fearless_simd` takes the level as a parameter of its vector traits (`SimdInt<S>`), so a
/// generic vector type does not determine its level, and an impl generic over the vector could
/// not name it. The associated type does, so a classifier needs a single impl for all widths.
///
/// The vector is its own byte representation, so it can index the swizzles.
pub trait ByteVector:
    SimdInt<<Self as ByteVector>::Simd, Element = u8, ByteVector = Self>
{
    /// The level of the vector.
    type Simd: StaticSimd;
}

impl<S: StaticSimd> ByteVector for u8x16<S> {
    type Simd = S;
}

impl<S: StaticSimd> ByteVector for u8x32<S> {
    type Simd = S;
}

impl<S: StaticSimd> ByteVector for u8x64<S> {
    type Simd = S;
}

/// The fearless_simd mask of a [`ByteVector`].
pub type Mask<V> = <V as SimdBase<<V as ByteVector>::Simd>>::Mask;

/// Trait for classifiers that take a chunk of input bytes and compute a mask identifying all
/// structural bytes contained in the input vector.
pub trait Classifier: Copy {
    /// The byte vector per classification, at most 64 so that a bitmask of them fits in a `u64`.
    type Vector: ByteVector;

    /// Number of bytes in a [`Vector`](Self::Vector).
    const LEN: usize = <Self::Vector as SimdBase<<Self::Vector as ByteVector>::Simd>>::LEN;

    /// Build the classifier for parser configuration.
    ///
    /// The structural bytes depend on the configuration, so implementations have to build their
    /// constants at runtime.
    fn new(config: &Config) -> Self;

    /// Load [`LEN`](Self::LEN) bytes.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` has a different length.
    #[inline(always)]
    fn load(bytes: &[u8]) -> Self::Vector {
        Self::Vector::from_slice(<Self::Vector as ByteVector>::Simd::static_token(), bytes)
    }

    /// Compute mask of [`structural_bytes`].
    fn structural(&self, v: Self::Vector) -> Mask<Self::Vector>;

    /// Compute mask of [`structural_bytes_in_field`].
    fn in_field(&self, v: Self::Vector) -> Mask<Self::Vector>;

    /// Comput mask of [`structural_bytes_in_quoted_field`].
    fn in_quoted_field(&self, v: Self::Vector) -> Mask<Self::Vector>;

    /// Compute `[in_field, in_quoted_field]`, for implementations that can share work between the
    /// two.
    #[inline(always)]
    fn per_state(&self, v: Self::Vector) -> [Mask<Self::Vector>; 2] {
        [self.in_field(v), self.in_quoted_field(v)]
    }
}

/// The bytes that can lead out of an output state.
///
/// Structural byte slots that are not needed for a specific parser configuration are filled up
/// with the delimiter byte.
pub fn structural_bytes(config: &Config) -> [u8; 5] {
    let delimiter = config.delimiter;
    let (cr, lf) = match config.terminator {
        RecordTerminator::CRLF => (b'\r', b'\n'),
        RecordTerminator::Byte(b) => (b, b),
    };
    [
        delimiter,
        cr,
        lf,
        config.quote.unwrap_or(delimiter),
        config.escape.unwrap_or(delimiter),
    ]
}

/// The bytes that can lead out of the `InField` state.
///
/// Structural byte slots that are not needed for a specific parser configuration are filled up
/// with the delimiter byte.
pub fn structural_bytes_in_field(config: &Config) -> [u8; 5] {
    let [delimiter, cr, lf, ..] = structural_bytes(config);
    let quote = config
        .quote
        .filter(|_| config.quote_style != QuoteStyle::Strict(true));
    let escape = config.escape.filter(|_| config.escape_in_unquoted_fields);
    [
        delimiter,
        cr,
        lf,
        quote.unwrap_or(delimiter),
        escape.unwrap_or(delimiter),
    ]
}

/// The bytes that can lead out of the `InQuotedFieldOutput` state.
///
/// If there is no escape character configured, the quote character is used twice.
pub fn structural_bytes_in_quoted_field(config: &Config) -> [u8; 2] {
    // When quoting is not enabled the InQuotedFieldOutput state cannot be reached anyways
    let quote = config.quote.unwrap_or(config.delimiter);
    [quote, config.escape.unwrap_or(quote)]
}

/// A byte that is not one of the [`structural_bytes`].
///
/// After a field end every such byte is output and starts an unquoted field, so a parser that finds
/// no structural byte after a field end can take the transition with this byte instead of the byte
/// itself, and jump on from there with the mask of the unquoted field.
pub fn ordinary_byte(config: &Config) -> u8 {
    let structural = structural_bytes(config);
    (0..=u8::MAX)
        .find(|b| !structural.contains(b))
        .expect("there are at most 5 structural bytes")
}

#[cfg(test)]
mod tests {
    use fearless_simd::SimdMask;

    use super::*;
    use crate::reference::{State, TransitionAction, transition_next_state};

    /// Every combination of the settings that change which bytes are structural.
    fn configs() -> Vec<Config> {
        let mut configs = Vec::new();
        for terminator in [RecordTerminator::CRLF, RecordTerminator::Byte(b';')] {
            for quote in [Some(b'"'), None] {
                for escape in [Some(b'\\'), None] {
                    for escape_in_unquoted_fields in [true, false] {
                        for quote_style in [
                            QuoteStyle::Toggle,
                            QuoteStyle::Strict(false),
                            QuoteStyle::Strict(true),
                        ] {
                            for double_quote_escape in [true, false] {
                                for comment in [Some(b'#'), None] {
                                    configs.push(
                                        Config::builder()
                                            .delimiter(b'\t')
                                            .terminator(terminator.clone())
                                            .quote(quote)
                                            .escape(escape)
                                            .escape_in_unquoted_fields(escape_in_unquoted_fields)
                                            .quote_style(quote_style.clone())
                                            .double_quote_escape(double_quote_escape)
                                            .comment(comment)
                                            .build()
                                            .unwrap(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        configs
    }

    #[test]
    fn non_structural_bytes_keep_the_output_states() {
        for config in configs() {
            let structural = structural_bytes(&config);
            for state in [State::InField, State::InQuotedFieldOutput] {
                for byte in (0..=255).filter(|b| !structural.contains(b)) {
                    assert_eq!(
                        transition_next_state(&config, state, byte),
                        (state, TransitionAction::Output),
                        "{state:?} --{byte:#04x}--> with {config:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn non_structural_bytes_start_an_unquoted_field() {
        for config in configs() {
            let structural = structural_bytes(&config);
            assert!(!structural.contains(&ordinary_byte(&config)));
            for byte in (0..=255).filter(|b| !structural.contains(b)) {
                assert_eq!(
                    transition_next_state(&config, State::EndField, byte),
                    (State::InField, TransitionAction::Output),
                    "EndField --{byte:#04x}--> with {config:?}",
                );
            }
        }
    }

    #[test]
    fn non_structural_bytes_keep_each_output_state() {
        for config in configs() {
            for (state, structural) in [
                (State::InField, structural_bytes_in_field(&config).to_vec()),
                (
                    State::InQuotedFieldOutput,
                    structural_bytes_in_quoted_field(&config).to_vec(),
                ),
            ] {
                for byte in (0..=255).filter(|b| !structural.contains(b)) {
                    assert_eq!(
                        transition_next_state(&config, state, byte),
                        (state, TransitionAction::Output),
                        "{state:?} --{byte:#04x}--> with {config:?}",
                    );
                }
            }
        }
    }

    /// Bytes in a test block, as many as the widest vector.
    const BLOCK: usize = 64;

    /// Bit `i` is set if `block[i]` is one of `needles`.
    fn scalar_mask(needles: &[u8], block: &[u8; BLOCK]) -> u64 {
        let mut mask = 0;
        for (i, byte) in block.iter().enumerate() {
            mask |= (needles.contains(byte) as u64) << i;
        }
        mask
    }

    /// Every byte value on both sides of each 16 byte boundary, then every byte value in every
    /// lane.
    fn blocks() -> impl Iterator<Item = [u8; BLOCK]> {
        let single = (0..=255).flat_map(|byte| {
            [0, 15, 16, 31, 32, 47, 48, 63].map(|position| {
                let mut block = [b'x'; BLOCK];
                block[position] = byte;
                block
            })
        });
        let lanes = (0..256)
            .step_by(BLOCK)
            .map(|start| std::array::from_fn(|i| (start + i) as u8));
        single.chain(lanes)
    }

    /// The bitmask of `classify` over `block`, one vector at a time.
    ///
    /// Also checks that `any_true` agrees with the bitmask of each vector.
    fn bitmask<C: Classifier>(block: &[u8; BLOCK], classify: impl Fn(C::Vector) -> Mask<C::Vector>) -> u64 {
        let mut mask = 0;
        for (i, chunk) in block.chunks_exact(C::LEN).enumerate() {
            let lanes = classify(C::load(chunk));
            assert_eq!(lanes.any_true(), lanes.to_bitmask() != 0, "{chunk:?}");
            mask |= lanes.to_bitmask() << (i * C::LEN);
        }
        mask
    }

    /// [`configs`] and configs whose structural bytes include zero: as the delimiter, or as the
    /// quote with and without an escape.
    fn configs_with_zero() -> Vec<Config> {
        let mut configs = configs();
        configs.push(Config::builder().delimiter(0).build().unwrap());
        configs.push(Config::builder().quote(Some(0)).escape(Some(b'\\')).build().unwrap());
        configs.push(Config::builder().quote(Some(0)).escape(None).build().unwrap());
        configs
    }

    /// Blocks with zero bytes before and after other bytes in the same 16 bytes, on top of
    /// [`blocks`], which put each byte value alone between ordinary bytes.
    fn blocks_with_zero() -> impl Iterator<Item = [u8; BLOCK]> {
        let pairs = (0..=255).flat_map(|byte| {
            [(0, 15), (15, 0), (3, 7), (7, 3), (15, 16), (16, 15)].map(move |(zero, position)| {
                let mut block = [b'x'; BLOCK];
                for offset in [0, 32] {
                    block[offset + zero] = 0;
                    block[offset + position] = byte;
                }
                block
            })
        });
        blocks().chain(pairs)
    }

    /// `C` marks exactly the bytes of each set, for every config, byte value and position, also
    /// through [`per_state`](Classifier::per_state).
    fn agrees_with_scalar<C: Classifier>() {
        for config in configs_with_zero() {
            let classifier = C::new(&config);
            let structural = structural_bytes(&config);
            let in_field = structural_bytes_in_field(&config);
            let in_quoted_field = structural_bytes_in_quoted_field(&config);
            for block in blocks_with_zero() {
                let check = |set: &str, needles: &[u8], classify: &dyn Fn(C::Vector) -> Mask<C::Vector>| {
                    let actual = bitmask::<C>(&block, classify);
                    let expected = scalar_mask(needles, &block);
                    assert_eq!(actual, expected, "{set} in {block:?} with {config:?}");
                };
                check("structural", &structural, &|v| classifier.structural(v));
                check("in field", &in_field, &|v| classifier.in_field(v));
                check("in quoted field", &in_quoted_field, &|v| classifier.in_quoted_field(v));
                check("per state in field", &in_field, &|v| classifier.per_state(v)[0]);
                check("per state in quoted field", &in_quoted_field, &|v| classifier.per_state(v)[1]);
            }
        }
    }

    #[test]
    fn cmp_or_agrees_with_scalar() {
        agrees_with_scalar::<CmpOr<u8x16<Native>>>();
        agrees_with_scalar::<CmpOr<u8x32<Native>>>();
        agrees_with_scalar::<CmpOr<u8x64<Native>>>();
    }

    #[test]
    fn nibble_map_agrees_with_scalar() {
        agrees_with_scalar::<NibbleMap<u8x16<Native>>>();
        agrees_with_scalar::<NibbleMap<u8x32<Native>>>();
        agrees_with_scalar::<NibbleMap<u8x64<Native>>>();
    }

    // the x86 levels below Native implement the swizzle differently (SSE2 emulates it)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn nibble_map_agrees_with_scalar_on_sse2() {
        agrees_with_scalar::<NibbleMap<u8x16<fearless_simd::Sse2>>>();
        agrees_with_scalar::<NibbleMap<u8x64<fearless_simd::Sse2>>>();
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse4.2"))]
    #[test]
    fn nibble_map_agrees_with_scalar_on_sse4_2() {
        agrees_with_scalar::<NibbleMap<u8x16<fearless_simd::Sse4_2>>>();
        agrees_with_scalar::<NibbleMap<u8x64<fearless_simd::Sse4_2>>>();
    }

    // for when Native is AVX-512, which has its own instructions for all of these
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
    #[test]
    fn agrees_with_scalar_on_avx2() {
        agrees_with_scalar::<CmpOr<u8x64<fearless_simd::Avx2>>>();
        agrees_with_scalar::<NibbleMap<u8x16<fearless_simd::Avx2>>>();
        agrees_with_scalar::<NibbleMap<u8x64<fearless_simd::Avx2>>>();
        agrees_with_scalar::<Pcmpistr<u8x64<fearless_simd::Avx2>>>();
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse4.2"))]
    #[test]
    fn pcmpistr_agrees_with_scalar() {
        agrees_with_scalar::<Pcmpistr<u8x16<fearless_simd::Sse4_2>>>();
        agrees_with_scalar::<Pcmpistr<u8x32<fearless_simd::Sse4_2>>>();
        agrees_with_scalar::<Pcmpistr<u8x64<fearless_simd::Sse4_2>>>();
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
    #[test]
    fn pcmpistr_agrees_with_scalar_on_native() {
        agrees_with_scalar::<Pcmpistr<u8x16<Native>>>();
        agrees_with_scalar::<Pcmpistr<u8x32<Native>>>();
        agrees_with_scalar::<Pcmpistr<u8x64<Native>>>();
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse4.2"))]
    #[test]
    fn pcmpestr_agrees_with_scalar() {
        agrees_with_scalar::<Pcmpestr<u8x16<fearless_simd::Sse4_2>>>();
        agrees_with_scalar::<Pcmpestr<u8x32<fearless_simd::Sse4_2>>>();
        agrees_with_scalar::<Pcmpestr<u8x64<fearless_simd::Sse4_2>>>();
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
    #[test]
    fn pcmpestr_agrees_with_scalar_on_native() {
        agrees_with_scalar::<Pcmpestr<u8x16<Native>>>();
        agrees_with_scalar::<Pcmpestr<u8x32<Native>>>();
        agrees_with_scalar::<Pcmpestr<u8x64<Native>>>();
    }
}
