//! Finding the bytes that can change the state inside a field, so the parser can skip the rest.
//!
//! The only states that loop on ordinary bytes are `InField` and `InQuotedFieldOutput`, which are
//! also exactly the output states ([`Dfa::is_output`](crate::dfa::Dfa::is_output)). In those two
//! states only the bytes from [`structural_bytes`] can lead to a different state, every other byte
//! is output and keeps the state. A parser in an output state can therefore copy everything up to
//! the next structural byte without running the DFA over it.
//!
//! The SIMD implementations use `fearless_simd` without runtime dispatch: they are generic over a
//! [`StaticSimd`] level whose instructions are enabled at compile time, usually [`Native`].

mod cmp_or;
mod nibble_map;

pub use cmp_or::CmpOr;
pub use nibble_map::NibbleMap;

use fearless_simd::{Level, Simd};

use crate::config::{Config, RecordTerminator};

/// Number of bytes a [`Skip`] classifies at once.
pub const BLOCK: usize = 32;

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

/// The best level for the compilation target: AVX2 with `-C target-cpu=native` on any AVX2 cpu,
/// SSE2 for the default x86-64 target, NEON on aarch64.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
pub type Native = fearless_simd::Avx2;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(target_feature = "avx2")))]
pub type Native = fearless_simd::Sse2;
#[cfg(target_arch = "aarch64")]
pub type Native = fearless_simd::Neon;

/// Classifies a block of input into bytes that can change the state of a field and bytes that
/// cannot.
///
/// The parser passes the skipper by value and keeps it in a local, so the vectors or tables an
/// implementation holds can stay in registers for the whole core loop.
pub trait Skip: Copy {
    /// Build the skipper for the dialect in `config`.
    ///
    /// The structural bytes depend on the config, so implementations have to build their
    /// constants (splatted needles, lookup tables) at runtime.
    fn new(config: &Config) -> Self;

    /// Bit `i` is set if `block[i]` is one of the [`structural_bytes`].
    ///
    /// Setting additional bits is allowed: it only stops a skip early.
    fn structural_mask(&self, block: &[u8; BLOCK]) -> u64;
}

/// The bytes that can lead out of an output state.
///
/// Missing bytes (no quote or escape, a single byte terminator) are filled with duplicates of
/// another entry, so the length is fixed.
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

/// Portable implementation without explicit SIMD, comparing every byte against every needle.
///
/// Serves as a baseline for the SIMD implementations and as the reference in the tests.
#[derive(Debug, Clone, Copy)]
pub struct ScalarSkip {
    needles: [u8; 5],
}

impl Skip for ScalarSkip {
    fn new(config: &Config) -> Self {
        Self {
            needles: structural_bytes(config),
        }
    }

    #[inline(always)]
    fn structural_mask(&self, block: &[u8; BLOCK]) -> u64 {
        let mut mask = 0;
        for (i, &byte) in block.iter().enumerate() {
            let structural = self.needles.iter().any(|&needle| needle == byte);
            mask |= (structural as u64) << i;
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::QuoteStyle,
        reference::{State, TransitionAction, transition_next_state},
    };

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
                                        Config::new()
                                            .delimiter(b'\t')
                                            .terminator(terminator.clone())
                                            .quote(quote)
                                            .escape(escape)
                                            .escape_in_unquoted_fields(escape_in_unquoted_fields)
                                            .quote_style(quote_style.clone())
                                            .double_quote_escape(double_quote_escape)
                                            .comment(comment),
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

    /// `skip` marks the same bytes as [`ScalarSkip`] for every config, byte value and position.
    fn agrees_with_scalar<S: Skip>() {
        for config in configs() {
            let scalar = ScalarSkip::new(&config);
            let skip = S::new(&config);

            // every byte value on both sides of each 16 and 32 byte vector boundary
            for byte in 0..=255 {
                for position in [0, 15, 16, 31] {
                    let mut block = [b'x'; BLOCK];
                    block[position] = byte;
                    assert_eq!(
                        skip.structural_mask(&block),
                        scalar.structural_mask(&block),
                        "{byte:#04x} at {position} with {config:?}",
                    );
                }
            }

            // every byte value in every lane
            for start in (0..256).step_by(BLOCK) {
                let block: [u8; BLOCK] = std::array::from_fn(|i| (start + i) as u8);
                assert_eq!(skip.structural_mask(&block), scalar.structural_mask(&block));
            }
        }
    }

    #[test]
    fn cmp_or_agrees_with_scalar() {
        agrees_with_scalar::<CmpOr<Native>>();
    }

    #[test]
    fn nibble_map_agrees_with_scalar() {
        agrees_with_scalar::<NibbleMap<Native>>();
    }

    // the x86 levels below Native implement the swizzle differently (SSE2 emulates it)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn nibble_map_agrees_with_scalar_on_sse2() {
        agrees_with_scalar::<NibbleMap<fearless_simd::Sse2>>();
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse4.2"))]
    #[test]
    fn nibble_map_agrees_with_scalar_on_sse4_2() {
        agrees_with_scalar::<NibbleMap<fearless_simd::Sse4_2>>();
    }
}
