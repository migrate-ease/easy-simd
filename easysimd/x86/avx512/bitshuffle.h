#if !defined(EASYSIMD_X86_AVX512_BITSHUFFLE_H)
#define EASYSIMD_X86_AVX512_BITSHUFFLE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_bitshuffle_epi64_mask (easysimd__m128i b, easysimd__m128i c) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_bitshuffle_epi64_mask(b, c);
  #else
    easysimd__m128i_private
      b_ = easysimd__m128i_to_private(b),
      c_ = easysimd__m128i_to_private(c);
    easysimd__mmask16 r = 0;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(b_.u64) rv = { 0, 0 };
      __typeof__(b_.u64) lshift = { 0, 8 };

      for (int8_t i = 0 ; i < 8 ; i++) {
        __typeof__(b_.u64) ct = (HEDLEY_REINTERPRET_CAST(__typeof__(ct), c_.u8) >> (i * 8)) & 63;
        rv |= ((b_.u64 >> ct) & 1) << lshift;
        lshift += 1;
      }

      r =
        HEDLEY_STATIC_CAST(easysimd__mmask16, rv[0]) |
        HEDLEY_STATIC_CAST(easysimd__mmask16, rv[1]);
    #else
      for (size_t i = 0 ; i < (sizeof(c_.m64_private) / sizeof(c_.m64_private[0])) ; i++) {
        EASYSIMD_VECTORIZE_REDUCTION(|:r)
        for (size_t j = 0 ; j < (sizeof(c_.m64_private[i].u8) / sizeof(c_.m64_private[i].u8[0])) ; j++) {
          r |= (((b_.u64[i] >> (c_.m64_private[i].u8[j]) & 63) & 1) << ((i * 8) + j));
        }
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_bitshuffle_epi64_mask
  #define _mm_bitshuffle_epi64_mask(b, c) easysimd_mm_bitshuffle_epi64_mask(b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_mask_bitshuffle_epi64_mask (easysimd__mmask16 k, easysimd__m128i b, easysimd__m128i c) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_bitshuffle_epi64_mask(k, b, c);
  #else
    return (k & easysimd_mm_bitshuffle_epi64_mask(b, c));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_bitshuffle_epi64_mask
  #define _mm_mask_bitshuffle_epi64_mask(k, b, c) easysimd_mm_mask_bitshuffle_epi64_mask(k, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_bitshuffle_epi64_mask (easysimd__m256i b, easysimd__m256i c) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_bitshuffle_epi64_mask(b, c);
  #else
    easysimd__m256i_private
      b_ = easysimd__m256i_to_private(b),
      c_ = easysimd__m256i_to_private(c);
    easysimd__mmask32 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < sizeof(b_.m128i) / sizeof(b_.m128i[0]) ; i++) {
        r |= (HEDLEY_STATIC_CAST(easysimd__mmask32, easysimd_mm_bitshuffle_epi64_mask(b_.m128i[i], c_.m128i[i])) << (i * 16));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(b_.u64) rv = { 0, 0, 0, 0 };
      __typeof__(b_.u64) lshift = { 0, 8, 16, 24 };

      for (int8_t i = 0 ; i < 8 ; i++) {
        __typeof__(b_.u64) ct = (HEDLEY_REINTERPRET_CAST(__typeof__(ct), c_.u8) >> (i * 8)) & 63;
        rv |= ((b_.u64 >> ct) & 1) << lshift;
        lshift += 1;
      }

      r =
        HEDLEY_STATIC_CAST(easysimd__mmask32, rv[0]) |
        HEDLEY_STATIC_CAST(easysimd__mmask32, rv[1]) |
        HEDLEY_STATIC_CAST(easysimd__mmask32, rv[2]) |
        HEDLEY_STATIC_CAST(easysimd__mmask32, rv[3]);
    #else
      for (size_t i = 0 ; i < (sizeof(c_.m128i_private) / sizeof(c_.m128i_private[0])) ; i++) {
        for (size_t j = 0 ; j < (sizeof(c_.m128i_private[i].m64_private) / sizeof(c_.m128i_private[i].m64_private[0])) ; j++) {
          EASYSIMD_VECTORIZE_REDUCTION(|:r)
          for (size_t k = 0 ; k < (sizeof(c_.m128i_private[i].m64_private[j].u8) / sizeof(c_.m128i_private[i].m64_private[j].u8[0])) ; k++) {
            r |= (((b_.m128i_private[i].u64[j] >> (c_.m128i_private[i].m64_private[j].u8[k]) & 63) & 1) << ((i * 16) + (j * 8) + k));
          }
        }
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_bitshuffle_epi64_mask
  #define _mm256_bitshuffle_epi64_mask(b, c) easysimd_mm256_bitshuffle_epi64_mask(b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_mask_bitshuffle_epi64_mask (easysimd__mmask32 k, easysimd__m256i b, easysimd__m256i c) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_bitshuffle_epi64_mask(k, b, c);
  #else
    return (k & easysimd_mm256_bitshuffle_epi64_mask(b, c));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_bitshuffle_epi64_mask
  #define _mm256_mask_bitshuffle_epi64_mask(k, b, c) easysimd_mm256_mask_bitshuffle_epi64_mask(k, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_bitshuffle_epi64_mask (easysimd__m512i b, easysimd__m512i c) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_bitshuffle_epi64_mask(b, c);
  #else
    easysimd__m512i_private
      b_ = easysimd__m512i_to_private(b),
      c_ = easysimd__m512i_to_private(c);
    easysimd__mmask64 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(b_.m128i) / sizeof(b_.m128i[0])) ; i++) {
        r |= (HEDLEY_STATIC_CAST(easysimd__mmask64, easysimd_mm_bitshuffle_epi64_mask(b_.m128i[i], c_.m128i[i])) << (i * 16));
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(b_.m256i) / sizeof(b_.m256i[0])) ; i++) {
        r |= (HEDLEY_STATIC_CAST(easysimd__mmask64, easysimd_mm256_bitshuffle_epi64_mask(b_.m256i[i], c_.m256i[i])) << (i * 32));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(b_.u64) rv = { 0, 0, 0, 0, 0, 0, 0, 0 };
      __typeof__(b_.u64) lshift = { 0, 8, 16, 24, 32, 40, 48, 56 };

      for (int8_t i = 0 ; i < 8 ; i++) {
        __typeof__(b_.u64) ct = (HEDLEY_REINTERPRET_CAST(__typeof__(ct), c_.u8) >> (i * 8)) & 63;
        rv |= ((b_.u64 >> ct) & 1) << lshift;
        lshift += 1;
      }

      r =
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[0]) |
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[1]) |
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[2]) |
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[3]) |
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[4]) |
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[5]) |
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[6]) |
        HEDLEY_STATIC_CAST(easysimd__mmask64, rv[7]);
    #else
      for (size_t i = 0 ; i < (sizeof(c_.m128i_private) / sizeof(c_.m128i_private[0])) ; i++) {
        for (size_t j = 0 ; j < (sizeof(c_.m128i_private[i].m64_private) / sizeof(c_.m128i_private[i].m64_private[0])) ; j++) {
          EASYSIMD_VECTORIZE_REDUCTION(|:r)
          for (size_t k = 0 ; k < (sizeof(c_.m128i_private[i].m64_private[j].u8) / sizeof(c_.m128i_private[i].m64_private[j].u8[0])) ; k++) {
            r |= (((b_.m128i_private[i].u64[j] >> (c_.m128i_private[i].m64_private[j].u8[k]) & 63) & 1) << ((i * 16) + (j * 8) + k));
          }
        }
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_bitshuffle_epi64_mask
  #define _mm512_bitshuffle_epi64_mask(b, c) easysimd_mm512_bitshuffle_epi64_mask(b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_mask_bitshuffle_epi64_mask (easysimd__mmask64 k, easysimd__m512i b, easysimd__m512i c) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_mask_bitshuffle_epi64_mask(k, b, c);
  #else
    return (k & easysimd_mm512_bitshuffle_epi64_mask(b, c));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_bitshuffle_epi64_mask
  #define _mm512_mask_bitshuffle_epi64_mask(k, b, c) easysimd_mm512_mask_bitshuffle_epi64_mask(k, b, c)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_BITSHUFFLE_H) */
