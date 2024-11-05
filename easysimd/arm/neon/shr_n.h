/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2020      Evan Nemerson <evan@nemerson.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_ARM_NEON_SHR_N_H)
#define EASYSIMD_ARM_NEON_SHR_N_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_x_vshrs_n_s32(int32_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  return a >> ((n == 32) ? 31 : n);
}

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_x_vshrs_n_u32(uint32_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  return (n == 32) ? 0 : a >> n;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vshrd_n_s64(int64_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  return a >> ((n == 64) ? 63 : n);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vshrd_n_s64(a, n) vshrd_n_s64(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vshrd_n_s64
  #define vshrd_n_s64(a, n) easysimd_vshrd_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vshrd_n_u64(uint64_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  return (n == 64) ? 0 : a >> n;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vshrd_n_u64(a, n) vshrd_n_u64(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vshrd_n_u64
  #define vshrd_n_u64(a, n) easysimd_vshrd_n_u64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vshr_n_s8 (const easysimd_int8x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_int8x8_private
    r_,
    a_ = easysimd_int8x8_to_private(a);
  int32_t n_ = (n == 8) ? 7 : n;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
    r_.values = a_.values >> n_;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int8_t, a_.values[i] >> n_);
    }
  #endif

  return easysimd_int8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_s8(a, n) vshr_n_s8((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshr_n_s8(a, n) \
    easysimd_int8x8_from_m64( \
      _mm_or_si64(_mm_andnot_si64(_mm_set1_pi16(0x00FF), _mm_srai_pi16(easysimd_int8x8_to_m64(a), (n))), \
      _mm_and_si64(_mm_set1_pi16(0x00FF), _mm_srai_pi16(_mm_slli_pi16(easysimd_int8x8_to_m64(a), 8), 8 + (n)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_s8
  #define vshr_n_s8(a, n) easysimd_vshr_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vshr_n_s16 (const easysimd_int16x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_int16x4_private
    r_,
    a_ = easysimd_int16x4_to_private(a);
  int32_t n_ = (n == 16) ? 15 : n;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
    r_.values = a_.values >> n_;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int16_t, a_.values[i] >> n_);
    }
  #endif

  return easysimd_int16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_s16(a, n) vshr_n_s16((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshr_n_s16(a, n) easysimd_int16x4_from_m64(_mm_srai_pi16(easysimd_int16x4_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_s16
  #define vshr_n_s16(a, n) easysimd_vshr_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vshr_n_s32 (const easysimd_int32x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_int32x2_private
    r_,
    a_ = easysimd_int32x2_to_private(a);
  int32_t n_ = (n == 32) ? 31 : n;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values >> n_;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] >> n_;
    }
  #endif

  return easysimd_int32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_s32(a, n) vshr_n_s32((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshr_n_s32(a, n) easysimd_int32x2_from_m64(_mm_srai_pi32(easysimd_int32x2_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_s32
  #define vshr_n_s32(a, n) easysimd_vshr_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vshr_n_s64 (const easysimd_int64x1_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_int64x1_private
    r_,
    a_ = easysimd_int64x1_to_private(a);
  int32_t n_ = (n == 64) ? 63 : n;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values >> n_;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] >> n_;
    }
  #endif

  return easysimd_int64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_s64(a, n) vshr_n_s64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_s64
  #define vshr_n_s64(a, n) easysimd_vshr_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vshr_n_u8 (const easysimd_uint8x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_uint8x8_private
    r_,
    a_ = easysimd_uint8x8_to_private(a);

  if (n == 8) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values >> n;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] >> n;
      }
    #endif
  }

  return easysimd_uint8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_u8(a, n) vshr_n_u8((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshr_n_u8(a, n) \
    easysimd_uint8x8_from_m64(_mm_and_si64(_mm_srli_si64(easysimd_uint8x8_to_m64(a), (n)), _mm_set1_pi8((1 << (8 - (n))) - 1)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_u8
  #define vshr_n_u8(a, n) easysimd_vshr_n_u8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vshr_n_u16 (const easysimd_uint16x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_uint16x4_private
    r_,
    a_ = easysimd_uint16x4_to_private(a);

  if (n == 16) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = a_.values >> n;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] >> n;
      }
    #endif
  }

  return easysimd_uint16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_u16(a, n) vshr_n_u16((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshr_n_u16(a, n) easysimd_uint16x4_from_m64(_mm_srli_pi16(easysimd_uint16x4_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_u16
  #define vshr_n_u16(a, n) easysimd_vshr_n_u16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vshr_n_u32 (const easysimd_uint32x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_uint32x2_private
    r_,
    a_ = easysimd_uint32x2_to_private(a);

  if (n == 32) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = a_.values >> n;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] >> n;
      }
    #endif
  }

  return easysimd_uint32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_u32(a, n) vshr_n_u32((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshr_n_u32(a, n) easysimd_uint32x2_from_m64(_mm_srli_pi32(easysimd_uint32x2_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_u32
  #define vshr_n_u32(a, n) easysimd_vshr_n_u32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vshr_n_u64 (const easysimd_uint64x1_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_uint64x1_private
    r_,
    a_ = easysimd_uint64x1_to_private(a);

  if (n == 64) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = a_.values >> n;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] >> n;
      }
    #endif
  }

  return easysimd_uint64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshr_n_u64(a, n) vshr_n_u64((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshr_n_u64(a, n) easysimd_uint64x1_from_m64(_mm_srli_si64(easysimd_uint64x1_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshr_n_u64
  #define vshr_n_u64(a, n) easysimd_vshr_n_u64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vshrq_n_s8 (const easysimd_int8x16_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_int8x16_private
    r_,
    a_ = easysimd_int8x16_to_private(a);

  #if defined(EASYSIMD_X86_GFNI_NATIVE)
    /* https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/ */
    const int shift = (n <= 7) ? n : 7;
    const uint64_t matrix = (UINT64_C(0x8182848890A0C000) << (shift * 8)) ^ UINT64_C(0x8080808080808080);
    r_.m128i = _mm_gf2p8affine_epi64_epi8(a_.m128i, _mm_set1_epi64x(HEDLEY_STATIC_CAST(int64_t, matrix)), 0);
  #elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
    r_.m128i =
      _mm_blendv_epi8(_mm_srai_epi16(a_.m128i, n),
                      _mm_srai_epi16(_mm_slli_epi16(a_.m128i, 8), 8 + (n)),
                      _mm_set1_epi16(0x00FF));
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i =
      _mm_or_si128(_mm_andnot_si128(_mm_set1_epi16(0x00FF), _mm_srai_epi16(a_.m128i, n)),
                  _mm_and_si128(_mm_set1_epi16(0x00FF), _mm_srai_epi16(_mm_slli_epi16(a_.m128i, 8), 8 + (n))));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values >> ((n == 8) ? 7 : n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int8_t, a_.values[i] >> ((n == 8) ? 7 : n));
    }
  #endif

  return easysimd_int8x16_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_s8(a, n) vshrq_n_s8((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_s8
  #define vshrq_n_s8(a, n) easysimd_vshrq_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vshrq_n_s16 (const easysimd_int16x8_t a, const int n)
  EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_int16x8_private
    r_,
    a_ = easysimd_int16x8_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_srai_epi16(a_.m128i, n);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values >> ((n == 16) ? 15 : n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int16_t, a_.values[i] >> ((n == 16) ? 15 : n));
    }
  #endif

  return easysimd_int16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_s16(a, n) vshrq_n_s16((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_s16
  #define vshrq_n_s16(a, n) easysimd_vshrq_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vshrq_n_s32 (const easysimd_int32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_int32x4_private
    r_,
    a_ = easysimd_int32x4_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_srai_epi32(a_.m128i, n);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values >> ((n == 32) ? 31 : n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] >> ((n == 32) ? 31 : n);
    }
  #endif

  return easysimd_int32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_s32(a, n) vshrq_n_s32((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_s32
  #define vshrq_n_s32(a, n) easysimd_vshrq_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vshrq_n_s64 (const easysimd_int64x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_int64x2_private
    r_,
    a_ = easysimd_int64x2_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values >> ((n == 64) ? 63 : n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] >> ((n == 64) ? 63 : n);
    }
  #endif

  return easysimd_int64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_s64(a, n) vshrq_n_s64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_s64
  #define vshrq_n_s64(a, n) easysimd_vshrq_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vshrq_n_u8 (const easysimd_uint8x16_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a);

    #if defined(EASYSIMD_X86_GFNI_NATIVE)
      /* https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/ */
      r_.m128i = (n > 7) ? _mm_setzero_si128() : _mm_gf2p8affine_epi64_epi8(a_.m128i, _mm_set1_epi64x(INT64_C(0x0102040810204080) << (n * 8)), 0);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_and_si128(_mm_srli_epi64(a_.m128i, (n)), _mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, (1 << (8 - (n))) - 1)));
    #else
      if (n == 8) {
        easysimd_memset(&r_, 0, sizeof(r_));
      } else {
        #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
          r_.values = a_.values >> n;
        #else
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
            r_.values[i] = a_.values[i] >> n;
          }
        #endif
      }
    #endif

    return easysimd_uint8x16_from_private(r_);\
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_u8(a, n) vshrq_n_u8((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_u8
  #define vshrq_n_u8(a, n) easysimd_vshrq_n_u8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vshrq_n_u16 (const easysimd_uint16x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_srli_epi16(a_.m128i, n);
    #else
      if (n == 16) {
        easysimd_memset(&r_, 0, sizeof(r_));
      } else {
        #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
          r_.values = a_.values >> n;
        #else
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
            r_.values[i] = a_.values[i] >> n;
          }
        #endif
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_u16(a, n) vshrq_n_u16((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_u16
  #define vshrq_n_u16(a, n) easysimd_vshrq_n_u16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vshrq_n_u32 (const easysimd_uint32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_uint32x4_private
    r_,
    a_ = easysimd_uint32x4_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_srli_epi32(a_.m128i, n);
  #else
    if (n == 32) {
      easysimd_memset(&r_, 0, sizeof(r_));
    } else {
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.values = a_.values >> n;
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
          r_.values[i] = a_.values[i] >> n;
        }
      #endif
    }
  #endif

  return easysimd_uint32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_u32(a, n) vshrq_n_u32((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_u32
  #define vshrq_n_u32(a, n) easysimd_vshrq_n_u32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vshrq_n_u64 (const easysimd_uint64x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_uint64x2_private
    r_,
    a_ = easysimd_uint64x2_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_srli_epi64(a_.m128i, n);
  #else
    if (n == 64) {
      easysimd_memset(&r_, 0, sizeof(r_));
    } else {
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_97248)
        r_.values = a_.values >> n;
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
          r_.values[i] = a_.values[i] >> n;
        }
      #endif
    }
  #endif

  return easysimd_uint64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrq_n_u64(a, n) vshrq_n_u64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrq_n_u64
  #define vshrq_n_u64(a, n) easysimd_vshrq_n_u64((a), (n))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_SHR_N_H) */
