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

#if !defined(EASYSIMD_ARM_NEON_SHL_N_H)
#define EASYSIMD_ARM_NEON_SHL_N_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vshld_n_s64 (const int64_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  return HEDLEY_STATIC_CAST(int64_t, a << n);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vshld_n_s64(a, n) vshld_n_s64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vshld_n_s64
  #define vshld_n_s64(a, n) easysimd_vshld_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vshld_n_u64 (const uint64_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  return HEDLEY_STATIC_CAST(uint64_t, a << n);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vshld_n_u64(a, n) vshld_n_u64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vshld_n_u64
  #define vshld_n_u64(a, n) easysimd_vshld_n_u64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vshl_n_s8 (const easysimd_int8x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  easysimd_int8x8_private
    r_,
    a_ = easysimd_int8x8_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
    r_.values = a_.values << HEDLEY_STATIC_CAST(int8_t, n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int8_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_s8(a, n) vshl_n_s8((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_s8(a, n) \
    easysimd_int8x8_from_m64(_mm_andnot_si64(_mm_set1_pi8((1 << n) - 1), _mm_slli_si64(easysimd_int8x8_to_m64(a), (n))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_s8
  #define vshl_n_s8(a, n) easysimd_vshl_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vshl_n_s16 (const easysimd_int16x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
  easysimd_int16x4_private
    r_,
    a_ = easysimd_int16x4_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << HEDLEY_STATIC_CAST(int16_t, n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int16_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_s16(a, n) vshl_n_s16((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_s16(a, n) easysimd_int16x4_from_m64(_mm_slli_pi16(easysimd_int16x4_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_s16
  #define vshl_n_s16(a, n) easysimd_vshl_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vshl_n_s32 (const easysimd_int32x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 31) {
  easysimd_int32x2_private
    r_,
    a_ = easysimd_int32x2_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int32_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_s32(a, n) vshl_n_s32((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_s32(a, n) easysimd_int32x2_from_m64(_mm_slli_pi32(easysimd_int32x2_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_s32
  #define vshl_n_s32(a, n) easysimd_vshl_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vshl_n_s64 (const easysimd_int64x1_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  easysimd_int64x1_private
    r_,
    a_ = easysimd_int64x1_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int64_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_s64(a, n) vshl_n_s64((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_s64(a, n) easysimd_int64x1_from_m64(_mm_slli_si64(easysimd_int64x1_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_s64
  #define vshl_n_s64(a, n) easysimd_vshl_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vshl_n_u8 (const easysimd_uint8x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  easysimd_uint8x8_private
    r_,
    a_ = easysimd_uint8x8_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
    r_.values = a_.values << HEDLEY_STATIC_CAST(uint8_t, n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, a_.values[i] << n);
    }
  #endif

  return easysimd_uint8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_u8(a, n) vshl_n_u8((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_u8(a, n) \
    easysimd_uint8x8_from_m64(_mm_andnot_si64(_mm_set1_pi8((1 << n) - 1), _mm_slli_si64(easysimd_uint8x8_to_m64(a), (n))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_u8
  #define vshl_n_u8(a, n) easysimd_vshl_n_u8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vshl_n_u16 (const easysimd_uint16x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
  easysimd_uint16x4_private
    r_,
    a_ = easysimd_uint16x4_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << HEDLEY_STATIC_CAST(uint16_t, n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, a_.values[i] << n);
    }
  #endif

  return easysimd_uint16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_u16(a, n) vshl_n_u16((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_u16(a, n) easysimd_uint16x4_from_m64(_mm_slli_pi16(easysimd_uint16x4_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_u16
  #define vshl_n_u16(a, n) easysimd_vshl_n_u16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vshl_n_u32 (const easysimd_uint32x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 31) {
  easysimd_uint32x2_private
    r_,
    a_ = easysimd_uint32x2_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint32_t, a_.values[i] << n);
    }
  #endif

  return easysimd_uint32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_u32(a, n) vshl_n_u32((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_u32(a, n) easysimd_uint32x2_from_m64(_mm_slli_pi32(easysimd_uint32x2_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_u32
  #define vshl_n_u32(a, n) easysimd_vshl_n_u32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vshl_n_u64 (const easysimd_uint64x1_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  easysimd_uint64x1_private
    r_,
    a_ = easysimd_uint64x1_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint64_t, a_.values[i] << n);
    }
  #endif

  return easysimd_uint64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshl_n_u64(a, n) vshl_n_u64((a), (n))
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #define easysimd_vshl_n_u64(a, n) easysimd_uint64x1_from_m64(_mm_slli_si64(easysimd_uint64x1_to_m64(a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshl_n_u64
  #define vshl_n_u64(a, n) easysimd_vshl_n_u64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vshlq_n_s8 (const easysimd_int8x16_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  easysimd_int8x16_private
    r_,
    a_ = easysimd_int8x16_to_private(a);

  #if defined(EASYSIMD_X86_GFNI_NATIVE)
    /* https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/ */
    r_.m128i = _mm_gf2p8affine_epi64_epi8(a_.m128i, _mm_set1_epi64x(INT64_C(0x0102040810204080) >> (n * 8)), 0);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_andnot_si128(_mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, (1 << n) - 1)), _mm_slli_epi64(a_.m128i, n));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << HEDLEY_STATIC_CAST(int8_t, n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int8_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int8x16_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_s8(a, n) vshlq_n_s8((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_s8
  #define vshlq_n_s8(a, n) easysimd_vshlq_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vshlq_n_s16 (const easysimd_int16x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
  easysimd_int16x8_private
    r_,
    a_ = easysimd_int16x8_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_slli_epi16(a_.m128i, (n));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << HEDLEY_STATIC_CAST(int16_t, n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int16_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_s16(a, n) vshlq_n_s16((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_s16
  #define vshlq_n_s16(a, n) easysimd_vshlq_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vshlq_n_s32 (const easysimd_int32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 31) {
  easysimd_int32x4_private
    r_,
    a_ = easysimd_int32x4_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_slli_epi32(a_.m128i, (n));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int32_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_s32(a, n) vshlq_n_s32((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_s32
  #define vshlq_n_s32(a, n) easysimd_vshlq_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vshlq_n_s64 (const easysimd_int64x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  easysimd_int64x2_private
    r_,
    a_ = easysimd_int64x2_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_slli_epi64(a_.m128i, (n));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(int64_t, a_.values[i] << n);
    }
  #endif

  return easysimd_int64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_s64(a, n) vshlq_n_s64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_s64
  #define vshlq_n_s64(a, n) easysimd_vshlq_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vshlq_n_u8 (const easysimd_uint8x16_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  easysimd_uint8x16_private
    r_,
    a_ = easysimd_uint8x16_to_private(a);

  #if defined(EASYSIMD_X86_GFNI_NATIVE)
    /* https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/ */
    r_.m128i = _mm_gf2p8affine_epi64_epi8(a_.m128i, _mm_set1_epi64x(INT64_C(0x0102040810204080) >> (n * 8)), 0);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_andnot_si128(_mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, (1 << n) - 1)), _mm_slli_epi64(a_.m128i, (n)));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << HEDLEY_STATIC_CAST(uint8_t, n);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, a_.values[i] << n);
    }
  #endif

  return easysimd_uint8x16_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_u8(a, n) vshlq_n_u8((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_u8
  #define vshlq_n_u8(a, n) easysimd_vshlq_n_u8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vshlq_n_u16 (const easysimd_uint16x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_slli_epi16(a_.m128i, (n));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = a_.values << HEDLEY_STATIC_CAST(uint16_t, n);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, a_.values[i] << n);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_u16(a, n) vshlq_n_u16((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_u16
  #define vshlq_n_u16(a, n) easysimd_vshlq_n_u16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vshlq_n_u32 (const easysimd_uint32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 31) {
  easysimd_uint32x4_private
    r_,
    a_ = easysimd_uint32x4_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_slli_epi32(a_.m128i, (n));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint32_t, a_.values[i] << n);
    }
  #endif

  return easysimd_uint32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_u32(a, n) vshlq_n_u32((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_u32
  #define vshlq_n_u32(a, n) easysimd_vshlq_n_u32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vshlq_n_u64 (const easysimd_uint64x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  easysimd_uint64x2_private
    r_,
    a_ = easysimd_uint64x2_to_private(a);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    r_.m128i = _mm_slli_epi64(a_.m128i, (n));
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.values = a_.values << n;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint64_t, a_.values[i] << n);
    }
  #endif

  return easysimd_uint64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshlq_n_u64(a, n) vshlq_n_u64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshlq_n_u64
  #define vshlq_n_u64(a, n) easysimd_vshlq_n_u64((a), (n))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_SHL_N_H) */
