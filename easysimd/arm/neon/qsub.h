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
 */

#if !defined(EASYSIMD_ARM_NEON_QSUB_H)
#define EASYSIMD_ARM_NEON_QSUB_H

#include "types.h"

#include "sub.h"
#include "bsl.h"
#include "cgt.h"
#include "dup_n.h"
#include "sub.h"

#include <limits.h>

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vqsubb_s8(int8_t a, int8_t b) {
  return easysimd_math_subs_i8(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubb_s8
  #define vqsubb_s8(a, b) easysimd_vqsubb_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vqsubh_s16(int16_t a, int16_t b) {
  return easysimd_math_subs_i16(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubh_s16
  #define vqsubh_s16(a, b) easysimd_vqsubh_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vqsubs_s32(int32_t a, int32_t b) {
  return easysimd_math_subs_i32(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubs_s32
  #define vqsubs_s32(a, b) easysimd_vqsubs_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vqsubd_s64(int64_t a, int64_t b) {
  return easysimd_math_subs_i64(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubd_s64
  #define vqsubd_s64(a, b) easysimd_vqsubd_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vqsubb_u8(uint8_t a, uint8_t b) {
  return easysimd_math_subs_u8(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubb_u8
  #define vqsubb_u8(a, b) easysimd_vqsubb_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vqsubh_u16(uint16_t a, uint16_t b) {
  return easysimd_math_subs_u16(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubh_u16
  #define vqsubh_u16(a, b) easysimd_vqsubh_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vqsubs_u32(uint32_t a, uint32_t b) {
  return easysimd_math_subs_u32(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubs_u32
  #define vqsubs_u32(a, b) easysimd_vqsubs_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vqsubd_u64(uint64_t a, uint64_t b) {
  return easysimd_math_subs_u64(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqsubd_u64
  #define vqsubd_u64(a, b) easysimd_vqsubd_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqsub_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_s8(a, b);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_subs_pi8(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT8_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 7;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubb_s8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_s8
  #define vqsub_s8(a, b) easysimd_vqsub_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vqsub_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_s16(a, b);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_subs_pi16(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT16_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 15;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubh_s16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_s16
  #define vqsub_s16(a, b) easysimd_vqsub_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vqsub_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_s32(a, b);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT32_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 31;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubs_s32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_s32
  #define vqsub_s32(a, b) easysimd_vqsub_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vqsub_s64(easysimd_int64x1_t a, easysimd_int64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_s64(a, b);
  #else
    easysimd_int64x1_private
      r_,
      a_ = easysimd_int64x1_to_private(a),
      b_ = easysimd_int64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT64_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 63;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubd_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_s64
  #define vqsub_s64(a, b) easysimd_vqsub_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqsub_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_u8(a, b);
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_subs_pu8(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (r_.values <= a_.values));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubb_u8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_u8
  #define vqsub_u8(a, b) easysimd_vqsub_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vqsub_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_u16(a, b);
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_subs_pu16(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (r_.values <= a_.values));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubh_u16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_u16
  #define vqsub_u16(a, b) easysimd_vqsub_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vqsub_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_u32(a, b);
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (r_.values <= a_.values));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubs_u32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_u32
  #define vqsub_u32(a, b) easysimd_vqsub_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vqsub_u64(easysimd_uint64x1_t a, easysimd_uint64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsub_u64(a, b);
  #else
    easysimd_uint64x1_private
      r_,
      a_ = easysimd_uint64x1_to_private(a),
      b_ = easysimd_uint64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (r_.values <= a_.values));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubd_u64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsub_u64
  #define vqsub_u64(a, b) easysimd_vqsub_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqsubq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_s8(a, b);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_subs_epi8(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT8_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 7;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubb_s8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_s8
  #define vqsubq_s8(a, b) easysimd_vqsubq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vqsubq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_s16(a, b);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_subs_epi16(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT16_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 15;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubh_s16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_s16
  #define vqsubq_s16(a, b) easysimd_vqsubq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vqsubq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_s32(a, b);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      const __m128i diff_sat = _mm_xor_si128(_mm_set1_epi32(INT32_MAX), _mm_cmpgt_epi32(b_.m128i, a_.m128i));
      const __m128i diff = _mm_sub_epi32(a_.m128i, b_.m128i);

      const __m128i t = _mm_xor_si128(diff_sat, diff);
      #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
        r_.m128i =
          _mm_castps_si128(
            _mm_blendv_ps(
              _mm_castsi128_ps(diff),
              _mm_castsi128_ps(diff_sat),
              _mm_castsi128_ps(t)
            )
          );
      #else
        r_.m128i = _mm_xor_si128(diff, _mm_and_si128(t, _mm_srai_epi32(t, 31)));
      #endif
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT32_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 31;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubs_s32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_s32
  #define vqsubq_s32(a, b) easysimd_vqsubq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vqsubq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_s64(a, b);
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      const __typeof__(r_.values) diff_sat = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (b_.values > a_.values) ^ INT64_MAX);
      const __typeof__(r_.values) diff = a_.values - b_.values;
      const __typeof__(r_.values) saturate = diff_sat ^ diff;
      const __typeof__(r_.values) m = saturate >> 63;
      r_.values = (diff_sat & m) | (diff & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubd_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_s64
  #define vqsubq_s64(a, b) easysimd_vqsubq_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vqsubq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_u8(a, b);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);
    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_subs_epu8(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values <= a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubb_u8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_u8
  #define vqsubq_u8(a, b) easysimd_vqsubq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vqsubq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_u16(a, b);
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_subs_epu16(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values <= a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubh_u16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_u16
  #define vqsubq_u16(a, b) easysimd_vqsubq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vqsubq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_u32(a, b);
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      const __m128i i32_min = _mm_set1_epi32(INT32_MIN);
      const __m128i difference = _mm_sub_epi32(a_.m128i, b_.m128i);
      r_.m128i =
        _mm_and_si128(
          difference,
          _mm_xor_si128(
            _mm_cmpgt_epi32(
              _mm_xor_si128(difference, i32_min),
              _mm_xor_si128(a_.m128i, i32_min)
            ),
            _mm_set1_epi32(~INT32_C(0))
          )
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (r_.values <= a_.values));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (r_.values <= a_.values));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubs_u32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_u32
  #define vqsubq_u32(a, b) easysimd_vqsubq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vqsubq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqsubq_u64(a, b);
  #else
    easysimd_uint64x2_private
      r_,
      a_ = easysimd_uint64x2_to_private(a),
      b_ = easysimd_uint64x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values  = a_.values - b_.values;
      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (r_.values <= a_.values));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqsubd_u64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqsubq_u64
  #define vqsubq_u64(a, b) easysimd_vqsubq_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QSUB_H) */
