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
 *   2020      Sean Maher <seanptmaher@gmail.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_QADD_H)
#define EASYSIMD_ARM_NEON_QADD_H

#include "types.h"

#include "add.h"
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
easysimd_vqaddb_s8(int8_t a, int8_t b) {
  return easysimd_math_adds_i8(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqaddb_s8
  #define vqaddb_s8(a, b) easysimd_vqaddb_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vqaddh_s16(int16_t a, int16_t b) {
  return easysimd_math_adds_i16(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqaddh_s16
  #define vqaddh_s16(a, b) easysimd_vqaddh_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vqadds_s32(int32_t a, int32_t b) {
  return easysimd_math_adds_i32(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqadds_s32
  #define vqadds_s32(a, b) easysimd_vqadds_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vqaddd_s64(int64_t a, int64_t b) {
  return easysimd_math_adds_i64(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqaddd_s64
  #define vqaddd_s64(a, b) easysimd_vqaddd_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vqaddb_u8(uint8_t a, uint8_t b) {
  return easysimd_math_adds_u8(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqaddb_u8
  #define vqaddb_u8(a, b) easysimd_vqaddb_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vqaddh_u16(uint16_t a, uint16_t b) {
  return easysimd_math_adds_u16(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqaddh_u16
  #define vqaddh_u16(a, b) easysimd_vqaddh_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vqadds_u32(uint32_t a, uint32_t b) {
  return easysimd_math_adds_u32(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqadds_u32
  #define vqadds_u32(a, b) easysimd_vqadds_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vqaddd_u64(uint64_t a, uint64_t b) {
  return easysimd_math_adds_u64(a, b);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqaddd_u64
  #define vqaddd_u64(a, b) easysimd_vqaddd_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqadd_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_s8(a, b);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_adds_pi8(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      uint8_t au EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint8_t bu EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint8_t ru EASYSIMD_VECTOR(8) = au + bu;

      au = (au >> 7) + INT8_MAX;

      uint8_t m EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddb_s8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_s8
  #define vqadd_s8(a, b) easysimd_vqadd_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vqadd_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_s16(a, b);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_adds_pi16(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      uint16_t au EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint16_t bu EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint16_t ru EASYSIMD_VECTOR(8) = au + bu;

      au = (au >> 15) + INT16_MAX;

      uint16_t m EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddh_s16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_s16
  #define vqadd_s16(a, b) easysimd_vqadd_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vqadd_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_s32(a, b);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      uint32_t au EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint32_t bu EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint32_t ru EASYSIMD_VECTOR(8) = au + bu;

      au = (au >> 31) + INT32_MAX;

      uint32_t m EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqadds_s32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_s32
  #define vqadd_s32(a, b) easysimd_vqadd_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vqadd_s64(easysimd_int64x1_t a, easysimd_int64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_s64(a, b);
  #else
    easysimd_int64x1_private
      r_,
      a_ = easysimd_int64x1_to_private(a),
      b_ = easysimd_int64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      uint64_t au EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint64_t bu EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint64_t ru EASYSIMD_VECTOR(8) = au + bu;

      au = (au >> 63) + INT64_MAX;

      uint64_t m EASYSIMD_VECTOR(8) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddd_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_s64
  #define vqadd_s64(a, b) easysimd_vqadd_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqadd_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_u8(a, b);
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_adds_pu8(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddb_u8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_u8
  #define vqadd_u8(a, b) easysimd_vqadd_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vqadd_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_u16(a, b);
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_adds_pu16(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddh_u16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_u16
  #define vqadd_u16(a, b) easysimd_vqadd_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vqadd_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_u32(a, b);
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqadds_u32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_u32
  #define vqadd_u32(a, b) easysimd_vqadd_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vqadd_u64(easysimd_uint64x1_t a, easysimd_uint64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqadd_u64(a, b);
  #else
    easysimd_uint64x1_private
      r_,
      a_ = easysimd_uint64x1_to_private(a),
      b_ = easysimd_uint64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddd_u64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqadd_u64
  #define vqadd_u64(a, b) easysimd_vqadd_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqaddq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_s8(a, b);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_adds_epi8(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SCALAR)
      uint8_t au EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint8_t bu EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint8_t ru EASYSIMD_VECTOR(16) = au + bu;

      au = (au >> 7) + INT8_MAX;

      uint8_t m EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddb_s8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_s8
  #define vqaddq_s8(a, b) easysimd_vqaddq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vqaddq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_s16(a, b);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_adds_epi16(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SCALAR)
      uint16_t au EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint16_t bu EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint16_t ru EASYSIMD_VECTOR(16) = au + bu;

      au = (au >> 15) + INT16_MAX;

      uint16_t m EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddh_s16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_s16
  #define vqaddq_s16(a, b) easysimd_vqaddq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vqaddq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_s32(a, b);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      /* https://stackoverflow.com/a/56544654/501126 */
      const __m128i int_max = _mm_set1_epi32(INT32_MAX);

      /* normal result (possibly wraps around) */
      const __m128i sum = _mm_add_epi32(a_.m128i, b_.m128i);

      /* If result saturates, it has the same sign as both a and b */
      const __m128i sign_bit = _mm_srli_epi32(a_.m128i, 31); /* shift sign to lowest bit */

      #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
        const __m128i overflow = _mm_ternarylogic_epi32(a_.m128i, b_.m128i, sum, 0x42);
      #else
        const __m128i sign_xor = _mm_xor_si128(a_.m128i, b_.m128i);
        const __m128i overflow = _mm_andnot_si128(sign_xor, _mm_xor_si128(a_.m128i, sum));
      #endif

      #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
        r_.m128i = _mm_mask_add_epi32(sum, _mm_movepi32_mask(overflow), int_max, sign_bit);
      #else
        const __m128i saturated = _mm_add_epi32(int_max, sign_bit);

        #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
          r_.m128i =
            _mm_castps_si128(
              _mm_blendv_ps(
                _mm_castsi128_ps(sum),
                _mm_castsi128_ps(saturated),
                _mm_castsi128_ps(overflow)
              )
            );
        #else
          const __m128i overflow_mask = _mm_srai_epi32(overflow, 31);
          r_.m128i =
            _mm_or_si128(
              _mm_and_si128(overflow_mask, saturated),
              _mm_andnot_si128(overflow_mask, sum)
            );
        #endif
      #endif
    #elif defined(EASYSIMD_VECTOR_SCALAR)
      uint32_t au EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint32_t bu EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint32_t ru EASYSIMD_VECTOR(16) = au + bu;

      au = (au >> 31) + INT32_MAX;

      uint32_t m EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqadds_s32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_s32
  #define vqaddq_s32(a, b) easysimd_vqaddq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vqaddq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_s64(a, b);
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      /* https://stackoverflow.com/a/56544654/501126 */
      const __m128i int_max = _mm_set1_epi64x(INT64_MAX);

      /* normal result (possibly wraps around) */
      const __m128i sum = _mm_add_epi64(a_.m128i, b_.m128i);

      /* If result saturates, it has the same sign as both a and b */
      const __m128i sign_bit = _mm_srli_epi64(a_.m128i, 63); /* shift sign to lowest bit */

      #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
        const __m128i overflow = _mm_ternarylogic_epi64(a_.m128i, b_.m128i, sum, 0x42);
      #else
        const __m128i sign_xor = _mm_xor_si128(a_.m128i, b_.m128i);
        const __m128i overflow = _mm_andnot_si128(sign_xor, _mm_xor_si128(a_.m128i, sum));
      #endif

      #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
        r_.m128i = _mm_mask_add_epi64(sum, _mm_movepi64_mask(overflow), int_max, sign_bit);
      #else
        const __m128i saturated = _mm_add_epi64(int_max, sign_bit);

        r_.m128i =
          _mm_castpd_si128(
            _mm_blendv_pd(
              _mm_castsi128_pd(sum),
              _mm_castsi128_pd(saturated),
              _mm_castsi128_pd(overflow)
            )
          );
      #endif
    #elif defined(EASYSIMD_VECTOR_SCALAR)
      uint64_t au EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.values);
      uint64_t bu EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.values);
      uint64_t ru EASYSIMD_VECTOR(16) = au + bu;

      au = (au >> 63) + INT64_MAX;

      uint64_t m EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddd_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_s64
  #define vqaddq_s64(a, b) easysimd_vqaddq_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vqaddq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_u8(a, b);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_adds_epu8(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddb_u8(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_u8
  #define vqaddq_u8(a, b) easysimd_vqaddq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vqaddq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_u16(a, b);
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);
    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_adds_epu16(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddh_u16(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_u16
  #define vqaddq_u16(a, b) easysimd_vqaddq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vqaddq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_u32(a, b);
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      #if defined(__AVX512VL__)
        __m128i notb = _mm_ternarylogic_epi32(b_.m128i, b_.m128i, b_.m128i, 0x0f);
      #else
        __m128i notb = _mm_xor_si128(b_.m128i, _mm_set1_epi32(~INT32_C(0)));
      #endif
      r_.m128i =
        _mm_add_epi32(
          b_.m128i,
          _mm_min_epu32(
            a_.m128i,
            notb
          )
        );
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      const __m128i sum = _mm_add_epi32(a_.m128i, b_.m128i);
      const __m128i i32min = _mm_set1_epi32(INT32_MIN);
      a_.m128i = _mm_xor_si128(a_.m128i, i32min);
      r_.m128i = _mm_or_si128(_mm_cmpgt_epi32(a_.m128i, _mm_xor_si128(i32min, sum)), sum);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqadds_u32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_u32
  #define vqaddq_u32(a, b) easysimd_vqaddq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vqaddq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqaddq_u64(a, b);
  #else
    easysimd_uint64x2_private
      r_,
      a_ = easysimd_uint64x2_to_private(a),
      b_ = easysimd_uint64x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT)
      r_.values = a_.values + b_.values;
      r_.values |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), r_.values < a_.values);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqaddd_u64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqaddq_u64
  #define vqaddq_u64(a, b) easysimd_vqaddq_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QADD_H) */
