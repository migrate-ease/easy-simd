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

#if !defined(EASYSIMD_ARM_NEON_ADD_H)
#define EASYSIMD_ARM_NEON_ADD_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float16
easysimd_vaddh_f16(easysimd_float16 a, easysimd_float16 b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vaddh_f16(a, b);
  #else
    easysimd_float32 af = easysimd_float16_to_float32(a);
    easysimd_float32 bf = easysimd_float16_to_float32(b);
    return easysimd_float16_from_float32(af + bf);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES)
  #undef vaddh_f16
  #define vaddh_f16(a, b) easysimd_vaddh_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vaddd_s64(int64_t a, int64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vaddd_s64(a, b);
  #else
    return a + b;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddd_s64
  #define vaddd_s64(a, b) easysimd_vaddd_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vaddd_u64(uint64_t a, uint64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vaddd_u64(a, b);
  #else
    return a + b;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddd_u64
  #define vaddd_u64(a, b) easysimd_vaddd_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float16x4_t
easysimd_vadd_f16(easysimd_float16x4_t a, easysimd_float16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vadd_f16(a, b);
  #else
    easysimd_float16x4_private
      r_,
      a_ = easysimd_float16x4_to_private(a),
      b_ = easysimd_float16x4_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_vaddh_f16(a_.values[i], b_.values[i]);
    }

    return easysimd_float16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES)
  #undef vadd_f16
  #define vadd_f16(a, b) easysimd_vadd_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vadd_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_f32(a, b);
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a),
      b_ = easysimd_float32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_f32
  #define vadd_f32(a, b) easysimd_vadd_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vadd_f64(easysimd_float64x1_t a, easysimd_float64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vadd_f64(a, b);
  #else
    easysimd_float64x1_private
      r_,
      a_ = easysimd_float64x1_to_private(a),
      b_ = easysimd_float64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vadd_f64
  #define vadd_f64(a, b) easysimd_vadd_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vadd_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_s8(a, b);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #elif defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_add_pi8(a_.m64, b_.m64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_s8
  #define vadd_s8(a, b) easysimd_vadd_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vadd_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_s16(a, b);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #elif defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_add_pi16(a_.m64, b_.m64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_s16
  #define vadd_s16(a, b) easysimd_vadd_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vadd_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_s32(a, b);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #elif defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_add_pi32(a_.m64, b_.m64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_s32
  #define vadd_s32(a, b) easysimd_vadd_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vadd_s64(easysimd_int64x1_t a, easysimd_int64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_s64(a, b);
  #else
    easysimd_int64x1_private
      r_,
      a_ = easysimd_int64x1_to_private(a),
      b_ = easysimd_int64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_s64
  #define vadd_s64(a, b) easysimd_vadd_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vadd_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_u8(a, b);
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_u8
  #define vadd_u8(a, b) easysimd_vadd_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vadd_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_u16(a, b);
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_u16
  #define vadd_u16(a, b) easysimd_vadd_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vadd_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_u32(a, b);
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_u32
  #define vadd_u32(a, b) easysimd_vadd_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vadd_u64(easysimd_uint64x1_t a, easysimd_uint64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vadd_u64(a, b);
  #else
    easysimd_uint64x1_private
      r_,
      a_ = easysimd_uint64x1_to_private(a),
      b_ = easysimd_uint64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vadd_u64
  #define vadd_u64(a, b) easysimd_vadd_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float16x8_t
easysimd_vaddq_f16(easysimd_float16x8_t a, easysimd_float16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vaddq_f16(a, b);
  #else
    easysimd_float16x8_private
      r_,
      a_ = easysimd_float16x8_to_private(a),
      b_ = easysimd_float16x8_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_vaddh_f16(a_.values[i], b_.values[i]);
    }

    return easysimd_float16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES)
  #undef vaddq_f16
  #define vaddq_f16(a, b) easysimd_vaddq_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vaddq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_f32(a, b);
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128 = _mm_add_ps(a_.m128, b_.m128);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_f32
  #define vaddq_f32(a, b) easysimd_vaddq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vaddq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vaddq_f64(a, b);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128d = _mm_add_pd(a_.m128d, b_.m128d);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddq_f64
  #define vaddq_f64(a, b) easysimd_vaddq_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vaddq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_s8(a, b);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_add_epi8(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_s8
  #define vaddq_s8(a, b) easysimd_vaddq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vaddq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_s16(a, b);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_add_epi16(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_s16
  #define vaddq_s16(a, b) easysimd_vaddq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vaddq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_s32(a, b);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_add_epi32(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_s32
  #define vaddq_s32(a, b) easysimd_vaddq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vaddq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_s64(a, b);
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_add_epi64(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_s64
  #define vaddq_s64(a, b) easysimd_vaddq_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vaddq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_u8(a, b);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_u8
  #define vaddq_u8(a, b) easysimd_vaddq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vaddq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_u16(a, b);
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_u16
  #define vaddq_u16(a, b) easysimd_vaddq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vaddq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_u32(a, b);
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_u32
  #define vaddq_u32(a, b) easysimd_vaddq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vaddq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddq_u64(a, b);
  #else
    easysimd_uint64x2_private
      r_,
      a_ = easysimd_uint64x2_to_private(a),
      b_ = easysimd_uint64x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values + b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] + b_.values[i];
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddq_u64
  #define vaddq_u64(a, b) easysimd_vaddq_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ADD_H) */
