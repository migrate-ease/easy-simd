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

#if !defined(EASYSIMD_ARM_NEON_UZP2_H)
#define EASYSIMD_ARM_NEON_UZP2_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vuzp2_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2_f32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    float32x2x2_t t = vuzp_f32(a, b);
    return t.val[1];
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a),
      b_ = easysimd_float32x2_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.values, b_.values, 1, 3);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2_f32
  #define vuzp2_f32(a, b) easysimd_vuzp2_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vuzp2_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2_s8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int8x8x2_t t = vuzp_s8(a, b);
    return t.val[1];
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 8, a_.values, b_.values, 1, 3, 5, 7, 9, 11, 13, 15);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2_s8
  #define vuzp2_s8(a, b) easysimd_vuzp2_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vuzp2_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2_s16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x4x2_t t = vuzp_s16(a, b);
    return t.val[1];
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.values, b_.values, 1, 3, 5, 7);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2_s16
  #define vuzp2_s16(a, b) easysimd_vuzp2_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vuzp2_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2_s32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x2x2_t t = vuzp_s32(a, b);
    return t.val[1];
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.values, b_.values, 1, 3);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2_s32
  #define vuzp2_s32(a, b) easysimd_vuzp2_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vuzp2_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2_u8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint8x8x2_t t = vuzp_u8(a, b);
    return t.val[1];
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 8, a_.values, b_.values, 1, 3, 5, 7, 9, 11, 13, 15);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2_u8
  #define vuzp2_u8(a, b) easysimd_vuzp2_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vuzp2_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2_u16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint16x4x2_t t = vuzp_u16(a, b);
    return t.val[1];
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.values, b_.values, 1, 3, 5, 7);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2_u16
  #define vuzp2_u16(a, b) easysimd_vuzp2_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vuzp2_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2_u32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint32x2x2_t t = vuzp_u32(a, b);
    return t.val[1];
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.values, b_.values, 1, 3);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2_u32
  #define vuzp2_u32(a, b) easysimd_vuzp2_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vuzp2q_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_f32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    float32x4x2_t t = vuzpq_f32(a, b);
    return t.val[1];
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128 = _mm_shuffle_ps(a_.m128, b_.m128, 0xdd);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.values, b_.values, 1, 3, 5, 7);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_f32
  #define vuzp2q_f32(a, b) easysimd_vuzp2q_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vuzp2q_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_f64(a, b);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128d = _mm_unpackhi_pd(a_.m128d, b_.m128d);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.values, b_.values, 1, 3);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_f64
  #define vuzp2q_f64(a, b) easysimd_vuzp2q_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vuzp2q_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_s8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int8x16x2_t t = vuzpq_s8(a, b);
    return t.val[1];
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.values, b_.values, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_s8
  #define vuzp2q_s8(a, b) easysimd_vuzp2q_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vuzp2q_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_s16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8x2_t t = vuzpq_s16(a, b);
    return t.val[1];
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.values, b_.values, 1, 3, 5, 7, 9, 11, 13, 15);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_s16
  #define vuzp2q_s16(a, b) easysimd_vuzp2q_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vuzp2q_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_s32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4x2_t t = vuzpq_s32(a, b);
    return t.val[1];
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(a_.m128i), _mm_castsi128_ps(b_.m128i), 0xdd));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.values, b_.values, 1, 3, 5, 7);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_s32
  #define vuzp2q_s32(a, b) easysimd_vuzp2q_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vuzp2q_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_s64(a, b);
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_unpackhi_epi64(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.values, b_.values, 1, 3);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_s64
  #define vuzp2q_s64(a, b) easysimd_vuzp2q_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vuzp2q_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_u8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint8x16x2_t t = vuzpq_u8(a, b);
    return t.val[1];
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.values, b_.values, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_u8
  #define vuzp2q_u8(a, b) easysimd_vuzp2q_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vuzp2q_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_u16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint16x8x2_t t = vuzpq_u16(a, b);
    return t.val[1];
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.values, b_.values, 1, 3, 5, 7, 9, 11, 13, 15);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_u16
  #define vuzp2q_u16(a, b) easysimd_vuzp2q_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vuzp2q_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_u32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint32x4x2_t t = vuzpq_u32(a, b);
    return t.val[1];
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(a_.m128i), _mm_castsi128_ps(b_.m128i), 0xdd));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.values, b_.values, 1, 3, 5, 7);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_u32
  #define vuzp2q_u32(a, b) easysimd_vuzp2q_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vuzp2q_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vuzp2q_u64(a, b);
  #else
    easysimd_uint64x2_private
      r_,
      a_ = easysimd_uint64x2_to_private(a),
      b_ = easysimd_uint64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_unpackhi_epi64(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.values, b_.values, 1, 3);
    #else
      const size_t halfway_point = sizeof(r_.values) / sizeof(r_.values[0]) / 2;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway_point ; i++) {
        const size_t idx = i << 1;
        r_.values[        i        ] = a_.values[idx | 1];
        r_.values[i + halfway_point] = b_.values[idx | 1];
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vuzp2q_u64
  #define vuzp2q_u64(a, b) easysimd_vuzp2q_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_UZP2_H) */
