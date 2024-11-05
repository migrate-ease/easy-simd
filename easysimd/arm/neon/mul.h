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

#if !defined(EASYSIMD_ARM_NEON_MUL_H)
#define EASYSIMD_ARM_NEON_MUL_H

#include "types.h"

#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vmul_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmul_f32(a, b);
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a),
      b_ = easysimd_float32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_f32
  #define vmul_f32(a, b) easysimd_vmul_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vmul_f64(easysimd_float64x1_t a, easysimd_float64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmul_f64(a, b);
  #else
    easysimd_float64x1_private
      r_,
      a_ = easysimd_float64x1_to_private(a),
      b_ = easysimd_float64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_f64
  #define vmul_f64(a, b) easysimd_vmul_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vmul_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmul_s8(a, b);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_s8
  #define vmul_s8(a, b) easysimd_vmul_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmul_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmul_s16(a, b);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _m_pmullw(a_.m64, b_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_s16
  #define vmul_s16(a, b) easysimd_vmul_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmul_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmul_s32(a, b);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_s32
  #define vmul_s32(a, b) easysimd_vmul_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_x_vmul_s64(easysimd_int64x1_t a, easysimd_int64x1_t b) {
  easysimd_int64x1_private
    r_,
    a_ = easysimd_int64x1_to_private(a),
    b_ = easysimd_int64x1_to_private(b);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.values = a_.values * b_.values;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] * b_.values[i];
    }
  #endif

  return easysimd_int64x1_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vmul_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmul_u8(a, b);
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_u8
  #define vmul_u8(a, b) easysimd_vmul_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmul_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmul_u16(a, b);
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_u16
  #define vmul_u16(a, b) easysimd_vmul_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmul_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmul_u32(a, b);
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_u32
  #define vmul_u32(a, b) easysimd_vmul_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_x_vmul_u64(easysimd_uint64x1_t a, easysimd_uint64x1_t b) {
  easysimd_uint64x1_private
    r_,
    a_ = easysimd_uint64x1_to_private(a),
    b_ = easysimd_uint64x1_to_private(b);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.values = a_.values * b_.values;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] * b_.values[i];
    }
  #endif

  return easysimd_uint64x1_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vmulq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmulq_f32(a, b);
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128 = _mm_mul_ps(a_.m128, b_.m128);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_f32
  #define vmulq_f32(a, b) easysimd_vmulq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vmulq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmulq_f64(a, b);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128d = _mm_mul_pd(a_.m128d, b_.m128d);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_f64
  #define vmulq_f64(a, b) easysimd_vmulq_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vmulq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmulq_s8(a, b);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      /* https://stackoverflow.com/a/29155682/501126 */
      const __m128i dst_even = _mm_mullo_epi16(a_.m128i, b_.m128i);
      r_.m128i =
        _mm_or_si128(
          _mm_slli_epi16(
            _mm_mullo_epi16(
              _mm_srli_epi16(a_.m128i, 8),
              _mm_srli_epi16(b_.m128i, 8)
            ),
            8
          ),
          #if defined(EASYSIMD_X86_AVX2_NATIVE)
            _mm_and_si128(dst_even, _mm_set1_epi16(0xFF))
          #else
            _mm_srli_epi16(
              _mm_slli_epi16(dst_even, 8),
              8
            )
          #endif
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_s8
  #define vmulq_s8(a, b) easysimd_vmulq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmulq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmulq_s16(a, b);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_mullo_epi16(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_s16
  #define vmulq_s16(a, b) easysimd_vmulq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmulq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmulq_s32(a, b);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = a_.values * b_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] * b_.values[i];
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_s32
  #define vmulq_s32(a, b) easysimd_vmulq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_x_vmulq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  easysimd_int64x2_private
    r_,
    a_ = easysimd_int64x2_to_private(a),
    b_ = easysimd_int64x2_to_private(b);

  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    r_.m128i = _mm_mullo_epi64(a_.m128i, b_.m128i);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.values = a_.values * b_.values;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] * b_.values[i];
    }
  #endif

  return easysimd_int64x2_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vmulq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmulq_u8(a, b);
  #else
    return
      easysimd_vreinterpretq_u8_s8(
        easysimd_vmulq_s8(
          easysimd_vreinterpretq_s8_u8(a),
          easysimd_vreinterpretq_s8_u8(b)
        )
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_u8
  #define vmulq_u8(a, b) easysimd_vmulq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmulq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmulq_u16(a, b);
  #else
    return
      easysimd_vreinterpretq_u16_s16(
        easysimd_vmulq_s16(
          easysimd_vreinterpretq_s16_u16(a),
          easysimd_vreinterpretq_s16_u16(b)
        )
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_u16
  #define vmulq_u16(a, b) easysimd_vmulq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmulq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmulq_u32(a, b);
  #else
    return
      easysimd_vreinterpretq_u32_s32(
        easysimd_vmulq_s32(
          easysimd_vreinterpretq_s32_u32(a),
          easysimd_vreinterpretq_s32_u32(b)
        )
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_u32
  #define vmulq_u32(a, b) easysimd_vmulq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_x_vmulq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  return
    easysimd_vreinterpretq_u64_s64(
      easysimd_x_vmulq_s64(
        easysimd_vreinterpretq_s64_u64(a),
        easysimd_vreinterpretq_s64_u64(b)
      )
    );
}

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MUL_H) */
