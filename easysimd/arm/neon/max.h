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

#if !defined(EASYSIMD_ARM_NEON_MAX_H)
#define EASYSIMD_ARM_NEON_MAX_H

#include "types.h"
#include "cgt.h"
#include "bsl.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vmax_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmax_f32(a, b);
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a),
      b_ = easysimd_float32x2_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      #if !defined(EASYSIMD_FAST_NANS)
        r_.values[i] = (a_.values[i] >= b_.values[i]) ? a_.values[i] : ((a_.values[i] < b_.values[i]) ? b_.values[i] : EASYSIMD_MATH_NANF);
      #else
        r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
      #endif
    }

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmax_f32
  #define vmax_f32(a, b) easysimd_vmax_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vmax_f64(easysimd_float64x1_t a, easysimd_float64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmax_f64(a, b);
  #else
    easysimd_float64x1_private
      r_,
      a_ = easysimd_float64x1_to_private(a),
      b_ = easysimd_float64x1_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      #if !defined(EASYSIMD_FAST_NANS)
        r_.values[i] = (a_.values[i] >= b_.values[i]) ? a_.values[i] : ((a_.values[i] < b_.values[i]) ? b_.values[i] : EASYSIMD_MATH_NAN);
      #else
        r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
      #endif
    }

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmax_f64
  #define vmax_f64(a, b) easysimd_vmax_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vmax_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmax_s8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vbsl_s8(easysimd_vcgt_s8(a, b), a, b);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
    }

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmax_s8
  #define vmax_s8(a, b) easysimd_vmax_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmax_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmax_s16(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vbsl_s16(easysimd_vcgt_s16(a, b), a, b);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
    }

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmax_s16
  #define vmax_s16(a, b) easysimd_vmax_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmax_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmax_s32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vbsl_s32(easysimd_vcgt_s32(a, b), a, b);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
    }

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmax_s32
  #define vmax_s32(a, b) easysimd_vmax_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_x_vmax_s64(easysimd_int64x1_t a, easysimd_int64x1_t b) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vbsl_s64(easysimd_vcgt_s64(a, b), a, b);
  #else
    easysimd_int64x1_private
      r_,
      a_ = easysimd_int64x1_to_private(a),
      b_ = easysimd_int64x1_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
    }

    return easysimd_int64x1_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vmax_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmax_u8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vbsl_u8(easysimd_vcgt_u8(a, b), a, b);
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
    }

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmax_u8
  #define vmax_u8(a, b) easysimd_vmax_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmax_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmax_u16(a, b);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && !defined(EASYSIMD_X86_SSE2_NATIVE)
    return easysimd_vbsl_u16(easysimd_vcgt_u16(a, b), a, b);
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      /* https://github.com/simd-everywhere/simde/issues/855#issuecomment-881656284 */
      r_.m64 = _mm_add_pi16(b_.m64, _mm_subs_pu16(a_.m64, b_.m64));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmax_u16
  #define vmax_u16(a, b) easysimd_vmax_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmax_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmax_u32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vbsl_u32(easysimd_vcgt_u32(a, b), a, b);
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
    }

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmax_u32
  #define vmax_u32(a, b) easysimd_vmax_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_x_vmax_u64(easysimd_uint64x1_t a, easysimd_uint64x1_t b) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vbsl_u64(easysimd_vcgt_u64(a, b), a, b);
  #else
    easysimd_uint64x1_private
      r_,
      a_ = easysimd_uint64x1_to_private(a),
      b_ = easysimd_uint64x1_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
    }

    return easysimd_uint64x1_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vmaxq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmaxq_f32(a, b);
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_FAST_NANS)
      r_.m128 = _mm_max_ps(a_.m128, b_.m128);
    #elif defined(EASYSIMD_X86_SSE_NATIVE)
      __m128 m = _mm_or_ps(_mm_cmpneq_ps(a_.m128, a_.m128), _mm_cmpgt_ps(a_.m128, b_.m128));
      #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
        r_.m128 = _mm_blendv_ps(b_.m128, a_.m128, m);
      #else
        r_.m128 =
          _mm_or_ps(
            _mm_and_ps(m, a_.m128),
            _mm_andnot_ps(m, b_.m128)
          );
      #endif
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        #if !defined(EASYSIMD_FAST_NANS)
          r_.values[i] = (a_.values[i] >= b_.values[i]) ? a_.values[i] : ((a_.values[i] < b_.values[i]) ? b_.values[i] : EASYSIMD_MATH_NANF);
        #else
          r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
        #endif
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_f32
  #define vmaxq_f32(a, b) easysimd_vmaxq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vmaxq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmaxq_f64(a, b);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_FAST_NANS)
      r_.m128d = _mm_max_pd(a_.m128d, b_.m128d);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      __m128d m = _mm_or_pd(_mm_cmpneq_pd(a_.m128d, a_.m128d), _mm_cmpgt_pd(a_.m128d, b_.m128d));
      #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
        r_.m128d = _mm_blendv_pd(b_.m128d, a_.m128d, m);
      #else
        r_.m128d =
          _mm_or_pd(
            _mm_and_pd(m, a_.m128d),
            _mm_andnot_pd(m, b_.m128d)
          );
      #endif
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        #if !defined(EASYSIMD_FAST_NANS)
          r_.values[i] = (a_.values[i] >= b_.values[i]) ? a_.values[i] : ((a_.values[i] < b_.values[i]) ? b_.values[i] : EASYSIMD_MATH_NAN);
        #else
          r_.values[i] = (a_.values[i] > b_.values[i]) ? a_.values[i] : b_.values[i];
        #endif
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_f64
  #define vmaxq_f64(a, b) easysimd_vmaxq_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vmaxq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmaxq_s8(a, b);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128i = _mm_max_epi8(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      __m128i m = _mm_cmpgt_epi8(a_.m128i, b_.m128i);
      r_.m128i = _mm_or_si128(_mm_and_si128(m, a_.m128i), _mm_andnot_si128(m, b_.m128i));
    #endif

    return easysimd_int8x16_from_private(r_);
  #else
    return easysimd_vbslq_s8(easysimd_vcgtq_s8(a, b), a, b);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_s8
  #define vmaxq_s8(a, b) easysimd_vmaxq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmaxq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmaxq_s16(a, b);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

      r_.m128i = _mm_max_epi16(a_.m128i, b_.m128i);

    return easysimd_int16x8_from_private(r_);
  #else
    return easysimd_vbslq_s16(easysimd_vcgtq_s16(a, b), a, b);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_s16
  #define vmaxq_s16(a, b) easysimd_vmaxq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmaxq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmaxq_s32(a, b);
  #elif \
      defined(EASYSIMD_X86_SSE4_1_NATIVE)
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);
      r_.m128i = _mm_max_epi32(a_.m128i, b_.m128i);

    return easysimd_int32x4_from_private(r_);
  #else
    return easysimd_vbslq_s32(easysimd_vcgtq_s32(a, b), a, b);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_s32
  #define vmaxq_s32(a, b) easysimd_vmaxq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_x_vmaxq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
    return easysimd_vbslq_s64(easysimd_vcgtq_s64(a, b), a, b);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vmaxq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmaxq_u8(a, b);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);
      r_.m128i = _mm_max_epu8(a_.m128i, b_.m128i);

    return easysimd_uint8x16_from_private(r_);
  #else
    return easysimd_vbslq_u8(easysimd_vcgtq_u8(a, b), a, b);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_u8
  #define vmaxq_u8(a, b) easysimd_vmaxq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmaxq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmaxq_u16(a, b);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE) 
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128i = _mm_max_epu16(a_.m128i, b_.m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      /* https://github.com/simd-everywhere/simde/issues/855#issuecomment-881656284 */
      r_.m128i = _mm_add_epi16(b_.m128i, _mm_subs_epu16(a_.m128i, b_.m128i));
    #endif

    return easysimd_uint16x8_from_private(r_);
  #else
    return easysimd_vbslq_u16(easysimd_vcgtq_u16(a, b), a, b);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_u16
  #define vmaxq_u16(a, b) easysimd_vmaxq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmaxq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmaxq_u32(a, b);
  #elif \
      defined(EASYSIMD_X86_SSE4_1_NATIVE) 
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);
      r_.m128i = _mm_max_epu32(a_.m128i, b_.m128i);

    return easysimd_uint32x4_from_private(r_);
  #else
    return easysimd_vbslq_u32(easysimd_vcgtq_u32(a, b), a, b);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmaxq_u32
  #define vmaxq_u32(a, b) easysimd_vmaxq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_x_vmaxq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
    return easysimd_vbslq_u64(easysimd_vcgtq_u64(a, b), a, b);
}

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MAX_H) */
