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

#if !defined(EASYSIMD_ARM_NEON_ABS_H)
#define EASYSIMD_ARM_NEON_ABS_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vabsd_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(9,1,0))
    return vabsd_s64(a);
  #else
    return a < 0 ? -a : a;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vabsd_s64
  #define vabsd_s64(a) easysimd_vabsd_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vabs_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabs_f32(a);
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
    }

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabs_f32
  #define vabs_f32(a) easysimd_vabs_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vabs_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabs_f64(a);
  #else
    easysimd_float64x1_private
      r_,
      a_ = easysimd_float64x1_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
    }

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vabs_f64
  #define vabs_f64(a) easysimd_vabs_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vabs_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabs_s8(a);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_abs_pi8(a_.m64);
    #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT8_C(0));
      r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabs_s8
  #define vabs_s8(a) easysimd_vabs_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vabs_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabs_s16(a);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_abs_pi16(a_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100761)
      __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT16_C(0));
      r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabs_s16
  #define vabs_s16(a) easysimd_vabs_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vabs_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabs_s32(a);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_abs_pi32(a_.m64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100761)
      __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT32_C(0));
      r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabs_s32
  #define vabs_s32(a) easysimd_vabs_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vabs_s64(easysimd_int64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabs_s64(a);
  #else
    easysimd_int64x1_private
      r_,
      a_ = easysimd_int64x1_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT64_C(0));
      r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabs_s64
  #define vabs_s64(a) easysimd_vabs_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vabsq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabsq_f32(a);
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a);

    #if defined(EASYSIMD_X86_SSE_NATIVE)
      easysimd_float32 mask_;
      uint32_t u32_ = UINT32_C(0x7FFFFFFF);
      easysimd_memcpy(&mask_, &u32_, sizeof(u32_));
      r_.m128 = _mm_and_ps(_mm_set1_ps(mask_), a_.m128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_math_fabsf(a_.values[i]);
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabsq_f32
  #define vabsq_f32(a) easysimd_vabsq_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vabsq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabsq_f64(a);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      easysimd_float64 mask_;
      uint64_t u64_ = UINT64_C(0x7FFFFFFFFFFFFFFF);
      easysimd_memcpy(&mask_, &u64_, sizeof(u64_));
      r_.m128d = _mm_and_pd(_mm_set1_pd(mask_), a_.m128d);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_math_fabs(a_.values[i]);
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vabsq_f64
  #define vabsq_f64(a) easysimd_vabsq_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vabsq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabsq_s8(a);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_abs_epi8(a_.m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_min_epu8(a_.m128i, _mm_sub_epi8(_mm_setzero_si128(), a_.m128i));
    #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT8_C(0));
        r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabsq_s8
  #define vabsq_s8(a) easysimd_vabsq_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vabsq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabsq_s16(a);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_abs_epi16(a_.m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_max_epi16(a_.m128i, _mm_sub_epi16(_mm_setzero_si128(), a_.m128i));
    #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT16_C(0));
      r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabsq_s16
  #define vabsq_s16(a) easysimd_vabsq_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vabsq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabsq_s32(a);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    r_.m128i = _mm_abs_epi32(a_.m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      const __m128i m = _mm_cmpgt_epi32(_mm_setzero_si128(), a_.m128i);
      r_.m128i = _mm_sub_epi32(_mm_xor_si128(a_.m128i, m), m);
    #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT32_C(0));
      r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabsq_s32
  #define vabsq_s32(a) easysimd_vabsq_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vabsq_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabsq_s64(a);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vbslq_s64(vreinterpretq_u64_s64(vshrq_n_s64(a, 63)), vsubq_s64(vdupq_n_s64(0), a), a);
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm_abs_epi64(a_.m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      const __m128i m = _mm_srai_epi32(_mm_shuffle_epi32(a_.m128i, 0xF5), 31);
      r_.m128i = _mm_sub_epi64(_mm_xor_si128(a_.m128i, m), m);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(r_.values) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < INT64_C(0));
      r_.values = (-a_.values & m) | (a_.values & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i] < 0 ? -a_.values[i] : a_.values[i];
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabsq_s64
  #define vabsq_s64(a) easysimd_vabsq_s64(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ABS_H) */
