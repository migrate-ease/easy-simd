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

#if !defined(EASYSIMD_ARM_NEON_RND_H)
#define EASYSIMD_ARM_NEON_RND_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vrnd_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
    return vrnd_f32(a);
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_math_truncf(a_.values[i]);
    }

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrnd_f32
  #define vrnd_f32(a) easysimd_vrnd_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vrnd_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrnd_f64(a);
  #else
    easysimd_float64x1_private
      r_,
      a_ = easysimd_float64x1_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_math_trunc(a_.values[i]);
    }

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrnd_f64
  #define vrnd_f64(a) easysimd_vrnd_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vrndq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
    return vrndq_f32(a);
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128 = _mm_round_ps(a_.m128, _MM_FROUND_TO_ZERO);
    #elif defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128 = _mm_trunc_ps(a_.m128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_math_truncf(a_.values[i]);
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrndq_f32
  #define vrndq_f32(a) easysimd_vrndq_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vrndq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrndq_f64(a);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128d = _mm_round_pd(a_.m128d, _MM_FROUND_TO_ZERO);
    #elif defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128d = _mm_trunc_ps(a_.m128d);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_math_trunc(a_.values[i]);
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrndq_f64
  #define vrndq_f64(a) easysimd_vrndq_f64(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_RND_H) */
