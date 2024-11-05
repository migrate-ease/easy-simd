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
 *   2021      Zhi An Ng <zhin@google.com> (Copyright owned by Google, LLC)
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_NEON_RSQRTS_H)
#define EASYSIMD_ARM_NEON_RSQRTS_H

#include "types.h"
#include "mls.h"
#include "mul_n.h"
#include "dup_n.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vrsqrtss_f32(easysimd_float32_t a, easysimd_float32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrtss_f32(a, b);
  #else
    return EASYSIMD_FLOAT32_C(0.5) * (EASYSIMD_FLOAT32_C(3.0) - (a * b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrtss_f32
  #define vrsqrtss_f32(a, b) easysimd_vrsqrtss_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vrsqrtsd_f64(easysimd_float64_t a, easysimd_float64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrtsd_f64(a, b);
  #else
    return EASYSIMD_FLOAT64_C(0.5) * (EASYSIMD_FLOAT64_C(3.0) - (a * b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrtsd_f64
  #define vrsqrtsd_f64(a, b) easysimd_vrsqrtsd_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vrsqrts_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrsqrts_f32(a, b);
  #else
    return
      easysimd_vmul_n_f32(
        easysimd_vmls_f32(
          easysimd_vdup_n_f32(EASYSIMD_FLOAT32_C(3.0)),
          a,
          b),
        EASYSIMD_FLOAT32_C(0.5)
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrsqrts_f32
  #define vrsqrts_f32(a, b) easysimd_vrsqrts_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vrsqrts_f64(easysimd_float64x1_t a, easysimd_float64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrts_f64(a, b);
  #else
    return
      easysimd_vmul_n_f64(
        easysimd_vmls_f64(
          easysimd_vdup_n_f64(EASYSIMD_FLOAT64_C(3.0)),
          a,
          b),
        EASYSIMD_FLOAT64_C(0.5)
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrts_f64
  #define vrsqrts_f64(a, b) easysimd_vrsqrts_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vrsqrtsq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrsqrtsq_f32(a, b);
  #else
    return
      easysimd_vmulq_n_f32(
        easysimd_vmlsq_f32(
          easysimd_vdupq_n_f32(EASYSIMD_FLOAT32_C(3.0)),
          a,
          b),
        EASYSIMD_FLOAT32_C(0.5)
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrsqrtsq_f32
  #define vrsqrtsq_f32(a, b) easysimd_vrsqrtsq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vrsqrtsq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrtsq_f64(a, b);
  #else
    return
      easysimd_vmulq_n_f64(
        easysimd_vmlsq_f64(
          easysimd_vdupq_n_f64(EASYSIMD_FLOAT64_C(3.0)),
          a,
          b),
        EASYSIMD_FLOAT64_C(0.5)
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrtsq_f64
  #define vrsqrtsq_f64(a, b) easysimd_vrsqrtsq_f64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP
#endif /* !defined(EASYSIMD_ARM_NEON_RSQRTS_H) */
