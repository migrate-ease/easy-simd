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
*   2021      Evan Nemerson <evan@nemerson.com>
*/

#if !defined(EASYSIMD_ARM_NEON_FMA_N_H)
#define EASYSIMD_ARM_NEON_FMA_N_H

#include "types.h"
#include "dup_n.h"
#include "fma.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vfma_n_f32(easysimd_float32x2_t a, easysimd_float32x2_t b, easysimd_float32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA) && (!defined(__clang__) || EASYSIMD_DETECT_CLANG_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_GCC_95399)
    return vfma_n_f32(a, b, c);
  #else
    return easysimd_vfma_f32(a, b, easysimd_vdup_n_f32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfma_n_f32
  #define vfma_n_f32(a, b, c) easysimd_vfma_n_f32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vfma_n_f64(easysimd_float64x1_t a, easysimd_float64x1_t b, easysimd_float64_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA) && (!defined(__clang__) || EASYSIMD_DETECT_CLANG_VERSION_CHECK(7,0,0))
    return vfma_n_f64(a, b, c);
  #else
    return easysimd_vfma_f64(a, b, easysimd_vdup_n_f64(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfma_n_f64
  #define vfma_n_f64(a, b, c) easysimd_vfma_n_f64(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vfmaq_n_f32(easysimd_float32x4_t a, easysimd_float32x4_t b, easysimd_float32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA) && (!defined(__clang__) || EASYSIMD_DETECT_CLANG_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_GCC_95399)
    return vfmaq_n_f32(a, b, c);
  #else
    return easysimd_vfmaq_f32(a, b, easysimd_vdupq_n_f32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_n_f32
  #define vfmaq_n_f32(a, b, c) easysimd_vfmaq_n_f32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vfmaq_n_f64(easysimd_float64x2_t a, easysimd_float64x2_t b, easysimd_float64_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA) && (!defined(__clang__) || EASYSIMD_DETECT_CLANG_VERSION_CHECK(7,0,0))
    return vfmaq_n_f64(a, b, c);
  #else
    return easysimd_vfmaq_f64(a, b, easysimd_vdupq_n_f64(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_n_f64
  #define vfmaq_n_f64(a, b, c) easysimd_vfmaq_n_f64(a, b, c)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_CMLA_H) */
