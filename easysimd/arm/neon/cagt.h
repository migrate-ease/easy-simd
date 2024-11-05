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

#if !defined(EASYSIMD_ARM_NEON_CAGT_H)
#define EASYSIMD_ARM_NEON_CAGT_H

#include "types.h"
#include "abs.h"
#include "cgt.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vcagth_f16(easysimd_float16_t a, easysimd_float16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vcagth_f16(a, b);
  #else
    easysimd_float32_t
      af = easysimd_float16_to_float32(a),
      bf = easysimd_float16_to_float32(b);
    return (easysimd_math_fabsf(af) > easysimd_math_fabsf(bf)) ? UINT16_MAX : UINT16_C(0);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcagth_f16
  #define vcagth_f16(a, b) easysimd_vcagth_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vcagts_f32(easysimd_float32_t a, easysimd_float32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcagts_f32(a, b);
  #else
    return (easysimd_math_fabsf(a) > easysimd_math_fabsf(b)) ? ~UINT32_C(0) : UINT32_C(0);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcagts_f32
  #define vcagts_f32(a, b) easysimd_vcagts_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vcagtd_f64(easysimd_float64_t a, easysimd_float64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcagtd_f64(a, b);
  #else
    return (easysimd_math_fabs(a) > easysimd_math_fabs(b)) ? ~UINT64_C(0) : UINT64_C(0);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcagtd_f64
  #define vcagtd_f64(a, b) easysimd_vcagtd_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vcagt_f16(easysimd_float16x4_t a, easysimd_float16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vcagt_f16(a, b);
  #else
    easysimd_uint16x4_private r_;
    easysimd_float16x4_private
      a_ = easysimd_float16x4_to_private(a),
      b_ = easysimd_float16x4_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_vcagth_f16(a_.values[i], b_.values[i]);
    }

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES)
  #undef vcagt_f16
  #define vcagt_f16(a, b) easysimd_vcagt_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vcagt_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vcagt_f32(a, b);
  #else
    return easysimd_vcgt_f32(easysimd_vabs_f32(a), easysimd_vabs_f32(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcagt_f32
  #define vcagt_f32(a, b) easysimd_vcagt_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vcagt_f64(easysimd_float64x1_t a, easysimd_float64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcagt_f64(a, b);
  #else
    return easysimd_vcgt_f64(easysimd_vabs_f64(a), easysimd_vabs_f64(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcagt_f64
  #define vcagt_f64(a, b) easysimd_vcagt_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vcagtq_f16(easysimd_float16x8_t a, easysimd_float16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vcagtq_f16(a, b);
  #else
    easysimd_uint16x8_private r_;
    easysimd_float16x8_private
      a_ = easysimd_float16x8_to_private(a),
      b_ = easysimd_float16x8_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_vcagth_f16(a_.values[i], b_.values[i]);
    }

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES)
  #undef vcagtq_f16
  #define vcagtq_f16(a, b) easysimd_vcagtq_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vcagtq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vcagtq_f32(a, b);
  #else
    return easysimd_vcgtq_f32(easysimd_vabsq_f32(a), easysimd_vabsq_f32(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcagtq_f32
  #define vcagtq_f32(a, b) easysimd_vcagtq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vcagtq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcagtq_f64(a, b);
  #else
    return easysimd_vcgtq_f64(easysimd_vabsq_f64(a), easysimd_vabsq_f64(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcagtq_f64
  #define vcagtq_f64(a, b) easysimd_vcagtq_f64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_CAGT_H) */
