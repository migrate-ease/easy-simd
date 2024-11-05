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
 *   2021      Zhi An Ng <zhin@google.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_RECPE_H)
#define EASYSIMD_ARM_NEON_RECPE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vrecpes_f32(easysimd_float32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrecpes_f32(a);
  #else
    return EASYSIMD_FLOAT32_C(1.0) / a;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrecpes_f32
  #define vrecpes_f32(a) easysimd_vrecpes_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vrecped_f64(easysimd_float64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrecped_f64(a);
  #else
    return EASYSIMD_FLOAT64_C(1.0) / a;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrecped_f64
  #define vrecped_f64(a) easysimd_vrecped_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vrecpe_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrecpe_f32(a);
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a);

    #if defined(EASYSIMD_IEEE754_STORAGE)
      /* https://stackoverflow.com/questions/12227126/division-as-multiply-and-lut-fast-float-division-reciprocal/12228234#12228234 */
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        int32_t ix;
        easysimd_float32 fx = a_.values[i];
        easysimd_memcpy(&ix, &fx, sizeof(ix));
        int32_t x = INT32_C(0x7EF311C3) - ix;
        easysimd_float32 temp;
        easysimd_memcpy(&temp, &x, sizeof(temp));
        r_.values[i] = temp * (EASYSIMD_FLOAT32_C(2.0) - temp * fx);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.f32 = 1.0f / a_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.values[i] = easysimd_vrecpes_f32(a_.values[i]);
      }
    #endif

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrecpe_f32
  #define vrecpe_f32(a) easysimd_vrecpe_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vrecpe_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrecpe_f64(a);
  #else
    easysimd_float64x1_private
      r_,
      a_ = easysimd_float64x1_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = 1.0 / a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vrecped_f64(a_.values[i]);
      }
    #endif

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrecpe_f64
  #define vrecpe_f64(a) easysimd_vrecpe_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vrecpeq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrecpeq_f64(a);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = 1.0 / a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vrecped_f64(a_.values[i]);
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrecpeq_f64
  #define vrecpeq_f64(a) easysimd_vrecpeq_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vrecpeq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrecpeq_f32(a);
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a);

    #if defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128 = _mm_rcp_ps(a_.m128);
    #elif defined(EASYSIMD_IEEE754_STORAGE)
      /* https://stackoverflow.com/questions/12227126/division-as-multiply-and-lut-fast-float-division-reciprocal/12228234#12228234 */
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        int32_t ix;
        easysimd_float32 fx = a_.values[i];
        easysimd_memcpy(&ix, &fx, sizeof(ix));
        int32_t x = INT32_C(0x7EF311C3) - ix;
        easysimd_float32 temp;
        easysimd_memcpy(&temp, &x, sizeof(temp));
        r_.values[i] = temp * (EASYSIMD_FLOAT32_C(2.0) - temp * fx);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.f32 = 1.0f / a_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.values[i] = easysimd_vrecpes_f32(a_.values[i]);
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrecpeq_f32
  #define vrecpeq_f32(a) easysimd_vrecpeq_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vrecpe_u32(easysimd_uint32x2_t a){
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrecpe_u32(a);
  #else
    easysimd_uint32x2_private
      a_ = easysimd_uint32x2_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for(size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      if(a_.values[i] <= 0x7FFFFFFF){
        r_.values[i] = UINT32_MAX;
      } else {
        uint32_t a_temp = (a_.values[i] >> 23) & 511;
        a_temp = a_temp * 2 + 1;
        uint32_t b = (1 << 19) / a_temp;
        r_.values[i] = (b+1) / 2;
        r_.values[i] = r_.values[i] << 23;
      }
    }

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrecpe_u32
  #define vrecpe_u32(a) easysimd_vrecpe_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vrecpeq_u32(easysimd_uint32x4_t a){
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrecpeq_u32(a);
  #else
    easysimd_uint32x4_private
      a_ = easysimd_uint32x4_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for(size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      if(a_.values[i] <= 0x7FFFFFFF){
        r_.values[i] = UINT32_MAX;
      } else {
        uint32_t a_temp = (a_.values[i] >> 23) & 511;
        a_temp = a_temp * 2 + 1;
        uint32_t b = (1 << 19) / a_temp;
        r_.values[i] = (b+1) / 2;
        r_.values[i] = r_.values[i] << 23;
      }
    }

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrecpeq_u32
  #define vrecpeq_u32(a) easysimd_vrecpeq_u32((a))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP
#endif /* !defined(EASYSIMD_ARM_NEON_RECPE_H) */
