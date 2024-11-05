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
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_ARM_NEON_CGEZ_H)
#define EASYSIMD_ARM_NEON_CGEZ_H

#include "cge.h"
#include "dup_n.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vcgezd_f64(easysimd_float64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vcgezd_f64(a));
  #else
    return (a >= EASYSIMD_FLOAT64_C(0.0)) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezd_f64
  #define vcgezd_f64(a) easysimd_vcgezd_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vcgezd_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vcgezd_s64(a));
  #else
    return (a >= 0) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezd_s64
  #define vcgezd_s64(a) easysimd_vcgezd_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vcgezs_f32(easysimd_float32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint32_t, vcgezs_f32(a));
  #else
    return (a >= EASYSIMD_FLOAT32_C(0.0)) ? UINT32_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezs_f32
  #define vcgezs_f32(a) easysimd_vcgezs_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vcgezq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgezq_f32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcgeq_f32(a, easysimd_vdupq_n_f32(EASYSIMD_FLOAT32_C(0.0)));
  #else
    easysimd_float32x4_private a_ = easysimd_float32x4_to_private(a);
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= EASYSIMD_FLOAT32_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vcgezs_f32(a_.values[i]);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezq_f32
  #define vcgezq_f32(a) easysimd_vcgezq_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vcgezq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgezq_f64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcgeq_f64(a, easysimd_vdupq_n_f64(EASYSIMD_FLOAT64_C(0.0)));
  #else
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);
    easysimd_uint64x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= EASYSIMD_FLOAT64_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vcgezd_f64(a_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezq_f64
  #define vcgezq_f64(a) easysimd_vcgezq_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vcgezq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgezq_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcgeq_s8(a, easysimd_vdupq_n_s8(0));
  #else
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);
    easysimd_uint8x16_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] >= 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezq_s8
  #define vcgezq_s8(a) easysimd_vcgezq_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vcgezq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgezq_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcgeq_s16(a, easysimd_vdupq_n_s16(0));
  #else
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);
    easysimd_uint16x8_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] >= 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezq_s16
  #define vcgezq_s16(a) easysimd_vcgezq_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vcgezq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgezq_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcgeq_s32(a, easysimd_vdupq_n_s32(0));
  #else
    easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] >= 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezq_s32
  #define vcgezq_s32(a) easysimd_vcgezq_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vcgezq_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgezq_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcgeq_s64(a, easysimd_vdupq_n_s64(0));
  #else
    easysimd_int64x2_private a_ = easysimd_int64x2_to_private(a);
    easysimd_uint64x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vcgezd_s64(a_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgezq_s64
  #define vcgezq_s64(a) easysimd_vcgezq_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vcgez_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgez_f32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcge_f32(a, easysimd_vdup_n_f32(EASYSIMD_FLOAT32_C(0.0)));
  #else
    easysimd_float32x2_private a_ = easysimd_float32x2_to_private(a);
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= EASYSIMD_FLOAT32_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vcgezs_f32(a_.values[i]);
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgez_f32
  #define vcgez_f32(a) easysimd_vcgez_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vcgez_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgez_f64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcge_f64(a, easysimd_vdup_n_f64(EASYSIMD_FLOAT64_C(0.0)));
  #else
    easysimd_float64x1_private a_ = easysimd_float64x1_to_private(a);
    easysimd_uint64x1_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values =  HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= EASYSIMD_FLOAT64_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vcgezd_f64(a_.values[i]);
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgez_f64
  #define vcgez_f64(a) easysimd_vcgez_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vcgez_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgez_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcge_s8(a, easysimd_vdup_n_s8(0));
  #else
    easysimd_int8x8_private a_ = easysimd_int8x8_to_private(a);
    easysimd_uint8x8_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] >= 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgez_s8
  #define vcgez_s8(a) easysimd_vcgez_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vcgez_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgez_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcge_s16(a, easysimd_vdup_n_s16(0));
  #else
    easysimd_int16x4_private a_ = easysimd_int16x4_to_private(a);
    easysimd_uint16x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] >= 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgez_s16
  #define vcgez_s16(a) easysimd_vcgez_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vcgez_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgez_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcge_s32(a, easysimd_vdup_n_s32(0));
  #else
    easysimd_int32x2_private a_ = easysimd_int32x2_to_private(a);
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] >= 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgez_s32
  #define vcgez_s32(a) easysimd_vcgez_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vcgez_s64(easysimd_int64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcgez_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcge_s64(a, easysimd_vdup_n_s64(0));
  #else
    easysimd_int64x1_private a_ = easysimd_int64x1_to_private(a);
    easysimd_uint64x1_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values >= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vcgezd_s64(a_.values[i]);
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcgez_s64
  #define vcgez_s64(a) easysimd_vcgez_s64(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_CGEZ_H) */
