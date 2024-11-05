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

#if !defined(EASYSIMD_ARM_NEON_CLEZ_H)
#define EASYSIMD_ARM_NEON_CLEZ_H

#include "cle.h"
#include "dup_n.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vclezd_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vclezd_s64(a));
  #else
    return (a <= 0) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezd_s64
  #define vclezd_s64(a) easysimd_vclezd_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vclezd_f64(easysimd_float64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vclezd_f64(a));
  #else
    return (a <= EASYSIMD_FLOAT64_C(0.0)) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezd_f64
  #define vclezd_f64(a) easysimd_vclezd_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vclezs_f32(easysimd_float32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint32_t, vclezs_f32(a));
  #else
    return (a <= EASYSIMD_FLOAT32_C(0.0)) ? UINT32_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezs_f32
  #define vclezs_f32(a) easysimd_vclezs_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vclezq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclezq_f32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcleq_f32(a, easysimd_vdupq_n_f32(EASYSIMD_FLOAT32_C(0.0)));
  #else
    easysimd_float32x4_private a_ = easysimd_float32x4_to_private(a);
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= EASYSIMD_FLOAT32_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= EASYSIMD_FLOAT32_C(0.0)) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezq_f32
  #define vclezq_f32(a) easysimd_vclezq_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vclezq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclezq_f64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcleq_f64(a, easysimd_vdupq_n_f64(EASYSIMD_FLOAT64_C(0.0)));
  #else
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);
    easysimd_uint64x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= EASYSIMD_FLOAT64_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= EASYSIMD_FLOAT64_C(0.0)) ? UINT64_MAX : 0;
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezq_f64
  #define vclezq_f64(a) easysimd_vclezq_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vclezq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclezq_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcleq_s8(a, easysimd_vdupq_n_s8(0));
  #else
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);
    easysimd_uint8x16_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezq_s8
  #define vclezq_s8(a) easysimd_vclezq_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vclezq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclezq_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcleq_s16(a, easysimd_vdupq_n_s16(0));
  #else
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);
    easysimd_uint16x8_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezq_s16
  #define vclezq_s16(a) easysimd_vclezq_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vclezq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclezq_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcleq_s32(a, easysimd_vdupq_n_s32(0));
  #else
    easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezq_s32
  #define vclezq_s32(a) easysimd_vclezq_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vclezq_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclezq_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcleq_s64(a, easysimd_vdupq_n_s64(0));
  #else
    easysimd_int64x2_private a_ = easysimd_int64x2_to_private(a);
    easysimd_uint64x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT64_MAX : 0;
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclezq_s64
  #define vclezq_s64(a) easysimd_vclezq_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vclez_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclez_f32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcle_f32(a, easysimd_vdup_n_f32(EASYSIMD_FLOAT32_C(0.0)));
  #else
    easysimd_float32x2_private a_ = easysimd_float32x2_to_private(a);
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= EASYSIMD_FLOAT32_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= EASYSIMD_FLOAT32_C(0.0)) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclez_f32
  #define vclez_f32(a) easysimd_vclez_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vclez_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclez_f64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcle_f64(a, easysimd_vdup_n_f64(EASYSIMD_FLOAT64_C(0.0)));
  #else
    easysimd_float64x1_private a_ = easysimd_float64x1_to_private(a);
    easysimd_uint64x1_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values =  HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= EASYSIMD_FLOAT64_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= EASYSIMD_FLOAT64_C(0.0)) ? UINT64_MAX : 0;
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclez_f64
  #define vclez_f64(a) easysimd_vclez_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vclez_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclez_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcle_s8(a, easysimd_vdup_n_s8(0));
  #else
    easysimd_int8x8_private a_ = easysimd_int8x8_to_private(a);
    easysimd_uint8x8_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclez_s8
  #define vclez_s8(a) easysimd_vclez_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vclez_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclez_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcle_s16(a, easysimd_vdup_n_s16(0));
  #else
    easysimd_int16x4_private a_ = easysimd_int16x4_to_private(a);
    easysimd_uint16x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclez_s16
  #define vclez_s16(a) easysimd_vclez_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vclez_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclez_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcle_s32(a, easysimd_vdup_n_s32(0));
  #else
    easysimd_int32x2_private a_ = easysimd_int32x2_to_private(a);
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclez_s32
  #define vclez_s32(a) easysimd_vclez_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vclez_s64(easysimd_int64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vclez_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcle_s64(a, easysimd_vdup_n_s64(0));
  #else
    easysimd_int64x1_private a_ = easysimd_int64x1_to_private(a);
    easysimd_uint64x1_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values <= 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] <= 0) ? UINT64_MAX : 0;
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vclez_s64
  #define vclez_s64(a) easysimd_vclez_s64(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_CLEZ_H) */
