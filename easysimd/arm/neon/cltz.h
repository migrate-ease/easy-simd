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

/* TODO: float fallbacks should use vclt(a, vdup_n(0.0)) */

#if !defined(EASYSIMD_ARM_NEON_CLTZ_H)
#define EASYSIMD_ARM_NEON_CLTZ_H

#include "types.h"
#include "shr_n.h"
#include "reinterpret.h"
#include "clt.h"
#include "dup_n.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vcltzd_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vcltzd_s64(a));
  #else
    return (a < 0) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcltzd_s64
  #define vcltzd_s64(a) easysimd_vcltzd_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vcltzd_f64(easysimd_float64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vcltzd_f64(a));
  #else
    return (a < EASYSIMD_FLOAT64_C(0.0)) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcltzd_f64
  #define vcltzd_f64(a) easysimd_vcltzd_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vcltzs_f32(easysimd_float32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint32_t, vcltzs_f32(a));
  #else
    return (a < EASYSIMD_FLOAT32_C(0.0)) ? UINT32_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcltzs_f32
  #define vcltzs_f32(a) easysimd_vcltzs_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vcltz_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltz_f32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vclt_f32(a, easysimd_vdup_n_f32(EASYSIMD_FLOAT32_C(0.0)));
  #else
    easysimd_float32x2_private a_ = easysimd_float32x2_to_private(a);
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < EASYSIMD_FLOAT32_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] < EASYSIMD_FLOAT32_C(0.0)) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltz_f32
  #define vcltz_f32(a) easysimd_vcltz_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vcltz_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltz_f64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vclt_f64(a, easysimd_vdup_n_f64(EASYSIMD_FLOAT64_C(0.0)));
  #else
    easysimd_float64x1_private a_ = easysimd_float64x1_to_private(a);
    easysimd_uint64x1_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < EASYSIMD_FLOAT64_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] < EASYSIMD_FLOAT64_C(0.0)) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcltz_f64
  #define vcltz_f64(a) easysimd_vcltz_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vcltz_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltz_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vclt_s8(a, easysimd_vdup_n_s8(0));
  #else
    return easysimd_vreinterpret_u8_s8(easysimd_vshr_n_s8(a, 7));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltz_s8
  #define vcltz_s8(a) easysimd_vcltz_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vcltz_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltz_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vclt_s16(a, easysimd_vdup_n_s16(0));
  #else
    return easysimd_vreinterpret_u16_s16(easysimd_vshr_n_s16(a, 15));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltz_s16
  #define vcltz_s16(a) easysimd_vcltz_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vcltz_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltz_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vclt_s32(a, easysimd_vdup_n_s32(0));
  #else
    return easysimd_vreinterpret_u32_s32(easysimd_vshr_n_s32(a, 31));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltz_s32
  #define vcltz_s32(a) easysimd_vcltz_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vcltz_s64(easysimd_int64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltz_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vclt_s64(a, easysimd_vdup_n_s64(0));
  #else
    return easysimd_vreinterpret_u64_s64(easysimd_vshr_n_s64(a, 63));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltz_s64
  #define vcltz_s64(a) easysimd_vcltz_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vcltzq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltzq_f32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcltq_f32(a, easysimd_vdupq_n_f32(EASYSIMD_FLOAT32_C(0.0)));
  #else
    easysimd_float32x4_private a_ = easysimd_float32x4_to_private(a);
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < EASYSIMD_FLOAT32_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] < EASYSIMD_FLOAT32_C(0.0)) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltzq_f32
  #define vcltzq_f32(a) easysimd_vcltzq_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vcltzq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltzq_f64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcltq_f64(a, easysimd_vdupq_n_f64(EASYSIMD_FLOAT64_C(0.0)));
  #else
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);
    easysimd_uint64x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values < EASYSIMD_FLOAT64_C(0.0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (a_.values[i] < EASYSIMD_FLOAT64_C(0.0)) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vcltzq_f64
  #define vcltzq_f64(a) easysimd_vcltzq_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vcltzq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltzq_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcltq_s8(a, easysimd_vdupq_n_s8(0));
  #else
    return easysimd_vreinterpretq_u8_s8(easysimd_vshrq_n_s8(a, 7));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltzq_s8
  #define vcltzq_s8(a) easysimd_vcltzq_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vcltzq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltzq_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcltq_s16(a, easysimd_vdupq_n_s16(0));
  #else
    return easysimd_vreinterpretq_u16_s16(easysimd_vshrq_n_s16(a, 15));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltzq_s16
  #define vcltzq_s16(a) easysimd_vcltzq_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vcltzq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltzq_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcltq_s32(a, easysimd_vdupq_n_s32(0));
  #else
    return easysimd_vreinterpretq_u32_s32(easysimd_vshrq_n_s32(a, 31));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltzq_s32
  #define vcltzq_s32(a) easysimd_vcltzq_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vcltzq_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vcltzq_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vcltq_s64(a, easysimd_vdupq_n_s64(0));
  #else
    return easysimd_vreinterpretq_u64_s64(easysimd_vshrq_n_s64(a, 63));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcltzq_s64
  #define vcltzq_s64(a) easysimd_vcltzq_s64(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_CLTZ_H) */
