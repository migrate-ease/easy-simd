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

#if !defined(EASYSIMD_ARM_NEON_MINV_H)
#define EASYSIMD_ARM_NEON_MINV_H

#include "types.h"
#include <float.h>

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vminv_f32(easysimd_float32x2_t a) {
  easysimd_float32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminv_f32(a);
  #else
    easysimd_float32x2_private a_ = easysimd_float32x2_to_private(a);

    r = EASYSIMD_MATH_INFINITYF;
    #if defined(EASYSIMD_FAST_NANS)
      EASYSIMD_VECTORIZE_REDUCTION(min:r)
    #else
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      #if defined(EASYSIMD_FAST_NANS)
        r = a_.values[i] < r ? a_.values[i] : r;
      #else
        r = (a_.values[i] < r) ? a_.values[i] : ((a_.values[i] >= r) ? r : ((a_.values[i] == a_.values[i]) ? r : a_.values[i]));
      #endif
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminv_f32
  #define vminv_f32(v) easysimd_vminv_f32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vminv_s8(easysimd_int8x8_t a) {
  int8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminv_s8(a);
  #else
    easysimd_int8x8_private a_ = easysimd_int8x8_to_private(a);

    r = INT8_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminv_s8
  #define vminv_s8(v) easysimd_vminv_s8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vminv_s16(easysimd_int16x4_t a) {
  int16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminv_s16(a);
  #else
    easysimd_int16x4_private a_ = easysimd_int16x4_to_private(a);

    r = INT16_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminv_s16
  #define vminv_s16(v) easysimd_vminv_s16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vminv_s32(easysimd_int32x2_t a) {
  int32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminv_s32(a);
  #else
    easysimd_int32x2_private a_ = easysimd_int32x2_to_private(a);

    r = INT32_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminv_s32
  #define vminv_s32(v) easysimd_vminv_s32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vminv_u8(easysimd_uint8x8_t a) {
  uint8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminv_u8(a);
  #else
    easysimd_uint8x8_private a_ = easysimd_uint8x8_to_private(a);

    r = UINT8_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminv_u8
  #define vminv_u8(v) easysimd_vminv_u8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vminv_u16(easysimd_uint16x4_t a) {
  uint16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminv_u16(a);
  #else
    easysimd_uint16x4_private a_ = easysimd_uint16x4_to_private(a);

    r = UINT16_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminv_u16
  #define vminv_u16(v) easysimd_vminv_u16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vminv_u32(easysimd_uint32x2_t a) {
  uint32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminv_u32(a);
  #else
    easysimd_uint32x2_private a_ = easysimd_uint32x2_to_private(a);

    r = UINT32_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminv_u32
  #define vminv_u32(v) easysimd_vminv_u32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vminvq_f32(easysimd_float32x4_t a) {
  easysimd_float32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_f32(a);
  #else
    easysimd_float32x4_private a_ = easysimd_float32x4_to_private(a);

    r = EASYSIMD_MATH_INFINITYF;
    #if defined(EASYSIMD_FAST_NANS)
      EASYSIMD_VECTORIZE_REDUCTION(min:r)
    #else
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      #if defined(EASYSIMD_FAST_NANS)
        r = a_.values[i] < r ? a_.values[i] : r;
      #else
        r = (a_.values[i] < r) ? a_.values[i] : ((a_.values[i] >= r) ? r : ((a_.values[i] == a_.values[i]) ? r : a_.values[i]));
      #endif
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_f32
  #define vminvq_f32(v) easysimd_vminvq_f32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vminvq_f64(easysimd_float64x2_t a) {
  easysimd_float64_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_f64(a);
  #else
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);

    r = EASYSIMD_MATH_INFINITY;
    #if defined(EASYSIMD_FAST_NANS)
      EASYSIMD_VECTORIZE_REDUCTION(min:r)
    #else
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      #if defined(EASYSIMD_FAST_NANS)
        r = a_.values[i] < r ? a_.values[i] : r;
      #else
        r = (a_.values[i] < r) ? a_.values[i] : ((a_.values[i] >= r) ? r : ((a_.values[i] == a_.values[i]) ? r : a_.values[i]));
      #endif
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_f64
  #define vminvq_f64(v) easysimd_vminvq_f64(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vminvq_s8(easysimd_int8x16_t a) {
  int8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_s8(a);
  #else
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);

    r = INT8_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_s8
  #define vminvq_s8(v) easysimd_vminvq_s8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vminvq_s16(easysimd_int16x8_t a) {
  int16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_s16(a);
  #else
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);

    r = INT16_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_s16
  #define vminvq_s16(v) easysimd_vminvq_s16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vminvq_s32(easysimd_int32x4_t a) {
  int32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_s32(a);
  #else
    easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);

    r = INT32_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_s32
  #define vminvq_s32(v) easysimd_vminvq_s32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vminvq_u8(easysimd_uint8x16_t a) {
  uint8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_u8(a);
  #else
    easysimd_uint8x16_private a_ = easysimd_uint8x16_to_private(a);

    r = UINT8_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_u8
  #define vminvq_u8(v) easysimd_vminvq_u8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vminvq_u16(easysimd_uint16x8_t a) {
  uint16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_u16(a);
  #else
    easysimd_uint16x8_private a_ = easysimd_uint16x8_to_private(a);

    r = UINT16_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_u16
  #define vminvq_u16(v) easysimd_vminvq_u16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vminvq_u32(easysimd_uint32x4_t a) {
  uint32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vminvq_u32(a);
  #else
    easysimd_uint32x4_private a_ = easysimd_uint32x4_to_private(a);

    r = UINT32_MAX;
    EASYSIMD_VECTORIZE_REDUCTION(min:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r = a_.values[i] < r ? a_.values[i] : r;
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminvq_u32
  #define vminvq_u32(v) easysimd_vminvq_u32(v)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MINV_H) */
