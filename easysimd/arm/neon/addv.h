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

#if !defined(EASYSIMD_ARM_NEON_ADDV_H)
#define EASYSIMD_ARM_NEON_ADDV_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vaddv_f32(easysimd_float32x2_t a) {
  easysimd_float32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddv_f32(a);
  #else
    easysimd_float32x2_private a_ = easysimd_float32x2_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddv_f32
  #define vaddv_f32(v) easysimd_vaddv_f32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vaddv_s8(easysimd_int8x8_t a) {
  int8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddv_s8(a);
  #else
    easysimd_int8x8_private a_ = easysimd_int8x8_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddv_s8
  #define vaddv_s8(v) easysimd_vaddv_s8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vaddv_s16(easysimd_int16x4_t a) {
  int16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddv_s16(a);
  #else
    easysimd_int16x4_private a_ = easysimd_int16x4_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddv_s16
  #define vaddv_s16(v) easysimd_vaddv_s16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vaddv_s32(easysimd_int32x2_t a) {
  int32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddv_s32(a);
  #else
    easysimd_int32x2_private a_ = easysimd_int32x2_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddv_s32
  #define vaddv_s32(v) easysimd_vaddv_s32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vaddv_u8(easysimd_uint8x8_t a) {
  uint8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddv_u8(a);
  #else
    easysimd_uint8x8_private a_ = easysimd_uint8x8_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddv_u8
  #define vaddv_u8(v) easysimd_vaddv_u8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vaddv_u16(easysimd_uint16x4_t a) {
  uint16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddv_u16(a);
  #else
    easysimd_uint16x4_private a_ = easysimd_uint16x4_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddv_u16
  #define vaddv_u16(v) easysimd_vaddv_u16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vaddv_u32(easysimd_uint32x2_t a) {
  uint32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddv_u32(a);
  #else
    easysimd_uint32x2_private a_ = easysimd_uint32x2_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddv_u32
  #define vaddv_u32(v) easysimd_vaddv_u32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vaddvq_f32(easysimd_float32x4_t a) {
  easysimd_float32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_f32(a);
  #else
    easysimd_float32x4_private a_ = easysimd_float32x4_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_f32
  #define vaddvq_f32(v) easysimd_vaddvq_f32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vaddvq_f64(easysimd_float64x2_t a) {
  easysimd_float64_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_f64(a);
  #else
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_f64
  #define vaddvq_f64(v) easysimd_vaddvq_f64(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vaddvq_s8(easysimd_int8x16_t a) {
  int8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_s8(a);
  #else
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_s8
  #define vaddvq_s8(v) easysimd_vaddvq_s8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vaddvq_s16(easysimd_int16x8_t a) {
  int16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_s16(a);
  #else
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_s16
  #define vaddvq_s16(v) easysimd_vaddvq_s16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vaddvq_s32(easysimd_int32x4_t a) {
  int32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_s32(a);
  #else
    easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_s32
  #define vaddvq_s32(v) easysimd_vaddvq_s32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vaddvq_s64(easysimd_int64x2_t a) {
  int64_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_s64(a);
  #else
    easysimd_int64x2_private a_ = easysimd_int64x2_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_s64
  #define vaddvq_s64(v) easysimd_vaddvq_s64(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vaddvq_u8(easysimd_uint8x16_t a) {
  uint8_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_u8(a);
  #else
    easysimd_uint8x16_private a_ = easysimd_uint8x16_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_u8
  #define vaddvq_u8(v) easysimd_vaddvq_u8(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vaddvq_u16(easysimd_uint16x8_t a) {
  uint16_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_u16(a);
  #else
    easysimd_uint16x8_private a_ = easysimd_uint16x8_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_u16
  #define vaddvq_u16(v) easysimd_vaddvq_u16(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vaddvq_u32(easysimd_uint32x4_t a) {
  uint32_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_u32(a);
  #else
    easysimd_uint32x4_private a_ = easysimd_uint32x4_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_u32
  #define vaddvq_u32(v) easysimd_vaddvq_u32(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vaddvq_u64(easysimd_uint64x2_t a) {
  uint64_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r = vaddvq_u64(a);
  #else
    easysimd_uint64x2_private a_ = easysimd_uint64x2_to_private(a);

    r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(+:r)
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      r += a_.values[i];
    }
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vaddvq_u64
  #define vaddvq_u64(v) easysimd_vaddvq_u64(v)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ADDV_H) */
