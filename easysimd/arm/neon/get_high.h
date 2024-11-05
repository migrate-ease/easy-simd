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

#if !defined(EASYSIMD_ARM_NEON_GET_HIGH_H)
#define EASYSIMD_ARM_NEON_GET_HIGH_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vget_high_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_f32(a);
  #else
    easysimd_float32x2_private r_;
    easysimd_float32x4_private a_ = easysimd_float32x4_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 2, 3);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_f32
  #define vget_high_f32(a) easysimd_vget_high_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vget_high_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vget_high_f64(a);
  #else
    easysimd_float64x1_private r_;
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vget_high_f64
  #define vget_high_f64(a) easysimd_vget_high_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vget_high_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_s8(a);
  #else
    easysimd_int8x8_private r_;
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 8, 9, 10, 11, 12, 13, 14, 15);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_s8
  #define vget_high_s8(a) easysimd_vget_high_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vget_high_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_s16(a);
  #else
    easysimd_int16x4_private r_;
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 4, 5, 6, 7);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_s16
  #define vget_high_s16(a) easysimd_vget_high_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vget_high_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_s32(a);
  #else
    easysimd_int32x2_private r_;
    easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 2, 3);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_s32
  #define vget_high_s32(a) easysimd_vget_high_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vget_high_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_s64(a);
  #else
    easysimd_int64x1_private r_;
    easysimd_int64x2_private a_ = easysimd_int64x2_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_s64
  #define vget_high_s64(a) easysimd_vget_high_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vget_high_u8(easysimd_uint8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_u8(a);
  #else
    easysimd_uint8x8_private r_;
    easysimd_uint8x16_private a_ = easysimd_uint8x16_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 8, 9, 10, 11, 12, 13, 14,15);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_u8
  #define vget_high_u8(a) easysimd_vget_high_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vget_high_u16(easysimd_uint16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_u16(a);
  #else
    easysimd_uint16x4_private r_;
    easysimd_uint16x8_private a_ = easysimd_uint16x8_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 4, 5, 6, 7);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_u16
  #define vget_high_u16(a) easysimd_vget_high_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vget_high_u32(easysimd_uint32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_u32(a);
  #else
    easysimd_uint32x2_private r_;
    easysimd_uint32x4_private a_ = easysimd_uint32x4_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 2, 3);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_u32
  #define vget_high_u32(a) easysimd_vget_high_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vget_high_u64(easysimd_uint64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vget_high_u64(a);
  #else
    easysimd_uint64x1_private r_;
    easysimd_uint64x2_private a_ = easysimd_uint64x2_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.values = __builtin_shufflevector(a_.values, a_.values, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i + (sizeof(r_.values) / sizeof(r_.values[0]))];
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_high_u64
  #define vget_high_u64(a) easysimd_vget_high_u64((a))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_GET_HIGH_H) */
