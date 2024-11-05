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

/* In older versions of clang, __builtin_neon_vld4_lane_v would
 * generate a diagnostic for most variants (those which didn't
 * use signed 8-bit integers).  I believe this was fixed by
 * 78ad22e0cc6390fcd44b2b7b5132f1b960ff975d.
 *
 * Since we have to use macros (due to the immediate-mode parameter)
 * we can't just disable it once in this file; we have to use statement
 * exprs and push / pop the stack for each macro. */

#if !defined(EASYSIMD_ARM_NEON_LD4_LANE_H)
#define EASYSIMD_ARM_NEON_LD4_LANE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if !defined(EASYSIMD_BUG_INTEL_857088)

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8x4_t
easysimd_vld4_lane_s8(int8_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int8x8x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int8x8x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int8x8_private tmp_ = easysimd_int8x8_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int8x8_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_s8(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_s8(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_s8(ptr, src, lane) vld4_lane_s8(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_s8
  #define vld4_lane_s8(ptr, src, lane) easysimd_vld4_lane_s8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4x4_t
easysimd_vld4_lane_s16(int16_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int16x4x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int16x4x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int16x4_private tmp_ = easysimd_int16x4_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int16x4_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_s16(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_s16(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_s16(ptr, src, lane) vld4_lane_s16(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_s16
  #define vld4_lane_s16(ptr, src, lane) easysimd_vld4_lane_s16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2x4_t
easysimd_vld4_lane_s32(int32_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int32x2x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int32x2x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int32x2_private tmp_ = easysimd_int32x2_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int32x2_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_s32(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_s32(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_s32(ptr, src, lane) vld4_lane_s32(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_s32
  #define vld4_lane_s32(ptr, src, lane) easysimd_vld4_lane_s32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1x4_t
easysimd_vld4_lane_s64(int64_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int64x1x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_int64x1x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int64x1_private tmp_ = easysimd_int64x1_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int64x1_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_s64(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_s64(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_s64(ptr, src, lane) vld4_lane_s64(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_s64
  #define vld4_lane_s64(ptr, src, lane) easysimd_vld4_lane_s64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8x4_t
easysimd_vld4_lane_u8(uint8_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint8x8x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint8x8x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint8x8_private tmp_ = easysimd_uint8x8_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint8x8_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_u8(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_u8(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_u8(ptr, src, lane) vld4_lane_u8(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_u8
  #define vld4_lane_u8(ptr, src, lane) easysimd_vld4_lane_u8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4x4_t
easysimd_vld4_lane_u16(uint16_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint16x4x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint16x4x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint16x4_private tmp_ = easysimd_uint16x4_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint16x4_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_u16(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_u16(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_u16(ptr, src, lane) vld4_lane_u16(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_u16
  #define vld4_lane_u16(ptr, src, lane) easysimd_vld4_lane_u16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2x4_t
easysimd_vld4_lane_u32(uint32_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint32x2x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint32x2x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint32x2_private tmp_ = easysimd_uint32x2_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint32x2_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_u32(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_u32(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_u32(ptr, src, lane) vld4_lane_u32(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_u32
  #define vld4_lane_u32(ptr, src, lane) easysimd_vld4_lane_u32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1x4_t
easysimd_vld4_lane_u64(uint64_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint64x1x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_uint64x1x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint64x1_private tmp_ = easysimd_uint64x1_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint64x1_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_u64(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_u64(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_u64(ptr, src, lane) vld4_lane_u64(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_u64
  #define vld4_lane_u64(ptr, src, lane) easysimd_vld4_lane_u64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2x4_t
easysimd_vld4_lane_f32(easysimd_float32_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float32x2x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float32x2x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_float32x2_private tmp_ = easysimd_float32x2_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_float32x2_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_f32(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_f32(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_f32(ptr, src, lane) vld4_lane_f32(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_f32
  #define vld4_lane_f32(ptr, src, lane) easysimd_vld4_lane_f32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1x4_t
easysimd_vld4_lane_f64(easysimd_float64_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float64x1x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_float64x1x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_float64x1_private tmp_ = easysimd_float64x1_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_float64x1_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4_lane_f64(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4_lane_f64(ptr, src, lane))
  #else
    #define easysimd_vld4_lane_f64(ptr, src, lane) vld4_lane_f64(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4_lane_f64
  #define vld4_lane_f64(ptr, src, lane) easysimd_vld4_lane_f64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16x4_t
easysimd_vld4q_lane_s8(int8_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int8x16x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  easysimd_int8x16x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int8x16_private tmp_ = easysimd_int8x16_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int8x16_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_s8(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_s8(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_s8(ptr, src, lane) vld4q_lane_s8(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_s8
  #define vld4q_lane_s8(ptr, src, lane) easysimd_vld4q_lane_s8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8x4_t
easysimd_vld4q_lane_s16(int16_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int16x8x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int16x8x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int16x8_private tmp_ = easysimd_int16x8_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int16x8_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_s16(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_s16(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_s16(ptr, src, lane) vld4q_lane_s16(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_s16
  #define vld4q_lane_s16(ptr, src, lane) easysimd_vld4q_lane_s16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4x4_t
easysimd_vld4q_lane_s32(int32_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int32x4x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int32x4x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int32x4_private tmp_ = easysimd_int32x4_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int32x4_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_s32(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_s32(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_s32(ptr, src, lane) vld4q_lane_s32(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_s32
  #define vld4q_lane_s32(ptr, src, lane) easysimd_vld4q_lane_s32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2x4_t
easysimd_vld4q_lane_s64(int64_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int64x2x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int64x2x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_int64x2_private tmp_ = easysimd_int64x2_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_int64x2_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_s64(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_s64(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_s64(ptr, src, lane) vld4q_lane_s64(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_s64
  #define vld4q_lane_s64(ptr, src, lane) easysimd_vld4q_lane_s64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16x4_t
easysimd_vld4q_lane_u8(uint8_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint8x16x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  easysimd_uint8x16x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint8x16_private tmp_ = easysimd_uint8x16_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint8x16_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_u8(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_u8(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_u8(ptr, src, lane) vld4q_lane_u8(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_u8
  #define vld4q_lane_u8(ptr, src, lane) easysimd_vld4q_lane_u8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8x4_t
easysimd_vld4q_lane_u16(uint16_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint16x8x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint16x8x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint16x8_private tmp_ = easysimd_uint16x8_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint16x8_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_u16(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_u16(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_u16(ptr, src, lane) vld4q_lane_u16(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_u16
  #define vld4q_lane_u16(ptr, src, lane) easysimd_vld4q_lane_u16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4x4_t
easysimd_vld4q_lane_u32(uint32_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint32x4x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint32x4x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint32x4_private tmp_ = easysimd_uint32x4_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint32x4_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_u32(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_u32(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_u32(ptr, src, lane) vld4q_lane_u32(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_u32
  #define vld4q_lane_u32(ptr, src, lane) easysimd_vld4q_lane_u32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2x4_t
easysimd_vld4q_lane_u64(uint64_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint64x2x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint64x2x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_uint64x2_private tmp_ = easysimd_uint64x2_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_uint64x2_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_u64(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_u64(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_u64(ptr, src, lane) vld4q_lane_u64(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_u64
  #define vld4q_lane_u64(ptr, src, lane) easysimd_vld4q_lane_u64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4x4_t
easysimd_vld4q_lane_f32(easysimd_float32_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float32x4x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_float32x4x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_float32x4_private tmp_ = easysimd_float32x4_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_float32x4_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_f32(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_f32(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_f32(ptr, src, lane) vld4q_lane_f32(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_f32
  #define vld4q_lane_f32(ptr, src, lane) easysimd_vld4q_lane_f32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2x4_t
easysimd_vld4q_lane_f64(easysimd_float64_t const ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float64x2x4_t src, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float64x2x4_t r;

  for (size_t i = 0 ; i < 4 ; i++) {
    easysimd_float64x2_private tmp_ = easysimd_float64x2_to_private(src.val[i]);
    tmp_.values[lane] = ptr[i];
    r.val[i] = easysimd_float64x2_from_private(tmp_);
  }

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0)
    #define easysimd_vld4q_lane_f64(ptr, src, lane) \
      EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vld4q_lane_f64(ptr, src, lane))
  #else
    #define easysimd_vld4q_lane_f64(ptr, src, lane) vld4q_lane_f64(ptr, src, lane)
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_lane_f64
  #define vld4q_lane_f64(ptr, src, lane) easysimd_vld4q_lane_f64((ptr), (src), (lane))
#endif

#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_LD4_LANE_H) */
