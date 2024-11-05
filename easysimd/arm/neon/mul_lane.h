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

#if !defined(EASYSIMD_ARM_NEON_MUL_LANE_H)
#define EASYSIMD_ARM_NEON_MUL_LANE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vmuld_lane_f64(easysimd_float64_t a, easysimd_float64x1_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  return a * easysimd_float64x1_to_private(b).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vmuld_lane_f64(a, b, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vmuld_lane_f64(a, b, lane))
  #else
    #define easysimd_vmuld_lane_f64(a, b, lane) vmuld_lane_f64((a), (b), (lane))
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmuld_lane_f64
  #define vmuld_lane_f64(a, b, lane) easysimd_vmuld_lane_f64(a, b, lane)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vmuld_laneq_f64(easysimd_float64_t a, easysimd_float64x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return a * easysimd_float64x2_to_private(b).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vmuld_laneq_f64(a, b, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vmuld_laneq_f64(a, b, lane))
  #else
    #define easysimd_vmuld_laneq_f64(a, b, lane) vmuld_laneq_f64((a), (b), (lane))
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmuld_laneq_f64
  #define vmuld_laneq_f64(a, b, lane) easysimd_vmuld_laneq_f64(a, b, lane)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vmuls_lane_f32(easysimd_float32_t a, easysimd_float32x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return a * easysimd_float32x2_to_private(b).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vmuls_lane_f32(a, b, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vmuls_lane_f32(a, b, lane))
  #else
    #define easysimd_vmuls_lane_f32(a, b, lane) vmuls_lane_f32((a), (b), (lane))
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmuls_lane_f32
  #define vmuls_lane_f32(a, b, lane) easysimd_vmuls_lane_f32(a, b, lane)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vmuls_laneq_f32(easysimd_float32_t a, easysimd_float32x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return a * easysimd_float32x4_to_private(b).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vmuls_laneq_f32(a, b, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vmuls_laneq_f32(a, b, lane))
  #else
    #define easysimd_vmuls_laneq_f32(a, b, lane) vmuls_laneq_f32((a), (b), (lane))
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmuls_laneq_f32
  #define vmuls_laneq_f32(a, b, lane) easysimd_vmuls_laneq_f32(a, b, lane)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vmul_lane_f32(easysimd_float32x2_t a, easysimd_float32x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float32x2_private
    r_,
    a_ = easysimd_float32x2_to_private(a),
    b_ = easysimd_float32x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmul_lane_f32(a, b, lane) vmul_lane_f32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_lane_f32
  #define vmul_lane_f32(a, b, lane) easysimd_vmul_lane_f32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vmul_lane_f64(easysimd_float64x1_t a, easysimd_float64x1_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_float64x1_private
    r_,
    a_ = easysimd_float64x1_to_private(a),
    b_ = easysimd_float64x1_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmul_lane_f64(a, b, lane) vmul_lane_f64((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_lane_f64
  #define vmul_lane_f64(a, b, lane) easysimd_vmul_lane_f64((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmul_lane_s16(easysimd_int16x4_t a, easysimd_int16x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int16x4_private
    r_,
    a_ = easysimd_int16x4_to_private(a),
    b_ = easysimd_int16x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmul_lane_s16(a, b, lane) vmul_lane_s16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_lane_s16
  #define vmul_lane_s16(a, b, lane) easysimd_vmul_lane_s16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmul_lane_s32(easysimd_int32x2_t a, easysimd_int32x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int32x2_private
    r_,
    a_ = easysimd_int32x2_to_private(a),
    b_ = easysimd_int32x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmul_lane_s32(a, b, lane) vmul_lane_s32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_lane_s32
  #define vmul_lane_s32(a, b, lane) easysimd_vmul_lane_s32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmul_lane_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint16x4_private
    r_,
    a_ = easysimd_uint16x4_to_private(a),
    b_ = easysimd_uint16x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmul_lane_u16(a, b, lane) vmul_lane_u16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_lane_u16
  #define vmul_lane_u16(a, b, lane) easysimd_vmul_lane_u16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmul_lane_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint32x2_private
    r_,
    a_ = easysimd_uint32x2_to_private(a),
    b_ = easysimd_uint32x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmul_lane_u32(a, b, lane) vmul_lane_u32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmul_lane_u32
  #define vmul_lane_u32(a, b, lane) easysimd_vmul_lane_u32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmul_laneq_s16(easysimd_int16x4_t a, easysimd_int16x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int16x4_private
    r_,
    a_ = easysimd_int16x4_to_private(a);
  easysimd_int16x8_private
    b_ = easysimd_int16x8_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmul_laneq_s16(a, b, lane) vmul_laneq_s16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_laneq_s16
  #define vmul_laneq_s16(a, b, lane) easysimd_vmul_laneq_s16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmul_laneq_s32(easysimd_int32x2_t a, easysimd_int32x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int32x2_private
    r_,
    a_ = easysimd_int32x2_to_private(a);
  easysimd_int32x4_private
    b_ = easysimd_int32x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmul_laneq_s32(a, b, lane) vmul_laneq_s32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_laneq_s32
  #define vmul_laneq_s32(a, b, lane) easysimd_vmul_laneq_s32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmul_laneq_u16(easysimd_uint16x4_t a, easysimd_uint16x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint16x4_private
    r_,
    a_ = easysimd_uint16x4_to_private(a);
  easysimd_uint16x8_private
    b_ = easysimd_uint16x8_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmul_laneq_u16(a, b, lane) vmul_laneq_u16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_laneq_u16
  #define vmul_laneq_u16(a, b, lane) easysimd_vmul_laneq_u16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmul_laneq_u32(easysimd_uint32x2_t a, easysimd_uint32x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint32x2_private
    r_,
    a_ = easysimd_uint32x2_to_private(a);
  easysimd_uint32x4_private
    b_ = easysimd_uint32x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmul_laneq_u32(a, b, lane) vmul_laneq_u32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_laneq_u32
  #define vmul_laneq_u32(a, b, lane) easysimd_vmul_laneq_u32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vmulq_lane_f32(easysimd_float32x4_t a, easysimd_float32x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float32x4_private
    r_,
    a_ = easysimd_float32x4_to_private(a);
  easysimd_float32x2_private b_ = easysimd_float32x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmulq_lane_f32(a, b, lane) vmulq_lane_f32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_lane_f32
  #define vmulq_lane_f32(a, b, lane) easysimd_vmulq_lane_f32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vmulq_lane_f64(easysimd_float64x2_t a, easysimd_float64x1_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_float64x2_private
    r_,
    a_ = easysimd_float64x2_to_private(a);
  easysimd_float64x1_private b_ = easysimd_float64x1_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmulq_lane_f64(a, b, lane) vmulq_lane_f64((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_lane_f64
  #define vmulq_lane_f64(a, b, lane) easysimd_vmulq_lane_f64((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmulq_lane_s16(easysimd_int16x8_t a, easysimd_int16x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int16x8_private
    r_,
    a_ = easysimd_int16x8_to_private(a);
  easysimd_int16x4_private b_ = easysimd_int16x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmulq_lane_s16(a, b, lane) vmulq_lane_s16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_lane_s16
  #define vmulq_lane_s16(a, b, lane) easysimd_vmulq_lane_s16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmulq_lane_s32(easysimd_int32x4_t a, easysimd_int32x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int32x4_private
    r_,
    a_ = easysimd_int32x4_to_private(a);
  easysimd_int32x2_private b_ = easysimd_int32x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmulq_lane_s32(a, b, lane) vmulq_lane_s32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_lane_s32
  #define vmulq_lane_s32(a, b, lane) easysimd_vmulq_lane_s32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmulq_lane_u16(easysimd_uint16x8_t a, easysimd_uint16x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint16x8_private
    r_,
    a_ = easysimd_uint16x8_to_private(a);
  easysimd_uint16x4_private b_ = easysimd_uint16x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmulq_lane_u16(a, b, lane) vmulq_lane_u16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_lane_u16
  #define vmulq_lane_u16(a, b, lane) easysimd_vmulq_lane_u16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmulq_lane_u32(easysimd_uint32x4_t a, easysimd_uint32x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint32x4_private
    r_,
    a_ = easysimd_uint32x4_to_private(a);
  easysimd_uint32x2_private b_ = easysimd_uint32x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vmulq_lane_u32(a, b, lane) vmulq_lane_u32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmulq_lane_u32
  #define vmulq_lane_u32(a, b, lane) easysimd_vmulq_lane_u32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vmulq_laneq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_float32x4_private
    r_,
    a_ = easysimd_float32x4_to_private(a),
    b_ = easysimd_float32x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmulq_laneq_f32(a, b, lane) vmulq_laneq_f32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_laneq_f32
  #define vmulq_laneq_f32(a, b, lane) easysimd_vmulq_laneq_f32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vmulq_laneq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float64x2_private
    r_,
    a_ = easysimd_float64x2_to_private(a),
    b_ = easysimd_float64x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmulq_laneq_f64(a, b, lane) vmulq_laneq_f64((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_laneq_f64
  #define vmulq_laneq_f64(a, b, lane) easysimd_vmulq_laneq_f64((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmulq_laneq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int16x8_private
    r_,
    a_ = easysimd_int16x8_to_private(a),
    b_ = easysimd_int16x8_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmulq_laneq_s16(a, b, lane) vmulq_laneq_s16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_laneq_s16
  #define vmulq_laneq_s16(a, b, lane) easysimd_vmulq_laneq_s16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmulq_laneq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int32x4_private
    r_,
    a_ = easysimd_int32x4_to_private(a),
    b_ = easysimd_int32x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_int32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmulq_laneq_s32(a, b, lane) vmulq_laneq_s32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_laneq_s32
  #define vmulq_laneq_s32(a, b, lane) easysimd_vmulq_laneq_s32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmulq_laneq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint16x8_private
    r_,
    a_ = easysimd_uint16x8_to_private(a),
    b_ = easysimd_uint16x8_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmulq_laneq_u16(a, b, lane) vmulq_laneq_u16((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_laneq_u16
  #define vmulq_laneq_u16(a, b, lane) easysimd_vmulq_laneq_u16((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmulq_laneq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint32x4_private
    r_,
    a_ = easysimd_uint32x4_to_private(a),
    b_ = easysimd_uint32x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_uint32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmulq_laneq_u32(a, b, lane) vmulq_laneq_u32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmulq_laneq_u32
  #define vmulq_laneq_u32(a, b, lane) easysimd_vmulq_laneq_u32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vmul_laneq_f32(easysimd_float32x2_t a, easysimd_float32x4_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_float32x2_private
    r_,
    a_ = easysimd_float32x2_to_private(a);
  easysimd_float32x4_private b_ = easysimd_float32x4_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmul_laneq_f32(a, b, lane) vmul_laneq_f32((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_laneq_f32
  #define vmul_laneq_f32(a, b, lane) easysimd_vmul_laneq_f32((a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vmul_laneq_f64(easysimd_float64x1_t a, easysimd_float64x2_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float64x1_private
    r_,
    a_ = easysimd_float64x1_to_private(a);
  easysimd_float64x2_private b_ = easysimd_float64x2_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = a_.values[i] * b_.values[lane];
  }

  return easysimd_float64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vmul_laneq_f64(a, b, lane) vmul_laneq_f64((a), (b), (lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmul_laneq_f64
  #define vmul_laneq_f64(a, b, lane) easysimd_vmul_laneq_f64((a), (b), (lane))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MUL_LANE_H) */
