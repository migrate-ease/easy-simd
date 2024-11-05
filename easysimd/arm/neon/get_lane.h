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

#if !defined(EASYSIMD_ARM_NEON_GET_LANE_H)
#define EASYSIMD_ARM_NEON_GET_LANE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vget_lane_f32(easysimd_float32x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float32_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vget_lane_f32, r, (HEDLEY_UNREACHABLE(), EASYSIMD_FLOAT32_C(0.0)), lane, v);
  #else
    easysimd_float32x2_private v_ = easysimd_float32x2_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_f32
  #define vget_lane_f32(v, lane) easysimd_vget_lane_f32((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vget_lane_f64(easysimd_float64x1_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_float64_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    (void) lane;
    return vget_lane_f64(v, 0);
  #else
    easysimd_float64x1_private v_ = easysimd_float64x1_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_f64
  #define vget_lane_f64(v, lane) easysimd_vget_lane_f64((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vget_lane_s8(easysimd_int8x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  int8_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vget_lane_s8, r, (HEDLEY_UNREACHABLE(), INT8_C(0)), lane, v);
  #else
    easysimd_int8x8_private v_ = easysimd_int8x8_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_s8
  #define vget_lane_s8(v, lane) easysimd_vget_lane_s8((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vget_lane_s16(easysimd_int16x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  int16_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vget_lane_s16, r, (HEDLEY_UNREACHABLE(), INT16_C(0)), lane, v);
  #else
    easysimd_int16x4_private v_ = easysimd_int16x4_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_s16
  #define vget_lane_s16(v, lane) easysimd_vget_lane_s16((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vget_lane_s32(easysimd_int32x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  int32_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vget_lane_s32, r, (HEDLEY_UNREACHABLE(), INT32_C(0)), lane, v);
  #else
    easysimd_int32x2_private v_ = easysimd_int32x2_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_s32
  #define vget_lane_s32(v, lane) easysimd_vget_lane_s32((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vget_lane_s64(easysimd_int64x1_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  int64_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    (void) lane;
    return vget_lane_s64(v, 0);
  #else
    easysimd_int64x1_private v_ = easysimd_int64x1_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_s64
  #define vget_lane_s64(v, lane) easysimd_vget_lane_s64((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vget_lane_u8(easysimd_uint8x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  uint8_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vget_lane_u8, r, (HEDLEY_UNREACHABLE(), UINT8_C(0)), lane, v);
  #else
    easysimd_uint8x8_private v_ = easysimd_uint8x8_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_u8
  #define vget_lane_u8(v, lane) easysimd_vget_lane_u8((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vget_lane_u16(easysimd_uint16x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  uint16_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vget_lane_u16, r, (HEDLEY_UNREACHABLE(), UINT16_C(0)), lane, v);
  #else
    easysimd_uint16x4_private v_ = easysimd_uint16x4_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_u16
  #define vget_lane_u16(v, lane) easysimd_vget_lane_u16((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vget_lane_u32(easysimd_uint32x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  uint32_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vget_lane_u32, r, (HEDLEY_UNREACHABLE(), UINT32_C(0)), lane, v);
  #else
    easysimd_uint32x2_private v_ = easysimd_uint32x2_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_u32
  #define vget_lane_u32(v, lane) easysimd_vget_lane_u32((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vget_lane_u64(easysimd_uint64x1_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  uint64_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    (void) lane;
    return vget_lane_u64(v, 0);
  #else
    easysimd_uint64x1_private v_ = easysimd_uint64x1_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vget_lane_u64
  #define vget_lane_u64(v, lane) easysimd_vget_lane_u64((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vgetq_lane_f32(easysimd_float32x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_float32_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vgetq_lane_f32, r, (HEDLEY_UNREACHABLE(), EASYSIMD_FLOAT32_C(0.0)), lane, v);
  #else
    easysimd_float32x4_private v_ = easysimd_float32x4_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_f32
  #define vgetq_lane_f32(v, lane) easysimd_vgetq_lane_f32((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vgetq_lane_f64(easysimd_float64x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float64_t r;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_CONSTIFY_2_(vgetq_lane_f64, r, (HEDLEY_UNREACHABLE(), EASYSIMD_FLOAT64_C(0.0)), lane, v);
  #else
    easysimd_float64x2_private v_ = easysimd_float64x2_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_f64
  #define vgetq_lane_f64(v, lane) easysimd_vgetq_lane_f64((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vgetq_lane_s8(easysimd_int8x16_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  int8_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_16_(vgetq_lane_s8, r, (HEDLEY_UNREACHABLE(), INT8_C(0)), lane, v);
  #else
    easysimd_int8x16_private v_ = easysimd_int8x16_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_s8
  #define vgetq_lane_s8(v, lane) easysimd_vgetq_lane_s8((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vgetq_lane_s16(easysimd_int16x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  int16_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vgetq_lane_s16, r, (HEDLEY_UNREACHABLE(), INT16_C(0)), lane, v);
  #else
    easysimd_int16x8_private v_ = easysimd_int16x8_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_s16
  #define vgetq_lane_s16(v, lane) easysimd_vgetq_lane_s16((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vgetq_lane_s32(easysimd_int32x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  int32_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vgetq_lane_s32, r, (HEDLEY_UNREACHABLE(), INT32_C(0)), lane, v);
  #else
    easysimd_int32x4_private v_ = easysimd_int32x4_to_private(v);
    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_s32
  #define vgetq_lane_s32(v, lane) easysimd_vgetq_lane_s32((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vgetq_lane_s64(easysimd_int64x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  int64_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vgetq_lane_s64, r, (HEDLEY_UNREACHABLE(), INT64_C(0)), lane, v);
  #else
    easysimd_int64x2_private v_ = easysimd_int64x2_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_s64
  #define vgetq_lane_s64(v, lane) easysimd_vgetq_lane_s64((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vgetq_lane_u8(easysimd_uint8x16_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  uint8_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_16_(vgetq_lane_u8, r, (HEDLEY_UNREACHABLE(), UINT8_C(0)), lane, v);
  #else
    easysimd_uint8x16_private v_ = easysimd_uint8x16_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_u8
  #define vgetq_lane_u8(v, lane) easysimd_vgetq_lane_u8((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vgetq_lane_u16(easysimd_uint16x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  uint16_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vgetq_lane_u16, r, (HEDLEY_UNREACHABLE(), UINT16_C(0)), lane, v);
  #else
    easysimd_uint16x8_private v_ = easysimd_uint16x8_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_u16
  #define vgetq_lane_u16(v, lane) easysimd_vgetq_lane_u16((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vgetq_lane_u32(easysimd_uint32x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  uint32_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vgetq_lane_u32, r, (HEDLEY_UNREACHABLE(), UINT32_C(0)), lane, v);
  #else
    easysimd_uint32x4_private v_ = easysimd_uint32x4_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_u32
  #define vgetq_lane_u32(v, lane) easysimd_vgetq_lane_u32((v), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vgetq_lane_u64(easysimd_uint64x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  uint64_t r;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vgetq_lane_u64, r, (HEDLEY_UNREACHABLE(), UINT64_C(0)), lane, v);
  #else
    easysimd_uint64x2_private v_ = easysimd_uint64x2_to_private(v);

    r = v_.values[lane];
  #endif

  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vgetq_lane_u64
  #define vgetq_lane_u64(v, lane) easysimd_vgetq_lane_u64((v), (lane))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_GET_LANE_H) */
