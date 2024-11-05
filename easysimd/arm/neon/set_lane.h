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

#if !defined(EASYSIMD_ARM_NEON_SET_LANE_H)
#define EASYSIMD_ARM_NEON_SET_LANE_H
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vset_lane_f32(easysimd_float32_t a, easysimd_float32x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float32x2_t r;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_CONSTIFY_2_(vset_lane_f32, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_float32x2_private v_ = easysimd_float32x2_to_private(v);
    v_.values[lane] = a;
    r = easysimd_float32x2_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_f32
  #define vset_lane_f32(a, b, c) easysimd_vset_lane_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vset_lane_f64(easysimd_float64_t a, easysimd_float64x1_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_float64x1_t r;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    (void) lane;
    r = vset_lane_f64(a, v, 0);
  #else
    easysimd_float64x1_private v_ = easysimd_float64x1_to_private(v);
    v_.values[lane] = a;
    r = easysimd_float64x1_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_f64
  #define vset_lane_f64(a, b, c) easysimd_vset_lane_f64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vset_lane_s8(int8_t a, easysimd_int8x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int8x8_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vset_lane_s8, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_int8x8_private v_ = easysimd_int8x8_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int8x8_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_s8
  #define vset_lane_s8(a, b, c) easysimd_vset_lane_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vset_lane_s16(int16_t a, easysimd_int16x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int16x4_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vset_lane_s16, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_int16x4_private v_ = easysimd_int16x4_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int16x4_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_s16
  #define vset_lane_s16(a, b, c) easysimd_vset_lane_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vset_lane_s32(int32_t a, easysimd_int32x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int32x2_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vset_lane_s32, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_int32x2_private v_ = easysimd_int32x2_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int32x2_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_s32
  #define vset_lane_s32(a, b, c) easysimd_vset_lane_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vset_lane_s64(int64_t a, easysimd_int64x1_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_int64x1_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    (void) lane;
    r = vset_lane_s64(a, v, 0);
  #else
    easysimd_int64x1_private v_ = easysimd_int64x1_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int64x1_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_s64
  #define vset_lane_s64(a, b, c) easysimd_vset_lane_s64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vset_lane_u8(uint8_t a, easysimd_uint8x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint8x8_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vset_lane_u8, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_uint8x8_private v_ = easysimd_uint8x8_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint8x8_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_u8
  #define vset_lane_u8(a, b, c) easysimd_vset_lane_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vset_lane_u16(uint16_t a, easysimd_uint16x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint16x4_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vset_lane_u16, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_uint16x4_private v_ = easysimd_uint16x4_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint16x4_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_u16
  #define vset_lane_u16(a, b, c) easysimd_vset_lane_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vset_lane_u32(uint32_t a, easysimd_uint32x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint32x2_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vset_lane_u32, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_uint32x2_private v_ = easysimd_uint32x2_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint32x2_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_u32
  #define vset_lane_u32(a, b, c) easysimd_vset_lane_u32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vset_lane_u64(uint64_t a, easysimd_uint64x1_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_uint64x1_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    (void) lane;
    r = vset_lane_u64(a, v, 0);
  #else
    easysimd_uint64x1_private v_ = easysimd_uint64x1_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint64x1_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vset_lane_u64
  #define vset_lane_u64(a, b, c) easysimd_vset_lane_u64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vsetq_lane_f32(easysimd_float32_t a, easysimd_float32x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_float32x4_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vsetq_lane_f32, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_float32x4_private v_ = easysimd_float32x4_to_private(v);
    v_.values[lane] = a;
    r = easysimd_float32x4_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_f32
  #define vsetq_lane_f32(a, b, c) easysimd_vsetq_lane_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vsetq_lane_f64(easysimd_float64_t a, easysimd_float64x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float64x2_t r;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_CONSTIFY_2_(vsetq_lane_f64, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_float64x2_private v_ = easysimd_float64x2_to_private(v);
    v_.values[lane] = a;
    r = easysimd_float64x2_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_f64
  #define vsetq_lane_f64(a, b, c) easysimd_vsetq_lane_f64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vsetq_lane_s8(int8_t a, easysimd_int8x16_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  easysimd_int8x16_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_16_(vsetq_lane_s8, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_int8x16_private v_ = easysimd_int8x16_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int8x16_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_s8
  #define vsetq_lane_s8(a, b, c) easysimd_vsetq_lane_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vsetq_lane_s16(int16_t a, easysimd_int16x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int16x8_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vsetq_lane_s16, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_int16x8_private v_ = easysimd_int16x8_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int16x8_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_s16
  #define vsetq_lane_s16(a, b, c) easysimd_vsetq_lane_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vsetq_lane_s32(int32_t a, easysimd_int32x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int32x4_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vsetq_lane_s32, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_int32x4_private v_ = easysimd_int32x4_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int32x4_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_s32
  #define vsetq_lane_s32(a, b, c) easysimd_vsetq_lane_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vsetq_lane_s64(int64_t a, easysimd_int64x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int64x2_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vsetq_lane_s64, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_int64x2_private v_ = easysimd_int64x2_to_private(v);
    v_.values[lane] = a;
    r = easysimd_int64x2_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_s64
  #define vsetq_lane_s64(a, b, c) easysimd_vsetq_lane_s64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vsetq_lane_u8(uint8_t a, easysimd_uint8x16_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  easysimd_uint8x16_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_16_(vsetq_lane_u8, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_uint8x16_private v_ = easysimd_uint8x16_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint8x16_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_u8
  #define vsetq_lane_u8(a, b, c) easysimd_vsetq_lane_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vsetq_lane_u16(uint16_t a, easysimd_uint16x8_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint16x8_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_(vsetq_lane_u16, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_uint16x8_private v_ = easysimd_uint16x8_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint16x8_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_u16
  #define vsetq_lane_u16(a, b, c) easysimd_vsetq_lane_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vsetq_lane_u32(uint32_t a, easysimd_uint32x4_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint32x4_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_(vsetq_lane_u32, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_uint32x4_private v_ = easysimd_uint32x4_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint32x4_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_u32
  #define vsetq_lane_u32(a, b, c) easysimd_vsetq_lane_u32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vsetq_lane_u64(uint64_t a, easysimd_uint64x2_t v, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint64x2_t r;
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_(vsetq_lane_u64, r, (HEDLEY_UNREACHABLE(), v), lane, a, v);
  #else
    easysimd_uint64x2_private v_ = easysimd_uint64x2_to_private(v);
    v_.values[lane] = a;
    r = easysimd_uint64x2_from_private(v_);
  #endif
  return r;
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vsetq_lane_u64
  #define vsetq_lane_u64(a, b, c) easysimd_vsetq_lane_u64((a), (b), (c))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_SET_LANE_H) */
