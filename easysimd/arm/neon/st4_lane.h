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
 *   2021      Evan Nemerson <evan@nemerson.com>
 *   2021      Zhi An Ng <zhin@google.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_ST4_LANE_H)
#define EASYSIMD_ARM_NEON_ST4_LANE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if !defined(EASYSIMD_BUG_INTEL_857088)

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_s8(int8_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int8x8x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_NO_RESULT_(vst4_lane_s8, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_int8x8_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int8x8_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_s8
  #define vst4_lane_s8(a, b, c) easysimd_vst4_lane_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_s16(int16_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int16x4x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_NO_RESULT_(vst4_lane_s16, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_int16x4_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int16x4_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_s16
  #define vst4_lane_s16(a, b, c) easysimd_vst4_lane_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_s32(int32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int32x2x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_NO_RESULT_(vst4_lane_s32, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_int32x2_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int32x2_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_s32
  #define vst4_lane_s32(a, b, c) easysimd_vst4_lane_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_s64(int64_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int64x1x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    (void) lane;
    vst4_lane_s64(ptr, val, 0);
  #else
    easysimd_int64x1_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int64x1_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_s64
  #define vst4_lane_s64(a, b, c) easysimd_vst4_lane_s64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_u8(uint8_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint8x8x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_NO_RESULT_(vst4_lane_u8, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_uint8x8_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint8x8_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_u8
  #define vst4_lane_u8(a, b, c) easysimd_vst4_lane_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_u16(uint16_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint16x4x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_NO_RESULT_(vst4_lane_u16, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_uint16x4_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint16x4_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_u16
  #define vst4_lane_u16(a, b, c) easysimd_vst4_lane_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_u32(uint32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint32x2x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_NO_RESULT_(vst4_lane_u32, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_uint32x2_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint32x2_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_u32
  #define vst4_lane_u32(a, b, c) easysimd_vst4_lane_u32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_u64(uint64_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint64x1x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    (void) lane;
    vst4_lane_u64(ptr, val, 0);
  #else
    easysimd_uint64x1_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint64x1_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_u64
  #define vst4_lane_u64(a, b, c) easysimd_vst4_lane_u64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_f32(easysimd_float32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float32x2x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_2_NO_RESULT_(vst4_lane_f32, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_float32x2_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_float32x2_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_f32
  #define vst4_lane_f32(a, b, c) easysimd_vst4_lane_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4_lane_f64(easysimd_float64_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float64x1x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    (void) lane;
    vst4_lane_f64(ptr, val, 0);
  #else
    easysimd_float64x1_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_float64x1_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4_lane_f64
  #define vst4_lane_f64(a, b, c) easysimd_vst4_lane_f64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_s8(int8_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int8x16x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_CONSTIFY_16_NO_RESULT_(vst4q_lane_s8, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_int8x16_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int8x16_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_s8
  #define vst4q_lane_s8(a, b, c) easysimd_vst4q_lane_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_s16(int16_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int16x8x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_NO_RESULT_(vst4q_lane_s16, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_int16x8_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int16x8_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_s16
  #define vst4q_lane_s16(a, b, c) easysimd_vst4q_lane_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_s32(int32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int32x4x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_NO_RESULT_(vst4q_lane_s32, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_int32x4_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int32x4_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_s32
  #define vst4q_lane_s32(a, b, c) easysimd_vst4q_lane_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_s64(int64_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int64x2x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_CONSTIFY_2_NO_RESULT_(vst4q_lane_s64, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_int64x2_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_int64x2_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_s64
  #define vst4q_lane_s64(a, b, c) easysimd_vst4q_lane_s64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_u8(uint8_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint8x16x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_CONSTIFY_16_NO_RESULT_(vst4q_lane_u8, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_uint8x16_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint8x16_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_u8
  #define vst4q_lane_u8(a, b, c) easysimd_vst4q_lane_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_u16(uint16_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint16x8x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_8_NO_RESULT_(vst4q_lane_u16, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_uint16x8_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint16x8_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_u16
  #define vst4q_lane_u16(a, b, c) easysimd_vst4q_lane_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_u32(uint32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint32x4x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_NO_RESULT_(vst4q_lane_u32, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_uint32x4_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint32x4_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_u32
  #define vst4q_lane_u32(a, b, c) easysimd_vst4q_lane_u32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_u64(uint64_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint64x2x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_CONSTIFY_2_NO_RESULT_(vst4q_lane_u64, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_uint64x2_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_uint64x2_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_u64
  #define vst4q_lane_u64(a, b, c) easysimd_vst4q_lane_u64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_f32(easysimd_float32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float32x4x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_CONSTIFY_4_NO_RESULT_(vst4q_lane_f32, HEDLEY_UNREACHABLE(), lane, ptr, val);
  #else
    easysimd_float32x4_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_float32x4_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_f32
  #define vst4q_lane_f32(a, b, c) easysimd_vst4q_lane_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst4q_lane_f64(easysimd_float64_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float64x2x4_t val, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    (void) lane;
    vst4q_lane_f64(ptr, val, 0);
  #else
    easysimd_float64x2_private r;
    for (size_t i = 0 ; i < 4 ; i++) {
      r = easysimd_float64x2_to_private(val.val[i]);
      ptr[i] = r.values[lane];
    }
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst4q_lane_f64
  #define vst4q_lane_f64(a, b, c) easysimd_vst4q_lane_f64((a), (b), (c))
#endif

#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ST4_LANE_H) */
