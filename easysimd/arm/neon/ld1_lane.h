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
 */

#if !defined(EASYSIMD_ARM_NEON_LD1_LANE_H)
#define EASYSIMD_ARM_NEON_LD1_LANE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t easysimd_vld1_lane_s8(int8_t const *ptr, easysimd_int8x8_t src,
                                    const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int8x8_private r = easysimd_int8x8_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int8x8_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_s8(ptr, src, lane) vld1_lane_s8(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_s8
  #define vld1_lane_s8(ptr, src, lane) easysimd_vld1_lane_s8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t easysimd_vld1_lane_s16(int16_t const *ptr, easysimd_int16x4_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int16x4_private r = easysimd_int16x4_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int16x4_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_s16(ptr, src, lane) vld1_lane_s16(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_s16
  #define vld1_lane_s16(ptr, src, lane) easysimd_vld1_lane_s16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t easysimd_vld1_lane_s32(int32_t const *ptr, easysimd_int32x2_t src,
                                      const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int32x2_private r = easysimd_int32x2_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int32x2_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_s32(ptr, src, lane) vld1_lane_s32(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_s32
  #define vld1_lane_s32(ptr, src, lane) easysimd_vld1_lane_s32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t easysimd_vld1_lane_s64(int64_t const *ptr, easysimd_int64x1_t src,
                                      const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_int64x1_private r = easysimd_int64x1_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int64x1_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_s64(ptr, src, lane) vld1_lane_s64(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_s64
  #define vld1_lane_s64(ptr, src, lane) easysimd_vld1_lane_s64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t easysimd_vld1_lane_u8(uint8_t const *ptr, easysimd_uint8x8_t src,
                                   const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint8x8_private r = easysimd_uint8x8_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint8x8_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_u8(ptr, src, lane) vld1_lane_u8(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_u8
  #define vld1_lane_u8(ptr, src, lane) easysimd_vld1_lane_u8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t easysimd_vld1_lane_u16(uint16_t const *ptr, easysimd_uint16x4_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint16x4_private r = easysimd_uint16x4_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint16x4_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_u16(ptr, src, lane) vld1_lane_u16(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_u16
  #define vld1_lane_u16(ptr, src, lane) easysimd_vld1_lane_u16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t easysimd_vld1_lane_u32(uint32_t const *ptr, easysimd_uint32x2_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint32x2_private r = easysimd_uint32x2_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint32x2_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_u32(ptr, src, lane) vld1_lane_u32(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_u32
  #define vld1_lane_u32(ptr, src, lane) easysimd_vld1_lane_u32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t easysimd_vld1_lane_u64(uint64_t const *ptr, easysimd_uint64x1_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_uint64x1_private r = easysimd_uint64x1_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint64x1_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_u64(ptr, src, lane) vld1_lane_u64(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_u64
  #define vld1_lane_u64(ptr, src, lane) easysimd_vld1_lane_u64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t easysimd_vld1_lane_f32(easysimd_float32_t const *ptr, easysimd_float32x2_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float32x2_private r = easysimd_float32x2_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_float32x2_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1_lane_f32(ptr, src, lane) vld1_lane_f32(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_f32
  #define vld1_lane_f32(ptr, src, lane) easysimd_vld1_lane_f32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t easysimd_vld1_lane_f64(easysimd_float64_t const *ptr, easysimd_float64x1_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  easysimd_float64x1_private r = easysimd_float64x1_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_float64x1_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vld1_lane_f64(ptr, src, lane) vld1_lane_f64(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld1_lane_f64
  #define vld1_lane_f64(ptr, src, lane) easysimd_vld1_lane_f64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t easysimd_vld1q_lane_s8(int8_t const *ptr, easysimd_int8x16_t src,
                                    const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  easysimd_int8x16_private r = easysimd_int8x16_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int8x16_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_s8(ptr, src, lane) vld1q_lane_s8(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_s8
  #define vld1q_lane_s8(ptr, src, lane) easysimd_vld1q_lane_s8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t easysimd_vld1q_lane_s16(int16_t const *ptr, easysimd_int16x8_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_int16x8_private r = easysimd_int16x8_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int16x8_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_s16(ptr, src, lane) vld1q_lane_s16(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_s16
  #define vld1q_lane_s16(ptr, src, lane) easysimd_vld1q_lane_s16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t easysimd_vld1q_lane_s32(int32_t const *ptr, easysimd_int32x4_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int32x4_private r = easysimd_int32x4_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int32x4_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_s32(ptr, src, lane) vld1q_lane_s32(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_s32
  #define vld1q_lane_s32(ptr, src, lane) easysimd_vld1q_lane_s32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t easysimd_vld1q_lane_s64(int64_t const *ptr, easysimd_int64x2_t src,
                                      const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int64x2_private r = easysimd_int64x2_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_int64x2_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_s64(ptr, src, lane) vld1q_lane_s64(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_s64
  #define vld1q_lane_s64(ptr, src, lane) easysimd_vld1q_lane_s64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t easysimd_vld1q_lane_u8(uint8_t const *ptr, easysimd_uint8x16_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  easysimd_uint8x16_private r = easysimd_uint8x16_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint8x16_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_u8(ptr, src, lane) vld1q_lane_u8(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_u8
  #define vld1q_lane_u8(ptr, src, lane) easysimd_vld1q_lane_u8((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t easysimd_vld1q_lane_u16(uint16_t const *ptr, easysimd_uint16x8_t src,
                                      const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  easysimd_uint16x8_private r = easysimd_uint16x8_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint16x8_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_u16(ptr, src, lane) vld1q_lane_u16(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_u16
  #define vld1q_lane_u16(ptr, src, lane) easysimd_vld1q_lane_u16((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t easysimd_vld1q_lane_u32(uint32_t const *ptr, easysimd_uint32x4_t src,
                                      const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint32x4_private r = easysimd_uint32x4_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint32x4_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_u32(ptr, src, lane) vld1q_lane_u32(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_u32
  #define vld1q_lane_u32(ptr, src, lane) easysimd_vld1q_lane_u32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t easysimd_vld1q_lane_u64(uint64_t const *ptr, easysimd_uint64x2_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint64x2_private r = easysimd_uint64x2_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_uint64x2_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_u64(ptr, src, lane) vld1q_lane_u64(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_u64
  #define vld1q_lane_u64(ptr, src, lane) easysimd_vld1q_lane_u64((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t easysimd_vld1q_lane_f32(easysimd_float32_t const *ptr, easysimd_float32x4_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_float32x4_private r = easysimd_float32x4_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_float32x4_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vld1q_lane_f32(ptr, src, lane) vld1q_lane_f32(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_f32
  #define vld1q_lane_f32(ptr, src, lane) easysimd_vld1q_lane_f32((ptr), (src), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t easysimd_vld1q_lane_f64(easysimd_float64_t const *ptr, easysimd_float64x2_t src,
                                     const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_float64x2_private r = easysimd_float64x2_to_private(src);
  r.values[lane] = *ptr;
  return easysimd_float64x2_from_private(r);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vld1q_lane_f64(ptr, src, lane) vld1q_lane_f64(ptr, src, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld1q_lane_f64
  #define vld1q_lane_f64(ptr, src, lane) easysimd_vld1q_lane_f64((ptr), (src), (lane))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_LD1_LANE_H) */
