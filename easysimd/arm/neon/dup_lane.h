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
 *   2020-2021 Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_NEON_DUP_LANE_H)
#define EASYSIMD_ARM_NEON_DUP_LANE_H

#include "dup_n.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vdups_lane_s32(easysimd_int32x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_int32x2_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdups_lane_s32(vec, lane) vdups_lane_s32(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdups_lane_s32
  #define vdups_lane_s32(vec, lane) easysimd_vdups_lane_s32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vdups_lane_u32(easysimd_uint32x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_uint32x2_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdups_lane_u32(vec, lane) vdups_lane_u32(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdups_lane_u32
  #define vdups_lane_u32(vec, lane) easysimd_vdups_lane_u32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vdups_lane_f32(easysimd_float32x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_float32x2_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdups_lane_f32(vec, lane) vdups_lane_f32(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdups_lane_f32
  #define vdups_lane_f32(vec, lane) easysimd_vdups_lane_f32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vdups_laneq_s32(easysimd_int32x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_int32x4_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdups_laneq_s32(vec, lane) vdups_laneq_s32(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdups_laneq_s32
  #define vdups_laneq_s32(vec, lane) easysimd_vdups_laneq_s32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vdups_laneq_u32(easysimd_uint32x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_uint32x4_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdups_laneq_u32(vec, lane) vdups_laneq_u32(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdups_laneq_u32
  #define vdups_laneq_u32(vec, lane) easysimd_vdups_laneq_u32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vdups_laneq_f32(easysimd_float32x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_float32x4_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdups_laneq_f32(vec, lane) vdups_laneq_f32(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdups_laneq_f32
  #define vdups_laneq_f32(vec, lane) easysimd_vdups_laneq_f32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vdupd_lane_s64(easysimd_int64x1_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  return easysimd_int64x1_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupd_lane_s64(vec, lane) vdupd_lane_s64(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupd_lane_s64
  #define vdupd_lane_s64(vec, lane) easysimd_vdupd_lane_s64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vdupd_lane_u64(easysimd_uint64x1_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  return easysimd_uint64x1_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupd_lane_u64(vec, lane) vdupd_lane_u64(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupd_lane_u64
  #define vdupd_lane_u64(vec, lane) easysimd_vdupd_lane_u64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vdupd_lane_f64(easysimd_float64x1_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  return easysimd_float64x1_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupd_lane_f64(vec, lane) vdupd_lane_f64(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupd_lane_f64
  #define vdupd_lane_f64(vec, lane) easysimd_vdupd_lane_f64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vdupd_laneq_s64(easysimd_int64x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_int64x2_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupd_laneq_s64(vec, lane) vdupd_laneq_s64(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupd_laneq_s64
  #define vdupd_laneq_s64(vec, lane) easysimd_vdupd_laneq_s64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vdupd_laneq_u64(easysimd_uint64x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_uint64x2_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupd_laneq_u64(vec, lane) vdupd_laneq_u64(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupd_laneq_u64
  #define vdupd_laneq_u64(vec, lane) easysimd_vdupd_laneq_u64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vdupd_laneq_f64(easysimd_float64x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_float64x2_to_private(vec).values[lane];
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupd_laneq_f64(vec, lane) vdupd_laneq_f64(vec, lane)
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupd_laneq_f64
  #define vdupd_laneq_f64(vec, lane) easysimd_vdupd_laneq_f64((vec), (lane))
#endif

//easysimd_vdup_lane_f32
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_f32(vec, lane) vdup_lane_f32(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vdup_lane_f32(vec, lane) (__extension__ ({ \
    easysimd_float32x2_private easysimd_vdup_lane_f32_vec_ = easysimd_float32x2_to_private(vec); \
    easysimd_float32x2_private easysimd_vdup_lane_f32_r_; \
    easysimd_vdup_lane_f32_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        32, 8, \
        easysimd_vdup_lane_f32_vec_.values, \
        easysimd_vdup_lane_f32_vec_.values, \
        lane, lane \
      ); \
    easysimd_float32x2_from_private(easysimd_vdup_lane_f32_r_); \
  }))
#else
  #define easysimd_vdup_lane_f32(vec, lane) easysimd_vdup_n_f32(easysimd_vdups_lane_f32(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_f32
  #define vdup_lane_f32(vec, lane) easysimd_vdup_lane_f32((vec), (lane))
#endif

//easysimd_vdup_lane_f64
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_lane_f64(vec, lane) vdup_lane_f64(vec, lane)
#else
  #define easysimd_vdup_lane_f64(vec, lane) easysimd_vdup_n_f64(easysimd_vdupd_lane_f64(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_f64
  #define vdup_lane_f64(vec, lane) easysimd_vdup_lane_f64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vdup_lane_s8(easysimd_int8x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdup_n_s8(easysimd_int8x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_s8(vec, lane) vdup_lane_s8(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vdup_lane_s8(vec, lane) (__extension__ ({ \
    easysimd_int8x8_private easysimd_vdup_lane_s8_vec_ = easysimd_int8x8_to_private(vec); \
    easysimd_int8x8_private easysimd_vdup_lane_s8_r_; \
    easysimd_vdup_lane_s8_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        8, 8, \
        easysimd_vdup_lane_s8_vec_.values, \
        easysimd_vdup_lane_s8_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_int8x8_from_private(easysimd_vdup_lane_s8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_s8
  #define vdup_lane_s8(vec, lane) easysimd_vdup_lane_s8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vdup_lane_s16(easysimd_int16x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_vdup_n_s16(easysimd_int16x4_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_s16(vec, lane) vdup_lane_s16(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vdup_lane_s16(vec, lane) (__extension__ ({ \
    easysimd_int16x4_private easysimd_vdup_lane_s16_vec_ = easysimd_int16x4_to_private(vec); \
    easysimd_int16x4_private easysimd_vdup_lane_s16_r_; \
    easysimd_vdup_lane_s16_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        16, 8, \
        easysimd_vdup_lane_s16_vec_.values, \
        easysimd_vdup_lane_s16_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_int16x4_from_private(easysimd_vdup_lane_s16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_s16
  #define vdup_lane_s16(vec, lane) easysimd_vdup_lane_s16((vec), (lane))
#endif

//easysimd_vdup_lane_s32
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_s32(vec, lane) vdup_lane_s32(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vdup_lane_s32(vec, lane) (__extension__ ({ \
    easysimd_int32x2_private easysimd_vdup_lane_s32_vec_ = easysimd_int32x2_to_private(vec); \
    easysimd_int32x2_private easysimd_vdup_lane_s32_r_; \
    easysimd_vdup_lane_s32_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        32, 8, \
        easysimd_vdup_lane_s32_vec_.values, \
        easysimd_vdup_lane_s32_vec_.values, \
        lane, lane \
      ); \
    easysimd_int32x2_from_private(easysimd_vdup_lane_s32_r_); \
  }))
#else
  #define easysimd_vdup_lane_s32(vec, lane) easysimd_vdup_n_s32(easysimd_vdups_lane_s32(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_s32
  #define vdup_lane_s32(vec, lane) easysimd_vdup_lane_s32((vec), (lane))
#endif

//easysimd_vdup_lane_s64
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_s64(vec, lane) vdup_lane_s64(vec, lane)
#else
  #define easysimd_vdup_lane_s64(vec, lane) easysimd_vdup_n_s64(easysimd_vdupd_lane_s64(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_s64
  #define vdup_lane_s64(vec, lane) easysimd_vdup_lane_s64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vdup_lane_u8(easysimd_uint8x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdup_n_u8(easysimd_uint8x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_u8(vec, lane) vdup_lane_u8(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vdup_lane_u8(vec, lane) (__extension__ ({ \
    easysimd_uint8x8_private easysimd_vdup_lane_u8_vec_ = easysimd_uint8x8_to_private(vec); \
    easysimd_uint8x8_private easysimd_vdup_lane_u8_r_; \
    easysimd_vdup_lane_u8_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        8, 8, \
        easysimd_vdup_lane_u8_vec_.values, \
        easysimd_vdup_lane_u8_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_uint8x8_from_private(easysimd_vdup_lane_u8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_u8
  #define vdup_lane_u8(vec, lane) easysimd_vdup_lane_u8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vdup_lane_u16(easysimd_uint16x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_vdup_n_u16(easysimd_uint16x4_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_u16(vec, lane) vdup_lane_u16(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vdup_lane_u16(vec, lane) (__extension__ ({ \
    easysimd_uint16x4_private easysimd_vdup_lane_u16_vec_ = easysimd_uint16x4_to_private(vec); \
    easysimd_uint16x4_private easysimd_vdup_lane_u16_r_; \
    easysimd_vdup_lane_u16_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        16, 8, \
        easysimd_vdup_lane_u16_vec_.values, \
        easysimd_vdup_lane_u16_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_uint16x4_from_private(easysimd_vdup_lane_u16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_u16
  #define vdup_lane_u16(vec, lane) easysimd_vdup_lane_u16((vec), (lane))
#endif

//easysimd_vdup_lane_u32
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_u32(vec, lane) vdup_lane_u32(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vdup_lane_u32(vec, lane) (__extension__ ({ \
    easysimd_uint32x2_private easysimd_vdup_lane_u32_vec_ = easysimd_uint32x2_to_private(vec); \
    easysimd_uint32x2_private easysimd_vdup_lane_u32_r_; \
    easysimd_vdup_lane_u32_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        32, 8, \
        easysimd_vdup_lane_u32_vec_.values, \
        easysimd_vdup_lane_u32_vec_.values, \
        lane, lane \
      ); \
    easysimd_uint32x2_from_private(easysimd_vdup_lane_u32_r_); \
  }))
#else
  #define easysimd_vdup_lane_u32(vec, lane) easysimd_vdup_n_u32(easysimd_vdups_lane_u32(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_u32
  #define vdup_lane_u32(vec, lane) easysimd_vdup_lane_u32((vec), (lane))
#endif

//easysimd_vdup_lane_u64
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdup_lane_u64(vec, lane) vdup_lane_u64(vec, lane)
#else
  #define easysimd_vdup_lane_u64(vec, lane) easysimd_vdup_n_u64(easysimd_vdupd_lane_u64(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_lane_u64
  #define vdup_lane_u64(vec, lane) easysimd_vdup_lane_u64((vec), (lane))
#endif

//easysimd_vdup_laneq_f32
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_f32(vec, lane) vdup_laneq_f32(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_f32(vec, lane) (__extension__ ({ \
    easysimd_float32x4_private easysimd_vdup_laneq_f32_vec_ = easysimd_float32x4_to_private(vec); \
    easysimd_float32x2_private easysimd_vdup_laneq_f32_r_; \
    easysimd_vdup_laneq_f32_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_f32_vec_.values, \
        easysimd_vdup_laneq_f32_vec_.values, \
        lane, lane \
      ); \
    easysimd_float32x2_from_private(easysimd_vdup_laneq_f32_r_); \
  }))
#else
  #define easysimd_vdup_laneq_f32(vec, lane) easysimd_vdup_n_f32(easysimd_vdups_laneq_f32(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_f32
  #define vdup_laneq_f32(vec, lane) easysimd_vdup_laneq_f32((vec), (lane))
#endif

#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_f64(vec, lane) vdup_laneq_f64(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_f64(vec, lane) (__extension__ ({ \
    easysimd_float64x2_private easysimd_vdup_laneq_f64_vec_ = easysimd_float64x2_to_private(vec); \
    easysimd_float64x1_private easysimd_vdup_laneq_f64_r_; \
    easysimd_vdup_laneq_f64_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_f64_vec_.values, \
        easysimd_vdup_laneq_f64_vec_.values, \
        lane \
      ); \
    easysimd_float64x1_from_private(easysimd_vdup_laneq_f64_r_); \
  }))
#else
  #define easysimd_vdup_laneq_f64(vec, lane) easysimd_vdup_n_f64(easysimd_vdupd_laneq_f64(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_f64
  #define vdup_laneq_f64(vec, lane) easysimd_vdup_laneq_f64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vdup_laneq_s8(easysimd_int8x16_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  return easysimd_vdup_n_s8(easysimd_int8x16_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_s8(vec, lane) vdup_laneq_s8(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_s8(vec, lane) (__extension__ ({ \
    easysimd_int8x16_private easysimd_vdup_laneq_s8_vec_ = easysimd_int8x16_to_private(vec); \
    easysimd_int8x8_private easysimd_vdup_laneq_s8_r_; \
    easysimd_vdup_laneq_s8_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_s8_vec_.values, \
        easysimd_vdup_laneq_s8_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_int8x8_from_private(easysimd_vdup_laneq_s8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_s8
  #define vdup_laneq_s8(vec, lane) easysimd_vdup_laneq_s8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vdup_laneq_s16(easysimd_int16x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdup_n_s16(easysimd_int16x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_s16(vec, lane) vdup_laneq_s16(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_s16(vec, lane) (__extension__ ({ \
    easysimd_int16x8_private easysimd_vdup_laneq_s16_vec_ = easysimd_int16x8_to_private(vec); \
    easysimd_int16x4_private easysimd_vdup_laneq_s16_r_; \
    easysimd_vdup_laneq_s16_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_s16_vec_.values, \
        easysimd_vdup_laneq_s16_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_int16x4_from_private(easysimd_vdup_laneq_s16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_s16
  #define vdup_laneq_s16(vec, lane) easysimd_vdup_laneq_s16((vec), (lane))
#endif

//easysimd_vdup_laneq_s32
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_s32(vec, lane) vdup_laneq_s32(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_s32(vec, lane) (__extension__ ({ \
    easysimd_int32x4_private easysimd_vdup_laneq_s32_vec_ = easysimd_int32x4_to_private(vec); \
    easysimd_int32x2_private easysimd_vdup_laneq_s32_r_; \
    easysimd_vdup_laneq_s32_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_s32_vec_.values, \
        easysimd_vdup_laneq_s32_vec_.values, \
        lane, lane \
      ); \
    easysimd_int32x2_from_private(easysimd_vdup_laneq_s32_r_); \
  }))
#else
  #define easysimd_vdup_laneq_s32(vec, lane) easysimd_vdup_n_s32(easysimd_vdups_laneq_s32(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_s32
  #define vdup_laneq_s32(vec, lane) easysimd_vdup_laneq_s32((vec), (lane))
#endif

//easysimd_vdup_laneq_s64
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_s64(vec, lane) vdup_laneq_s64(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_s64(vec, lane) (__extension__ ({ \
    easysimd_int64x2_private easysimd_vdup_laneq_s64_vec_ = easysimd_int64x2_to_private(vec); \
    easysimd_int64x1_private easysimd_vdup_laneq_s64_r_; \
    easysimd_vdup_laneq_s64_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_s64_vec_.values, \
        easysimd_vdup_laneq_s64_vec_.values, \
        lane \
      ); \
    easysimd_int64x1_from_private(easysimd_vdup_laneq_s64_r_); \
  }))
#else
  #define easysimd_vdup_laneq_s64(vec, lane) easysimd_vdup_n_s64(easysimd_vdupd_laneq_s64(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_s64
  #define vdup_laneq_s64(vec, lane) easysimd_vdup_laneq_s64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vdup_laneq_u8(easysimd_uint8x16_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  return easysimd_vdup_n_u8(easysimd_uint8x16_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_u8(vec, lane) vdup_laneq_u8(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_u8(vec, lane) (__extension__ ({ \
    easysimd_uint8x16_private easysimd_vdup_laneq_u8_vec_ = easysimd_uint8x16_to_private(vec); \
    easysimd_uint8x8_private easysimd_vdup_laneq_u8_r_; \
    easysimd_vdup_laneq_u8_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_u8_vec_.values, \
        easysimd_vdup_laneq_u8_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_uint8x8_from_private(easysimd_vdup_laneq_u8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_u8
  #define vdup_laneq_u8(vec, lane) easysimd_vdup_laneq_u8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vdup_laneq_u16(easysimd_uint16x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdup_n_u16(easysimd_uint16x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_u16(vec, lane) vdup_laneq_u16(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_u16(vec, lane) (__extension__ ({ \
    easysimd_uint16x8_private easysimd_vdup_laneq_u16_vec_ = easysimd_uint16x8_to_private(vec); \
    easysimd_uint16x4_private easysimd_vdup_laneq_u16_r_; \
    easysimd_vdup_laneq_u16_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_u16_vec_.values, \
        easysimd_vdup_laneq_u16_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_uint16x4_from_private(easysimd_vdup_laneq_u16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_u16
  #define vdup_laneq_u16(vec, lane) easysimd_vdup_laneq_u16((vec), (lane))
#endif

//easysimd_vdup_laneq_u32
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_u32(vec, lane) vdup_laneq_u32(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_u32(vec, lane) (__extension__ ({ \
    easysimd_uint32x4_private easysimd_vdup_laneq_u32_vec_ = easysimd_uint32x4_to_private(vec); \
    easysimd_uint32x2_private easysimd_vdup_laneq_u32_r_; \
    easysimd_vdup_laneq_u32_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_u32_vec_.values, \
        easysimd_vdup_laneq_u32_vec_.values, \
        lane, lane \
      ); \
    easysimd_uint32x2_from_private(easysimd_vdup_laneq_u32_r_); \
  }))
#else
  #define easysimd_vdup_laneq_u32(vec, lane) easysimd_vdup_n_u32(easysimd_vdups_laneq_u32(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_u32
  #define vdup_laneq_u32(vec, lane) easysimd_vdup_laneq_u32((vec), (lane))
#endif

//easysimd_vdup_laneq_u64
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdup_laneq_u64(vec, lane) vdup_laneq_u64(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdup_laneq_u64(vec, lane) (__extension__ ({ \
    easysimd_uint64x2_private easysimd_vdup_laneq_u64_vec_ = easysimd_uint64x2_to_private(vec); \
    easysimd_uint64x1_private easysimd_vdup_laneq_u64_r_; \
    easysimd_vdup_laneq_u64_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdup_laneq_u64_vec_.values, \
        easysimd_vdup_laneq_u64_vec_.values, \
        lane \
      ); \
    easysimd_uint64x1_from_private(easysimd_vdup_laneq_u64_r_); \
  }))
#else
  #define easysimd_vdup_laneq_u64(vec, lane) easysimd_vdup_n_u64(easysimd_vdupd_laneq_u64(vec, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_laneq_u64
  #define vdup_laneq_u64(vec, lane) easysimd_vdup_laneq_u64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vdupq_lane_f32(easysimd_float32x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_vdupq_n_f32(easysimd_float32x2_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_f32(vec, lane) vdupq_lane_f32(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_f32(vec, lane) (__extension__ ({ \
    easysimd_float32x2_private easysimd_vdupq_lane_f32_vec_ = easysimd_float32x2_to_private(vec); \
    easysimd_float32x4_private easysimd_vdupq_lane_f32_r_; \
    easysimd_vdupq_lane_f32_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_f32_vec_.values, \
        easysimd_vdupq_lane_f32_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_float32x4_from_private(easysimd_vdupq_lane_f32_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_f32
  #define vdupq_lane_f32(vec, lane) easysimd_vdupq_lane_f32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vdupq_lane_f64(easysimd_float64x1_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  return easysimd_vdupq_n_f64(easysimd_float64x1_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_lane_f64(vec, lane) vdupq_lane_f64(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_f64(vec, lane) (__extension__ ({ \
    easysimd_float64x1_private easysimd_vdupq_lane_f64_vec_ = easysimd_float64x1_to_private(vec); \
    easysimd_float64x2_private easysimd_vdupq_lane_f64_r_; \
    easysimd_vdupq_lane_f64_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_f64_vec_.values, \
        easysimd_vdupq_lane_f64_vec_.values, \
        lane, lane \
      ); \
    easysimd_float64x2_from_private(easysimd_vdupq_lane_f64_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_f64
  #define vdupq_lane_f64(vec, lane) easysimd_vdupq_lane_f64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vdupq_lane_s8(easysimd_int8x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdupq_n_s8(easysimd_int8x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_s8(vec, lane) vdupq_lane_s8(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_s8(vec, lane) (__extension__ ({ \
    easysimd_int8x8_private easysimd_vdupq_lane_s8_vec_ = easysimd_int8x8_to_private(vec); \
    easysimd_int8x16_private easysimd_vdupq_lane_s8_r_; \
    easysimd_vdupq_lane_s8_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_s8_vec_.values, \
        easysimd_vdupq_lane_s8_vec_.values, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane \
      ); \
    easysimd_int8x16_from_private(easysimd_vdupq_lane_s8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_s8
  #define vdupq_lane_s8(vec, lane) easysimd_vdupq_lane_s8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vdupq_lane_s16(easysimd_int16x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_vdupq_n_s16(easysimd_int16x4_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_s16(vec, lane) vdupq_lane_s16(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_s16(vec, lane) (__extension__ ({ \
    easysimd_int16x4_private easysimd_vdupq_lane_s16_vec_ = easysimd_int16x4_to_private(vec); \
    easysimd_int16x8_private easysimd_vdupq_lane_s16_r_; \
    easysimd_vdupq_lane_s16_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_s16_vec_.values, \
        easysimd_vdupq_lane_s16_vec_.values, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane \
      ); \
    easysimd_int16x8_from_private(easysimd_vdupq_lane_s16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_s16
  #define vdupq_lane_s16(vec, lane) easysimd_vdupq_lane_s16((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vdupq_lane_s32(easysimd_int32x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_vdupq_n_s32(easysimd_int32x2_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_s32(vec, lane) vdupq_lane_s32(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_s32(vec, lane) (__extension__ ({ \
    easysimd_int32x2_private easysimd_vdupq_lane_s32_vec_ = easysimd_int32x2_to_private(vec); \
    easysimd_int32x4_private easysimd_vdupq_lane_s32_r_; \
    easysimd_vdupq_lane_s32_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_s32_vec_.values, \
        easysimd_vdupq_lane_s32_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_int32x4_from_private(easysimd_vdupq_lane_s32_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_s32
  #define vdupq_lane_s32(vec, lane) easysimd_vdupq_lane_s32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vdupq_lane_s64(easysimd_int64x1_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  return easysimd_vdupq_n_s64(easysimd_int64x1_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_s64(vec, lane) vdupq_lane_s64(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_s64(vec, lane) (__extension__ ({ \
    easysimd_int64x1_private easysimd_vdupq_lane_s64_vec_ = easysimd_int64x1_to_private(vec); \
    easysimd_int64x2_private easysimd_vdupq_lane_s64_r_; \
    easysimd_vdupq_lane_s64_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_s64_vec_.values, \
        easysimd_vdupq_lane_s64_vec_.values, \
        lane, lane \
      ); \
    easysimd_int64x2_from_private(easysimd_vdupq_lane_s64_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_s64
  #define vdupq_lane_s64(vec, lane) easysimd_vdupq_lane_s64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vdupq_lane_u8(easysimd_uint8x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdupq_n_u8(easysimd_uint8x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_u8(vec, lane) vdupq_lane_u8(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_u8(vec, lane) (__extension__ ({ \
    easysimd_uint8x8_private easysimd_vdupq_lane_u8_vec_ = easysimd_uint8x8_to_private(vec); \
    easysimd_uint8x16_private easysimd_vdupq_lane_u8_r_; \
    easysimd_vdupq_lane_u8_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_u8_vec_.values, \
        easysimd_vdupq_lane_u8_vec_.values, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane \
      ); \
    easysimd_uint8x16_from_private(easysimd_vdupq_lane_u8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_u8
  #define vdupq_lane_u8(vec, lane) easysimd_vdupq_lane_u8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vdupq_lane_u16(easysimd_uint16x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_vdupq_n_u16(easysimd_uint16x4_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_u16(vec, lane) vdupq_lane_u16(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_u16(vec, lane) (__extension__ ({ \
    easysimd_uint16x4_private easysimd_vdupq_lane_u16_vec_ = easysimd_uint16x4_to_private(vec); \
    easysimd_uint16x8_private easysimd_vdupq_lane_u16_r_; \
    easysimd_vdupq_lane_u16_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_u16_vec_.values, \
        easysimd_vdupq_lane_u16_vec_.values, \
        lane, lane, lane, lane, \
        lane, lane, lane, lane \
      ); \
    easysimd_uint16x8_from_private(easysimd_vdupq_lane_u16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_u16
  #define vdupq_lane_u16(vec, lane) easysimd_vdupq_lane_u16((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vdupq_lane_u32(easysimd_uint32x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_vdupq_n_u32(easysimd_uint32x2_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_u32(vec, lane) vdupq_lane_u32(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_u32(vec, lane) (__extension__ ({ \
    easysimd_uint32x2_private easysimd_vdupq_lane_u32_vec_ = easysimd_uint32x2_to_private(vec); \
    easysimd_uint32x4_private easysimd_vdupq_lane_u32_r_; \
    easysimd_vdupq_lane_u32_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_u32_vec_.values, \
        easysimd_vdupq_lane_u32_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_uint32x4_from_private(easysimd_vdupq_lane_u32_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_u32
  #define vdupq_lane_u32(vec, lane) easysimd_vdupq_lane_u32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vdupq_lane_u64(easysimd_uint64x1_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 0) {
  return easysimd_vdupq_n_u64(easysimd_uint64x1_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vdupq_lane_u64(vec, lane) vdupq_lane_u64(vec, lane)
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vdupq_lane_u64(vec, lane) (__extension__ ({ \
    easysimd_uint64x1_private easysimd_vdupq_lane_u64_vec_ = easysimd_uint64x1_to_private(vec); \
    easysimd_uint64x2_private easysimd_vdupq_lane_u64_r_; \
    easysimd_vdupq_lane_u64_r_.values = \
      __builtin_shufflevector( \
        easysimd_vdupq_lane_u64_vec_.values, \
        easysimd_vdupq_lane_u64_vec_.values, \
        lane, lane \
      ); \
    easysimd_uint64x2_from_private(easysimd_vdupq_lane_u64_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_lane_u64
  #define vdupq_lane_u64(vec, lane) easysimd_vdupq_lane_u64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vdupq_laneq_f32(easysimd_float32x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_vdupq_n_f32(easysimd_float32x4_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_f32(vec, lane) vdupq_laneq_f32(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_f32(vec, lane) (__extension__ ({ \
    easysimd_float32x4_private easysimd_vdupq_laneq_f32_vec_ = easysimd_float32x4_to_private(vec); \
    easysimd_float32x4_private easysimd_vdupq_laneq_f32_r_; \
    easysimd_vdupq_laneq_f32_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        32, 16, \
        easysimd_vdupq_laneq_f32_vec_.values, \
        easysimd_vdupq_laneq_f32_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_float32x4_from_private(easysimd_vdupq_laneq_f32_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_f32
  #define vdupq_laneq_f32(vec, lane) easysimd_vdupq_laneq_f32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vdupq_laneq_f64(easysimd_float64x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_vdupq_n_f64(easysimd_float64x2_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_f64(vec, lane) vdupq_laneq_f64(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_f64(vec, lane) (__extension__ ({ \
    easysimd_float64x2_private easysimd_vdupq_laneq_f64_vec_ = easysimd_float64x2_to_private(vec); \
    easysimd_float64x2_private easysimd_vdupq_laneq_f64_r_; \
    easysimd_vdupq_laneq_f64_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        64, 16, \
        easysimd_vdupq_laneq_f64_vec_.values, \
        easysimd_vdupq_laneq_f64_vec_.values, \
        lane, lane \
      ); \
    easysimd_float64x2_from_private(easysimd_vdupq_laneq_f64_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_f64
  #define vdupq_laneq_f64(vec, lane) easysimd_vdupq_laneq_f64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vdupq_laneq_s8(easysimd_int8x16_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  return easysimd_vdupq_n_s8(easysimd_int8x16_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_s8(vec, lane) vdupq_laneq_s8(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_s8(vec, lane) (__extension__ ({ \
    easysimd_int8x16_private easysimd_vdupq_laneq_s8_vec_ = easysimd_int8x16_to_private(vec); \
    easysimd_int8x16_private easysimd_vdupq_laneq_s8_r_; \
    easysimd_vdupq_laneq_s8_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        8, 16, \
        easysimd_vdupq_laneq_s8_vec_.values, \
        easysimd_vdupq_laneq_s8_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_int8x16_from_private(easysimd_vdupq_laneq_s8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_s8
  #define vdupq_laneq_s8(vec, lane) easysimd_vdupq_laneq_s8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vdupq_laneq_s16(easysimd_int16x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdupq_n_s16(easysimd_int16x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_s16(vec, lane) vdupq_laneq_s16(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_s16(vec, lane) (__extension__ ({ \
    easysimd_int16x8_private easysimd_vdupq_laneq_s16_vec_ = easysimd_int16x8_to_private(vec); \
    easysimd_int16x8_private easysimd_vdupq_laneq_s16_r_; \
    easysimd_vdupq_laneq_s16_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        16, 16, \
        easysimd_vdupq_laneq_s16_vec_.values, \
        easysimd_vdupq_laneq_s16_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_int16x8_from_private(easysimd_vdupq_laneq_s16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_s16
  #define vdupq_laneq_s16(vec, lane) easysimd_vdupq_laneq_s16((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vdupq_laneq_s32(easysimd_int32x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_vdupq_n_s32(easysimd_int32x4_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_s32(vec, lane) vdupq_laneq_s32(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_s32(vec, lane) (__extension__ ({ \
    easysimd_int32x4_private easysimd_vdupq_laneq_s32_vec_ = easysimd_int32x4_to_private(vec); \
    easysimd_int32x4_private easysimd_vdupq_laneq_s32_r_; \
    easysimd_vdupq_laneq_s32_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        32, 16, \
        easysimd_vdupq_laneq_s32_vec_.values, \
        easysimd_vdupq_laneq_s32_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_int32x4_from_private(easysimd_vdupq_laneq_s32_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_s32
  #define vdupq_laneq_s32(vec, lane) easysimd_vdupq_laneq_s32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vdupq_laneq_s64(easysimd_int64x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_vdupq_n_s64(easysimd_int64x2_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_s64(vec, lane) vdupq_laneq_s64(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_s64(vec, lane) (__extension__ ({ \
    easysimd_int64x2_private easysimd_vdupq_laneq_s64_vec_ = easysimd_int64x2_to_private(vec); \
    easysimd_int64x2_private easysimd_vdupq_laneq_s64_r_; \
    easysimd_vdupq_laneq_s64_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        64, 16, \
        easysimd_vdupq_laneq_s64_vec_.values, \
        easysimd_vdupq_laneq_s64_vec_.values, \
        lane, lane \
      ); \
    easysimd_int64x2_from_private(easysimd_vdupq_laneq_s64_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_s64
  #define vdupq_laneq_s64(vec, lane) easysimd_vdupq_laneq_s64((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vdupq_laneq_u8(easysimd_uint8x16_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 15) {
  return easysimd_vdupq_n_u8(easysimd_uint8x16_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_u8(vec, lane) vdupq_laneq_u8(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_u8(vec, lane) (__extension__ ({ \
    easysimd_uint8x16_private easysimd_vdupq_laneq_u8_vec_ = easysimd_uint8x16_to_private(vec); \
    easysimd_uint8x16_private easysimd_vdupq_laneq_u8_r_; \
    easysimd_vdupq_laneq_u8_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        8, 16, \
        easysimd_vdupq_laneq_u8_vec_.values, \
        easysimd_vdupq_laneq_u8_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_uint8x16_from_private(easysimd_vdupq_laneq_u8_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_u8
  #define vdupq_laneq_u8(vec, lane) easysimd_vdupq_laneq_u8((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vdupq_laneq_u16(easysimd_uint16x8_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 7) {
  return easysimd_vdupq_n_u16(easysimd_uint16x8_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_u16(vec, lane) vdupq_laneq_u16(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_u16(vec, lane) (__extension__ ({ \
    easysimd_uint16x8_private easysimd_vdupq_laneq_u16_vec_ = easysimd_uint16x8_to_private(vec); \
    easysimd_uint16x8_private easysimd_vdupq_laneq_u16_r_; \
    easysimd_vdupq_laneq_u16_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        16, 16, \
        easysimd_vdupq_laneq_u16_vec_.values, \
        easysimd_vdupq_laneq_u16_vec_.values, \
        lane, lane, lane, lane, lane, lane, lane, lane \
      ); \
    easysimd_uint16x8_from_private(easysimd_vdupq_laneq_u16_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_u16
  #define vdupq_laneq_u16(vec, lane) easysimd_vdupq_laneq_u16((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vdupq_laneq_u32(easysimd_uint32x4_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  return easysimd_vdupq_n_u32(easysimd_uint32x4_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_u32(vec, lane) vdupq_laneq_u32(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_u32(vec, lane) (__extension__ ({ \
    easysimd_uint32x4_private easysimd_vdupq_laneq_u32_vec_ = easysimd_uint32x4_to_private(vec); \
    easysimd_uint32x4_private easysimd_vdupq_laneq_u32_r_; \
    easysimd_vdupq_laneq_u32_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        32, 16, \
        easysimd_vdupq_laneq_u32_vec_.values, \
        easysimd_vdupq_laneq_u32_vec_.values, \
        lane, lane, lane, lane \
      ); \
    easysimd_uint32x4_from_private(easysimd_vdupq_laneq_u32_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_u32
  #define vdupq_laneq_u32(vec, lane) easysimd_vdupq_laneq_u32((vec), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vdupq_laneq_u64(easysimd_uint64x2_t vec, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  return easysimd_vdupq_n_u64(easysimd_uint64x2_to_private(vec).values[lane]);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vdupq_laneq_u64(vec, lane) vdupq_laneq_u64(vec, lane)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_vdupq_laneq_u64(vec, lane) (__extension__ ({ \
    easysimd_uint64x2_private easysimd_vdupq_laneq_u64_vec_ = easysimd_uint64x2_to_private(vec); \
    easysimd_uint64x2_private easysimd_vdupq_laneq_u64_r_; \
    easysimd_vdupq_laneq_u64_r_.values = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        64, 16, \
        easysimd_vdupq_laneq_u64_vec_.values, \
        easysimd_vdupq_laneq_u64_vec_.values, \
        lane, lane \
      ); \
    easysimd_uint64x2_from_private(easysimd_vdupq_laneq_u64_r_); \
  }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_laneq_u64
  #define vdupq_laneq_u64(vec, lane) easysimd_vdupq_laneq_u64((vec), (lane))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_DUP_LANE_H) */
