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

#if !defined(EASYSIMD_ARM_NEON_DOT_LANE_H)
#define EASYSIMD_ARM_NEON_DOT_LANE_H

#include "types.h"

#include "add.h"
#include "dup_lane.h"
#include "paddl.h"
#include "movn.h"
#include "mull.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vdot_lane_s32(easysimd_int32x2_t r, easysimd_int8x8_t a, easysimd_int8x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int32x2_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_2_(vdot_lane_s32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int32x2_t
      b_lane,
      b_32 = vreinterpret_s32_s8(b);

    EASYSIMD_CONSTIFY_2_(vdup_lane_s32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);
    result =
      vadd_s32(
        r,
        vmovn_s64(
          vpaddlq_s32(
            vpaddlq_s16(
              vmull_s8(a, vreinterpret_s8_s32(b_lane))
            )
          )
        )
      );
  #else
    easysimd_int32x2_private r_ = easysimd_int32x2_to_private(r);
    easysimd_int8x8_private
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    for (int i = 0 ; i < 2 ; i++) {
      int32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(int32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(int32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_int32x2_from_private(r_);
  #endif

  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdot_lane_s32
  #define vdot_lane_s32(r, a, b, lane) easysimd_vdot_lane_s32((r), (a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vdot_lane_u32(easysimd_uint32x2_t r, easysimd_uint8x8_t a, easysimd_uint8x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint32x2_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_2_(vdot_lane_u32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint32x2_t
      b_lane,
      b_32 = vreinterpret_u32_u8(b);

    EASYSIMD_CONSTIFY_2_(vdup_lane_u32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);
    result =
      vadd_u32(
        r,
        vmovn_u64(
          vpaddlq_u32(
            vpaddlq_u16(
              vmull_u8(a, vreinterpret_u8_u32(b_lane))
            )
          )
        )
      );
  #else
    easysimd_uint32x2_private r_ = easysimd_uint32x2_to_private(r);
    easysimd_uint8x8_private
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    for (int i = 0 ; i < 2 ; i++) {
      uint32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(uint32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(uint32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_uint32x2_from_private(r_);
  #endif

  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdot_lane_u32
  #define vdot_lane_u32(r, a, b, lane) easysimd_vdot_lane_u32((r), (a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vdot_laneq_s32(easysimd_int32x2_t r, easysimd_int8x8_t a, easysimd_int8x16_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int32x2_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_4_(vdot_laneq_s32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int32x2_t b_lane;
    easysimd_int32x4_t b_32 = vreinterpretq_s32_s8(b);

    EASYSIMD_CONSTIFY_4_(easysimd_vdup_laneq_s32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);
    result =
      vadd_s32(
        r,
        vmovn_s64(
          vpaddlq_s32(
            vpaddlq_s16(
              vmull_s8(a, vreinterpret_s8_s32(b_lane))
            )
          )
        )
      );
  #else
    easysimd_int32x2_private r_ = easysimd_int32x2_to_private(r);
    easysimd_int8x8_private a_ = easysimd_int8x8_to_private(a);
    easysimd_int8x16_private b_ = easysimd_int8x16_to_private(b);

    for (int i = 0 ; i < 2 ; i++) {
      int32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(int32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(int32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_int32x2_from_private(r_);
  #endif

  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdot_laneq_s32
  #define vdot_laneq_s32(r, a, b, lane) easysimd_vdot_laneq_s32((r), (a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vdot_laneq_u32(easysimd_uint32x2_t r, easysimd_uint8x8_t a, easysimd_uint8x16_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint32x2_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_4_(vdot_laneq_u32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint32x2_t b_lane;
    easysimd_uint32x4_t b_32 = vreinterpretq_u32_u8(b);

    EASYSIMD_CONSTIFY_4_(easysimd_vdup_laneq_u32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);
    result =
      vadd_u32(
        r,
        vmovn_u64(
          vpaddlq_u32(
            vpaddlq_u16(
              vmull_u8(a, vreinterpret_u8_u32(b_lane))
            )
          )
        )
      );
  #else
    easysimd_uint32x2_private r_ = easysimd_uint32x2_to_private(r);
    easysimd_uint8x8_private a_ = easysimd_uint8x8_to_private(a);
    easysimd_uint8x16_private b_ = easysimd_uint8x16_to_private(b);

    for (int i = 0 ; i < 2 ; i++) {
      uint32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(uint32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(uint32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_uint32x2_from_private(r_);
  #endif
  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdot_laneq_u32
  #define vdot_laneq_u32(r, a, b, lane) easysimd_vdot_laneq_u32((r), (a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vdotq_laneq_u32(easysimd_uint32x4_t r, easysimd_uint8x16_t a, easysimd_uint8x16_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_uint32x4_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_4_(vdotq_laneq_u32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint32x4_t
      b_lane,
      b_32 = vreinterpretq_u32_u8(b);
    EASYSIMD_CONSTIFY_4_(easysimd_vdupq_laneq_u32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);

    result =
      vcombine_u32(
        vadd_u32(
          vget_low_u32(r),
          vmovn_u64(
            vpaddlq_u32(
              vpaddlq_u16(
                vmull_u8(vget_low_u8(a), vget_low_u8(vreinterpretq_u8_u32(b_lane)))
              )
            )
          )
        ),
        vadd_u32(
          vget_high_u32(r),
          vmovn_u64(
            vpaddlq_u32(
              vpaddlq_u16(
                vmull_u8(vget_high_u8(a), vget_high_u8(vreinterpretq_u8_u32(b_lane)))
              )
            )
          )
        )
      );
  #else
    easysimd_uint32x4_private r_ = easysimd_uint32x4_to_private(r);
    easysimd_uint8x16_private
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);

    for(int i = 0 ; i < 4 ; i++) {
      uint32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for(int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(uint32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(uint32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_uint32x4_from_private(r_);
  #endif
  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdotq_laneq_u32
  #define vdotq_laneq_u32(r, a, b, lane) easysimd_vdotq_laneq_u32((r), (a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vdotq_laneq_s32(easysimd_int32x4_t r, easysimd_int8x16_t a, easysimd_int8x16_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 3) {
  easysimd_int32x4_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_4_(vdotq_laneq_s32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int32x4_t
      b_lane,
      b_32 = vreinterpretq_s32_s8(b);
    EASYSIMD_CONSTIFY_4_(easysimd_vdupq_laneq_s32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);

    result =
      vcombine_s32(
        vadd_s32(
          vget_low_s32(r),
          vmovn_s64(
            vpaddlq_s32(
              vpaddlq_s16(
                vmull_s8(vget_low_s8(a), vget_low_s8(vreinterpretq_s8_s32(b_lane)))
              )
            )
          )
        ),
        vadd_s32(
          vget_high_s32(r),
          vmovn_s64(
            vpaddlq_s32(
              vpaddlq_s16(
                vmull_s8(vget_high_s8(a), vget_high_s8(vreinterpretq_s8_s32(b_lane)))
              )
            )
          )
        )
      );
  #else
    easysimd_int32x4_private r_ = easysimd_int32x4_to_private(r);
    easysimd_int8x16_private
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    for(int i = 0 ; i < 4 ; i++) {
      int32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for(int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(int32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(int32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_int32x4_from_private(r_);
  #endif
  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdotq_laneq_s32
  #define vdotq_laneq_s32(r, a, b, lane) easysimd_vdotq_laneq_s32((r), (a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vdotq_lane_u32(easysimd_uint32x4_t r, easysimd_uint8x16_t a, easysimd_uint8x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_uint32x4_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_2_(vdotq_lane_u32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint32x2_t
      b_lane,
      b_32 = vreinterpret_u32_u8(b);
    EASYSIMD_CONSTIFY_2_(easysimd_vdup_lane_u32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);

    result =
      vcombine_u32(
        vadd_u32(
          vget_low_u32(r),
          vmovn_u64(
            vpaddlq_u32(
              vpaddlq_u16(
                vmull_u8(vget_low_u8(a), vreinterpret_u8_u32(b_lane))
              )
            )
          )
        ),
        vadd_u32(
          vget_high_u32(r),
          vmovn_u64(
            vpaddlq_u32(
              vpaddlq_u16(
                vmull_u8(vget_high_u8(a), vreinterpret_u8_u32(b_lane))
              )
            )
          )
        )
      );
  #else
    easysimd_uint32x4_private r_ = easysimd_uint32x4_to_private(r);
    easysimd_uint8x16_private a_ = easysimd_uint8x16_to_private(a);
    easysimd_uint8x8_private b_ = easysimd_uint8x8_to_private(b);

    for(int i = 0 ; i < 4 ; i++) {
      uint32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for(int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(uint32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(uint32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_uint32x4_from_private(r_);
  #endif
  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdotq_lane_u32
  #define vdotq_lane_u32(r, a, b, lane) easysimd_vdotq_lane_u32((r), (a), (b), (lane))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vdotq_lane_s32(easysimd_int32x4_t r, easysimd_int8x16_t a, easysimd_int8x8_t b, const int lane)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lane, 0, 1) {
  easysimd_int32x4_t result;
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    EASYSIMD_CONSTIFY_2_(vdotq_lane_s32, result, (HEDLEY_UNREACHABLE(), result), lane, r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int32x2_t
      b_lane,
      b_32 = vreinterpret_s32_s8(b);
    EASYSIMD_CONSTIFY_2_(easysimd_vdup_lane_s32, b_lane, (HEDLEY_UNREACHABLE(), b_lane), lane, b_32);

    result =
      vcombine_s32(
        vadd_s32(
          vget_low_s32(r),
          vmovn_s64(
            vpaddlq_s32(
              vpaddlq_s16(
                vmull_s8(vget_low_s8(a), vreinterpret_s8_s32(b_lane))
              )
            )
          )
        ),
        vadd_s32(
          vget_high_s32(r),
          vmovn_s64(
            vpaddlq_s32(
              vpaddlq_s16(
                vmull_s8(vget_high_s8(a), vreinterpret_s8_s32(b_lane))
              )
            )
          )
        )
      );
  #else
    easysimd_int32x4_private r_ = easysimd_int32x4_to_private(r);
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);
    easysimd_int8x8_private b_ = easysimd_int8x8_to_private(b);

    for(int i = 0 ; i < 4 ; i++) {
      int32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for(int j = 0 ; j < 4 ; j++) {
        const int idx_b = j + (lane << 2);
        const int idx_a = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(int32_t, a_.values[idx_a]) * HEDLEY_STATIC_CAST(int32_t, b_.values[idx_b]);
      }
      r_.values[i] += acc;
    }

    result = easysimd_int32x4_from_private(r_);
  #endif
  return result;
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdotq_lane_s32
  #define vdotq_lane_s32(r, a, b, lane) easysimd_vdotq_lane_s32((r), (a), (b), (lane))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_DOT_LANE_H) */
