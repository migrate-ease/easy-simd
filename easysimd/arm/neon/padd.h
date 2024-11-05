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
 *   2020      Sean Maher <seanptmaher@gmail.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_PADD_H)
#define EASYSIMD_ARM_NEON_PADD_H

#include "add.h"
#include "uzp1.h"
#include "uzp2.h"
#include "types.h"
#include "get_lane.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vpaddd_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddd_s64(a);
  #else
    return easysimd_vaddd_s64(easysimd_vgetq_lane_s64(a, 0), easysimd_vgetq_lane_s64(a, 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpaddd_s64
  #define vpaddd_s64(a) easysimd_vpaddd_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vpaddd_u64(easysimd_uint64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddd_u64(a);
  #else
    return easysimd_vaddd_u64(easysimd_vgetq_lane_u64(a, 0), easysimd_vgetq_lane_u64(a, 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpaddd_u64
  #define vpaddd_u64(a) easysimd_vpaddd_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vpaddd_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddd_f64(a);
  #else
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);
    return a_.values[0] + a_.values[1];
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpaddd_f64
  #define vpaddd_f64(a) easysimd_vpaddd_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vpadds_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpadds_f32(a);
  #else
    easysimd_float32x2_private a_ = easysimd_float32x2_to_private(a);
    return a_.values[0] + a_.values[1];
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpadds_f32
  #define vpadds_f32(a) easysimd_vpadds_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vpadd_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !EASYSIMD_DETECT_CLANG_VERSION_NOT(9,0,0)
    return vpadd_f32(a, b);
  #else
    return easysimd_vadd_f32(easysimd_vuzp1_f32(a, b), easysimd_vuzp2_f32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadd_f32
  #define vpadd_f32(a, b) easysimd_vpadd_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vpadd_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadd_s8(a, b);
  #else
    return easysimd_vadd_s8(easysimd_vuzp1_s8(a, b), easysimd_vuzp2_s8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadd_s8
  #define vpadd_s8(a, b) easysimd_vpadd_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vpadd_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadd_s16(a, b);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return easysimd_int16x4_from_m64(_mm_hadd_pi16(easysimd_int16x4_to_m64(a), easysimd_int16x4_to_m64(b)));
  #else
    return easysimd_vadd_s16(easysimd_vuzp1_s16(a, b), easysimd_vuzp2_s16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadd_s16
  #define vpadd_s16(a, b) easysimd_vpadd_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vpadd_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadd_s32(a, b);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return easysimd_int32x2_from_m64(_mm_hadd_pi32(easysimd_int32x2_to_m64(a), easysimd_int32x2_to_m64(b)));
  #else
    return easysimd_vadd_s32(easysimd_vuzp1_s32(a, b), easysimd_vuzp2_s32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadd_s32
  #define vpadd_s32(a, b) easysimd_vpadd_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vpadd_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadd_u8(a, b);
  #else
    return easysimd_vadd_u8(easysimd_vuzp1_u8(a, b), easysimd_vuzp2_u8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadd_u8
  #define vpadd_u8(a, b) easysimd_vpadd_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vpadd_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadd_u16(a, b);
  #else
    return easysimd_vadd_u16(easysimd_vuzp1_u16(a, b), easysimd_vuzp2_u16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadd_u16
  #define vpadd_u16(a, b) easysimd_vpadd_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vpadd_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadd_u32(a, b);
  #else
    return easysimd_vadd_u32(easysimd_vuzp1_u32(a, b), easysimd_vuzp2_u32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadd_u32
  #define vpadd_u32(a, b) easysimd_vpadd_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vpaddq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_f32(a, b);
  #elif defined(EASYSIMD_X86_SSE3_NATIVE)
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE3_NATIVE)
      r_.m128 = _mm_hadd_ps(a_.m128, b_.m128);
    #endif

    return easysimd_float32x4_from_private(r_);
  #else
    return easysimd_vaddq_f32(easysimd_vuzp1q_f32(a, b), easysimd_vuzp2q_f32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_f32
  #define vpaddq_f32(a, b) easysimd_vpaddq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vpaddq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_f64(a, b);
  #elif defined(EASYSIMD_X86_SSE3_NATIVE)
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE3_NATIVE)
      r_.m128d = _mm_hadd_pd(a_.m128d, b_.m128d);
    #endif

    return easysimd_float64x2_from_private(r_);
  #else
    return easysimd_vaddq_f64(easysimd_vuzp1q_f64(a, b), easysimd_vuzp2q_f64(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_f64
  #define vpaddq_f64(a, b) easysimd_vpaddq_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vpaddq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_s8(a, b);
  #else
    return easysimd_vaddq_s8(easysimd_vuzp1q_s8(a, b), easysimd_vuzp2q_s8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_s8
  #define vpaddq_s8(a, b) easysimd_vpaddq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vpaddq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_s16(a, b);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_hadd_epi16(a_.m128i, b_.m128i);
    #endif

    return easysimd_int16x8_from_private(r_);
  #else
    return easysimd_vaddq_s16(easysimd_vuzp1q_s16(a, b), easysimd_vuzp2q_s16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_s16
  #define vpaddq_s16(a, b) easysimd_vpaddq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vpaddq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_s32(a, b);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_hadd_epi32(a_.m128i, b_.m128i);
    #endif

    return easysimd_int32x4_from_private(r_);
  #else
    return easysimd_vaddq_s32(easysimd_vuzp1q_s32(a, b), easysimd_vuzp2q_s32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_s32
  #define vpaddq_s32(a, b) easysimd_vpaddq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vpaddq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_s64(a, b);
  #else
    return easysimd_vaddq_s64(easysimd_vuzp1q_s64(a, b), easysimd_vuzp2q_s64(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_s64
  #define vpaddq_s64(a, b) easysimd_vpaddq_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vpaddq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_u8(a, b);
  #else
    return easysimd_vaddq_u8(easysimd_vuzp1q_u8(a, b), easysimd_vuzp2q_u8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_u8
  #define vpaddq_u8(a, b) easysimd_vpaddq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vpaddq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_u16(a, b);
  #else
    return easysimd_vaddq_u16(easysimd_vuzp1q_u16(a, b), easysimd_vuzp2q_u16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_u16
  #define vpaddq_u16(a, b) easysimd_vpaddq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vpaddq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_u32(a, b);
  #else
    return easysimd_vaddq_u32(easysimd_vuzp1q_u32(a, b), easysimd_vuzp2q_u32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_u32
  #define vpaddq_u32(a, b) easysimd_vpaddq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vpaddq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpaddq_u64(a, b);
  #else
    return easysimd_vaddq_u64(easysimd_vuzp1q_u64(a, b), easysimd_vuzp2q_u64(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddq_u64
  #define vpaddq_u64(a, b) easysimd_vpaddq_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_PADD_H) */
