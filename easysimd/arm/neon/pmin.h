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

#if !defined(EASYSIMD_ARM_NEON_PMIN_H)
#define EASYSIMD_ARM_NEON_PMIN_H

#include "types.h"
#include "min.h"
#include "uzp1.h"
#include "uzp2.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vpmins_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpmins_f32(a);
  #else
    easysimd_float32x2_private a_ = easysimd_float32x2_to_private(a);
    return (a_.values[0] < a_.values[1]) ? a_.values[0] : a_.values[1];
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpmins_f32
  #define vpmins_f32(a) easysimd_vpmins_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vpminqd_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminqd_f64(a);
  #else
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(a);
    return (a_.values[0] < a_.values[1]) ? a_.values[0] : a_.values[1];
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpminqd_f64
  #define vpminqd_f64(a) easysimd_vpminqd_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vpmin_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpmin_f32(a, b);
  #else
    return easysimd_vmin_f32(easysimd_vuzp1_f32(a, b), easysimd_vuzp2_f32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpmin_f32
  #define vpmin_f32(a, b) easysimd_vpmin_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vpmin_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpmin_s8(a, b);
  #else
    return easysimd_vmin_s8(easysimd_vuzp1_s8(a, b), easysimd_vuzp2_s8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpmin_s8
  #define vpmin_s8(a, b) easysimd_vpmin_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vpmin_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpmin_s16(a, b);
  #else
    return easysimd_vmin_s16(easysimd_vuzp1_s16(a, b), easysimd_vuzp2_s16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpmin_s16
  #define vpmin_s16(a, b) easysimd_vpmin_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vpmin_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpmin_s32(a, b);
  #else
    return easysimd_vmin_s32(easysimd_vuzp1_s32(a, b), easysimd_vuzp2_s32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpmin_s32
  #define vpmin_s32(a, b) easysimd_vpmin_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vpmin_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpmin_u8(a, b);
  #else
    return easysimd_vmin_u8(easysimd_vuzp1_u8(a, b), easysimd_vuzp2_u8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpmin_u8
  #define vpmin_u8(a, b) easysimd_vpmin_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vpmin_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpmin_u16(a, b);
  #else
    return easysimd_vmin_u16(easysimd_vuzp1_u16(a, b), easysimd_vuzp2_u16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpmin_u16
  #define vpmin_u16(a, b) easysimd_vpmin_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vpmin_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpmin_u32(a, b);
  #else
    return easysimd_vmin_u32(easysimd_vuzp1_u32(a, b), easysimd_vuzp2_u32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpmin_u32
  #define vpmin_u32(a, b) easysimd_vpmin_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vpminq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_f32(a, b);
  #elif defined(EASYSIMD_X86_SSE3_NATIVE)
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE3_NATIVE)
      __m128 e = _mm_shuffle_ps(a_.m128, b_.m128, _MM_SHUFFLE(2, 0, 2, 0));
      __m128 o = _mm_shuffle_ps(a_.m128, b_.m128, _MM_SHUFFLE(3, 1, 3, 1));
      r_.m128 = _mm_min_ps(e, o);
    #endif

    return easysimd_float32x4_from_private(r_);
  #else
    return easysimd_vminq_f32(easysimd_vuzp1q_f32(a, b), easysimd_vuzp2q_f32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpminq_f32
  #define vpminq_f32(a, b) easysimd_vpminq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vpminq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_f64(a, b);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      __m128d e = _mm_unpacklo_pd(a_.m128d, b_.m128d);
      __m128d o = _mm_unpackhi_pd(a_.m128d, b_.m128d);
      r_.m128d = _mm_min_pd(e, o);
    #endif

    return easysimd_float64x2_from_private(r_);
  #else
    return easysimd_vminq_f64(easysimd_vuzp1q_f64(a, b), easysimd_vuzp2q_f64(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vpminq_f64
  #define vpminq_f64(a, b) easysimd_vpminq_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vpminq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_s8(a, b);
  #else
    return easysimd_vminq_s8(easysimd_vuzp1q_s8(a, b), easysimd_vuzp2q_s8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpminq_s8
  #define vpminq_s8(a, b) easysimd_vpminq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vpminq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_s16(a, b);
  #else
    return easysimd_vminq_s16(easysimd_vuzp1q_s16(a, b), easysimd_vuzp2q_s16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpminq_s16
  #define vpminq_s16(a, b) easysimd_vpminq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vpminq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_s32(a, b);
  #else
    return easysimd_vminq_s32(easysimd_vuzp1q_s32(a, b), easysimd_vuzp2q_s32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpminq_s32
  #define vpminq_s32(a, b) easysimd_vpminq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vpminq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_u8(a, b);
  #else
    return easysimd_vminq_u8(easysimd_vuzp1q_u8(a, b), easysimd_vuzp2q_u8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpminq_u8
  #define vpminq_u8(a, b) easysimd_vpminq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vpminq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_u16(a, b);
  #else
    return easysimd_vminq_u16(easysimd_vuzp1q_u16(a, b), easysimd_vuzp2q_u16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpminq_u16
  #define vpminq_u16(a, b) easysimd_vpminq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vpminq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vpminq_u32(a, b);
  #else
    return easysimd_vminq_u32(easysimd_vuzp1q_u32(a, b), easysimd_vuzp2q_u32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpminq_u32
  #define vpminq_u32(a, b) easysimd_vpminq_u32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_PMIN_H) */
