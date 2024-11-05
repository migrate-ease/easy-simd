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

#if !defined(EASYSIMD_ARM_NEON_MLS_H)
#define EASYSIMD_ARM_NEON_MLS_H

#include "mul.h"
#include "sub.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vmls_f32(easysimd_float32x2_t a, easysimd_float32x2_t b, easysimd_float32x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_f32(a, b, c);
  #else
    return easysimd_vsub_f32(a, easysimd_vmul_f32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_f32
  #define vmls_f32(a, b, c) easysimd_vmls_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vmls_f64(easysimd_float64x1_t a, easysimd_float64x1_t b, easysimd_float64x1_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmls_f64(a, b, c);
  #else
    return easysimd_vsub_f64(a, easysimd_vmul_f64(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_f64
  #define vmls_f64(a, b, c) easysimd_vmls_f64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vmls_s8(easysimd_int8x8_t a, easysimd_int8x8_t b, easysimd_int8x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_s8(a, b, c);
  #else
    return easysimd_vsub_s8(a, easysimd_vmul_s8(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_s8
  #define vmls_s8(a, b, c) easysimd_vmls_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmls_s16(easysimd_int16x4_t a, easysimd_int16x4_t b, easysimd_int16x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_s16(a, b, c);
  #else
    return easysimd_vsub_s16(a, easysimd_vmul_s16(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_s16
  #define vmls_s16(a, b, c) easysimd_vmls_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmls_s32(easysimd_int32x2_t a, easysimd_int32x2_t b, easysimd_int32x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_s32(a, b, c);
  #else
    return easysimd_vsub_s32(a, easysimd_vmul_s32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_s32
  #define vmls_s32(a, b, c) easysimd_vmls_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vmls_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b, easysimd_uint8x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_u8(a, b, c);
  #else
    return easysimd_vsub_u8(a, easysimd_vmul_u8(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_u8
  #define vmls_u8(a, b, c) easysimd_vmls_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmls_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b, easysimd_uint16x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_u16(a, b, c);
  #else
    return easysimd_vsub_u16(a, easysimd_vmul_u16(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_u16
  #define vmls_u16(a, b, c) easysimd_vmls_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmls_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b, easysimd_uint32x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_u32(a, b, c);
  #else
    return easysimd_vsub_u32(a, easysimd_vmul_u32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_u32
  #define vmls_u32(a, b, c) easysimd_vmls_u32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vmlsq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b, easysimd_float32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_f32(a, b, c);
  #elif \
      defined(EASYSIMD_X86_FMA_NATIVE)
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b),
      c_ = easysimd_float32x4_to_private(c);

    #if defined(EASYSIMD_X86_FMA_NATIVE)
      r_.m128 = _mm_fnmadd_ps(b_.m128, c_.m128, a_.m128);
    #endif

    return easysimd_float32x4_from_private(r_);
  #else
    return easysimd_vsubq_f32(a, easysimd_vmulq_f32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_f32
  #define vmlsq_f32(a, b, c) easysimd_vmlsq_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vmlsq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b, easysimd_float64x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmlsq_f64(a, b, c);
  #elif \
      defined(EASYSIMD_X86_FMA_NATIVE)
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b),
      c_ = easysimd_float64x2_to_private(c);

    #if defined(EASYSIMD_X86_FMA_NATIVE)
      r_.m128d = _mm_fnmadd_pd(b_.m128d, c_.m128d, a_.m128d);
    #endif

    return easysimd_float64x2_from_private(r_);
  #else
    return easysimd_vsubq_f64(a, easysimd_vmulq_f64(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_f64
  #define vmlsq_f64(a, b, c) easysimd_vmlsq_f64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vmlsq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b, easysimd_int8x16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_s8(a, b, c);
  #else
    return easysimd_vsubq_s8(a, easysimd_vmulq_s8(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_s8
  #define vmlsq_s8(a, b, c) easysimd_vmlsq_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmlsq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b, easysimd_int16x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_s16(a, b, c);
  #else
    return easysimd_vsubq_s16(a, easysimd_vmulq_s16(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_s16
  #define vmlsq_s16(a, b, c) easysimd_vmlsq_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmlsq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b, easysimd_int32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_s32(a, b, c);
  #else
    return easysimd_vsubq_s32(a, easysimd_vmulq_s32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_s32
  #define vmlsq_s32(a, b, c) easysimd_vmlsq_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vmlsq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b, easysimd_uint8x16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_u8(a, b, c);
  #else
    return easysimd_vsubq_u8(a, easysimd_vmulq_u8(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_u8
  #define vmlsq_u8(a, b, c) easysimd_vmlsq_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmlsq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b, easysimd_uint16x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_u16(a, b, c);
  #else
    return easysimd_vsubq_u16(a, easysimd_vmulq_u16(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_u16
  #define vmlsq_u16(a, b, c) easysimd_vmlsq_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmlsq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b, easysimd_uint32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_u32(a, b, c);
  #else
    return easysimd_vsubq_u32(a, easysimd_vmulq_u32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_u32
  #define vmlsq_u32(a, b, c) easysimd_vmlsq_u32((a), (b), (c))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MLS_H) */
