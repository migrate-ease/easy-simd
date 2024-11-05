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

#if !defined(EASYSIMD_ARM_NEON_MLA_H)
#define EASYSIMD_ARM_NEON_MLA_H

#include "types.h"
#include "add.h"
#include "mul.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vmla_f32(easysimd_float32x2_t a, easysimd_float32x2_t b, easysimd_float32x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmla_f32(a, b, c);
  #else
    return easysimd_vadd_f32(easysimd_vmul_f32(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmla_f32
  #define vmla_f32(a, b, c) easysimd_vmla_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vmla_f64(easysimd_float64x1_t a, easysimd_float64x1_t b, easysimd_float64x1_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmla_f64(a, b, c);
  #else
    return easysimd_vadd_f64(easysimd_vmul_f64(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmla_f64
  #define vmla_f64(a, b, c) easysimd_vmla_f64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vmla_s8(easysimd_int8x8_t a, easysimd_int8x8_t b, easysimd_int8x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmla_s8(a, b, c);
  #else
    return easysimd_vadd_s8(easysimd_vmul_s8(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmla_s8
  #define vmla_s8(a, b, c) easysimd_vmla_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmla_s16(easysimd_int16x4_t a, easysimd_int16x4_t b, easysimd_int16x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmla_s16(a, b, c);
  #else
    return easysimd_vadd_s16(easysimd_vmul_s16(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmla_s16
  #define vmla_s16(a, b, c) easysimd_vmla_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmla_s32(easysimd_int32x2_t a, easysimd_int32x2_t b, easysimd_int32x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmla_s32(a, b, c);
  #else
    return easysimd_vadd_s32(easysimd_vmul_s32(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmla_s32
  #define vmla_s32(a, b, c) easysimd_vmla_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vmla_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b, easysimd_uint8x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmla_u8(a, b, c);
  #else
    return easysimd_vadd_u8(easysimd_vmul_u8(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmla_u8
  #define vmla_u8(a, b, c) easysimd_vmla_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmla_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b, easysimd_uint16x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmla_u16(a, b, c);
  #else
    return easysimd_vadd_u16(easysimd_vmul_u16(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmla_u16
  #define vmla_u16(a, b, c) easysimd_vmla_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmla_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b, easysimd_uint32x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmla_u32(a, b, c);
  #else
    return easysimd_vadd_u32(easysimd_vmul_u32(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmla_u32
  #define vmla_u32(a, b, c) easysimd_vmla_u32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vmlaq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b, easysimd_float32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlaq_f32(a, b, c);
  #elif \
      defined(EASYSIMD_X86_FMA_NATIVE)
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b),
      c_ = easysimd_float32x4_to_private(c);

    #if defined(EASYSIMD_X86_FMA_NATIVE)
      r_.m128 = _mm_fmadd_ps(b_.m128, c_.m128, a_.m128);
    #endif

    return easysimd_float32x4_from_private(r_);
  #else
    return easysimd_vaddq_f32(easysimd_vmulq_f32(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_f32
  #define vmlaq_f32(a, b, c) easysimd_vmlaq_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vmlaq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b, easysimd_float64x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmlaq_f64(a, b, c);
  #elif \
      defined(EASYSIMD_X86_FMA_NATIVE)
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b),
      c_ = easysimd_float64x2_to_private(c);

    #if defined(EASYSIMD_X86_FMA_NATIVE)
      r_.m128d = _mm_fmadd_pd(b_.m128d, c_.m128d, a_.m128d);
    #endif

    return easysimd_float64x2_from_private(r_);
  #else
    return easysimd_vaddq_f64(easysimd_vmulq_f64(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_f64
  #define vmlaq_f64(a, b, c) easysimd_vmlaq_f64((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vmlaq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b, easysimd_int8x16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlaq_s8(a, b, c);
  #else
    return easysimd_vaddq_s8(easysimd_vmulq_s8(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_s8
  #define vmlaq_s8(a, b, c) easysimd_vmlaq_s8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmlaq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b, easysimd_int16x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlaq_s16(a, b, c);
  #else
    return easysimd_vaddq_s16(easysimd_vmulq_s16(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_s16
  #define vmlaq_s16(a, b, c) easysimd_vmlaq_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmlaq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b, easysimd_int32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlaq_s32(a, b, c);
  #else
    return easysimd_vaddq_s32(easysimd_vmulq_s32(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_s32
  #define vmlaq_s32(a, b, c) easysimd_vmlaq_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vmlaq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b, easysimd_uint8x16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlaq_u8(a, b, c);
  #else
    return easysimd_vaddq_u8(easysimd_vmulq_u8(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_u8
  #define vmlaq_u8(a, b, c) easysimd_vmlaq_u8((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmlaq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b, easysimd_uint16x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlaq_u16(a, b, c);
  #else
    return easysimd_vaddq_u16(easysimd_vmulq_u16(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_u16
  #define vmlaq_u16(a, b, c) easysimd_vmlaq_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmlaq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b, easysimd_uint32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlaq_u32(a, b, c);
  #else
    return easysimd_vaddq_u32(easysimd_vmulq_u32(b, c), a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlaq_u32
  #define vmlaq_u32(a, b, c) easysimd_vmlaq_u32((a), (b), (c))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MLA_H) */
