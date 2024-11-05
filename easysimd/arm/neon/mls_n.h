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

#if !defined(EASYSIMD_ARM_NEON_MLS_N_H)
#define EASYSIMD_ARM_NEON_MLS_N_H

#include "sub.h"
#include "dup_n.h"
#include "mls.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vmls_n_f32(easysimd_float32x2_t a, easysimd_float32x2_t b, easysimd_float32 c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_n_f32(a, b, c);
  #else
    return easysimd_vmls_f32(a, b, easysimd_vdup_n_f32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_n_f32
  #define vmls_n_f32(a, b, c) easysimd_vmls_n_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmls_n_s16(easysimd_int16x4_t a, easysimd_int16x4_t b, int16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_n_s16(a, b, c);
  #else
    return easysimd_vmls_s16(a, b, easysimd_vdup_n_s16(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_n_s16
  #define vmls_n_s16(a, b, c) easysimd_vmls_n_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmls_n_s32(easysimd_int32x2_t a, easysimd_int32x2_t b, int32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_n_s32(a, b, c);
  #else
    return easysimd_vmls_s32(a, b, easysimd_vdup_n_s32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_n_s32
  #define vmls_n_s32(a, b, c) easysimd_vmls_n_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmls_n_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b, uint16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_n_u16(a, b, c);
  #else
    return easysimd_vmls_u16(a, b, easysimd_vdup_n_u16(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_n_u16
  #define vmls_n_u16(a, b, c) easysimd_vmls_n_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmls_n_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b, uint32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmls_n_u32(a, b, c);
  #else
    return easysimd_vmls_u32(a, b, easysimd_vdup_n_u32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmls_n_u32
  #define vmls_n_u32(a, b, c) easysimd_vmls_n_u32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vmlsq_n_f32(easysimd_float32x4_t a, easysimd_float32x4_t b, easysimd_float32 c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_n_f32(a, b, c);
  #else
    return easysimd_vmlsq_f32(a, b, easysimd_vdupq_n_f32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_n_f32
  #define vmlsq_n_f32(a, b, c) easysimd_vmlsq_n_f32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmlsq_n_s16(easysimd_int16x8_t a, easysimd_int16x8_t b, int16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_n_s16(a, b, c);
  #else
    return easysimd_vmlsq_s16(a, b, easysimd_vdupq_n_s16(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_n_s16
  #define vmlsq_n_s16(a, b, c) easysimd_vmlsq_n_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmlsq_n_s32(easysimd_int32x4_t a, easysimd_int32x4_t b, int32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_n_s32(a, b, c);
  #else
    return easysimd_vmlsq_s32(a, b, easysimd_vdupq_n_s32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_n_s32
  #define vmlsq_n_s32(a, b, c) easysimd_vmlsq_n_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmlsq_n_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b, uint16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_n_u16(a, b, c);
  #else
    return easysimd_vmlsq_u16(a, b, easysimd_vdupq_n_u16(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_n_u16
  #define vmlsq_n_u16(a, b, c) easysimd_vmlsq_n_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmlsq_n_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b, uint32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlsq_n_u32(a, b, c);
  #else
    return easysimd_vmlsq_u32(a, b, easysimd_vdupq_n_u32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlsq_n_u32
  #define vmlsq_n_u32(a, b, c) easysimd_vmlsq_n_u32((a), (b), (c))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MLS_N_H) */
