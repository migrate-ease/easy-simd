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

#if !defined(EASYSIMD_ARM_NEON_CLS_H)
#define EASYSIMD_ARM_NEON_CLS_H

#include "types.h"
#include "bsl.h"
#include "clz.h"
#include "cltz.h"
#include "dup_n.h"
#include "mvn.h"
#include "sub.h"
#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vcls_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vcls_s8(a);
  #else
    return easysimd_vsub_s8(easysimd_vclz_s8(easysimd_vbsl_s8(easysimd_vcltz_s8(a), easysimd_vmvn_s8(a), a)), easysimd_vdup_n_s8(INT8_C(1)));
  #endif
}
#define easysimd_vcls_u8(a) easysimd_vcls_s8(easysimd_vreinterpret_s8_u8(a))
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcls_s8
  #define vcls_s8(a) easysimd_vcls_s8(a)
  #undef vcls_u8
  #define vcls_u8(a) easysimd_vcls_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vcls_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vcls_s16(a);
  #else
    return easysimd_vsub_s16(easysimd_vclz_s16(easysimd_vbsl_s16(easysimd_vcltz_s16(a), easysimd_vmvn_s16(a), a)), easysimd_vdup_n_s16(INT16_C(1)));
  #endif
}
#define easysimd_vcls_u16(a) easysimd_vcls_s16(easysimd_vreinterpret_s16_u16(a))
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcls_s16
  #define vcls_s16(a) easysimd_vcls_s16(a)
  #undef vcls_u16
  #define vcls_u16(a) easysimd_vcls_u16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vcls_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vcls_s32(a);
  #else
    return easysimd_vsub_s32(easysimd_vclz_s32(easysimd_vbsl_s32(easysimd_vcltz_s32(a), easysimd_vmvn_s32(a), a)), easysimd_vdup_n_s32(INT32_C(1)));
  #endif
}
#define easysimd_vcls_u32(a) easysimd_vcls_s32(easysimd_vreinterpret_s32_u32(a))
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vcls_s32
  #define vcls_s32(a) easysimd_vcls_s32(a)
  #undef vcls_u32
  #define vcls_u32(a) easysimd_vcls_u32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vclsq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vclsq_s8(a);
  #else
    return easysimd_vsubq_s8(easysimd_vclzq_s8(easysimd_vbslq_s8(easysimd_vcltzq_s8(a), easysimd_vmvnq_s8(a), a)), easysimd_vdupq_n_s8(INT8_C(1)));
  #endif
}
#define easysimd_vclsq_u8(a) easysimd_vclsq_s8(easysimd_vreinterpretq_s8_u8(a))
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vclsq_s8
  #define vclsq_s8(a) easysimd_vclsq_s8(a)
  #undef vclsq_u8
  #define vclsq_u8(a) easysimd_vclsq_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vclsq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vclsq_s16(a);
  #else
    return easysimd_vsubq_s16(easysimd_vclzq_s16(easysimd_vbslq_s16(easysimd_vcltzq_s16(a), easysimd_vmvnq_s16(a), a)), easysimd_vdupq_n_s16(INT16_C(1)));
  #endif
}
#define easysimd_vclsq_u16(a) easysimd_vclsq_s16(easysimd_vreinterpretq_s16_u16(a))
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vclsq_s16
  #define vclsq_s16(a) easysimd_vclsq_s16(a)
  #undef vclsq_u16
  #define vclsq_u16(a) easysimd_vclsq_u16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vclsq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vclsq_s32(a);
  #else
    return easysimd_vsubq_s32(easysimd_vclzq_s32(easysimd_vbslq_s32(easysimd_vcltzq_s32(a), easysimd_vmvnq_s32(a), a)), easysimd_vdupq_n_s32(INT32_C(1)));
  #endif
}
#define easysimd_vclsq_u32(a) easysimd_vclsq_s32(easysimd_vreinterpretq_s32_u32(a))
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vclsq_s32
  #define vclsq_s32(a) easysimd_vclsq_s32(a)
  #undef vclsq_u32
  #define vclsq_u32(a) easysimd_vclsq_u32(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_CLS_H) */
