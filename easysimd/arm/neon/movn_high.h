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

#if !defined(EASYSIMD_ARM_NEON_MOVN_HIGH_H)
#define EASYSIMD_ARM_NEON_MOVN_HIGH_H

#include "types.h"
#include "movn.h"
#include "combine.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vmovn_high_s16(easysimd_int8x8_t r, easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmovn_high_s16(r, a);
  #else
    return easysimd_vcombine_s8(r, easysimd_vmovn_s16(a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmovn_high_s16
  #define vmovn_high_s16(r, a) easysimd_vmovn_high_s16((r), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmovn_high_s32(easysimd_int16x4_t r, easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmovn_high_s32(r, a);
  #else
    return easysimd_vcombine_s16(r, easysimd_vmovn_s32(a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmovn_high_s32
  #define vmovn_high_s32(r, a) easysimd_vmovn_high_s32((r), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmovn_high_s64(easysimd_int32x2_t r, easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmovn_high_s64(r, a);
  #else
    return easysimd_vcombine_s32(r, easysimd_vmovn_s64(a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmovn_high_s64
  #define vmovn_high_s64(r, a) easysimd_vmovn_high_s64((r), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vmovn_high_u16(easysimd_uint8x8_t r, easysimd_uint16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmovn_high_u16(r, a);
  #else
    return easysimd_vcombine_u8(r, easysimd_vmovn_u16(a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmovn_high_u16
  #define vmovn_high_u16(r, a) easysimd_vmovn_high_u16((r), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmovn_high_u32(easysimd_uint16x4_t r, easysimd_uint32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmovn_high_u32(r, a);
  #else
    return easysimd_vcombine_u16(r, easysimd_vmovn_u32(a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmovn_high_u32
  #define vmovn_high_u32(r, a) easysimd_vmovn_high_u32((r), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmovn_high_u64(easysimd_uint32x2_t r, easysimd_uint64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vmovn_high_u64(r, a);
  #else
    return easysimd_vcombine_u32(r, easysimd_vmovn_u64(a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vmovn_high_u64
  #define vmovn_high_u64(r, a) easysimd_vmovn_high_u64((r), (a))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MOVN_HIGH_H) */
