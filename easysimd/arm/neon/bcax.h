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
 *   2021      Atharva Nimbalkar <atharvakn@gmail.com>
 */

#if !defined(EASYSIMD_ARM_NEON_BCAX_H)
#define EASYSIMD_ARM_NEON_BCAX_H

#include "types.h"

#include "eor.h"
#include "bic.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vbcaxq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b, easysimd_uint8x16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_u8(a, b, c);
  #else
    return easysimd_veorq_u8(a, easysimd_vbicq_u8(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_u8
  #define vbcaxq_u8(a, b, c) easysimd_vbcaxq_u8(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vbcaxq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b, easysimd_uint16x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_u16(a, b, c);
  #else
    return easysimd_veorq_u16(a, easysimd_vbicq_u16(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_u16
  #define vbcaxq_u16(a, b, c) easysimd_vbcaxq_u16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vbcaxq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b, easysimd_uint32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_u32(a, b, c);
  #else
    return easysimd_veorq_u32(a, easysimd_vbicq_u32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_u32
  #define vbcaxq_u32(a, b, c) easysimd_vbcaxq_u32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vbcaxq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b, easysimd_uint64x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_u64(a, b, c);
  #else
    return easysimd_veorq_u64(a, easysimd_vbicq_u64(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_u64
  #define vbcaxq_u64(a, b, c) easysimd_vbcaxq_u64(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vbcaxq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b, easysimd_int8x16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_s8(a, b, c);
  #else
    return easysimd_veorq_s8(a, easysimd_vbicq_s8(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_s8
  #define vbcaxq_s8(a, b, c) easysimd_vbcaxq_s8(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vbcaxq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b, easysimd_int16x8_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_s16(a, b, c);
  #else
    return easysimd_veorq_s16(a,easysimd_vbicq_s16(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_s16
  #define vbcaxq_s16(a, b, c) easysimd_vbcaxq_s16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vbcaxq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b, easysimd_int32x4_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_s32(a, b, c);
  #else
    return easysimd_veorq_s32(a, easysimd_vbicq_s32(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_s32
  #define vbcaxq_s32(a, b, c) easysimd_vbcaxq_s32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vbcaxq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b, easysimd_int64x2_t c) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_SHA3)
    return vbcaxq_s64(a, b, c);
  #else
    return easysimd_veorq_s64(a, easysimd_vbicq_s64(b, c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_SHA3))
  #undef vbcaxq_s64
  #define vbcaxq_s64(a, b, c) easysimd_vbcaxq_s64(a, b, c)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_BCAX_H) */
