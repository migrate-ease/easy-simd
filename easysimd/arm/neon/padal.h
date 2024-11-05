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

#if !defined(EASYSIMD_ARM_NEON_PADAL_H)
#define EASYSIMD_ARM_NEON_PADAL_H

#include "types.h"

#include "add.h"
#include "paddl.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vpadal_s8(easysimd_int16x4_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadal_s8(a, b);
  #else
    return easysimd_vadd_s16(a, easysimd_vpaddl_s8(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadal_s8
  #define vpadal_s8(a, b) easysimd_vpadal_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vpadal_s16(easysimd_int32x2_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadal_s16(a, b);
  #else
    return easysimd_vadd_s32(a, easysimd_vpaddl_s16(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadal_s16
  #define vpadal_s16(a, b) easysimd_vpadal_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vpadal_s32(easysimd_int64x1_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadal_s32(a, b);
  #else
    return easysimd_vadd_s64(a, easysimd_vpaddl_s32(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadal_s32
  #define vpadal_s32(a, b) easysimd_vpadal_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vpadal_u8(easysimd_uint16x4_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadal_u8(a, b);
  #else
    return easysimd_vadd_u16(a, easysimd_vpaddl_u8(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadal_u8
  #define vpadal_u8(a, b) easysimd_vpadal_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vpadal_u16(easysimd_uint32x2_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadal_u16(a, b);
  #else
    return easysimd_vadd_u32(a, easysimd_vpaddl_u16(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadal_u16
  #define vpadal_u16(a, b) easysimd_vpadal_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vpadal_u32(easysimd_uint64x1_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadal_u32(a, b);
  #else
    return easysimd_vadd_u64(a, easysimd_vpaddl_u32(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadal_u32
  #define vpadal_u32(a, b) easysimd_vpadal_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vpadalq_s8(easysimd_int16x8_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadalq_s8(a, b);
  #else
    return easysimd_vaddq_s16(a, easysimd_vpaddlq_s8(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadalq_s8
  #define vpadalq_s8(a, b) easysimd_vpadalq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vpadalq_s16(easysimd_int32x4_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadalq_s16(a, b);
  #else
    return easysimd_vaddq_s32(a, easysimd_vpaddlq_s16(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadalq_s16
  #define vpadalq_s16(a, b) easysimd_vpadalq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vpadalq_s32(easysimd_int64x2_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadalq_s32(a, b);
  #else
    return easysimd_vaddq_s64(a, easysimd_vpaddlq_s32(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadalq_s32
  #define vpadalq_s32(a, b) easysimd_vpadalq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vpadalq_u8(easysimd_uint16x8_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadalq_u8(a, b);
  #else
    return easysimd_vaddq_u16(a, easysimd_vpaddlq_u8(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadalq_u8
  #define vpadalq_u8(a, b) easysimd_vpadalq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vpadalq_u16(easysimd_uint32x4_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadalq_u16(a, b);
  #else
    return easysimd_vaddq_u32(a, easysimd_vpaddlq_u16(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadalq_u16
  #define vpadalq_u16(a, b) easysimd_vpadalq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vpadalq_u32(easysimd_uint64x2_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpadalq_u32(a, b);
  #else
    return easysimd_vaddq_u64(a, easysimd_vpaddlq_u32(b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpadalq_u32
  #define vpadalq_u32(a, b) easysimd_vpadalq_u32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_NEON_PADAL_H */
