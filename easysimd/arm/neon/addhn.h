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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_NEON_ADDHN_H)
#define EASYSIMD_ARM_NEON_ADDHN_H

#include "add.h"
#include "shr_n.h"
#include "movn.h"

#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vaddhn_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddhn_s16(a, b);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
    easysimd_int8x8_private r_;
    easysimd_int8x16_private tmp_ =
      easysimd_int8x16_to_private(
        easysimd_vreinterpretq_s8_s16(
          easysimd_vaddq_s16(a, b)
        )
      );
    #if EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3, 5, 7, 9, 11, 13, 15);
    #else
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 0, 2, 4, 6, 8, 10, 12, 14);
    #endif
    return easysimd_int8x8_from_private(r_);
  #else
    return easysimd_vmovn_s16(easysimd_vshrq_n_s16(easysimd_vaddq_s16(a, b), 8));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddhn_s16
  #define vaddhn_s16(a, b) easysimd_vaddhn_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vaddhn_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddhn_s32(a, b);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
    easysimd_int16x4_private r_;
    easysimd_int16x8_private tmp_ =
      easysimd_int16x8_to_private(
        easysimd_vreinterpretq_s16_s32(
          easysimd_vaddq_s32(a, b)
        )
      );
    #if EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3, 5, 7);
    #else
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 0, 2, 4, 6);
    #endif
    return easysimd_int16x4_from_private(r_);
  #else
    return easysimd_vmovn_s32(easysimd_vshrq_n_s32(easysimd_vaddq_s32(a, b), 16));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddhn_s32
  #define vaddhn_s32(a, b) easysimd_vaddhn_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vaddhn_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddhn_s64(a, b);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
    easysimd_int32x2_private r_;
    easysimd_int32x4_private tmp_ =
      easysimd_int32x4_to_private(
        easysimd_vreinterpretq_s32_s64(
          easysimd_vaddq_s64(a, b)
        )
      );
    #if EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3);
    #else
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 0, 2);
    #endif
    return easysimd_int32x2_from_private(r_);
  #else
    return easysimd_vmovn_s64(easysimd_vshrq_n_s64(easysimd_vaddq_s64(a, b), 32));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddhn_s64
  #define vaddhn_s64(a, b) easysimd_vaddhn_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vaddhn_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddhn_u16(a, b);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
    easysimd_uint8x8_private r_;
    easysimd_uint8x16_private tmp_ =
      easysimd_uint8x16_to_private(
        easysimd_vreinterpretq_u8_u16(
          easysimd_vaddq_u16(a, b)
        )
      );
    #if EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3, 5, 7, 9, 11, 13, 15);
    #else
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 0, 2, 4, 6, 8, 10, 12, 14);
    #endif
    return easysimd_uint8x8_from_private(r_);
  #else
    return easysimd_vmovn_u16(easysimd_vshrq_n_u16(easysimd_vaddq_u16(a, b), 8));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddhn_u16
  #define vaddhn_u16(a, b) easysimd_vaddhn_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vaddhn_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddhn_u32(a, b);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
    easysimd_uint16x4_private r_;
    easysimd_uint16x8_private tmp_ =
      easysimd_uint16x8_to_private(
        easysimd_vreinterpretq_u16_u32(
          easysimd_vaddq_u32(a, b)
        )
      );
    #if EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3, 5, 7);
    #else
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 0, 2, 4, 6);
    #endif
    return easysimd_uint16x4_from_private(r_);
  #else
    return easysimd_vmovn_u32(easysimd_vshrq_n_u32(easysimd_vaddq_u32(a, b), 16));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddhn_u32
  #define vaddhn_u32(a, b) easysimd_vaddhn_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vaddhn_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vaddhn_u64(a, b);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
    easysimd_uint32x2_private r_;
    easysimd_uint32x4_private tmp_ =
      easysimd_uint32x4_to_private(
        easysimd_vreinterpretq_u32_u64(
          easysimd_vaddq_u64(a, b)
        )
      );
    #if EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3);
    #else
      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 0, 2);
    #endif
    return easysimd_uint32x2_from_private(r_);
  #else
    return easysimd_vmovn_u64(easysimd_vshrq_n_u64(easysimd_vaddq_u64(a, b), 32));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vaddhn_u64
  #define vaddhn_u64(a, b) easysimd_vaddhn_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ADDHN_H) */
