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

#if !defined(EASYSIMD_ARM_NEON_ABDL_H)
#define EASYSIMD_ARM_NEON_ABDL_H

#include "abs.h"
#include "subl.h"
#include "movl.h"
#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vabdl_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdl_s8(a, b);
  #else
    return easysimd_vabsq_s16(easysimd_vsubl_s8(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdl_s8
  #define vabdl_s8(a, b) easysimd_vabdl_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vabdl_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdl_s16(a, b);
  #else
    return easysimd_vabsq_s32(easysimd_vsubl_s16(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdl_s16
  #define vabdl_s16(a, b) easysimd_vabdl_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vabdl_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdl_s32(a, b);
  #else
    return easysimd_vabsq_s64(easysimd_vsubl_s32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdl_s32
  #define vabdl_s32(a, b) easysimd_vabdl_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vabdl_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdl_u8(a, b);
  #else
    return easysimd_vreinterpretq_u16_s16(
      easysimd_vabsq_s16(
        easysimd_vsubq_s16(
          easysimd_vreinterpretq_s16_u16(easysimd_vmovl_u8(a)),
          easysimd_vreinterpretq_s16_u16(easysimd_vmovl_u8(b))
        )
      )
    );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdl_u8
  #define vabdl_u8(a, b) easysimd_vabdl_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vabdl_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdl_u16(a, b);
  #else
    return easysimd_vreinterpretq_u32_s32(
      easysimd_vabsq_s32(
        easysimd_vsubq_s32(
          easysimd_vreinterpretq_s32_u32(easysimd_vmovl_u16(a)),
          easysimd_vreinterpretq_s32_u32(easysimd_vmovl_u16(b))
        )
      )
    );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdl_u16
  #define vabdl_u16(a, b) easysimd_vabdl_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vabdl_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdl_u32(a, b);
  #else
    return easysimd_vreinterpretq_u64_s64(
      easysimd_vabsq_s64(
        easysimd_vsubq_s64(
          easysimd_vreinterpretq_s64_u64(easysimd_vmovl_u32(a)),
          easysimd_vreinterpretq_s64_u64(easysimd_vmovl_u32(b))
        )
      )
    );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdl_u32
  #define vabdl_u32(a, b) easysimd_vabdl_u32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ABDL_H) */
