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

#if !defined(EASYSIMD_ARM_NEON_PADDL_H)
#define EASYSIMD_ARM_NEON_PADDL_H

#include "add.h"
#include "get_high.h"
#include "get_low.h"
#include "movl.h"
#include "movl_high.h"
#include "padd.h"
#include "reinterpret.h"
#include "shl_n.h"
#include "shr_n.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vpaddl_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddl_s8(a);
  #else
    easysimd_int16x8_t tmp = easysimd_vmovl_s8(a);
    return easysimd_vpadd_s16(easysimd_vget_low_s16(tmp), easysimd_vget_high_s16(tmp));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddl_s8
  #define vpaddl_s8(a) easysimd_vpaddl_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vpaddl_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddl_s16(a);
  #else
    easysimd_int32x4_t tmp = easysimd_vmovl_s16(a);
    return easysimd_vpadd_s32(easysimd_vget_low_s32(tmp), easysimd_vget_high_s32(tmp));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddl_s16
  #define vpaddl_s16(a) easysimd_vpaddl_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vpaddl_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddl_s32(a);
  #else
    easysimd_int64x2_t tmp = easysimd_vmovl_s32(a);
    return easysimd_vadd_s64(easysimd_vget_low_s64(tmp), easysimd_vget_high_s64(tmp));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddl_s32
  #define vpaddl_s32(a) easysimd_vpaddl_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vpaddl_u8(easysimd_uint8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddl_u8(a);
  #else
    easysimd_uint16x8_t tmp = easysimd_vmovl_u8(a);
    return easysimd_vpadd_u16(easysimd_vget_low_u16(tmp), easysimd_vget_high_u16(tmp));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddl_u8
  #define vpaddl_u8(a) easysimd_vpaddl_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vpaddl_u16(easysimd_uint16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddl_u16(a);
  #else
    easysimd_uint32x4_t tmp = easysimd_vmovl_u16(a);
    return easysimd_vpadd_u32(easysimd_vget_low_u32(tmp), easysimd_vget_high_u32(tmp));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddl_u16
  #define vpaddl_u16(a) easysimd_vpaddl_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vpaddl_u32(easysimd_uint32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddl_u32(a);
  #else
    easysimd_uint64x2_t tmp = easysimd_vmovl_u32(a);
    return easysimd_vadd_u64(easysimd_vget_low_u64(tmp), easysimd_vget_high_u64(tmp));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddl_u32
  #define vpaddl_u32(a) easysimd_vpaddl_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vpaddlq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddlq_s8(a);
  #elif \
      defined(EASYSIMD_X86_XOP_NATIVE) || \
      defined(EASYSIMD_X86_SSSE3_NATIVE)
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);
    easysimd_int16x8_private r_;

    #if defined(EASYSIMD_X86_XOP_NATIVE)
      r_.m128i = _mm_haddw_epi8(a_.m128i);
    #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_maddubs_epi16(_mm_set1_epi8(INT8_C(1)), a_.m128i);
    #endif

    return easysimd_int16x8_from_private(r_);
  #else
    easysimd_int16x8_t lo = easysimd_vshrq_n_s16(easysimd_vshlq_n_s16(easysimd_vreinterpretq_s16_s8(a), 8), 8);
    easysimd_int16x8_t hi = easysimd_vshrq_n_s16(easysimd_vreinterpretq_s16_s8(a), 8);
    return easysimd_vaddq_s16(lo, hi);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddlq_s8
  #define vpaddlq_s8(a) easysimd_vpaddlq_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vpaddlq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddlq_s16(a);
  #elif \
      defined(EASYSIMD_X86_XOP_NATIVE) || \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);
    easysimd_int32x4_private r_;

    #if defined(EASYSIMD_X86_XOP_NATIVE)
      r_.m128i = _mm_haddd_epi16(a_.m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_madd_epi16(a_.m128i, _mm_set1_epi16(INT8_C(1)));
    #endif

    return easysimd_int32x4_from_private(r_);
  #else
    easysimd_int32x4_t lo = easysimd_vshrq_n_s32(easysimd_vshlq_n_s32(easysimd_vreinterpretq_s32_s16(a), 16), 16);
    easysimd_int32x4_t hi = easysimd_vshrq_n_s32(easysimd_vreinterpretq_s32_s16(a), 16);
    return easysimd_vaddq_s32(lo, hi);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddlq_s16
  #define vpaddlq_s16(a) easysimd_vpaddlq_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vpaddlq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddlq_s32(a);
  #else
    easysimd_int64x2_t lo = easysimd_vshrq_n_s64(easysimd_vshlq_n_s64(easysimd_vreinterpretq_s64_s32(a), 32), 32);
    easysimd_int64x2_t hi = easysimd_vshrq_n_s64(easysimd_vreinterpretq_s64_s32(a), 32);
    return easysimd_vaddq_s64(lo, hi);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddlq_s32
  #define vpaddlq_s32(a) easysimd_vpaddlq_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vpaddlq_u8(easysimd_uint8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddlq_u8(a);
  #elif \
      defined(EASYSIMD_X86_XOP_NATIVE) || \
      defined(EASYSIMD_X86_SSSE3_NATIVE)
    easysimd_uint8x16_private a_ = easysimd_uint8x16_to_private(a);
    easysimd_uint16x8_private r_;

    #if defined(EASYSIMD_X86_XOP_NATIVE)
      r_.m128i = _mm_haddw_epu8(a_.m128i);
    #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_maddubs_epi16(a_.m128i, _mm_set1_epi8(INT8_C(1)));
    #endif

    return easysimd_uint16x8_from_private(r_);
  #else
    easysimd_uint16x8_t lo = easysimd_vshrq_n_u16(easysimd_vshlq_n_u16(easysimd_vreinterpretq_u16_u8(a), 8), 8);
    easysimd_uint16x8_t hi = easysimd_vshrq_n_u16(easysimd_vreinterpretq_u16_u8(a), 8);
    return easysimd_vaddq_u16(lo, hi);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddlq_u8
  #define vpaddlq_u8(a) easysimd_vpaddlq_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vpaddlq_u16(easysimd_uint16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddlq_u16(a);
  #elif \
      defined(EASYSIMD_X86_XOP_NATIVE) || \
      defined(EASYSIMD_X86_SSSE3_NATIVE)
    easysimd_uint16x8_private a_ = easysimd_uint16x8_to_private(a);
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_X86_XOP_NATIVE)
      r_.sse_m128i = _mm_haddd_epu16(a_.sse_m128i);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i =
        _mm_add_epi32(
          _mm_srli_epi32(a_.m128i, 16),
          _mm_and_si128(a_.m128i, _mm_set1_epi32(INT32_C(0x0000ffff)))
        );
    #endif

    return easysimd_uint32x4_from_private(r_);
  #else
    easysimd_uint32x4_t lo = easysimd_vshrq_n_u32(easysimd_vshlq_n_u32(easysimd_vreinterpretq_u32_u16(a), 16), 16);
    easysimd_uint32x4_t hi = easysimd_vshrq_n_u32(easysimd_vreinterpretq_u32_u16(a), 16);
    return easysimd_vaddq_u32(lo, hi);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddlq_u16
  #define vpaddlq_u16(a) easysimd_vpaddlq_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vpaddlq_u32(easysimd_uint32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vpaddlq_u32(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_uint32x4_private a_ = easysimd_uint32x4_to_private(a);
    easysimd_uint64x2_private r_;

    r_.m128i =
      _mm_add_epi64(
        _mm_srli_epi64(a_.m128i, 32),
        _mm_and_si128(a_.m128i, _mm_set1_epi64x(INT64_C(0x00000000ffffffff)))
      );

    return easysimd_uint64x2_from_private(r_);
  #else
    easysimd_uint64x2_t lo = easysimd_vshrq_n_u64(easysimd_vshlq_n_u64(easysimd_vreinterpretq_u64_u32(a), 32), 32);
    easysimd_uint64x2_t hi = easysimd_vshrq_n_u64(easysimd_vreinterpretq_u64_u32(a), 32);
    return easysimd_vaddq_u64(lo, hi);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vpaddlq_u32
  #define vpaddlq_u32(a) easysimd_vpaddlq_u32((a))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_NEON_PADDL_H */
