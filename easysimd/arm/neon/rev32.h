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
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_ARM_NEON_REV32_H)
#define EASYSIMD_ARM_NEON_REV32_H

#include "reinterpret.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vrev32_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32_s8(a);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_shuffle_pi8(a_.m64, _mm_set_pi8(4, 5, 6, 7, 0, 1, 2, 3));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 8, a_.values, a_.values, 3, 2, 1, 0, 7, 6, 5, 4);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 3];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32_s8
  #define vrev32_s8(a) easysimd_vrev32_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vrev32_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32_s16(a);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a);

    #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_shuffle_pi16(a_.m64, (2 << 6) | (3 << 4) | (0 << 2) | (1 << 0));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.values, a_.values, 1, 0, 3, 2);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 1];
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32_s16
  #define vrev32_s16(a) easysimd_vrev32_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vrev32_u8(easysimd_uint8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32_u8(a);
  #else
    return easysimd_vreinterpret_u8_s8(easysimd_vrev32_s8(easysimd_vreinterpret_s8_u8(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32_u8
  #define vrev32_u8(a) easysimd_vrev32_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vrev32_u16(easysimd_uint16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32_u16(a);
  #else
    return easysimd_vreinterpret_u16_s16(easysimd_vrev32_s16(easysimd_vreinterpret_s16_u16(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32_u16
  #define vrev32_u16(a) easysimd_vrev32_u16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vrev32q_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32q_s8(a);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_shuffle_epi8(a_.m128i, _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11,
                                                          4,  5,  6,  7, 0, 1,  2,  3));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.values, a_.values, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 3];
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32q_s8
  #define vrev32q_s8(a) easysimd_vrev32q_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vrev32q_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32q_s16(a);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_shuffle_epi8(a_.m128i, _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10,
                                                          5,  4,  7,  6, 1, 0,  3,  2));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_shufflehi_epi16(_mm_shufflelo_epi16(a_.m128i,
                                     (2 << 6) | (3 << 4) | (0 << 2) | (1 << 0)),
                                     (2 << 6) | (3 << 4) | (0 << 2) | (1 << 0));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.values, a_.values, 1, 0, 3, 2, 5, 4, 7, 6);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 1];
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32q_s16
  #define vrev32q_s16(a) easysimd_vrev32q_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vrev32q_u8(easysimd_uint8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32q_u8(a);
  #else
    return easysimd_vreinterpretq_u8_s8(easysimd_vrev32q_s8(easysimd_vreinterpretq_s8_u8(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32q_u8
  #define vrev32q_u8(a) easysimd_vrev32q_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vrev32q_u16(easysimd_uint16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev32q_u16(a);
  #else
    return easysimd_vreinterpretq_u16_s16(easysimd_vrev32q_s16(easysimd_vreinterpretq_s16_u16(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev32q_u16
  #define vrev32q_u16(a) easysimd_vrev32q_u16(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_REV32_H) */
