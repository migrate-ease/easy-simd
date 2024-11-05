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

#if !defined(EASYSIMD_ARM_NEON_REV16_H)
#define EASYSIMD_ARM_NEON_REV16_H

#include "reinterpret.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vrev16_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev16_s8(a);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_shuffle_pi8(a_.m64, _mm_set_pi8(6, 7, 4, 5, 2, 3, 0, 1));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 8, a_.values, a_.values, 1, 0, 3, 2, 5, 4, 7, 6);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 1];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev16_s8
  #define vrev16_s8(a) easysimd_vrev16_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vrev16_u8(easysimd_uint8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev16_u8(a);
  #else
    return easysimd_vreinterpret_u8_s8(easysimd_vrev16_s8(easysimd_vreinterpret_s8_u8(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev16_u8
  #define vrev16_u8(a) easysimd_vrev16_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vrev16q_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev16q_s8(a);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_shuffle_epi8(a_.m128i, _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.values, a_.values, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 1];
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev16q_s8
  #define vrev16q_s8(a) easysimd_vrev16q_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vrev16q_u8(easysimd_uint8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev16q_u8(a);
  #else
    return easysimd_vreinterpretq_u8_s8(easysimd_vrev16q_s8(easysimd_vreinterpretq_s8_u8(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev16q_u8
  #define vrev16q_u8(a) easysimd_vrev16q_u8(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_REV16_H) */
