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

/* N.B. CM: vrev64_f16 and vrev64q_f16 are omitted as
 * SIMDe has no 16-bit floating point support. */

#if !defined(EASYSIMD_ARM_NEON_REV64_H)
#define EASYSIMD_ARM_NEON_REV64_H

#include "reinterpret.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vrev64_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64_s8(a);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_shuffle_pi8(a_.m64, _mm_set_pi8(0, 1, 2, 3, 4, 5, 6, 7));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 8, a_.values, a_.values, 7, 6, 5, 4, 3, 2, 1, 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 7];
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64_s8
  #define vrev64_s8(a) easysimd_vrev64_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vrev64_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64_s16(a);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a);

    #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_shuffle_pi16(a_.m64, (0 << 6) | (1 << 4) | (2 << 2) | (3 << 0));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.values, a_.values, 3, 2, 1, 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 3];
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64_s16
  #define vrev64_s16(a) easysimd_vrev64_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vrev64_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64_s32(a);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a);

    #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 =  _mm_shuffle_pi16(a_.m64, (1 << 6) | (0 << 4) | (3 << 2) | (2 << 0));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.values, a_.values, 1, 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 1];
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64_s32
  #define vrev64_s32(a) easysimd_vrev64_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vrev64_u8(easysimd_uint8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64_u8(a);
  #else
    return easysimd_vreinterpret_u8_s8(easysimd_vrev64_s8(easysimd_vreinterpret_s8_u8(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64_u8
  #define vrev64_u8(a) easysimd_vrev64_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vrev64_u16(easysimd_uint16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64_u16(a);
  #else
    return easysimd_vreinterpret_u16_s16(easysimd_vrev64_s16(easysimd_vreinterpret_s16_u16(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64_u16
  #define vrev64_u16(a) easysimd_vrev64_u16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vrev64_u32(easysimd_uint32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64_u32(a);
  #else
    return easysimd_vreinterpret_u32_s32(easysimd_vrev64_s32(easysimd_vreinterpret_s32_u32(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64_u32
  #define vrev64_u32(a) easysimd_vrev64_u32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vrev64_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64_f32(a);
  #else
    return easysimd_vreinterpret_f32_s32(easysimd_vrev64_s32(easysimd_vreinterpret_s32_f32(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64_f32
  #define vrev64_f32(a) easysimd_vrev64_f32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vrev64q_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64q_s8(a);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_shuffle_epi8(a_.m128i, _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15,
                                                         0, 1,  2,  3,  4,  5,  6,  7));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.values, a_.values, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 7];
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64q_s8
  #define vrev64q_s8(a) easysimd_vrev64q_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vrev64q_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64q_s16(a);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i = _mm_shuffle_epi8(a_.m128i, _mm_set_epi8(9, 8, 11, 10, 13, 12, 15, 14,
                                                         1, 0,  3,  2,  5,  4,  7,  6));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_shufflehi_epi16(_mm_shufflelo_epi16(a_.m128i,
                                                        (0 << 6) | (1 << 4) | (2 << 2) | (3 << 0)),
                                                        (0 << 6) | (1 << 4) | (2 << 2) | (3 << 0));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.values, a_.values, 3, 2, 1, 0, 7, 6, 5, 4);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 3];
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64q_s16
  #define vrev64q_s16(a) easysimd_vrev64q_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vrev64q_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64q_s32(a);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_shuffle_epi32(a_.m128i, (2 << 6) | (3 << 4) | (0 << 2) | (1 << 0));
   #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.values, a_.values, 1, 0, 3, 2);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = a_.values[i ^ 1];
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64q_s32
  #define vrev64q_s32(a) easysimd_vrev64q_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vrev64q_u8(easysimd_uint8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64q_u8(a);
  #else
    return easysimd_vreinterpretq_u8_s8(easysimd_vrev64q_s8(easysimd_vreinterpretq_s8_u8(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64q_u8
  #define vrev64q_u8(a) easysimd_vrev64q_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vrev64q_u16(easysimd_uint16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64q_u16(a);
  #else
    return easysimd_vreinterpretq_u16_s16(easysimd_vrev64q_s16(easysimd_vreinterpretq_s16_u16(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64q_u16
  #define vrev64q_u16(a) easysimd_vrev64q_u16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vrev64q_u32(easysimd_uint32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64q_u32(a);
  #else
    return easysimd_vreinterpretq_u32_s32(easysimd_vrev64q_s32(easysimd_vreinterpretq_s32_u32(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64q_u32
  #define vrev64q_u32(a) easysimd_vrev64q_u32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vrev64q_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrev64q_f32(a);
  #else
    return easysimd_vreinterpretq_f32_s32(easysimd_vrev64q_s32(easysimd_vreinterpretq_s32_f32(a)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrev64q_f32
  #define vrev64q_f32(a) easysimd_vrev64q_f32(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_REV64_H) */
