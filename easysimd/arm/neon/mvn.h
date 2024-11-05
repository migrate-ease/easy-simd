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

#if !defined(EASYSIMD_ARM_NEON_MVN_H)
#define EASYSIMD_ARM_NEON_MVN_H

#include "combine.h"
#include "get_low.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vmvnq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvnq_s8(a);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm_ternarylogic_epi32(a_.m128i, a_.m128i, a_.m128i, 0x55);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_andnot_si128(a_.m128i, _mm_cmpeq_epi8(a_.m128i, a_.m128i));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvnq_s8
  #define vmvnq_s8(a) easysimd_vmvnq_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmvnq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvnq_s16(a);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm_ternarylogic_epi32(a_.m128i, a_.m128i, a_.m128i, 0x55);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_andnot_si128(a_.m128i, _mm_cmpeq_epi16(a_.m128i, a_.m128i));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvnq_s16
  #define vmvnq_s16(a) easysimd_vmvnq_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmvnq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvnq_s32(a);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm_ternarylogic_epi32(a_.m128i, a_.m128i, a_.m128i, 0x55);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_andnot_si128(a_.m128i, _mm_cmpeq_epi32(a_.m128i, a_.m128i));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvnq_s32
  #define vmvnq_s32(a) easysimd_vmvnq_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vmvnq_u8(easysimd_uint8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvnq_u8(a);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm_ternarylogic_epi32(a_.m128i, a_.m128i, a_.m128i, 0x55);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_andnot_si128(a_.m128i, _mm_cmpeq_epi8(a_.m128i, a_.m128i));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvnq_u8
  #define vmvnq_u8(a) easysimd_vmvnq_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmvnq_u16(easysimd_uint16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvnq_u16(a);
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm_ternarylogic_epi32(a_.m128i, a_.m128i, a_.m128i, 0x55);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_andnot_si128(a_.m128i, _mm_cmpeq_epi16(a_.m128i, a_.m128i));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvnq_u16
  #define vmvnq_u16(a) easysimd_vmvnq_u16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmvnq_u32(easysimd_uint32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvnq_u32(a);
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm_ternarylogic_epi32(a_.m128i, a_.m128i, a_.m128i, 0x55);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_andnot_si128(a_.m128i, _mm_cmpeq_epi32(a_.m128i, a_.m128i));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvnq_u32
  #define vmvnq_u32(a) easysimd_vmvnq_u32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vmvn_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvn_s8(a);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_andnot_si64(a_.m64, _mm_cmpeq_pi8(a_.m64, a_.m64));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvn_s8
  #define vmvn_s8(a) easysimd_vmvn_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vmvn_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvn_s16(a);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_andnot_si64(a_.m64, _mm_cmpeq_pi16(a_.m64, a_.m64));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvn_s16
  #define vmvn_s16(a) easysimd_vmvn_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vmvn_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvn_s32(a);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_andnot_si64(a_.m64, _mm_cmpeq_pi32(a_.m64, a_.m64));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvn_s32
  #define vmvn_s32(a) easysimd_vmvn_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vmvn_u8(easysimd_uint8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvn_u8(a);
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_andnot_si64(a_.m64, _mm_cmpeq_pi8(a_.m64, a_.m64));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvn_u8
  #define vmvn_u8(a) easysimd_vmvn_u8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vmvn_u16(easysimd_uint16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvn_u16(a);
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_andnot_si64(a_.m64, _mm_cmpeq_pi16(a_.m64, a_.m64));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvn_u16
  #define vmvn_u16(a) easysimd_vmvn_u16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vmvn_u32(easysimd_uint32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmvn_u32(a);
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a);

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_andnot_si64(a_.m64, _mm_cmpeq_pi32(a_.m64, a_.m64));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = ~a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ~(a_.values[i]);
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmvn_u32
  #define vmvn_u32(a) easysimd_vmvn_u32(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MVN_H) */
