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

#if !defined(EASYSIMD_ARM_SVE_ADD_H)
#define EASYSIMD_ARM_SVE_ADD_H

#include "types.h"
#include "sel.h"
#include "dup.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svadd_s8_x(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s8_x(pg, op1, op2);
  #else
    easysimd_svint8_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_s8(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi8(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi8(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi8(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi8(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
      r.altivec = op1.altivec + op2.altivec;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s8_x
  #define svadd_s8_x(pg, op1, op2) easysimd_svadd_s8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svadd_s8_z(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s8_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_s8_z(pg, easysimd_svadd_s8_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s8_z
  #define svadd_s8_z(pg, op1, op2) easysimd_svadd_s8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svadd_s8_m(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s8_m(pg, op1, op2);
  #else
    return easysimd_svsel_s8(pg, easysimd_svadd_s8_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s8_m
  #define svadd_s8_m(pg, op1, op2) easysimd_svadd_s8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svadd_n_s8_x(easysimd_svbool_t pg, easysimd_svint8_t op1, int8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s8_x(pg, op1, op2);
  #else
    return easysimd_svadd_s8_x(pg, op1, easysimd_svdup_n_s8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s8_x
  #define svadd_n_s8_x(pg, op1, op2) easysimd_svadd_n_s8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svadd_n_s8_z(easysimd_svbool_t pg, easysimd_svint8_t op1, int8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s8_z(pg, op1, op2);
  #else
    return easysimd_svadd_s8_z(pg, op1, easysimd_svdup_n_s8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s8_z
  #define svadd_n_s8_z(pg, op1, op2) easysimd_svadd_n_s8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svadd_n_s8_m(easysimd_svbool_t pg, easysimd_svint8_t op1, int8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s8_m(pg, op1, op2);
  #else
    return easysimd_svadd_s8_m(pg, op1, easysimd_svdup_n_s8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s8_m
  #define svadd_n_s8_m(pg, op1, op2) easysimd_svadd_n_s8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svadd_s16_x(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s16_x(pg, op1, op2);
  #else
    easysimd_svint16_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_s16(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi16(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi16(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi16(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi16(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
      r.altivec = op1.altivec + op2.altivec;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s16_x
  #define svadd_s16_x(pg, op1, op2) easysimd_svadd_s16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svadd_s16_z(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s16_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_s16_z(pg, easysimd_svadd_s16_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s16_z
  #define svadd_s16_z(pg, op1, op2) easysimd_svadd_s16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svadd_s16_m(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s16_m(pg, op1, op2);
  #else
    return easysimd_svsel_s16(pg, easysimd_svadd_s16_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s16_m
  #define svadd_s16_m(pg, op1, op2) easysimd_svadd_s16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svadd_n_s16_x(easysimd_svbool_t pg, easysimd_svint16_t op1, int16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s16_x(pg, op1, op2);
  #else
    return easysimd_svadd_s16_x(pg, op1, easysimd_svdup_n_s16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s16_x
  #define svadd_n_s16_x(pg, op1, op2) easysimd_svadd_n_s16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svadd_n_s16_z(easysimd_svbool_t pg, easysimd_svint16_t op1, int16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s16_z(pg, op1, op2);
  #else
    return easysimd_svadd_s16_z(pg, op1, easysimd_svdup_n_s16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s16_z
  #define svadd_n_s16_z(pg, op1, op2) easysimd_svadd_n_s16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svadd_n_s16_m(easysimd_svbool_t pg, easysimd_svint16_t op1, int16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s16_m(pg, op1, op2);
  #else
    return easysimd_svadd_s16_m(pg, op1, easysimd_svdup_n_s16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s16_m
  #define svadd_n_s16_m(pg, op1, op2) easysimd_svadd_n_s16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svadd_s32_x(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s32_x(pg, op1, op2);
  #else
    easysimd_svint32_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_s32(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi32(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi32(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi32(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi32(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
      r.altivec = op1.altivec + op2.altivec;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s32_x
  #define svadd_s32_x(pg, op1, op2) easysimd_svadd_s32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svadd_s32_z(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s32_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_s32_z(pg, easysimd_svadd_s32_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s32_z
  #define svadd_s32_z(pg, op1, op2) easysimd_svadd_s32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svadd_s32_m(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s32_m(pg, op1, op2);
  #else
    return easysimd_svsel_s32(pg, easysimd_svadd_s32_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s32_m
  #define svadd_s32_m(pg, op1, op2) easysimd_svadd_s32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svadd_n_s32_x(easysimd_svbool_t pg, easysimd_svint32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s32_x(pg, op1, op2);
  #else
    return easysimd_svadd_s32_x(pg, op1, easysimd_svdup_n_s32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s32_x
  #define svadd_n_s32_x(pg, op1, op2) easysimd_svadd_n_s32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svadd_n_s32_z(easysimd_svbool_t pg, easysimd_svint32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s32_z(pg, op1, op2);
  #else
    return easysimd_svadd_s32_z(pg, op1, easysimd_svdup_n_s32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s32_z
  #define svadd_n_s32_z(pg, op1, op2) easysimd_svadd_n_s32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svadd_n_s32_m(easysimd_svbool_t pg, easysimd_svint32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s32_m(pg, op1, op2);
  #else
    return easysimd_svadd_s32_m(pg, op1, easysimd_svdup_n_s32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s32_m
  #define svadd_n_s32_m(pg, op1, op2) easysimd_svadd_n_s32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svadd_s64_x(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s64_x(pg, op1, op2);
  #else
    easysimd_svint64_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_s64(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi64(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi64(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi64(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi64(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
      r.altivec = op1.altivec + op2.altivec;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s64_x
  #define svadd_s64_x(pg, op1, op2) easysimd_svadd_s64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svadd_s64_z(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s64_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_s64_z(pg, easysimd_svadd_s64_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s64_z
  #define svadd_s64_z(pg, op1, op2) easysimd_svadd_s64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svadd_s64_m(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_s64_m(pg, op1, op2);
  #else
    return easysimd_svsel_s64(pg, easysimd_svadd_s64_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_s64_m
  #define svadd_s64_m(pg, op1, op2) easysimd_svadd_s64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svadd_n_s64_x(easysimd_svbool_t pg, easysimd_svint64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s64_x(pg, op1, op2);
  #else
    return easysimd_svadd_s64_x(pg, op1, easysimd_svdup_n_s64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s64_x
  #define svadd_n_s64_x(pg, op1, op2) easysimd_svadd_n_s64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svadd_n_s64_z(easysimd_svbool_t pg, easysimd_svint64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s64_z(pg, op1, op2);
  #else
    return easysimd_svadd_s64_z(pg, op1, easysimd_svdup_n_s64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s64_z
  #define svadd_n_s64_z(pg, op1, op2) easysimd_svadd_n_s64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svadd_n_s64_m(easysimd_svbool_t pg, easysimd_svint64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_s64_m(pg, op1, op2);
  #else
    return easysimd_svadd_s64_m(pg, op1, easysimd_svdup_n_s64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_s64_m
  #define svadd_n_s64_m(pg, op1, op2) easysimd_svadd_n_s64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svadd_u8_x(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u8_x(pg, op1, op2);
  #else
    easysimd_svuint8_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_u8(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi8(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi8(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi8(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi8(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u8_x
  #define svadd_u8_x(pg, op1, op2) easysimd_svadd_u8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svadd_u8_z(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u8_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_u8_z(pg, easysimd_svadd_u8_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u8_z
  #define svadd_u8_z(pg, op1, op2) easysimd_svadd_u8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svadd_u8_m(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u8_m(pg, op1, op2);
  #else
    return easysimd_svsel_u8(pg, easysimd_svadd_u8_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u8_m
  #define svadd_u8_m(pg, op1, op2) easysimd_svadd_u8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svadd_n_u8_x(easysimd_svbool_t pg, easysimd_svuint8_t op1, uint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u8_x(pg, op1, op2);
  #else
    return easysimd_svadd_u8_x(pg, op1, easysimd_svdup_n_u8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u8_x
  #define svadd_n_u8_x(pg, op1, op2) easysimd_svadd_n_u8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svadd_n_u8_z(easysimd_svbool_t pg, easysimd_svuint8_t op1, uint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u8_z(pg, op1, op2);
  #else
    return easysimd_svadd_u8_z(pg, op1, easysimd_svdup_n_u8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u8_z
  #define svadd_n_u8_z(pg, op1, op2) easysimd_svadd_n_u8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svadd_n_u8_m(easysimd_svbool_t pg, easysimd_svuint8_t op1, uint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u8_m(pg, op1, op2);
  #else
    return easysimd_svadd_u8_m(pg, op1, easysimd_svdup_n_u8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u8_m
  #define svadd_n_u8_m(pg, op1, op2) easysimd_svadd_n_u8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svadd_u16_x(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u16_x(pg, op1, op2);
  #else
    easysimd_svuint16_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_u16(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi16(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi16(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi16(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi16(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u16_x
  #define svadd_u16_x(pg, op1, op2) easysimd_svadd_u16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svadd_u16_z(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u16_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_u16_z(pg, easysimd_svadd_u16_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u16_z
  #define svadd_u16_z(pg, op1, op2) easysimd_svadd_u16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svadd_u16_m(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u16_m(pg, op1, op2);
  #else
    return easysimd_svsel_u16(pg, easysimd_svadd_u16_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u16_m
  #define svadd_u16_m(pg, op1, op2) easysimd_svadd_u16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svadd_n_u16_x(easysimd_svbool_t pg, easysimd_svuint16_t op1, uint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u16_x(pg, op1, op2);
  #else
    return easysimd_svadd_u16_x(pg, op1, easysimd_svdup_n_u16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u16_x
  #define svadd_n_u16_x(pg, op1, op2) easysimd_svadd_n_u16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svadd_n_u16_z(easysimd_svbool_t pg, easysimd_svuint16_t op1, uint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u16_z(pg, op1, op2);
  #else
    return easysimd_svadd_u16_z(pg, op1, easysimd_svdup_n_u16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u16_z
  #define svadd_n_u16_z(pg, op1, op2) easysimd_svadd_n_u16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svadd_n_u16_m(easysimd_svbool_t pg, easysimd_svuint16_t op1, uint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u16_m(pg, op1, op2);
  #else
    return easysimd_svadd_u16_m(pg, op1, easysimd_svdup_n_u16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u16_m
  #define svadd_n_u16_m(pg, op1, op2) easysimd_svadd_n_u16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svadd_u32_x(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u32_x(pg, op1, op2);
  #else
    easysimd_svuint32_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_u32(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi32(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi32(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi32(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi32(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u32_x
  #define svadd_u32_x(pg, op1, op2) easysimd_svadd_u32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svadd_u32_z(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u32_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_u32_z(pg, easysimd_svadd_u32_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u32_z
  #define svadd_u32_z(pg, op1, op2) easysimd_svadd_u32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svadd_u32_m(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u32_m(pg, op1, op2);
  #else
    return easysimd_svsel_u32(pg, easysimd_svadd_u32_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u32_m
  #define svadd_u32_m(pg, op1, op2) easysimd_svadd_u32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svadd_n_u32_x(easysimd_svbool_t pg, easysimd_svuint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u32_x(pg, op1, op2);
  #else
    return easysimd_svadd_u32_x(pg, op1, easysimd_svdup_n_u32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u32_x
  #define svadd_n_u32_x(pg, op1, op2) easysimd_svadd_n_u32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svadd_n_u32_z(easysimd_svbool_t pg, easysimd_svuint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u32_z(pg, op1, op2);
  #else
    return easysimd_svadd_u32_z(pg, op1, easysimd_svdup_n_u32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u32_z
  #define svadd_n_u32_z(pg, op1, op2) easysimd_svadd_n_u32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svadd_n_u32_m(easysimd_svbool_t pg, easysimd_svuint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u32_m(pg, op1, op2);
  #else
    return easysimd_svadd_u32_m(pg, op1, easysimd_svdup_n_u32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u32_m
  #define svadd_n_u32_m(pg, op1, op2) easysimd_svadd_n_u32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svadd_u64_x(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u64_x(pg, op1, op2);
  #else
    easysimd_svuint64_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_u64(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_add_epi64(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_add_epi64(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_add_epi64(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_add_epi64(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u64_x
  #define svadd_u64_x(pg, op1, op2) easysimd_svadd_u64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svadd_u64_z(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u64_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_u64_z(pg, easysimd_svadd_u64_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u64_z
  #define svadd_u64_z(pg, op1, op2) easysimd_svadd_u64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svadd_u64_m(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_u64_m(pg, op1, op2);
  #else
    return easysimd_svsel_u64(pg, easysimd_svadd_u64_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_u64_m
  #define svadd_u64_m(pg, op1, op2) easysimd_svadd_u64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svadd_n_u64_x(easysimd_svbool_t pg, easysimd_svuint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u64_x(pg, op1, op2);
  #else
    return easysimd_svadd_u64_x(pg, op1, easysimd_svdup_n_u64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u64_x
  #define svadd_n_u64_x(pg, op1, op2) easysimd_svadd_n_u64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svadd_n_u64_z(easysimd_svbool_t pg, easysimd_svuint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u64_z(pg, op1, op2);
  #else
    return easysimd_svadd_u64_z(pg, op1, easysimd_svdup_n_u64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u64_z
  #define svadd_n_u64_z(pg, op1, op2) easysimd_svadd_n_u64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svadd_n_u64_m(easysimd_svbool_t pg, easysimd_svuint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_u64_m(pg, op1, op2);
  #else
    return easysimd_svadd_u64_m(pg, op1, easysimd_svdup_n_u64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_u64_m
  #define svadd_n_u64_m(pg, op1, op2) easysimd_svadd_n_u64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svadd_f32_x(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_f32_x(pg, op1, op2);
  #else
    easysimd_svfloat32_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vaddq_f32(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512 = _mm512_add_ps(op1.m512, op2.m512);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256[0] = _mm256_add_ps(op1.m256[0], op2.m256[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256) / sizeof(r.m256[0])) ; i++) {
        r.m256[i] = _mm256_add_ps(op1.m256[i], op2.m256[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128) / sizeof(r.m128[0])) ; i++) {
        r.m128[i] = _mm_add_ps(op1.m128[i], op2.m128[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_f32_x
  #define svadd_f32_x(pg, op1, op2) easysimd_svadd_f32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svadd_f32_z(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_f32_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_f32_z(pg, easysimd_svadd_f32_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_f32_z
  #define svadd_f32_z(pg, op1, op2) easysimd_svadd_f32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svadd_f32_m(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_f32_m(pg, op1, op2);
  #else
    return easysimd_svsel_f32(pg, easysimd_svadd_f32_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_f32_m
  #define svadd_f32_m(pg, op1, op2) easysimd_svadd_f32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svadd_n_f32_x(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_float32 op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_f32_x(pg, op1, op2);
  #else
    return easysimd_svadd_f32_x(pg, op1, easysimd_svdup_n_f32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_f32_x
  #define svadd_n_f32_x(pg, op1, op2) easysimd_svadd_n_f32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svadd_n_f32_z(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_float32 op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_f32_z(pg, op1, op2);
  #else
    return easysimd_svadd_f32_z(pg, op1, easysimd_svdup_n_f32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_f32_z
  #define svadd_n_f32_z(pg, op1, op2) easysimd_svadd_n_f32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svadd_n_f32_m(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_float32 op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_f32_m(pg, op1, op2);
  #else
    return easysimd_svadd_f32_m(pg, op1, easysimd_svdup_n_f32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_f32_m
  #define svadd_n_f32_m(pg, op1, op2) easysimd_svadd_n_f32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svadd_f64_x(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_f64_x(pg, op1, op2);
  #else
    easysimd_svfloat64_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r.neon = vaddq_f64(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512d = _mm512_add_pd(op1.m512d, op2.m512d);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256d[0] = _mm256_add_pd(op1.m256d[0], op2.m256d[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256d) / sizeof(r.m256d[0])) ; i++) {
        r.m256d[i] = _mm256_add_pd(op1.m256d[i], op2.m256d[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128d) / sizeof(r.m128d[0])) ; i++) {
        r.m128d[i] = _mm_add_pd(op1.m128d[i], op2.m128d[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values + op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] + op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_f64_x
  #define svadd_f64_x(pg, op1, op2) easysimd_svadd_f64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svadd_f64_z(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_f64_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_f64_z(pg, easysimd_svadd_f64_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_f64_z
  #define svadd_f64_z(pg, op1, op2) easysimd_svadd_f64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svadd_f64_m(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_f64_m(pg, op1, op2);
  #else
    return easysimd_svsel_f64(pg, easysimd_svadd_f64_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_f64_m
  #define svadd_f64_m(pg, op1, op2) easysimd_svadd_f64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svadd_n_f64_x(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_float64 op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_f64_x(pg, op1, op2);
  #else
    return easysimd_svadd_f64_x(pg, op1, easysimd_svdup_n_f64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_f64_x
  #define svadd_n_f64_x(pg, op1, op2) easysimd_svadd_n_f64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svadd_n_f64_z(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_float64 op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_f64_z(pg, op1, op2);
  #else
    return easysimd_svadd_f64_z(pg, op1, easysimd_svdup_n_f64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_f64_z
  #define svadd_n_f64_z(pg, op1, op2) easysimd_svadd_n_f64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svadd_n_f64_m(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_float64 op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svadd_n_f64_m(pg, op1, op2);
  #else
    return easysimd_svadd_f64_m(pg, op1, easysimd_svdup_n_f64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svadd_n_f64_m
  #define svadd_n_f64_m(pg, op1, op2) easysimd_svadd_n_f64_m(pg, op1, op2)
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svadd_x(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svadd_s8_x   (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svadd_s16_x  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svadd_s32_x  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svadd_s64_x  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svadd_u8_x   (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svadd_x(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svadd_u16_x  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svadd_x(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svadd_u32_x  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svadd_x(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svadd_u64_x  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svadd_x(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) { return easysimd_svadd_f32_x  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svadd_x(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) { return easysimd_svadd_f64_x  (pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svadd_z(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svadd_s8_z   (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svadd_s16_z  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svadd_s32_z  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svadd_s64_z  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svadd_u8_z   (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svadd_z(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svadd_u16_z  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svadd_z(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svadd_u32_z  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svadd_z(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svadd_u64_z  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svadd_z(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) { return easysimd_svadd_f32_z  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svadd_z(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) { return easysimd_svadd_f64_z  (pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svadd_m(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svadd_s8_m   (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svadd_s16_m  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svadd_s32_m  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svadd_s64_m  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svadd_u8_m   (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svadd_m(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svadd_u16_m  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svadd_m(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svadd_u32_m  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svadd_m(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svadd_u64_m  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svadd_m(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) { return easysimd_svadd_f32_m  (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svadd_m(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) { return easysimd_svadd_f64_m  (pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svadd_x(easysimd_svbool_t pg,    easysimd_svint8_t op1,            int8_t op2) { return easysimd_svadd_n_s8_x (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svint16_t op1,           int16_t op2) { return easysimd_svadd_n_s16_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svint32_t op1,           int32_t op2) { return easysimd_svadd_n_s32_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svint64_t op1,           int64_t op2) { return easysimd_svadd_n_s64_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svadd_x(easysimd_svbool_t pg,   easysimd_svuint8_t op1,           uint8_t op2) { return easysimd_svadd_n_u8_x (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svadd_x(easysimd_svbool_t pg,  easysimd_svuint16_t op1,          uint16_t op2) { return easysimd_svadd_n_u16_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svadd_x(easysimd_svbool_t pg,  easysimd_svuint32_t op1,          uint32_t op2) { return easysimd_svadd_n_u32_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svadd_x(easysimd_svbool_t pg,  easysimd_svuint64_t op1,          uint64_t op2) { return easysimd_svadd_n_u64_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svadd_x(easysimd_svbool_t pg, easysimd_svfloat32_t op1,     easysimd_float32 op2) { return easysimd_svadd_n_f32_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svadd_x(easysimd_svbool_t pg, easysimd_svfloat64_t op1,     easysimd_float64 op2) { return easysimd_svadd_n_f64_x(pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svadd_z(easysimd_svbool_t pg,    easysimd_svint8_t op1,            int8_t op2) { return easysimd_svadd_n_s8_z (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svint16_t op1,           int16_t op2) { return easysimd_svadd_n_s16_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svint32_t op1,           int32_t op2) { return easysimd_svadd_n_s32_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svint64_t op1,           int64_t op2) { return easysimd_svadd_n_s64_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svadd_z(easysimd_svbool_t pg,   easysimd_svuint8_t op1,           uint8_t op2) { return easysimd_svadd_n_u8_z (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svadd_z(easysimd_svbool_t pg,  easysimd_svuint16_t op1,          uint16_t op2) { return easysimd_svadd_n_u16_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svadd_z(easysimd_svbool_t pg,  easysimd_svuint32_t op1,          uint32_t op2) { return easysimd_svadd_n_u32_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svadd_z(easysimd_svbool_t pg,  easysimd_svuint64_t op1,          uint64_t op2) { return easysimd_svadd_n_u64_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svadd_z(easysimd_svbool_t pg, easysimd_svfloat32_t op1,     easysimd_float32 op2) { return easysimd_svadd_n_f32_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svadd_z(easysimd_svbool_t pg, easysimd_svfloat64_t op1,     easysimd_float64 op2) { return easysimd_svadd_n_f64_z(pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svadd_m(easysimd_svbool_t pg,    easysimd_svint8_t op1,            int8_t op2) { return easysimd_svadd_n_s8_m (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svint16_t op1,           int16_t op2) { return easysimd_svadd_n_s16_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svint32_t op1,           int32_t op2) { return easysimd_svadd_n_s32_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svint64_t op1,           int64_t op2) { return easysimd_svadd_n_s64_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svadd_m(easysimd_svbool_t pg,   easysimd_svuint8_t op1,           uint8_t op2) { return easysimd_svadd_n_u8_m (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svadd_m(easysimd_svbool_t pg,  easysimd_svuint16_t op1,          uint16_t op2) { return easysimd_svadd_n_u16_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svadd_m(easysimd_svbool_t pg,  easysimd_svuint32_t op1,          uint32_t op2) { return easysimd_svadd_n_u32_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svadd_m(easysimd_svbool_t pg,  easysimd_svuint64_t op1,          uint64_t op2) { return easysimd_svadd_n_u64_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svadd_m(easysimd_svbool_t pg, easysimd_svfloat32_t op1,     easysimd_float32 op2) { return easysimd_svadd_n_f32_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svadd_m(easysimd_svbool_t pg, easysimd_svfloat64_t op1,     easysimd_float64 op2) { return easysimd_svadd_n_f64_m(pg, op1, op2); }
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_svadd_x(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op2), \
         easysimd_svint8_t: easysimd_svadd_s8_x, \
        easysimd_svint16_t: easysimd_svadd_s16_x, \
        easysimd_svint32_t: easysimd_svadd_s32_x, \
        easysimd_svint64_t: easysimd_svadd_s64_x, \
        easysimd_svuint8_t: easysimd_svadd_u8_x, \
       easysimd_svuint16_t: easysimd_svadd_u16_x, \
       easysimd_svuint32_t: easysimd_svadd_u32_x, \
       easysimd_svuint64_t: easysimd_svadd_u64_x, \
      easysimd_svfloat32_t: easysimd_svadd_f32_x, \
      easysimd_svfloat64_t: easysimd_svadd_f64_x, \
                 int8_t: easysimd_svadd_n_s8_x, \
                int16_t: easysimd_svadd_n_s16_x, \
                int32_t: easysimd_svadd_n_s32_x, \
                int64_t: easysimd_svadd_n_s64_x, \
                uint8_t: easysimd_svadd_n_u8_x, \
               uint16_t: easysimd_svadd_n_u16_x, \
               uint32_t: easysimd_svadd_n_u32_x, \
               uint64_t: easysimd_svadd_n_u64_x, \
          easysimd_float32: easysimd_svadd_n_f32_x, \
          easysimd_float64: easysimd_svadd_n_f64_x)((pg), (op1), (op2)))

  #define easysimd_svadd_z(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op2), \
         easysimd_svint8_t: easysimd_svadd_s8_z, \
        easysimd_svint16_t: easysimd_svadd_s16_z, \
        easysimd_svint32_t: easysimd_svadd_s32_z, \
        easysimd_svint64_t: easysimd_svadd_s64_z, \
        easysimd_svuint8_t: easysimd_svadd_u8_z, \
       easysimd_svuint16_t: easysimd_svadd_u16_z, \
       easysimd_svuint32_t: easysimd_svadd_u32_z, \
       easysimd_svuint64_t: easysimd_svadd_u64_z, \
      easysimd_svfloat32_t: easysimd_svadd_f32_z, \
      easysimd_svfloat64_t: easysimd_svadd_f64_z, \
                 int8_t: easysimd_svadd_n_s8_z, \
                int16_t: easysimd_svadd_n_s16_z, \
                int32_t: easysimd_svadd_n_s32_z, \
                int64_t: easysimd_svadd_n_s64_z, \
                uint8_t: easysimd_svadd_n_u8_z, \
               uint16_t: easysimd_svadd_n_u16_z, \
               uint32_t: easysimd_svadd_n_u32_z, \
               uint64_t: easysimd_svadd_n_u64_z, \
          easysimd_float32: easysimd_svadd_n_f32_z, \
          easysimd_float64: easysimd_svadd_n_f64_z)((pg), (op1), (op2)))

  #define easysimd_svadd_m(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op2), \
         easysimd_svint8_t: easysimd_svadd_s8_m, \
        easysimd_svint16_t: easysimd_svadd_s16_m, \
        easysimd_svint32_t: easysimd_svadd_s32_m, \
        easysimd_svint64_t: easysimd_svadd_s64_m, \
        easysimd_svuint8_t: easysimd_svadd_u8_m, \
       easysimd_svuint16_t: easysimd_svadd_u16_m, \
       easysimd_svuint32_t: easysimd_svadd_u32_m, \
       easysimd_svuint64_t: easysimd_svadd_u64_m, \
      easysimd_svfloat32_t: easysimd_svadd_f32_m, \
      easysimd_svfloat64_t: easysimd_svadd_f64_m, \
                 int8_t: easysimd_svadd_n_s8_m, \
                int16_t: easysimd_svadd_n_s16_m, \
                int32_t: easysimd_svadd_n_s32_m, \
                int64_t: easysimd_svadd_n_s64_m, \
                uint8_t: easysimd_svadd_n_u8_m, \
               uint16_t: easysimd_svadd_n_u16_m, \
               uint32_t: easysimd_svadd_n_u32_m, \
               uint64_t: easysimd_svadd_n_u64_m, \
          easysimd_float32: easysimd_svadd_n_f32_m, \
          easysimd_float64: easysimd_svadd_n_f64_m)((pg), (op1), (op2)))
#endif
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef svadd_x
  #undef svadd_z
  #undef svadd_m
  #undef svadd_n_x
  #undef svadd_n_z
  #undef svadd_n_m
  #define svadd_x(pg, op1, op2) easysimd_svadd_x((pg), (op1), (op2))
  #define svadd_z(pg, op1, op2) easysimd_svadd_z((pg), (op1), (op2))
  #define svadd_m(pg, op1, op2) easysimd_svadd_m((pg), (op1), (op2))
  #define svadd_n_x(pg, op1, op2) easysimd_svadd_n_x((pg), (op1), (op2))
  #define svadd_n_z(pg, op1, op2) easysimd_svadd_n_z((pg), (op1), (op2))
  #define svadd_n_m(pg, op1, op2) easysimd_svadd_n_m((pg), (op1), (op2))
#endif


#if defined (EASYSIMD_ARM_SVE_NATIVE)
  #define easysimd_svqadd_u8_z(pg, op1, op2)          svqadd_u8_z((pg), (op1), (op2))
  #define easysimd_svqadd_u16_z(pg, op1, op2)         svqadd_u16_z((pg), (op1), (op2))
  #define easysimd_svqadd_s8_z(pg, op1, op2)          svqadd_s8_z((pg), (op1), (op2))
  #define easysimd_svqadd_s16_z(pg, op1, op2)         svqadd_s16_z((pg), (op1), (op2))
#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_ADD_H */
