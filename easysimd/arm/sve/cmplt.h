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

#if !defined(EASYSIMD_ARM_SVE_CMPLT_H)
#define EASYSIMD_ARM_SVE_CMPLT_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_s8(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_s8(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask64(_mm512_mask_cmplt_epi8_mask(easysimd_svbool_to_mmask64(pg), op1.m512i, op2.m512i));
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask32(_mm256_mask_cmplt_epi8_mask(easysimd_svbool_to_mmask32(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon_i8 = vandq_s8(pg.neon_i8, vreinterpretq_s8_u8(vcltq_s8(op1.neon, op2.neon)));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(pg.m128i[i], _mm_cmplt_epi8(op1.m128i[i], op2.m128i[i]));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_i8 = pg.values_i8 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_i8), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_i8) / sizeof(r.values_i8[0])) ; i++) {
        r.values_i8[i] = pg.values_i8[i] & ((op1.values[i] < op2.values[i]) ? ~INT8_C(0) : INT8_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_s8
  #define svcmplt_s8(pg, op1, op2) easysimd_svcmplt_s8(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_s16(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_s16(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask32(_mm512_mask_cmplt_epi16_mask(easysimd_svbool_to_mmask32(pg), op1.m512i, op2.m512i));
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask16(_mm256_mask_cmplt_epi16_mask(easysimd_svbool_to_mmask16(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon_i16 = vandq_s16(pg.neon_i16, vreinterpretq_s16_u16(vcltq_s16(op1.neon, op2.neon)));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(pg.m128i[i], _mm_cmplt_epi16(op1.m128i[i], op2.m128i[i]));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_i16 = pg.values_i16 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_i16), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_i16) / sizeof(r.values_i16[0])) ; i++) {
        r.values_i16[i] = pg.values_i16[i] & ((op1.values[i] < op2.values[i]) ? ~INT16_C(0) : INT16_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_s16
  #define svcmplt_s16(pg, op1, op2) easysimd_svcmplt_s16(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_s32(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_s32(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask16(_mm512_mask_cmplt_epi32_mask(easysimd_svbool_to_mmask16(pg), op1.m512i, op2.m512i));
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask8(_mm256_mask_cmplt_epi32_mask(easysimd_svbool_to_mmask8(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon_i32 = vandq_s32(pg.neon_i32, vreinterpretq_s32_u32(vcltq_s32(op1.neon, op2.neon)));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(pg.m128i[i], _mm_cmplt_epi32(op1.m128i[i], op2.m128i[i]));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_i32 = pg.values_i32 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_i32), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_i32) / sizeof(r.values_i32[0])) ; i++) {
        r.values_i32[i] = pg.values_i32[i] & ((op1.values[i] < op2.values[i]) ? ~INT32_C(0) : INT32_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_s32
  #define svcmplt_s32(pg, op1, op2) easysimd_svcmplt_s32(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_s64(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_s64(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask8(_mm512_mask_cmplt_epi64_mask(easysimd_svbool_to_mmask8(pg), op1.m512i, op2.m512i));
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask4(_mm256_mask_cmplt_epi64_mask(easysimd_svbool_to_mmask4(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r.neon_i64 = vandq_s64(pg.neon_i64, vreinterpretq_s64_u64(vcltq_s64(op1.neon, op2.neon)));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_i64 = pg.values_i64 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_i64), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_i64) / sizeof(r.values_i64[0])) ; i++) {
        r.values_i64[i] = pg.values_i64[i] & ((op1.values[i] < op2.values[i]) ? ~INT64_C(0) : INT64_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_s64
  #define svcmplt_s64(pg, op1, op2) easysimd_svcmplt_s64(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_u8(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_u8(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask64(_mm512_mask_cmplt_epu8_mask(easysimd_svbool_to_mmask64(pg), op1.m512i, op2.m512i));
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask32(_mm256_mask_cmplt_epu8_mask(easysimd_svbool_to_mmask32(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon_u8 = vandq_u8(pg.neon_u8, vcltq_u8(op1.neon, op2.neon));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_u8 = pg.values_u8 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_u8), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_u8) / sizeof(r.values_u8[0])) ; i++) {
        r.values_u8[i] = pg.values_u8[i] & ((op1.values[i] < op2.values[i]) ? ~UINT8_C(0) : UINT8_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_u8
  #define svcmplt_u8(pg, op1, op2) easysimd_svcmplt_u8(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_u16(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_u16(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask32(_mm512_mask_cmplt_epu16_mask(easysimd_svbool_to_mmask32(pg), op1.m512i, op2.m512i));
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask16(_mm256_mask_cmplt_epu16_mask(easysimd_svbool_to_mmask16(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon_u16 = vandq_u16(pg.neon_u16, vcltq_u16(op1.neon, op2.neon));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_u16 = pg.values_u16 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_u16), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_u16) / sizeof(r.values_u16[0])) ; i++) {
        r.values_u16[i] = pg.values_u16[i] & ((op1.values[i] < op2.values[i]) ? ~UINT16_C(0) : UINT16_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_u16
  #define svcmplt_u16(pg, op1, op2) easysimd_svcmplt_u16(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_u32(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_u32(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask16(_mm512_mask_cmplt_epu32_mask(easysimd_svbool_to_mmask16(pg), op1.m512i, op2.m512i));
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask8(_mm256_mask_cmplt_epu32_mask(easysimd_svbool_to_mmask8(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon_u32 = vandq_u32(pg.neon_u32, vcltq_u32(op1.neon, op2.neon));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_u32 = pg.values_u32 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_u32), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_u32) / sizeof(r.values_u32[0])) ; i++) {
        r.values_u32[i] = pg.values_u32[i] & ((op1.values[i] < op2.values[i]) ? ~UINT32_C(0) : UINT32_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_u32
  #define svcmplt_u32(pg, op1, op2) easysimd_svcmplt_u32(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_u64(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_u64(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask8(_mm512_mask_cmplt_epu64_mask(easysimd_svbool_to_mmask8(pg), op1.m512i, op2.m512i));
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask4(_mm256_mask_cmplt_epu64_mask(easysimd_svbool_to_mmask4(pg), op1.m256i[0], op2.m256i[0]));
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r.neon_u64 = vandq_u64(pg.neon_u64, vcltq_u64(op1.neon, op2.neon));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_u64 = pg.values_u64 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_u64), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_u64) / sizeof(r.values_u64[0])) ; i++) {
        r.values_u64[i] = pg.values_u64[i] & ((op1.values[i] < op2.values[i]) ? ~UINT64_C(0) : UINT64_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_u64
  #define svcmplt_u64(pg, op1, op2) easysimd_svcmplt_u64(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_f32(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_f32(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask16(_mm512_mask_cmp_ps_mask(easysimd_svbool_to_mmask16(pg), op1.m512, op2.m512, _CMP_LT_OQ));
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask8(_mm256_mask_cmp_ps_mask(easysimd_svbool_to_mmask8(pg), op1.m256[0], op2.m256[0], _CMP_LT_OQ));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon_u32 = vandq_u32(pg.neon_u32, vcltq_f32(op1.neon, op2.neon));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(pg.m128i[i]), _mm_cmplt_ps(op1.m128[i], op2.m128[i])));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_i32 = pg.values_i32 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_i32), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_i32) / sizeof(r.values_i32[0])) ; i++) {
        r.values_i32[i] = pg.values_i32[i] & ((op1.values[i] < op2.values[i]) ? ~INT32_C(0) : INT32_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_f32
  #define svcmplt_f32(pg, op1, op2) easysimd_svcmplt_f32(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svcmplt_f64(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcmplt_f64(pg, op1, op2);
  #else
    easysimd_svbool_t r;

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r = easysimd_svbool_from_mmask8(_mm512_mask_cmp_pd_mask(easysimd_svbool_to_mmask8(pg), op1.m512d, op2.m512d, _CMP_LT_OQ));
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r = easysimd_svbool_from_mmask4(_mm256_mask_cmp_pd_mask(easysimd_svbool_to_mmask4(pg), op1.m256d[0], op2.m256d[0], _CMP_LT_OQ));
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r.neon_u64 = vandq_u64(pg.neon_u64, vcltq_f64(op1.neon, op2.neon));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_castpd_si128(_mm_and_pd(_mm_castsi128_pd(pg.m128i[i]), _mm_cmplt_pd(op1.m128d[i], op2.m128d[i])));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values_i64 = pg.values_i64 & HEDLEY_REINTERPRET_CAST(__typeof__(r.values_i64), op1.values < op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values_i64) / sizeof(r.values_i64[0])) ; i++) {
        r.values_i64[i] = pg.values_i64[i] & ((op1.values[i] < op2.values[i]) ? ~INT64_C(0) : INT64_C(0));
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcmplt_f64
  #define svcmplt_f64(pg, op1, op2) easysimd_svcmplt_f64(pg, op1, op2)
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return  easysimd_svcmplt_s8(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svcmplt_s16(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svcmplt_s32(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svcmplt_s64(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return  easysimd_svcmplt_u8(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svcmplt_u16(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svcmplt_u32(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svcmplt_u64(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) { return easysimd_svcmplt_f32(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svcmplt(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) { return easysimd_svcmplt_f64(pg, op1, op2); }

  #if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,    svint8_t op1,    svint8_t op2) { return  svcmplt_s8(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,   svint16_t op1,   svint16_t op2) { return svcmplt_s16(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,   svint32_t op1,   svint32_t op2) { return svcmplt_s32(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,   svint64_t op1,   svint64_t op2) { return svcmplt_s64(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,   svuint8_t op1,   svuint8_t op2) { return  svcmplt_u8(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,  svuint16_t op1,  svuint16_t op2) { return svcmplt_u16(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,  svuint32_t op1,  svuint32_t op2) { return svcmplt_u32(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg,  svuint64_t op1,  svuint64_t op2) { return svcmplt_u64(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg, svfloat32_t op1, svfloat32_t op2) { return svcmplt_f32(pg, op1, op2); }
    EASYSIMD_FUNCTION_ATTRIBUTES svbool_t svcmplt(svbool_t pg, svfloat64_t op1, svfloat64_t op2) { return svcmplt_f64(pg, op1, op2); }
  #endif
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_svcmplt(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op1), \
        easysimd_svint8_t:  easysimd_svcmplt_s8)(pg, op1, op2), \
       easysimd_svint16_t: easysimd_svcmplt_s16)(pg, op1, op2), \
       easysimd_svint32_t: easysimd_svcmplt_s32)(pg, op1, op2), \
       easysimd_svint64_t: easysimd_svcmplt_s64)(pg, op1, op2), \
       easysimd_svuint8_t:  easysimd_svcmplt_u8)(pg, op1, op2), \
      easysimd_svuint16_t: easysimd_svcmplt_u16)(pg, op1, op2), \
      easysimd_svuint32_t: easysimd_svcmplt_u32)(pg, op1, op2), \
      easysimd_svuint64_t: easysimd_svcmplt_u64)(pg, op1, op2), \
       easysimd_svint32_t: easysimd_svcmplt_f32)(pg, op1, op2), \
       easysimd_svint64_t: easysimd_svcmplt_f64)(pg, op1, op2))

  #if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
    #define svcmplt(pg, op1, op2) \
      (EASYSIMD_GENERIC_((op1), \
          svint8_t:  svcmplt_s8)(pg, op1, op2), \
         svint16_t: svcmplt_s16)(pg, op1, op2), \
         svint32_t: svcmplt_s32)(pg, op1, op2), \
         svint64_t: svcmplt_s64)(pg, op1, op2), \
         svuint8_t:  svcmplt_u8)(pg, op1, op2), \
        svuint16_t: svcmplt_u16)(pg, op1, op2), \
        svuint32_t: svcmplt_u32)(pg, op1, op2), \
        svuint64_t: svcmplt_u64)(pg, op1, op2), \
         svint32_t: svcmplt_f32)(pg, op1, op2), \
         svint64_t: svcmplt_f64)(pg, op1, op2))
  #endif
#endif
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef svcmplt
  #define svcmplt(pg, op1, op2) easysimd_svcmplt((pg), (op1), (op2))
#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_CMPLT_H */
