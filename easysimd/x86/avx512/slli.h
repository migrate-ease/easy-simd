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
 *   2020      Hidayat Khan <huk2209@gmail.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_X86_AVX512_SLLI_H)
#define EASYSIMD_X86_AVX512_SLLI_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#include "setzero.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_slli_epi16 (easysimd__m512i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (defined(HEDLEY_GCC_VERSION) && ((__GNUC__ == 5 && __GNUC_MINOR__ == 5) || (__GNUC__ == 6 && __GNUC_MINOR__ >= 4)))
    easysimd__m512i r;

    EASYSIMD_CONSTIFY_16_(_mm512_slli_epi16, r, easysimd_mm512_setzero_si512(), imm8, a);

    return r;
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return EASYSIMD_BUG_IGNORE_SIGN_CONVERSION(_mm512_slli_epi16(a, imm8));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    if (imm8 > 15) {
      return easysimd_mm512_setzero_si512();
    }
    uint16_t imm8_ = HEDLEY_STATIC_CAST(uint16_t, imm8);

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], imm8_);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], imm8_);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], imm8_);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], imm8_);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    if (imm8 > 15) {
      easysimd_memset(&r_, 0, sizeof(r_));
    } else {
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
          r_.i16 = HEDLEY_STATIC_CAST(__typeof__(r_.i16), (a_.i16 << HEDLEY_STATIC_CAST(int16_t, imm8)));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (imm8 & 0xff));
        }
      #endif
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm512_slli_epi16(a, imm8) ({ \
    easysimd__m512i r; \
    if (imm8 > 15) { \
      r = easysimd_mm512_setzero_si512(); \
    } else {\
      r.m128i[0].neon_u16 = vshlq_n_u16(a.m128i[0].neon_u16, imm8); \
      r.m128i[1].neon_u16 = vshlq_n_u16(a.m128i[1].neon_u16, imm8); \
      r.m128i[2].neon_u16 = vshlq_n_u16(a.m128i[2].neon_u16, imm8); \
      r.m128i[3].neon_u16 = vshlq_n_u16(a.m128i[3].neon_u16, imm8); \
    } \
    r; \
  })
#endif

#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_slli_epi16
  #define _mm512_slli_epi16(a, imm8) easysimd_mm512_slli_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_slli_epi32 (easysimd__m512i a, unsigned int imm8) {
  /* I guess the restriction was added in 6.4, back-ported to 5.5, then
   * removed (fixed) in 7? */
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (defined(HEDLEY_GCC_VERSION) && ((__GNUC__ == 5 && __GNUC_MINOR__ == 5) || (__GNUC__ == 6 && __GNUC_MINOR__ >= 4)))
    easysimd__m512i r;

    EASYSIMD_CONSTIFY_32_(_mm512_slli_epi32, r, easysimd_mm512_setzero_si512(), imm8, a);

    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return EASYSIMD_BUG_IGNORE_SIGN_CONVERSION(_mm512_slli_epi32(a, imm8));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    if (imm8 > 31) {
      return easysimd_mm512_setzero_si512();
    }
    uint32_t imm8_ = HEDLEY_STATIC_CAST(uint32_t, imm8);

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], imm8_);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], imm8_);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], imm8_);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], imm8_);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    /* The Intel Intrinsics Guide says that only the 8 LSBits of imm8 are
    * used.  In this case we should do "imm8 &= 0xff".  However in
    * practice all bits are used. */
    if (imm8 > 31) {
      easysimd_memset(&r_, 0, sizeof(r_));
    } else {
      #if defined(EASYSIMD_X86_AVX2_NATIVE)
        r_.m256i[0] = easysimd_mm256_slli_epi32(a_.m256i[0], HEDLEY_STATIC_CAST(int, imm8));
        r_.m256i[1] = easysimd_mm256_slli_epi32(a_.m256i[1], HEDLEY_STATIC_CAST(int, imm8));
      #elif defined(EASYSIMD_X86_SSE2_NATIVE)
        r_.m128i[0] = easysimd_mm_slli_epi32(a_.m128i[0], HEDLEY_STATIC_CAST(int, imm8));
        r_.m128i[1] = easysimd_mm_slli_epi32(a_.m128i[1], HEDLEY_STATIC_CAST(int, imm8));
        r_.m128i[2] = easysimd_mm_slli_epi32(a_.m128i[2], HEDLEY_STATIC_CAST(int, imm8));
        r_.m128i[3] = easysimd_mm_slli_epi32(a_.m128i[3], HEDLEY_STATIC_CAST(int, imm8));
      #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.u32 = a_.u32 << imm8;
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = a_.u32[i] << imm8;
        }
      #endif
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm512_slli_epi32(a, imm8) ({ \
    easysimd__m512i r; \
    if (imm8 > 31) { \
      r = easysimd_mm512_setzero_si512(); \
    } else { \
      r.m128i[0].neon_u32 = vshlq_n_u32(a.m128i[0].neon_u32, imm8); \
      r.m128i[1].neon_u32 = vshlq_n_u32(a.m128i[1].neon_u32, imm8); \
      r.m128i[2].neon_u32 = vshlq_n_u32(a.m128i[2].neon_u32, imm8); \
      r.m128i[3].neon_u32 = vshlq_n_u32(a.m128i[3].neon_u32, imm8); \
    } \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_slli_epi32
  #define _mm512_slli_epi32(a, imm8) easysimd_mm512_slli_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_slli_epi64 (easysimd__m512i a, unsigned int imm8) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (defined(HEDLEY_GCC_VERSION) && ((__GNUC__ == 5 && __GNUC_MINOR__ == 5) || (__GNUC__ == 6 && __GNUC_MINOR__ >= 4)))
    easysimd__m512i r;

    EASYSIMD_CONSTIFY_64_(_mm512_slli_epi64, r, easysimd_mm512_setzero_si512(), imm8, a);

    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return EASYSIMD_BUG_IGNORE_SIGN_CONVERSION(_mm512_slli_epi64(a, imm8));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    if (imm8 > 63) {
      return easysimd_mm512_setzero_si512();
    }
    uint64_t imm8_ = HEDLEY_STATIC_CAST(uint64_t, imm8);

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], imm8_);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], imm8_);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], imm8_);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], imm8_);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    /* The Intel Intrinsics Guide says that only the 8 LSBits of imm8 are
    * used.  In this case we should do "imm8 &= 0xff".  However in
    * practice all bits are used. */
    if (imm8 > 63) {
      easysimd_memset(&r_, 0, sizeof(r_));
    } else {
      #if defined(EASYSIMD_X86_AVX2_NATIVE)
        r_.m256i[0] = easysimd_mm256_slli_epi64(a_.m256i[0], HEDLEY_STATIC_CAST(int, imm8));
        r_.m256i[1] = easysimd_mm256_slli_epi64(a_.m256i[1], HEDLEY_STATIC_CAST(int, imm8));
      #elif defined(EASYSIMD_X86_SSE2_NATIVE)
        r_.m128i[0] = easysimd_mm_slli_epi64(a_.m128i[0], HEDLEY_STATIC_CAST(int, imm8));
        r_.m128i[1] = easysimd_mm_slli_epi64(a_.m128i[1], HEDLEY_STATIC_CAST(int, imm8));
        r_.m128i[2] = easysimd_mm_slli_epi64(a_.m128i[2], HEDLEY_STATIC_CAST(int, imm8));
        r_.m128i[3] = easysimd_mm_slli_epi64(a_.m128i[3], HEDLEY_STATIC_CAST(int, imm8));
      #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_97248)
        r_.u64 = a_.u64 << imm8;
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = a_.u64[i] << imm8;
        }
      #endif
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm512_slli_epi64(a, imm8) ({ \
    easysimd__m512i r; \
    if (HEDLEY_LIKELY(imm8 < 64)) { \
        r.m128i[0].neon_i64 = vshlq_n_s64(a.m128i[0].neon_i64, imm8); \
        r.m128i[1].neon_i64 = vshlq_n_s64(a.m128i[1].neon_i64, imm8); \
        r.m128i[2].neon_i64 = vshlq_n_s64(a.m128i[2].neon_i64, imm8); \
        r.m128i[3].neon_i64 = vshlq_n_s64(a.m128i[3].neon_i64, imm8); \
    } else { \
        r.m128i[0].neon_i64 = vdupq_n_s64(0); \
        r.m128i[1].neon_i64 = vdupq_n_s64(0); \
        r.m128i[2].neon_i64 = vdupq_n_s64(0); \
        r.m128i[3].neon_i64 = vdupq_n_s64(0); \
    } \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_slli_epi64
  #define _mm512_slli_epi64(a, imm8) easysimd_mm512_slli_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_slli_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm512_mask_slli_epi16(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b16();
  r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], imm8), src.sve_i16[EASYSIMD_SV_INDEX_0]);
  r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], imm8), src.sve_i16[EASYSIMD_SV_INDEX_1]);
  r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], imm8), src.sve_i16[EASYSIMD_SV_INDEX_2]);
  r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], imm8), src.sve_i16[EASYSIMD_SV_INDEX_3]);
  return r;
#else
  easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
  if (imm8 > 15) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (imm8 & 0xff));
    }
  }
  return easysimd_mm512_mask_mov_epi16(src, k, easysimd__m512i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_slli_epi16
  #define _mm512_mask_slli_epi16(src, k, a, imm8) easysimd_mm512_mask_slli_epi16(src, k, a, imm8)
#endif 

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_slli_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_slli_epi32(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b32();
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], imm8), src.sve_i32[EASYSIMD_SV_INDEX_0]);
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], imm8), src.sve_i32[EASYSIMD_SV_INDEX_1]);
  r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], imm8), src.sve_i32[EASYSIMD_SV_INDEX_2]);
  r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], imm8), src.sve_i32[EASYSIMD_SV_INDEX_3]);
  return r;
#else
  easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
  if (imm8 > 31) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm512_mask_mov_epi32(src, k, easysimd__m512i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_slli_epi32
  #define _mm512_mask_slli_epi32(src, k, a, imm8) easysimd_mm512_mask_slli_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_slli_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_slli_epi64(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b64();
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], imm8), src.sve_i64[EASYSIMD_SV_INDEX_0]);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], imm8), src.sve_i64[EASYSIMD_SV_INDEX_1]);
  r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], imm8), src.sve_i64[EASYSIMD_SV_INDEX_2]);
  r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], imm8), src.sve_i64[EASYSIMD_SV_INDEX_3]);
  return r;
#else
  easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
  if (imm8 > 63) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = a_.u64[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm512_mask_mov_epi64(src, k, easysimd__m512i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_slli_epi64
  #define _mm512_mask_slli_epi64(src, k, a, imm8) easysimd_mm512_mask_slli_epi64(src, k, a, imm8)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_slli_epi16 (easysimd__mmask32 k, easysimd__m512i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm512_maskz_slli_epi16(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  r.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], imm8);
  r.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], imm8);
  r.sve_i16[EASYSIMD_SV_INDEX_2] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2], imm8);
  r.sve_i16[EASYSIMD_SV_INDEX_3] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3], imm8);
  return r;
#else
  easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
  if (imm8 > 15) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (imm8 & 0xff));
    }
  }
  return easysimd_mm512_maskz_mov_epi16(k, easysimd__m512i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_slli_epi16
  #define _mm512_maskz_slli_epi16(k, a, imm8) easysimd_mm512_maskz_slli_epi16(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_slli_epi32 (easysimd__mmask16 k, easysimd__m512i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_maskz_slli_epi32(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], imm8);
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], imm8);
  r.sve_i32[EASYSIMD_SV_INDEX_2] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], imm8);
  r.sve_i32[EASYSIMD_SV_INDEX_3] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], imm8);
  return r;
#else
  easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
  if (imm8 > 31) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm512_maskz_mov_epi32(k, easysimd__m512i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_slli_epi32
  #define _mm512_maskz_slli_epi32(k, a, imm8) easysimd_mm512_maskz_slli_epi32(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_slli_epi64 (easysimd__mmask8 k, easysimd__m512i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_maskz_slli_epi64(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], imm8);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], imm8);
  r.sve_i64[EASYSIMD_SV_INDEX_2] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], imm8);
  r.sve_i64[EASYSIMD_SV_INDEX_3] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], imm8);
  return r;
#else
  easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
  if (imm8 > 63) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = a_.u64[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm512_maskz_mov_epi64(k, easysimd__m512i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_slli_epi64
  #define _mm512_maskz_slli_epi64(k, a, imm8) easysimd_mm512_maskz_slli_epi64(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_slli_epi16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm256_mask_slli_epi16(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svbool_t pg = svptrue_b16();
  r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], imm8), src.sve_i16[EASYSIMD_SV_INDEX_0]);
  r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], imm8), src.sve_i16[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
  if (imm8 > 15) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (imm8 & 0xff));
    }
  }
  return easysimd_mm256_mask_mov_epi16(src, k, easysimd__m256i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_slli_epi16
  #define _mm256_mask_slli_epi16(src, k, a, imm8) easysimd_mm256_mask_slli_epi16(src, k, a, imm8)
#endif 

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_slli_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm256_mask_slli_epi32(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svbool_t pg = svptrue_b32();
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], imm8), src.sve_i32[EASYSIMD_SV_INDEX_0]);
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], imm8), src.sve_i32[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
  if (imm8 > 31) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm256_mask_mov_epi32(src, k, easysimd__m256i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_slli_epi32
  #define _mm256_mask_slli_epi32(src, k, a, imm8) easysimd_mm256_mask_slli_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_slli_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm256_mask_slli_epi64(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svbool_t pg = svptrue_b64();
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], imm8), src.sve_i64[EASYSIMD_SV_INDEX_0]);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], imm8), src.sve_i64[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
  if (imm8 > 63) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm256_mask_mov_epi64(src, k, easysimd__m256i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_slli_epi64
  #define _mm256_mask_slli_epi64(src, k, a, imm8) easysimd_mm256_mask_slli_epi64(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_slli_epi16 (easysimd__mmask16 k, easysimd__m256i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm256_maskz_slli_epi16(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], imm8);
  r.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], imm8);
  return r;
#else
  easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
  if (imm8 > 15) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (imm8 & 0xff));
    }
  }
  return easysimd_mm256_maskz_mov_epi16(k, easysimd__m256i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_slli_epi16
  #define _mm256_maskz_slli_epi16(k, a, imm8) easysimd_mm256_maskz_slli_epi16(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_slli_epi32 (easysimd__mmask8 k, easysimd__m256i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm256_maskz_slli_epi32(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], imm8);
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], imm8);
  return r;
#else
  easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
  if (imm8 > 31) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm256_maskz_mov_epi32(k, easysimd__m256i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_slli_epi32
  #define _mm256_maskz_slli_epi32(k, a, imm8) easysimd_mm256_maskz_slli_epi32(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_slli_epi64 (easysimd__mmask8 k, easysimd__m256i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm256_maskz_slli_epi64(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], imm8);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], imm8);
  return r;
#else
  easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
  if (imm8 > 63) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm256_maskz_mov_epi64(k, easysimd__m256i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_slli_epi64
  #define _mm256_maskz_slli_epi64(k, a, imm8) easysimd_mm256_maskz_slli_epi64(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_slli_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm_mask_slli_epi16(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  svbool_t pg = svptrue_b16();
  r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, 0), svlsl_n_s16_x(pg, a.sve_i16, imm8), src.sve_i16);
  return r;
#else
  easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);
  const int s = (imm8 > HEDLEY_STATIC_CAST(int, sizeof(r_.i16[0]) * CHAR_BIT) - 1) ? 0 : imm8;
  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
    r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << s);
  }
  return easysimd_mm_mask_mov_epi16(src, k, easysimd__m128i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_slli_epi16
  #define _mm_mask_slli_epi16(src, k, a, imm8) easysimd_mm_mask_slli_epi16(src, k, a, imm8)
#endif 

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_slli_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm_mask_slli_epi32(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  svbool_t pg = svptrue_b32();
  r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, 0), svlsl_n_s32_x(pg, a.sve_i32, imm8), src.sve_i32);
  return r;
#else
  easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);
  if (imm8 > 31) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm_mask_mov_epi32(src, k, easysimd__m128i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_slli_epi32
  #define _mm_mask_slli_epi32(src, k, a, imm8) easysimd_mm_mask_slli_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_slli_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm_mask_slli_epi64(src, k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  svbool_t pg = svptrue_b64();
  r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, 0), svlsl_n_s64_x(pg, a.sve_i64, imm8), src.sve_i64);
  return r;
#else
  easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);
  if (imm8 > 63) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm_mask_mov_epi64(src, k, easysimd__m128i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_slli_epi64
  #define _mm_mask_slli_epi64(src, k, a, imm8) easysimd_mm_mask_slli_epi64(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_slli_epi16 (easysimd__mmask8 k, easysimd__m128i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm_maskz_slli_epi16(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  r.sve_i16 = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, 0), a.sve_i16, imm8);
  return r;
#else
  easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);
  const int s = (imm8 > HEDLEY_STATIC_CAST(int, sizeof(r_.i16[0]) * CHAR_BIT) - 1) ? 0 : imm8;
  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
    r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << s);
  }
  return easysimd_mm_maskz_mov_epi16(k, easysimd__m128i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_slli_epi16
  #define _mm_maskz_slli_epi16(k, a, imm8) easysimd_mm_maskz_slli_epi16(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_slli_epi32 (easysimd__mmask8 k, easysimd__m128i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm_maskz_slli_epi32(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  r.sve_i32 = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, 0), a.sve_i32, imm8);
  return r;
#else
  easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);
  if (imm8 > 31) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm_maskz_mov_epi32(k, easysimd__m128i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_slli_epi32
  #define _mm_maskz_slli_epi32(k, a, imm8) easysimd_mm_maskz_slli_epi32(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_slli_epi64 (easysimd__mmask8 k, easysimd__m128i a, const unsigned int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm_maskz_slli_epi64(k, a, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  r.sve_i64 = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, 0), a.sve_i64, imm8);
  return r;
#else
  easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);
  if (imm8 > 63) {
    easysimd_memset(&r_, 0, sizeof(r_));
  } else {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] << (imm8 & 0xff);
    }
  }
  return easysimd_mm_maskz_mov_epi64(k, easysimd__m128i_from_private(r_));
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_slli_epi64
  #define _mm_maskz_slli_epi64(k, a, imm8) easysimd_mm_maskz_slli_epi64(k, a, imm8)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SLLI_H) */
