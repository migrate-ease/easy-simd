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

#if !defined(EASYSIMD_X86_AVX512_MOV_H)
#define EASYSIMD_X86_AVX512_MOV_H

#include "types.h"
#include "cast.h"
#include "set.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_mov_epi8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mov_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8, src.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] : src_.i8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mov_epi8
  #define _mm_mask_mov_epi8(src, k, a) easysimd_mm_mask_mov_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_mov_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mov_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16, src.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? a_.i16[i] : src_.i16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mov_epi16
  #define _mm_mask_mov_epi16(src, k, a) easysimd_mm_mask_mov_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_mov_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mov_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[i] : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mov_epi32
  #define _mm_mask_mov_epi32(src, k, a) easysimd_mm_mask_mov_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_mov_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mov_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64, src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      r_;

    /* N.B. CM: No fallbacks as there are only two elements */
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[i] : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mov_epi64
  #define _mm_mask_mov_epi64(src, k, a) easysimd_mm_mask_mov_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_mov_pd(easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mov_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64, src.sve_f64);
    return r;
  #else
    return easysimd_mm_castsi128_pd(easysimd_mm_mask_mov_epi64(easysimd_mm_castpd_si128(src), k, easysimd_mm_castpd_si128(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mov_pd
  #define _mm_mask_mov_pd(src, k, a) easysimd_mm_mask_mov_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_mov_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mov_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, src.sve_f32);
    return r;
  #else
    return easysimd_mm_castsi128_ps(easysimd_mm_mask_mov_epi32(easysimd_mm_castps_si128(src), k, easysimd_mm_castps_si128(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mov_ps
  #define _mm_mask_mov_ps(src, k, a) easysimd_mm_mask_mov_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_mov_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_mov_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], src.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i[0] = easysimd_mm_mask_mov_epi8(src_.m128i[0], HEDLEY_STATIC_CAST(easysimd__mmask16, k      ), a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_mask_mov_epi8(src_.m128i[1], HEDLEY_STATIC_CAST(easysimd__mmask16, k >> 16), a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] : src_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mov_epi8
  #define _mm256_mask_mov_epi8(src, k, a) easysimd_mm256_mask_mov_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_mov_epi16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_mov_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], src.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i[0] = easysimd_mm_mask_mov_epi16(src_.m128i[0], HEDLEY_STATIC_CAST(easysimd__mmask8, k     ), a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_mask_mov_epi16(src_.m128i[1], HEDLEY_STATIC_CAST(easysimd__mmask8, k >> 8), a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = ((k >> i) & 1) ? a_.i16[i] : src_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mov_epi16
  #define _mm256_mask_mov_epi16(src, k, a) easysimd_mm256_mask_mov_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_mov_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_mov_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i[0] = easysimd_mm_mask_mov_epi32(src_.m128i[0], k     , a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_mask_mov_epi32(src_.m128i[1], k >> 4, a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((k >> i) & 1) ? a_.i32[i] : src_.i32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mov_epi32
  #define _mm256_mask_mov_epi32(src, k, a) easysimd_mm256_mask_mov_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_mov_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_mov_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      r_;

    /* N.B. CM: This fallback may not be faster as there are only four elements */
    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i[0] = easysimd_mm_mask_mov_epi64(src_.m128i[0], k     , a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_mask_mov_epi64(src_.m128i[1], k >> 2, a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = ((k >> i) & 1) ? a_.i64[i] : src_.i64[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mov_epi64
  #define _mm256_mask_mov_epi64(src, k, a) easysimd_mm256_mask_mov_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_mov_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_mov_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], src.sve_f64[EASYSIMD_SV_INDEX_0]);

    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_castsi256_pd(easysimd_mm256_mask_mov_epi64(easysimd_mm256_castpd_si256(src), k, easysimd_mm256_castpd_si256(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mov_pd
  #define _mm256_mask_mov_pd(src, k, a) easysimd_mm256_mask_mov_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_mov_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_mov_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], src.sve_f32[EASYSIMD_SV_INDEX_0]);

    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_castsi256_ps(easysimd_mm256_mask_mov_epi32(easysimd_mm256_castps_si256(src), k, easysimd_mm256_castps_si256(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mov_ps
  #define _mm256_mask_mov_ps(src, k, a) easysimd_mm256_mask_mov_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_mov_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_mov_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], src.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_i8[EASYSIMD_SV_INDEX_2], src.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_i8[EASYSIMD_SV_INDEX_3], src.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    static easysimd__m128i mask = {
      .u8 = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}};

    uint8x8_t mk = vcreate_u8(k);
    uint8x16_t k_vec[4];
    k_vec[0] = vcombine_u8(vdup_lane_u8(mk, 0), vdup_lane_u8(mk, 1));
    k_vec[1] = vcombine_u8(vdup_lane_u8(mk, 2), vdup_lane_u8(mk, 3));
    k_vec[2] = vcombine_u8(vdup_lane_u8(mk, 4), vdup_lane_u8(mk, 5));
    k_vec[3] = vcombine_u8(vdup_lane_u8(mk, 6), vdup_lane_u8(mk, 7));
    a.m128i[0].neon_u8 = vbslq_u8(vtstq_u8(k_vec[0], mask.neon_u8), a.m128i[0].neon_u8, src.m128i[0].neon_u8);
    a.m128i[1].neon_u8 = vbslq_u8(vtstq_u8(k_vec[1], mask.neon_u8), a.m128i[1].neon_u8, src.m128i[1].neon_u8);
    a.m128i[2].neon_u8 = vbslq_u8(vtstq_u8(k_vec[2], mask.neon_u8), a.m128i[2].neon_u8, src.m128i[2].neon_u8);
    a.m128i[3].neon_u8 = vbslq_u8(vtstq_u8(k_vec[3], mask.neon_u8), a.m128i[3].neon_u8, src.m128i[3].neon_u8);
    return a;
  #else
    easysimd__m512i_private
      src_ = easysimd__m512i_to_private(src),
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m256i[0] = easysimd_mm256_mask_mov_epi8(src_.m256i[0], HEDLEY_STATIC_CAST(easysimd__mmask32, k      ), a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_mask_mov_epi8(src_.m256i[1], HEDLEY_STATIC_CAST(easysimd__mmask32, k >> 32), a_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] : src_.i8[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mov_epi8
  #define _mm512_mask_mov_epi8(src, k, a) easysimd_mm512_mask_mov_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_mov_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_mov_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], src.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2], src.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3], src.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    static easysimd__m128i mask = {
      .u16 = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}};

    uint16x8_t mk = vmovl_u8(vcreate_u8(k));
    uint16x8_t k_vec[4];
    k_vec[0] = vdupq_laneq_u16(mk, 0);
    k_vec[1] = vdupq_laneq_u16(mk, 1);
    k_vec[2] = vdupq_laneq_u16(mk, 2);
    k_vec[3] = vdupq_laneq_u16(mk, 3);

    a.m128i[0].neon_u16 = vbslq_u16(vtstq_u16(k_vec[0], mask.neon_u16), a.m128i[0].neon_u16, src.m128i[0].neon_u16);
    a.m128i[1].neon_u16 = vbslq_u16(vtstq_u16(k_vec[1], mask.neon_u16), a.m128i[1].neon_u16, src.m128i[1].neon_u16);
    a.m128i[2].neon_u16 = vbslq_u16(vtstq_u16(k_vec[2], mask.neon_u16), a.m128i[2].neon_u16, src.m128i[2].neon_u16);
    a.m128i[3].neon_u16 = vbslq_u16(vtstq_u16(k_vec[3], mask.neon_u16), a.m128i[3].neon_u16, src.m128i[3].neon_u16);
    return a;
  #else
    easysimd__m512i_private
      src_ = easysimd__m512i_to_private(src),
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m256i[0] = easysimd_mm256_mask_mov_epi16(src_.m256i[0], HEDLEY_STATIC_CAST(easysimd__mmask16, k      ), a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_mask_mov_epi16(src_.m256i[1], HEDLEY_STATIC_CAST(easysimd__mmask16, k >> 16), a_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = ((k >> i) & 1) ? a_.i16[i] : src_.i16[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mov_epi16
  #define _mm512_mask_mov_epi16(src, k, a) easysimd_mm512_mask_mov_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_mov_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mov_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    static easysimd__m128i mask = {
      .u32 = {0x01, 0x02, 0x04, 0x08}};

    uint32x4_t k_vec[4];
    k_vec[0] = vdupq_n_u32(k & 0xFFFFFFFF);
    k_vec[1] = vdupq_n_u32((k >> 4) & 0xFFFFFFFF);
    k_vec[2] = vdupq_n_u32((k >> 8) & 0xFFFFFFFF);
    k_vec[3] = vdupq_n_u32((k >> 12) & 0xFFFFFFFF);

    a.m128i[0].neon_u32 = vbslq_u32(vtstq_u32(k_vec[0], mask.neon_u32), a.m128i[0].neon_u32, src.m128i[0].neon_u32);
    a.m128i[1].neon_u32 = vbslq_u32(vtstq_u32(k_vec[1], mask.neon_u32), a.m128i[1].neon_u32, src.m128i[1].neon_u32);
    a.m128i[2].neon_u32 = vbslq_u32(vtstq_u32(k_vec[2], mask.neon_u32), a.m128i[2].neon_u32, src.m128i[2].neon_u32);
    a.m128i[3].neon_u32 = vbslq_u32(vtstq_u32(k_vec[3], mask.neon_u32), a.m128i[3].neon_u32, src.m128i[3].neon_u32);
    return a;
  #else
    easysimd__m512i_private
      src_ = easysimd__m512i_to_private(src),
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m256i[0] = easysimd_mm256_mask_mov_epi32(src_.m256i[0], HEDLEY_STATIC_CAST(easysimd__mmask8, k     ), a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_mask_mov_epi32(src_.m256i[1], HEDLEY_STATIC_CAST(easysimd__mmask8, k >> 8), a_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((k >> i) & 1) ? a_.i32[i] : src_.i32[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mov_epi32
  #define _mm512_mask_mov_epi32(src, k, a) easysimd_mm512_mask_mov_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_mov_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mov_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], src.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], src.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      src_ = easysimd__m512i_to_private(src),
      a_ = easysimd__m512i_to_private(a),
      r_;

    /* N.B. CM: Without AVX2 this fallback may not be faster as there are only eight elements */
    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m256i[0] = easysimd_mm256_mask_mov_epi64(src_.m256i[0], k     , a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_mask_mov_epi64(src_.m256i[1], k >> 4, a_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = ((k >> i) & 1) ? a_.i64[i] : src_.i64[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mov_epi64
  #define _mm512_mask_mov_epi64(src, k, a) easysimd_mm512_mask_mov_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_mov_pd (easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mov_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_castsi512_pd(easysimd_mm512_mask_mov_epi64(easysimd_mm512_castpd_si512(src), k, easysimd_mm512_castpd_si512(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mov_pd
  #define _mm512_mask_mov_pd(src, k, a) easysimd_mm512_mask_mov_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_mov_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mov_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_castsi512_ps(easysimd_mm512_mask_mov_epi32(easysimd_mm512_castps_si512(src), k, easysimd_mm512_castps_si512(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mov_ps
  #define _mm512_mask_mov_ps(src, k, a) easysimd_mm512_mask_mov_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_mov_epi8 (easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mov_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8, svdup_n_s8(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] : INT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mov_epi8
  #define _mm_maskz_mov_epi8(k, a) easysimd_mm_maskz_mov_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_mov_epi16 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mov_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16, svdup_n_s16(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? a_.i16[i] : INT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mov_epi16
  #define _mm_maskz_mov_epi16(k, a) easysimd_mm_maskz_mov_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_mov_epi32 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mov_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, svdup_n_s32(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[i] : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mov_epi32
  #define _mm_maskz_mov_epi32(k, a) easysimd_mm_maskz_mov_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_mov_epi64 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mov_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64, svdup_n_s64(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      r_;

    /* N.B. CM: No fallbacks as there are only two elements */
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[i] : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mov_epi64
  #define _mm_maskz_mov_epi64(k, a) easysimd_mm_maskz_mov_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_mov_pd (easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mov_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64, svdup_n_f64(0.0));
    return r;
  #else
    return easysimd_mm_castsi128_pd(easysimd_mm_maskz_mov_epi64(k, easysimd_mm_castpd_si128(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mov_pd
  #define _mm_maskz_mov_pd(k, a) easysimd_mm_maskz_mov_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_mov_ps (easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mov_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, svdup_n_f32(0.0));
    return r;
  #else
    return easysimd_mm_castsi128_ps(easysimd_mm_maskz_mov_epi32(k, easysimd_mm_castps_si128(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mov_ps
  #define _mm_maskz_mov_ps(k, a) easysimd_mm_maskz_mov_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_mov_epi8 (easysimd__mmask32 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mov_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], svdup_n_s8(0));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], svdup_n_s8(0));
    return r;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m128i[0] = easysimd_mm_maskz_mov_epi8(HEDLEY_STATIC_CAST(easysimd__mmask16, k      ), a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_maskz_mov_epi8(HEDLEY_STATIC_CAST(easysimd__mmask16, k >> 16), a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] : INT8_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mov_epi8
  #define _mm256_maskz_mov_epi8(k, a) easysimd_mm256_maskz_mov_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_mov_epi16 (easysimd__mmask16 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mov_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], svdup_n_s16(0));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], svdup_n_s16(0));
    return r;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i[0] = easysimd_mm_maskz_mov_epi16(HEDLEY_STATIC_CAST(easysimd__mmask8, k     ), a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_maskz_mov_epi16(HEDLEY_STATIC_CAST(easysimd__mmask8, k >> 8), a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = ((k >> i) & 1) ? a_.i16[i] : INT16_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mov_epi16
  #define _mm256_maskz_mov_epi16(k, a) easysimd_mm256_maskz_mov_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_mov_epi32 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mov_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], svdup_n_s32(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], svdup_n_s32(0));
    return r;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i[0] = easysimd_mm_maskz_mov_epi32(k     , a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_maskz_mov_epi32(k >> 4, a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((k >> i) & 1) ? a_.i32[i] : INT32_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mov_epi32
  #define _mm256_maskz_mov_epi32(k, a) easysimd_mm256_maskz_mov_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_mov_epi64 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mov_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], svdup_n_s64(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(0));
    return r;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      r_;

    /* N.B. CM: This fallback may not be faster as there are only four elements */
    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i[0] = easysimd_mm_maskz_mov_epi64(k     , a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_maskz_mov_epi64(k >> 2, a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = ((k >> i) & 1) ? a_.i64[i] : INT64_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mov_epi64
  #define _mm256_maskz_mov_epi64(k, a) easysimd_mm256_maskz_mov_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_mov_pd (easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mov_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], svdup_n_f64(0.0));
    return r;
  #else
    return easysimd_mm256_castsi256_pd(easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_castpd_si256(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mov_pd
  #define _mm256_maskz_mov_pd(k, a) easysimd_mm256_maskz_mov_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_mov_ps (easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mov_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], svdup_n_f32(0.0));
    return r;
  #else
    return easysimd_mm256_castsi256_ps(easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_castps_si256(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mov_ps
  #define _mm256_maskz_mov_ps(k, a) easysimd_mm256_maskz_mov_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_mov_epi8 (easysimd__mmask64 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_mov_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svint8_t svzero = svdup_n_s8(0);
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_i8[EASYSIMD_SV_INDEX_2], svzero);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_i8[EASYSIMD_SV_INDEX_3], svzero);
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSSE3_NATIVE)
      r_.m256i[0] = easysimd_mm256_maskz_mov_epi8(HEDLEY_STATIC_CAST(easysimd__mmask32, k      ), a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_maskz_mov_epi8(HEDLEY_STATIC_CAST(easysimd__mmask32, k >> 32), a_.m256i[1]);
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] : INT8_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mov_epi8
  #define _mm512_maskz_mov_epi8(k, a) easysimd_mm512_maskz_mov_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_mov_epi16 (easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_mov_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svint16_t svzero = svdup_n_s16(0);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2], svzero);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3], svzero);
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m256i[0] = easysimd_mm256_maskz_mov_epi16(HEDLEY_STATIC_CAST(easysimd__mmask16, k      ), a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_maskz_mov_epi16(HEDLEY_STATIC_CAST(easysimd__mmask16, k >> 16), a_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = ((k >> i) & 1) ? a_.i16[i] : INT16_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mov_epi16
  #define _mm512_maskz_mov_epi16(k, a) easysimd_mm512_maskz_mov_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_mov_epi32 (easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mov_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svint32_t svzero = svdup_n_s32(0);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], svzero);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], svzero);
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m256i[0] = easysimd_mm256_maskz_mov_epi32(HEDLEY_STATIC_CAST(easysimd__mmask8, k     ), a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_maskz_mov_epi32(HEDLEY_STATIC_CAST(easysimd__mmask8, k >> 8), a_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((k >> i) & 1) ? a_.i32[i] : INT32_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mov_epi32
  #define _mm512_maskz_mov_epi32(k, a) easysimd_mm512_maskz_mov_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_mov_epi64 (easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mov_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svint64_t svzero = svdup_n_s64(0);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], svzero);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], svzero);
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      r_;

    /* N.B. CM: Without AVX2 this fallback may not be faster as there are only eight elements */
    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m256i[0] = easysimd_mm256_maskz_mov_epi64(k     , a_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_maskz_mov_epi64(k >> 4, a_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = ((k >> i) & 1) ? a_.i64[i] : INT64_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mov_epi64
  #define _mm512_maskz_mov_epi64(k, a) easysimd_mm512_maskz_mov_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_mov_pd (easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mov_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svfloat64_t svzero = svdup_n_f64(0);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], svzero);
    return r;
  #else
    return easysimd_mm512_castsi512_pd(easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_castpd_si512(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mov_pd
  #define _mm512_maskz_mov_pd(k, a) easysimd_mm512_maskz_mov_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_mov_ps (easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mov_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svfloat32_t svzero = svdup_n_f32(0);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], svzero);
    return r;
  #else
    return easysimd_mm512_castsi512_ps(easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_castps_si512(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mov_ps
  #define _mm512_maskz_mov_ps(k, a) easysimd_mm512_maskz_mov_ps(k, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_MOV_H) */
