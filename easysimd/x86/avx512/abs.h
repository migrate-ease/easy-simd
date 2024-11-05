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
 */

#if !defined(EASYSIMD_X86_AVX512_ABS_H)
#define EASYSIMD_X86_AVX512_ABS_H

#include "types.h"
#include "mov.h"
#include "../avx2.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_abs_epi8(easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_abs_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = EASYSIMD_MASK_TO_B8(k, 0);
    r.sve_i8 = svsel_s8(pg, svabs_s8_x(pg, a.sve_i8), src.sve_i8);
    return r;
  #else
    return easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_abs_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_abs_epi8
  #define _mm_mask_abs_epi8(src, k, a) easysimd_mm_mask_abs_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_abs_epi8(easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_abs_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svabs_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_abs_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_abs_epi8
  #define _mm_maskz_abs_epi8(k, a) easysimd_mm_maskz_abs_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_abs_epi16(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_abs_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = EASYSIMD_MASK_TO_B16(k, 0);
    r.sve_i16 = svsel_s16(pg, svabs_s16_x(pg, a.sve_i16), src.sve_i16);
    return r;
  #else
    return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_abs_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_abs_epi16
  #define _mm_mask_abs_epi16(src, k, a) easysimd_mm_mask_abs_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_abs_epi16(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_abs_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svabs_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_abs_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_abs_epi16
  #define _mm_maskz_abs_epi16(k, a) easysimd_mm_maskz_abs_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_abs_epi32(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_abs_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = EASYSIMD_MASK_TO_B32(k, 0);
    r.sve_i32 = svsel_s32(pg, svabs_s32_x(pg, a.sve_i32), src.sve_i32);
    return r;
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_abs_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_abs_epi32
  #define _mm_mask_abs_epi32(src, k, a) easysimd_mm_mask_abs_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_abs_epi32(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_abs_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svabs_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_abs_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_abs_epi32
  #define _mm_maskz_abs_epi32(k, a) easysimd_mm_maskz_abs_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_abs_epi64(easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_abs_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svabs_s64_z(svptrue_b64(), a.sve_i64);
    return r;
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    const __m128i m = _mm_srai_epi32(_mm_shuffle_epi32(a, 0xF5), 31);
    return _mm_sub_epi64(_mm_xor_si128(a, m), m);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i64 = vabsq_s64(a_.neon_i64);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      const int64x2_t m = vshrq_n_s64(a_.neon_i64, 63);
      r_.neon_i64 = vsubq_s64(veorq_s64(a_.neon_i64, m), m);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      __typeof__(r_.i64) z = { 0, };
      __typeof__(r_.i64) m = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 < z);
      r_.i64 = (-a_.i64 & m) | (a_.i64 & ~m);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
        r_.i64[i] = (a_.i64[i] < INT64_C(0)) ? -a_.i64[i] : a_.i64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_abs_epi64
  #define _mm_abs_epi64(a) easysimd_mm_abs_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_abs_epi64(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_abs_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = EASYSIMD_MASK_TO_B64(k, 0);
    r.sve_i64 = svsel_s64(pg, svabs_s64_x(pg, a.sve_i64), src.sve_i64);
    return r;
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_abs_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_abs_epi64
  #define _mm_mask_abs_epi64(src, k, a) easysimd_mm_mask_abs_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_abs_epi64(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_abs_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svabs_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_abs_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_abs_epi64
  #define _mm_maskz_abs_epi64(k, a) easysimd_mm_maskz_abs_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_abs_epi8(easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_abs_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_abs_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_abs_epi8
  #define _mm256_mask_abs_epi8(src, k, a) easysimd_mm256_mask_abs_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_abs_epi8(easysimd__mmask32 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_abs_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svabs_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svabs_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_abs_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_abs_epi8
  #define _mm256_maskz_abs_epi8(k, a) easysimd_mm256_maskz_abs_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_abs_epi16(easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_abs_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi16(src, k, easysimd_mm256_abs_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_abs_epi16
  #define _mm256_mask_abs_epi16(src, k, a) easysimd_mm256_mask_abs_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_abs_epi16(easysimd__mmask16 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_abs_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svabs_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svabs_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_abs_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_abs_epi16
  #define _mm256_maskz_abs_epi16(k, a) easysimd_mm256_maskz_abs_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_abs_epi32(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_abs_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_abs_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_abs_epi32
  #define _mm256_mask_abs_epi32(src, k, a) easysimd_mm256_mask_abs_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_abs_epi32(easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_abs_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svabs_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svabs_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_abs_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_abs_epi32
  #define _mm256_maskz_abs_epi32(k, a) easysimd_mm256_maskz_abs_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_abs_epi64(easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_abs_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_abs_epi64(a_.m128i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
        r_.i64[i] = (a_.i64[i] < INT64_C(0)) ? -a_.i64[i] : a_.i64[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_abs_epi64
  #define _mm256_abs_epi64(a) easysimd_mm256_abs_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_abs_epi64(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_abs_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_abs_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_abs_epi64
  #define _mm256_mask_abs_epi64(src, k, a) easysimd_mm256_mask_abs_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_abs_epi64(easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_abs_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svabs_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svabs_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_abs_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_abs_epi64
  #define _mm256_maskz_abs_epi64(k, a) easysimd_mm256_maskz_abs_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_abs_epi8 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_abs_epi8(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i8 = vabsq_s8(a.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vabsq_s8(a.m128i[1].neon_i8);
    r.m128i[2].neon_i8 = vabsq_s8(a.m128i[2].neon_i8);
    r.m128i[3].neon_i8 = vabsq_s8(a.m128i[3].neon_i8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_abs_epi8(a_.m256i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] < INT32_C(0)) ? -a_.i8[i] : a_.i8[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_abs_epi8
  #define _mm512_abs_epi8(a) easysimd_mm512_abs_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_abs_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_abs_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svabs_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svabs_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), svabs_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_2]), src.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), svabs_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_3]), src.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_abs_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_abs_epi8
  #define _mm512_mask_abs_epi8(src, k, a) easysimd_mm512_mask_abs_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_abs_epi8 (easysimd__mmask64 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_abs_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svabs_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svabs_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svabs_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svabs_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_abs_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_abs_epi8
  #define _mm512_maskz_abs_epi8(k, a) easysimd_mm512_maskz_abs_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_abs_epi16 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_abs_epi16(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i16 = vabsq_s16(a.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vabsq_s16(a.m128i[1].neon_i16);
    r.m128i[2].neon_i16 = vabsq_s16(a.m128i[2].neon_i16);
    r.m128i[3].neon_i16 = vabsq_s16(a.m128i[3].neon_i16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_abs_epi16(a_.m256i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] < INT32_C(0)) ? -a_.i16[i] : a_.i16[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_abs_epi16
  #define _mm512_abs_epi16(a) easysimd_mm512_abs_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_abs_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_abs_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svabs_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svabs_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svabs_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_2]), src.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svabs_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_3]), src.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_abs_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_abs_epi16
  #define _mm512_mask_abs_epi16(src, k, a) easysimd_mm512_mask_abs_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_abs_epi16 (easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_abs_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svabs_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svabs_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svabs_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svabs_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_abs_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_abs_epi16
  #define _mm512_maskz_abs_epi16(k, a) easysimd_mm512_maskz_abs_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_abs_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_abs_epi32(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vabsq_s32(a.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vabsq_s32(a.m128i[1].neon_i32);
    r.m128i[2].neon_i32 = vabsq_s32(a.m128i[2].neon_i32);
    r.m128i[3].neon_i32 = vabsq_s32(a.m128i[3].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_abs_epi32(a_.m256i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i32) / sizeof(r_.i32[0])); i++) {
        r_.i32[i] = (a_.i32[i] < INT64_C(0)) ? -a_.i32[i] : a_.i32[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_abs_epi32
  #define _mm512_abs_epi32(a) easysimd_mm512_abs_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_abs_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_abs_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svabs_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svabs_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svabs_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2]), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svabs_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3]), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_abs_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_abs_epi32
  #define _mm512_mask_abs_epi32(src, k, a) easysimd_mm512_mask_abs_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_abs_epi32(easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_abs_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svabs_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svabs_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svabs_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svabs_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_abs_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_abs_epi32
  #define _mm512_maskz_abs_epi32(k, a) easysimd_mm512_maskz_abs_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_abs_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_abs_epi64(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i64 = vabsq_s64(a.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vabsq_s64(a.m128i[1].neon_i64);
    r.m128i[2].neon_i64 = vabsq_s64(a.m128i[2].neon_i64);
    r.m128i[3].neon_i64 = vabsq_s64(a.m128i[3].neon_i64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svabs_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_abs_epi64(a_.m256i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
        r_.i64[i] = (a_.i64[i] < INT64_C(0)) ? -a_.i64[i] : a_.i64[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_abs_epi64
  #define _mm512_abs_epi64(a) easysimd_mm512_abs_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_abs_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_abs_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svabs_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svabs_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svabs_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]), src.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svabs_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]), src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_abs_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_abs_epi64
  #define _mm512_mask_abs_epi64(src, k, a) easysimd_mm512_mask_abs_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_abs_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_abs_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svabs_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svabs_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svabs_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svabs_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_abs_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_abs_epi64
  #define _mm512_maskz_abs_epi64(k, a) easysimd_mm512_maskz_abs_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_abs_ps(easysimd__m512 v2) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
    return _mm512_abs_ps(v2);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vabsq_f32(v2.m128[0].neon_f32);
    r.m128[1].neon_f32 = vabsq_f32(v2.m128[1].neon_f32);
    r.m128[2].neon_f32 = vabsq_f32(v2.m128[2].neon_f32);
    r.m128[3].neon_f32 = vabsq_f32(v2.m128[3].neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svabs_f32_z(pg, v2.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svabs_f32_z(pg, v2.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svabs_f32_z(pg, v2.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svabs_f32_z(pg, v2.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      v2_ = easysimd__m512_to_private(v2);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      for (size_t i = 0 ; i < (sizeof(r_.m128_private) / sizeof(r_.m128_private[0])) ; i++) {
        r_.m128_private[i].neon_f32 = vabsq_f32(v2_.m128_private[i].neon_f32);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])); i++) {
        r_.f32[i] = (v2_.f32[i] < INT64_C(0)) ? -v2_.f32[i] : v2_.f32[i];
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_abs_ps
  #define _mm512_abs_ps(v2) easysimd_mm512_abs_ps(v2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_abs_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 v2) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
    return _mm512_mask_abs_ps(src, k, v2);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svabs_f32_x(pg, v2.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svabs_f32_x(pg, v2.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svabs_f32_x(pg, v2.sve_f32[EASYSIMD_SV_INDEX_2]), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svabs_f32_x(pg, v2.sve_f32[EASYSIMD_SV_INDEX_3]), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_abs_ps(v2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_abs_ps
  #define _mm512_mask_abs_ps(src, k, v2) easysimd_mm512_mask_abs_ps(src, k, v2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_abs_pd(easysimd__m512d v2) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,3,0))
    return _mm512_abs_pd(v2);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
    /* gcc bug: https://gcc.gnu.org/legacy-ml/gcc-patches/2018-01/msg01962.html */
    return _mm512_abs_pd(_mm512_castpd_ps(v2));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vabsq_f64(v2.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vabsq_f64(v2.m128d[1].neon_f64);
    r.m128d[2].neon_f64 = vabsq_f64(v2.m128d[2].neon_f64);
    r.m128d[3].neon_f64 = vabsq_f64(v2.m128d[3].neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svabs_f64_z(pg, v2.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svabs_f64_z(pg, v2.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svabs_f64_z(pg, v2.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svabs_f64_z(pg, v2.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d_private
      r_,
      v2_ = easysimd__m512d_to_private(v2);

    EASYSIMD_VECTORIZE
    for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])); i++) {
      r_.f64[i] = (v2_.f64[i] < INT64_C(0)) ? -v2_.f64[i] : v2_.f64[i];
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_abs_pd
  #define _mm512_abs_pd(v2) easysimd_mm512_abs_pd(v2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_abs_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d v2) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,3,0))
    return _mm512_mask_abs_pd(src, k, v2);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
    /* gcc bug: https://gcc.gnu.org/legacy-ml/gcc-patches/2018-01/msg01962.html */
    return _mm512_mask_abs_pd(src, k, _mm512_castpd_ps(v2));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svabs_f64_x(pg, v2.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svabs_f64_x(pg, v2.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svabs_f64_x(pg, v2.sve_f64[EASYSIMD_SV_INDEX_2]), src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svabs_f64_x(pg, v2.sve_f64[EASYSIMD_SV_INDEX_3]), src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_abs_pd(v2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_abs_pd
  #define _mm512_mask_abs_pd(src, k, v2) easysimd_mm512_mask_abs_pd(src, k, v2)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_ABS_H) */
