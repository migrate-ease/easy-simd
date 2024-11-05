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

#if !defined(EASYSIMD_X86_AVX512_INSERT_H)
#define EASYSIMD_X86_AVX512_INSERT_H

#include "types.h"
#include "mov.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_insertf32x4 (easysimd__m512 a, easysimd__m128 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_4_(_mm512_insertf32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[imm8 & 3] = b.sve_f32;
    return a;
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);

    a_.m128[imm8 & 3] = b;

    return easysimd__m512_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_insertf32x4
  #define _mm512_insertf32x4(a, b, imm8) easysimd_mm512_insertf32x4(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_insertf32x4 (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m128 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,0,0))
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_4_(_mm512_mask_insertf32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[imm8 & 0x03] = b.sve_f32;
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], src.sve_f32[EASYSIMD_SV_INDEX_0]);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], src.sve_f32[EASYSIMD_SV_INDEX_1]);
    a.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], src.sve_f32[EASYSIMD_SV_INDEX_2]);
    a.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_insertf32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, a, b);
    return easysimd_mm512_mask_mov_ps(src, k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_insertf32x4
  #define _mm512_mask_insertf32x4(src, k, a, b, imm8) easysimd_mm512_mask_insertf32x4(src, k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_insertf32x4 (easysimd__mmask16 k, easysimd__m512 a, easysimd__m128 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,0,0))
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_4_(_mm512_maskz_insertf32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[imm8 & 0x03] = b.sve_f32;
    svfloat32_t svzero = svdup_n_f32(0);
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_insertf32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_ps(k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_insertf32x4
  #define _mm512_maskz_insertf32x4(k, a, b, imm8) easysimd_mm512_maskz_insertf32x4(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_insertf64x4 (easysimd__m512d a, easysimd__m256d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f64[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)] = b.sve_f64[EASYSIMD_SV_INDEX_0];
    a.sve_f64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)] = b.sve_f64[EASYSIMD_SV_INDEX_1];
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    uint64x2_t vmask = vceqq_s64(vdupq_n_s64(imm8 & 1), vdupq_n_s64(0));
    r.m128[0].neon_f64 = vbslq_f64(vmask, b.m128[0].neon_f64, a.m128[0].neon_f64);
    r.m128[1].neon_f64 = vbslq_f64(vmask, b.m128[1].neon_f64, a.m128[1].neon_f64);
    r.m128[2].neon_f64 = vbslq_f64(vmask, a.m128[2].neon_f64, b.m128[0].neon_f64);
    r.m128[3].neon_f64 = vbslq_f64(vmask, a.m128[3].neon_f64, b.m128[1].neon_f64);
    return r;
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    a_.m256d[imm8 & 1] = b;
    return easysimd__m512d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_insertf64x4(a, b, imm8) _mm512_insertf64x4(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_insertf64x4
  #define _mm512_insertf64x4(a, b, imm8) easysimd_mm512_insertf64x4(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_insertf64x4 (easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m256d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_2_(_mm512_mask_insertf64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256d[imm8 & 0x01] = b;
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], src.sve_f64[EASYSIMD_SV_INDEX_0]);
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], src.sve_f64[EASYSIMD_SV_INDEX_1]);
    a.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], src.sve_f64[EASYSIMD_SV_INDEX_2]);
    a.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_insertf64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, a, b);
    return easysimd_mm512_mask_mov_pd(src, k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_insertf64x4
  #define _mm512_mask_insertf64x4(src, k, a, b, imm8) easysimd_mm512_mask_insertf64x4(src, k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_insertf64x4 (easysimd__mmask8 k, easysimd__m512d a, easysimd__m256d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
  easysimd__m512d r;
    EASYSIMD_CONSTIFY_2_(_mm512_maskz_insertf64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256d[imm8 & 0x01] = b;
    svfloat64_t svzero = svdup_n_f64(0);
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
  easysimd__m512d r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_insertf64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_pd(k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_insertf64x4
  #define _mm512_maskz_insertf64x4(k, a, b, imm8) easysimd_mm512_maskz_insertf64x4(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_inserti32x4 (easysimd__m512i a, easysimd__m128i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  a.sve_i32[imm8 & 3] = b.sve_i32;
  return a;
#else
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  a_.m128i[imm8 & 3] = b;

  return easysimd__m512i_from_private(a_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_inserti32x4(a, b, imm8) _mm512_inserti32x4(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_inserti32x4
  #define _mm512_inserti32x4(a, b, imm8) easysimd_mm512_inserti32x4(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_inserti32x4 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m128i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,0,0))
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(_mm512_mask_inserti32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i32[imm8 & 0x03] = b.sve_i32;
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], src.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], src.sve_i32[EASYSIMD_SV_INDEX_1]);
    a.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], src.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_inserti32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, a, b);
    return easysimd_mm512_mask_mov_epi32(src, k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_inserti32x4
  #define _mm512_mask_inserti32x4(src, k, a, b, imm8) easysimd_mm512_mask_inserti32x4(src, k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_inserti32x4 (easysimd__mmask16 k, easysimd__m512i a, easysimd__m128i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,0,0))
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(_mm512_maskz_inserti32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i32[imm8 & 0x03] = b.sve_i32;
    svint32_t svzero = svdup_n_s32(0);
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_inserti32x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_epi32(k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_inserti32x4
  #define _mm512_maskz_inserti32x4(k, a, b, imm8) easysimd_mm512_maskz_inserti32x4(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_inserti64x4 (easysimd__m512i a, easysimd__m256i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i64[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)] = b.sve_i64[EASYSIMD_SV_INDEX_0];
    a.sve_i64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)] = b.sve_i64[EASYSIMD_SV_INDEX_1];
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint64x2_t vmask = vceqq_s64(vdupq_n_s64(imm8 & 1), vdupq_n_s64(0));
    r.m128i[0].neon_i64 = vbslq_s64(vmask, b.m128i[0].neon_i64, a.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vbslq_s64(vmask, b.m128i[1].neon_i64, a.m128i[1].neon_i64);
    r.m128i[2].neon_i64 = vbslq_s64(vmask, a.m128i[2].neon_i64, b.m128i[0].neon_i64);
    r.m128i[3].neon_i64 = vbslq_s64(vmask, a.m128i[3].neon_i64, b.m128i[1].neon_i64);
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    a_.m256i[imm8 & 1] = b;
    return easysimd__m512i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_inserti64x4(a, b, imm8) _mm512_inserti64x4(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_inserti64x4
  #define _mm512_inserti64x4(a, b, imm8) easysimd_mm512_inserti64x4(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_inserti64x4 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m256i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 2) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(_mm512_mask_inserti64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256i[imm8 & 0x01] = b;
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], src.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], src.sve_i64[EASYSIMD_SV_INDEX_1]);
    a.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], src.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_inserti64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, a, b);
    return easysimd_mm512_mask_mov_epi64(src, k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_inserti64x4
  #define _mm512_mask_inserti64x4(src, k, a, b, imm8) easysimd_mm512_mask_inserti64x4(src, k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_inserti64x4 (easysimd__mmask8 k, easysimd__m512i a, easysimd__m256i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 2) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(_mm512_maskz_inserti64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256i[imm8 & 0x01] = b;
    svint64_t svzero = svdup_n_s64(0);
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_inserti64x4, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_epi64(k, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_inserti64x4
  #define _mm512_maskz_inserti64x4(k, a, b, imm8) easysimd_mm512_maskz_inserti64x4(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_insertf32x8 (easysimd__m512 a, easysimd__m256 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)] = b.sve_f32[EASYSIMD_SV_INDEX_0];
    a.sve_f32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)] = b.sve_f32[EASYSIMD_SV_INDEX_1];
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    uint32x4_t vmask = vceqq_s32(vdupq_n_s32(imm8 & 1), vdupq_n_s32(0));
    r.m128[0].neon_f32 = vbslq_f32(vmask, b.m128[0].neon_f32, a.m128[0].neon_f32);
    r.m128[1].neon_f32 = vbslq_f32(vmask, b.m128[1].neon_f32, a.m128[1].neon_f32);
    r.m128[2].neon_f32 = vbslq_f32(vmask, a.m128[2].neon_f32, b.m128[0].neon_f32);
    r.m128[3].neon_f32 = vbslq_f32(vmask, a.m128[3].neon_f32, b.m128[1].neon_f32);
    return r;
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    a_.m256[imm8 & 1] = b;
    return easysimd__m512_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_insertf32x8(a, b, imm8) _mm512_insertf32x8(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_insertf32x8
  #define _mm512_insertf32x8(a, b, imm8) easysimd_mm512_insertf32x8(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_insertf32x8(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m256 b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_2_(_mm512_mask_insertf32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256[imm8 & 0x01] = b;
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], src.sve_f32[EASYSIMD_SV_INDEX_0]);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], src.sve_f32[EASYSIMD_SV_INDEX_1]);
    a.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], src.sve_f32[EASYSIMD_SV_INDEX_2]);
    a.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_insertf32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, a, b);
    return easysimd_mm512_mask_mov_ps(src, k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_insertf32x8
  #define _mm512_mask_insertf32x8(src, k, a, b, imm8) easysimd_mm512_mask_insertf32x8(src, k, a, b, imms8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_insertf32x8(easysimd__mmask16 k, easysimd__m512 a, easysimd__m256 b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_2_(_mm512_maskz_insertf32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256[imm8 & 0x01] = b;
    svfloat32_t svzero = svdup_n_f32(0);
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_insertf32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_ps ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_ps(k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_insertf32x8
  #define _mm512_maskz_insertf32x8(k, a, b, imm8) easysimd_mm512_maskz_insertf32x8(k, a, b, imms8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_insertf64x2 (easysimd__m512d a, easysimd__m128d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f64[imm8 & 3] = b.sve_f64;
    return a;
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

    a_.m128d[imm8 & 3] = b;

    return easysimd__m512d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_insertf64x2(a, b, imm8) _mm512_insertf64x2(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_insertf64x2
  #define _mm512_insertf64x2(a, b, imm8) easysimd_mm512_insertf64x2(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_insertf64x2(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m128d b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_4_(_mm512_mask_insertf64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f64[imm8 & 0x03] = b.sve_f64;
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], src.sve_f64[EASYSIMD_SV_INDEX_0]);
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], src.sve_f64[EASYSIMD_SV_INDEX_1]);
    a.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], src.sve_f64[EASYSIMD_SV_INDEX_2]);
    a.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_insertf64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, a, b);
    return easysimd_mm512_mask_mov_pd(src, k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_insertf64x2
  #define _mm512_mask_insertf64x2(src, k, a, b, imm8) easysimd_mm512_mask_insertf64x2(src, k, a, b, imms8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_insertf64x2(easysimd__mmask8 k, easysimd__m512d a, easysimd__m128d b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_4_(_mm512_maskz_insertf64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f64[imm8 & 0x03] = b.sve_f64;
    svfloat64_t svzero = svdup_n_f64(0);
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_insertf64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_pd ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_pd(k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_insertf64x2
  #define _mm512_maskz_insertf64x2(k, a, b, imm8) easysimd_mm512_maskz_insertf64x2(k, a, b, imms8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_inserti32x8 (easysimd__m512i a, easysimd__m256i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i32[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)] = b.sve_i32[EASYSIMD_SV_INDEX_0];
    a.sve_i32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)] = b.sve_i32[EASYSIMD_SV_INDEX_1];
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint32x4_t vmask = vceqq_s32(vdupq_n_s32(imm8 & 1), vdupq_n_s32(0));
    r.m128i[0].neon_i32 = vbslq_s32(vmask, b.m128i[0].neon_i32, a.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vbslq_s32(vmask, b.m128i[1].neon_i32, a.m128i[1].neon_i32);
    r.m128i[2].neon_i32 = vbslq_s32(vmask, a.m128i[2].neon_i32, b.m128i[0].neon_i32);
    r.m128i[3].neon_i32 = vbslq_s32(vmask, a.m128i[3].neon_i32, b.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    a_.m256i[imm8 & 1] = b;
    return easysimd__m512i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_inserti32x8(a, b, imm8) _mm512_inserti32x8(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_inserti32x8
  #define _mm512_inserti32x8(a, b, imm8) easysimd_mm512_inserti32x8(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_inserti32x8(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m256i b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(_mm512_mask_inserti32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_epi32 ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256i[imm8 & 0x01] = b;
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], src.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], src.sve_i32[EASYSIMD_SV_INDEX_1]);
    a.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], src.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_inserti32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_epi32 ()), imm8, a, b);
    return easysimd_mm512_mask_mov_epi32(src, k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_inserti32x8
  #define _mm512_mask_inserti32x8(src, k, a, b, imm8) easysimd_mm512_mask_inserti32x8(src, k, a, b, imms8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_inserti32x8(easysimd__mmask16 k, easysimd__m512i a, easysimd__m256i b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(_mm512_maskz_inserti32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_epi32 ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.m256i[imm8 & 0x01] = b;
    svint32_t svzero = svdup_n_s32(0);
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_2_(easysimd_mm512_inserti32x8, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_epi32 ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_epi32(k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_inserti32x8
  #define _mm512_maskz_inserti32x8(k, a, b, imm8) easysimd_mm512_maskz_inserti32x8(k, a, b, imms8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_inserti64x2 (easysimd__m512i a, easysimd__m128i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  a_.m128i[imm8 & 3] = b;

  return easysimd__m512i_from_private(a_);
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_inserti64x2(a, b, imm8) _mm512_inserti64x2(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_inserti64x2
  #define _mm512_inserti64x2(a, b, imm8) easysimd_mm512_inserti64x2(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_inserti64x2(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m128i b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(_mm512_mask_inserti64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, src, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i64[imm8 & 0x03] = b.sve_i64;
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], src.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], src.sve_i64[EASYSIMD_SV_INDEX_1]);
    a.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], src.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_inserti64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, a, b);
    return easysimd_mm512_mask_mov_epi64(src, k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_inserti64x2
  #define _mm512_mask_inserti64x2(src, k, a, b, imm8) easysimd_mm512_mask_inserti64x2(src, k, a, b, imms8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_inserti64x2(easysimd__mmask8 k, easysimd__m512i a, easysimd__m128i b, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(_mm512_maskz_inserti64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, k, a, b);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i64[imm8 & 0x03] = b.sve_i64;
    svint64_t svzero = svdup_n_s64(0);
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], svzero);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], svzero);
    a.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], svzero);
    a.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], svzero);
    return a;
  #else
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_4_(easysimd_mm512_inserti64x2, r, (HEDLEY_UNREACHABLE(), easysimd_mm512_setzero_si512 ()), imm8, a, b);
    return easysimd_mm512_maskz_mov_epi64(k, r);
  #endif
 }
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_inserti64x2
  #define _mm512_maskz_inserti64x2(k, a, b, imm8) easysimd_mm512_maskz_inserti64x2(k, a, b, imms8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_INSERT_H) */
