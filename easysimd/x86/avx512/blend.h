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

#if !defined(EASYSIMD_X86_AVX512_BLEND_H)
#define EASYSIMD_X86_AVX512_BLEND_H

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
easysimd__m128i
easysimd_mm_mask_blend_epi8(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_blend_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), b.sve_i8, a.sve_i8);
    return r;
  #else
    return easysimd_mm_mask_mov_epi8(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_blend_epi8
  #define _mm_mask_blend_epi8(k, a, b) easysimd_mm_mask_blend_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_blend_epi16(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_blend_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), b.sve_i16, a.sve_i16);
    return r;
  #else
    return easysimd_mm_mask_mov_epi16(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_blend_epi16
  #define _mm_mask_blend_epi16(k, a, b) easysimd_mm_mask_blend_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_blend_epi32(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_blend_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_i32, a.sve_i32);
    return r;
  #else
    return easysimd_mm_mask_mov_epi32(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_blend_epi32
  #define _mm_mask_blend_epi32(k, a, b) easysimd_mm_mask_blend_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_blend_epi64(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_blend_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_i64, a.sve_i64);
    return r;
  #else
    return easysimd_mm_mask_mov_epi64(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_blend_epi64
  #define _mm_mask_blend_epi64(k, a, b) easysimd_mm_mask_blend_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_blend_ps(easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_blend_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_f32, a.sve_f32);
    return r;
  #else
    return easysimd_mm_mask_mov_ps(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_blend_ps
  #define _mm_mask_blend_ps(k, a, b) easysimd_mm_mask_blend_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_blend_pd(easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_blend_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_f64, a.sve_f64);
    return r;
  #else
    return easysimd_mm_mask_mov_pd(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_blend_pd
  #define _mm_mask_blend_pd(k, a, b) easysimd_mm_mask_blend_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_blend_epi8(easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_blend_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), b.sve_i8[EASYSIMD_SV_INDEX_0], a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), b.sve_i8[EASYSIMD_SV_INDEX_1], a.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi8(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_blend_epi8
  #define _mm256_mask_blend_epi8(k, a, b) easysimd_mm256_mask_blend_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_blend_epi16(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_blend_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), b.sve_i16[EASYSIMD_SV_INDEX_0], a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), b.sve_i16[EASYSIMD_SV_INDEX_1], a.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi16(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_blend_epi16
  #define _mm256_mask_blend_epi16(k, a, b) easysimd_mm256_mask_blend_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_blend_epi32(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_blend_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi32(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_blend_epi32
  #define _mm256_mask_blend_epi32(k, a, b) easysimd_mm256_mask_blend_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_blend_epi64(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_blend_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi64(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_blend_epi64
  #define _mm256_mask_blend_epi64(k, a, b) easysimd_mm256_mask_blend_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_blend_ps(easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_blend_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), b.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_ps(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_blend_ps
  #define _mm256_mask_blend_ps(k, a, b) easysimd_mm256_mask_blend_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_blend_pd(easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_blend_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), b.sve_f64[EASYSIMD_SV_INDEX_1], a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_pd(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_blend_pd
  #define _mm256_mask_blend_pd(k, a, b) easysimd_mm256_mask_blend_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_blend_epi8(easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_blend_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), b.sve_i8[EASYSIMD_SV_INDEX_0], a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), b.sve_i8[EASYSIMD_SV_INDEX_1], a.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), b.sve_i8[EASYSIMD_SV_INDEX_2], a.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), b.sve_i8[EASYSIMD_SV_INDEX_3], a.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_blend_epi8
  #define _mm512_mask_blend_epi8(k, a, b) easysimd_mm512_mask_blend_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_blend_epi16(easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_blend_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), b.sve_i16[EASYSIMD_SV_INDEX_0], a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), b.sve_i16[EASYSIMD_SV_INDEX_1], a.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), b.sve_i16[EASYSIMD_SV_INDEX_2], a.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), b.sve_i16[EASYSIMD_SV_INDEX_3], a.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_blend_epi16
  #define _mm512_mask_blend_epi16(k, a, b) easysimd_mm512_mask_blend_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_blend_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    uint32x4_t vect_imm = vdupq_n_u32(k);
    uint32x4_t flag[4];
    flag[0] = vtstq_u32(vect_imm, vect_mask);
    flag[1] = vtstq_u32(vshrq_n_u32(vect_imm, 4), vect_mask);
    flag[2] = vtstq_u32(vshrq_n_u32(vect_imm, 8), vect_mask);
    flag[3] = vtstq_u32(vshrq_n_u32(vect_imm, 12), vect_mask);
    r.m128i[0].neon_i32 = vbslq_s32(flag[0], b.m128i[0].neon_i32, a.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vbslq_s32(flag[1], b.m128i[1].neon_i32, a.m128i[1].neon_i32);
    r.m128i[2].neon_i32 = vbslq_s32(flag[2], b.m128i[2].neon_i32, a.m128i[2].neon_i32);
    r.m128i[3].neon_i32 = vbslq_s32(flag[3], b.m128i[3].neon_i32, a.m128i[3].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),b.sve_i32[EASYSIMD_SV_INDEX_2], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),b.sve_i32[EASYSIMD_SV_INDEX_3], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_blend_epi32(k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_blend_epi32
  #define _mm512_mask_blend_epi32(k, a, b) easysimd_mm512_mask_blend_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_blend_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_blend_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), b.sve_i64[EASYSIMD_SV_INDEX_2], a.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), b.sve_i64[EASYSIMD_SV_INDEX_3], a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi64(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_blend_epi64
  #define _mm512_mask_blend_epi64(k, a, b) easysimd_mm512_mask_blend_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_blend_ps(easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    uint32x4_t vect_imm = vdupq_n_u32(k);
    uint32x4_t flag[4];
    flag[0] = vtstq_u32(vect_imm, vect_mask);
    flag[1] = vtstq_u32(vshrq_n_u32(vect_imm, 4), vect_mask);
    flag[2] = vtstq_u32(vshrq_n_u32(vect_imm, 8), vect_mask);
    flag[3] = vtstq_u32(vshrq_n_u32(vect_imm, 12), vect_mask);
    r.m128[0].neon_f32 = vbslq_f32(flag[0], b.m128[0].neon_f32, a.m128[0].neon_f32);
    r.m128[1].neon_f32 = vbslq_f32(flag[1], b.m128[1].neon_f32, a.m128[1].neon_f32);
    r.m128[2].neon_f32 = vbslq_f32(flag[2], b.m128[2].neon_f32, a.m128[2].neon_f32);
    r.m128[3].neon_f32 = vbslq_f32(flag[3], b.m128[3].neon_f32, a.m128[3].neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),b.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),b.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),b.sve_f32[EASYSIMD_SV_INDEX_2], a.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),b.sve_f32[EASYSIMD_SV_INDEX_3], a.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_blend_ps(k, a, b);
  #else
    return easysimd_mm512_mask_mov_ps(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_blend_ps
  #define _mm512_mask_blend_ps(k, a, b) easysimd_mm512_mask_blend_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_blend_pd(easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    uint64_t g_mask_epi64[2] __attribute__((aligned(16))) = {0x01, 0x02};
    uint64x2_t vect_mask = vld1q_u64(g_mask_epi64);
    uint64x2_t vect_imm = vdupq_n_u64(k);
    uint64x2_t flag[4];
    flag[0] = vtstq_u64(vect_imm, vect_mask);
    flag[1] = vtstq_u64(vshrq_n_u64(vect_imm, 2), vect_mask);
    flag[2] = vtstq_u64(vshrq_n_u64(vect_imm, 4), vect_mask);
    flag[3] = vtstq_u64(vshrq_n_u64(vect_imm, 6), vect_mask);
    r.m128d[0].neon_f64 = vbslq_f64(flag[0], b.m128d[0].neon_f64, a.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vbslq_f64(flag[1], b.m128d[1].neon_f64, a.m128d[1].neon_f64);
    r.m128d[2].neon_f64 = vbslq_f64(flag[2], b.m128d[2].neon_f64, a.m128d[2].neon_f64);  
    r.m128d[3].neon_f64 = vbslq_f64(flag[3], b.m128d[3].neon_f64, a.m128d[3].neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),b.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),b.sve_f64[EASYSIMD_SV_INDEX_1], a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2),b.sve_f64[EASYSIMD_SV_INDEX_2], a.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3),b.sve_f64[EASYSIMD_SV_INDEX_3], a.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_blend_pd(k, a, b);
  #else
    return easysimd_mm512_mask_mov_pd(a, k, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_blend_pd
  #define _mm512_mask_blend_pd(k, a, b) easysimd_mm512_mask_blend_pd(k, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_BLEND_H) */
