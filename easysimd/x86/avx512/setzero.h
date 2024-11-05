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
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 *   2020      Hidayat Khan <huk2209@gmail.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_X86_AVX512_SETZERO_H)
#define EASYSIMD_X86_AVX512_SETZERO_H

#include "types.h"
#include "cast.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_setzero_si512(void) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_setzero_si512();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdup_n_s32(0);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32(0);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svdup_n_s32(0);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svdup_n_s32(0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vdupq_n_s32(0);
    r.m128i[1].neon_i32 = r.m128i[0].neon_i32;
    r.m128i[2].neon_i32 = r.m128i[0].neon_i32;
    r.m128i[3].neon_i32 = r.m128i[0].neon_i32;
    return r;
  #else
    easysimd__m512i r;
    easysimd_memset(&r, 0, sizeof(r));
    return r;
  #endif
}
#define easysimd_mm512_setzero_epi32() easysimd_mm512_setzero_si512()
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_setzero_si512
  #define _mm512_setzero_si512() easysimd_mm512_setzero_si512()
  #undef _mm512_setzero_epi32
  #define _mm512_setzero_epi32() easysimd_mm512_setzero_si512()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_setzero_ps(void) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_setzero_ps();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdup_n_f32(0.0);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32(0.0);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svdup_n_f32(0.0);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svdup_n_f32(0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vdupq_n_f32(0.0);
    r.m128[1].neon_f32 = vdupq_n_f32(0.0);
    r.m128[2].neon_f32 = vdupq_n_f32(0.0);
    r.m128[3].neon_f32 = vdupq_n_f32(0.0);
    return r;
  #else
    return easysimd_mm512_castsi512_ps(easysimd_mm512_setzero_si512());
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_setzero_si512
  #define _mm512_setzero_si512() easysimd_mm512_setzero_si512()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_setzero_pd(void) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_setzero_pd();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdup_n_f64(0.0);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(0.0);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svdup_n_f64(0.0);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svdup_n_f64(0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vdupq_n_f64(0.0);
    r.m128d[1].neon_f64 = vdupq_n_f64(0.0);
    r.m128d[2].neon_f64 = vdupq_n_f64(0.0);
    r.m128d[3].neon_f64 = vdupq_n_f64(0.0);
    return r;
  #else
    return easysimd_mm512_castsi512_pd(easysimd_mm512_setzero_si512());
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_setzero_si512
  #define _mm512_setzero_si512() easysimd_mm512_setzero_si512()
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SETZERO_H) */
