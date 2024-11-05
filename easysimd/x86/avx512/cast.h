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

#if !defined(EASYSIMD_X86_AVX512_CAST_H)
#define EASYSIMD_X86_AVX512_CAST_H

#include "types.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_castpd_ps (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castpd_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svreinterpret_f32_f64(a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svreinterpret_f32_f64(a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svreinterpret_f32_f64(a.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svreinterpret_f32_f64(a.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512 r;
    easysimd_memcpy(&r, &a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castpd_ps
  #define _mm512_castpd_ps(a) easysimd_mm512_castpd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_castpd_si512 (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castpd_si512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svreinterpret_s32_f64(a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svreinterpret_s32_f64(a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svreinterpret_s32_f64(a.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svreinterpret_s32_f64(a.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i r;
    easysimd_memcpy(&r, &a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castpd_si512
  #define _mm512_castpd_si512(a) easysimd_mm512_castpd_si512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_castps_pd (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castps_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svreinterpret_f64_f32(a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svreinterpret_f64_f32(a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svreinterpret_f64_f32(a.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svreinterpret_f64_f32(a.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d r;
    easysimd_memcpy(&r, &a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castps_pd
  #define _mm512_castps_pd(a) easysimd_mm512_castps_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_castps_si512 (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castps_si512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svreinterpret_s32_f32(a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svreinterpret_s32_f32(a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svreinterpret_s32_f32(a.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svreinterpret_s32_f32(a.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i r;
    easysimd_memcpy(&r, &a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castps_si512
  #define _mm512_castps_si512(a) easysimd_mm512_castps_si512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_castsi512_ps (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castsi512_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svreinterpret_f32_s32(a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svreinterpret_f32_s32(a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svreinterpret_f32_s32(a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svreinterpret_f32_s32(a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512 r;
    easysimd_memcpy(&r, &a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castsi512_ps
  #define _mm512_castsi512_ps(a) easysimd_mm512_castsi512_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_castsi512_pd (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castsi512_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svreinterpret_f64_s32(a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svreinterpret_f64_s32(a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svreinterpret_f64_s32(a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svreinterpret_f64_s32(a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d r;
    easysimd_memcpy(&r, &a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castsi512_pd
  #define _mm512_castsi512_pd(a) easysimd_mm512_castsi512_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_castpd128_pd512 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castpd128_pd512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0] = a;
    return r;
  #else
    easysimd__m512d_private r_;
    r_.m128d[0] = a;
    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castpd128_pd512
  #define _mm512_castpd128_pd512(a) easysimd_mm512_castpd128_pd512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_castpd256_pd512 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castpd256_pd512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64[EASYSIMD_SV_INDEX_0];
    r.sve_f64[EASYSIMD_SV_INDEX_1] = a.sve_f64[EASYSIMD_SV_INDEX_1];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m256d[0] = a;
    return r;
  #else
    easysimd__m512d_private r_;
    r_.m256d[0] = a;
    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castpd256_pd512
  #define _mm512_castpd256_pd512(a) easysimd_mm512_castpd256_pd512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm512_castpd512_pd128 (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castpd512_pd128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = a.sve_f64[EASYSIMD_SV_INDEX_0];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128d[0];
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    return a_.m128d[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castpd512_pd128
  #define _mm512_castpd512_pd128(a) easysimd_mm512_castpd512_pd128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm512_castpd512_pd256 (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castpd512_pd256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64[EASYSIMD_SV_INDEX_0];
    r.sve_f64[EASYSIMD_SV_INDEX_1] = a.sve_f64[EASYSIMD_SV_INDEX_1];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m256d[0];
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    return a_.m256d[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castpd512_pd256
  #define _mm512_castpd512_pd256(a) easysimd_mm512_castpd512_pd256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_castps128_ps512 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castps128_ps512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0] = a;
    return r;
  #else
    easysimd__m512_private r_;
    r_.m128[0] = a;
    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castps128_ps512
  #define _mm512_castps128_ps512(a) easysimd_mm512_castps128_ps512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_castps256_ps512 (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castps256_ps512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[EASYSIMD_SV_INDEX_0];
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[EASYSIMD_SV_INDEX_1];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m256[0] = a;
    return r;
  #else
    easysimd__m512_private r_;
    r_.m256[0] = a;
    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castps256_ps512
  #define _mm512_castps256_ps512(a) easysimd_mm512_castps256_ps512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm512_castps512_ps128 (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castps512_ps128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = a.sve_f32[EASYSIMD_SV_INDEX_0];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128[0];
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    return a_.m128[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castps512_ps128
  #define _mm512_castps512_ps128(a) easysimd_mm512_castps512_ps128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_castps512_ps256 (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castps512_ps256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[EASYSIMD_SV_INDEX_0];
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[EASYSIMD_SV_INDEX_1];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m256[0];
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    return a_.m256[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castps512_ps256
  #define _mm512_castps512_ps256(a) easysimd_mm512_castps512_ps256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_castsi128_si512 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castsi128_si512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0] = a;
    return r;
  #else
    easysimd__m512i_private r_;
    r_.m128i[0] = a;
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castsi128_si512
  #define _mm512_castsi128_si512(a) easysimd_mm512_castsi128_si512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_castsi256_si512 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castsi256_si512(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32[EASYSIMD_SV_INDEX_0];
    r.sve_i32[EASYSIMD_SV_INDEX_1] = a.sve_i32[EASYSIMD_SV_INDEX_1];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m256i[0] = a;
    return r;
  #else
    easysimd__m512i_private r_;
    r_.m256i[0] = a;
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castsi256_si512
  #define _mm512_castsi256_si512(a) easysimd_mm512_castsi256_si512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm512_castsi512_si128 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castsi512_si128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = a.sve_i32[EASYSIMD_SV_INDEX_0];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128i[0];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    return a_.m128i[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castsi512_si128
  #define _mm512_castsi512_si128(a) easysimd_mm512_castsi512_si128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_castsi512_si256 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_castsi512_si256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32[EASYSIMD_SV_INDEX_0];
    r.sve_i32[EASYSIMD_SV_INDEX_1] = a.sve_i32[EASYSIMD_SV_INDEX_1];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m256i[0];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    return a_.m256i[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_castsi512_si256
  #define _mm512_castsi512_si256(a) easysimd_mm512_castsi512_si256(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_CAST_H) */
