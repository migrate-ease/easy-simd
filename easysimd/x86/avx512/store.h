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
 */

#if !defined(EASYSIMD_X86_AVX512_STORE_H)
#define EASYSIMD_X86_AVX512_STORE_H

#include "types.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store_epi32 (void const *mem_addr, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
  return _mm_store_epi32(mem_addr, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1_s32(svptrue_b32(), (int32_t *)mem_addr, a.sve_i32);
  return;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_s32((int32_t *)mem_addr, a.neon_i32);
  return;
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
  for(size_t i = 0; i < sizeof(a_.i32) / sizeof(a_.i32[0]); i++){
      *((int32_t *)mem_addr + i) = a_.i32[i];
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_store_epi32
  #define _mm_store_epi32(mem_addr, a) easysimd_mm_store_epi32(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store_epi64 (void const *mem_addr, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm_store_epi64((void*)mem_addr, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1_s64(svptrue_b64(), (int64_t *)mem_addr, a.sve_i64);
  return ;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_s64((int64_t *)mem_addr, a.neon_i64);
  return ;
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
  for(size_t i = 0; i < sizeof(a_.i64) / sizeof(a_.i64[0]); i++){
      *((int64_t *)mem_addr + i) = a_.i64[i];
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_store_epi64
  #define _mm_store_epi64(mem_addr, a) easysimd_mm_store_epi64(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_store_ps (void * mem_addr, easysimd__m512 a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svst1_f32(pg, HEDLEY_STATIC_CAST(float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    svst1_f32(pg, HEDLEY_STATIC_CAST(float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    svst1_f32(pg, HEDLEY_STATIC_CAST(float32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f32[EASYSIMD_SV_INDEX_2]);
    svst1_f32(pg, HEDLEY_STATIC_CAST(float32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f32[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_f32(HEDLEY_STATIC_CAST(float32_t*, mem_addr) + 0, a.m128[0].neon_f32);
    vst1q_f32(HEDLEY_STATIC_CAST(float32_t*, mem_addr) + 4, a.m128[1].neon_f32);
    vst1q_f32(HEDLEY_STATIC_CAST(float32_t*, mem_addr) + 8, a.m128[2].neon_f32);
    vst1q_f32(HEDLEY_STATIC_CAST(float32_t*, mem_addr) + 12, a.m128[3].neon_f32);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_store_ps(mem_addr, a);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512), &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_store_ps
  #define _mm512_store_ps(mem_addr, a) easysimd_mm512_store_ps(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_store_pd (void * mem_addr, easysimd__m512d a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    svst1_f64(pg, HEDLEY_STATIC_CAST(float64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    svst1_f64(pg, HEDLEY_STATIC_CAST(float64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_f64[EASYSIMD_SV_INDEX_1]);
    svst1_f64(pg, HEDLEY_STATIC_CAST(float64_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_f64[EASYSIMD_SV_INDEX_2]);
    svst1_f64(pg, HEDLEY_STATIC_CAST(float64_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_f64[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_f64(HEDLEY_STATIC_CAST(float64_t*, mem_addr) + 0, a.m128d[0].neon_f64);
    vst1q_f64(HEDLEY_STATIC_CAST(float64_t*, mem_addr) + 2, a.m128d[1].neon_f64);
    vst1q_f64(HEDLEY_STATIC_CAST(float64_t*, mem_addr) + 4, a.m128d[2].neon_f64);
    vst1q_f64(HEDLEY_STATIC_CAST(float64_t*, mem_addr) + 6, a.m128d[3].neon_f64);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
    _mm512_store_pd(mem_addr, a);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512d), &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_store_pd
  #define _mm512_store_pd(mem_addr, a) easysimd_mm512_store_pd(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_store_si512 (void * mem_addr, easysimd__m512i a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_s32(HEDLEY_STATIC_CAST(int32_t*, mem_addr) + 0, a.m128i[0].neon_i32);
    vst1q_s32(HEDLEY_STATIC_CAST(int32_t*, mem_addr) + 4, a.m128i[1].neon_i32);
    vst1q_s32(HEDLEY_STATIC_CAST(int32_t*, mem_addr) + 8, a.m128i[2].neon_i32);
    vst1q_s32(HEDLEY_STATIC_CAST(int32_t*, mem_addr) + 12, a.m128i[3].neon_i32);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_store_si512(HEDLEY_REINTERPRET_CAST(void*, mem_addr), a);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512i), &a, sizeof(a));
  #endif
}
#define easysimd_mm512_store_epi8(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
#define easysimd_mm512_store_epi16(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
#define easysimd_mm512_store_epi32(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
#define easysimd_mm512_store_epi64(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_store_epi8
  #undef _mm512_store_epi16
  #undef _mm512_store_epi32
  #undef _mm512_store_epi64
  #undef _mm512_store_si512
  #define _mm512_store_si512(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
  #define _mm512_store_epi8(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
  #define _mm512_store_epi16(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
  #define _mm512_store_epi32(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
  #define _mm512_store_epi64(mem_addr, a) easysimd_mm512_store_si512(mem_addr, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_STORE_H) */
