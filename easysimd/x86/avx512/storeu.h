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

#if !defined(EASYSIMD_X86_AVX512_STOREU_H)
#define EASYSIMD_X86_AVX512_STOREU_H

#include "types.h"

#if defined(EASYSIMD_ARM_SVE_NATIVE)
#include <arm_sve.h>
#endif
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_storeu_ph (void * mem_addr, easysimd__m512 a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE_NOT_USE)
    svbool_t pg = svptrue_b16();
    svst1_f16(pg, HEDLEY_STATIC_CAST(float16_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f16[EASYSIMD_SV_INDEX_0]);
    svst1_f16(pg, HEDLEY_STATIC_CAST(float16_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f16[EASYSIMD_SV_INDEX_1]);
    svst1_f16(pg, HEDLEY_STATIC_CAST(float16_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f16[EASYSIMD_SV_INDEX_2]);
    svst1_f16(pg, HEDLEY_STATIC_CAST(float16_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f16[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE_NOT_USE)
    vst1q_f16(HEDLEY_STATIC_CAST(float16_t*, mem_addr) + 0, a.m128[0].neon_f16);
    vst1q_f16(HEDLEY_STATIC_CAST(float16_t*, mem_addr) + 4, a.m128[1].neon_f16);
    vst1q_f16(HEDLEY_STATIC_CAST(float16_t*, mem_addr) + 8, a.m128[2].neon_f16);
    vst1q_f16(HEDLEY_STATIC_CAST(float16_t*, mem_addr) + 12, a.m128[3].neon_f16);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE_NOT_USE)
    _mm512_storeu_ph(mem_addr, a);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_storeu_ph
  #define _mm512_storeu_ph(mem_addr, a) easysimd_mm512_storeu_ph(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_storeu_ps (void * mem_addr, easysimd__m512 a) {
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
    _mm512_storeu_ps(mem_addr, a);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_storeu_ps
  #define _mm512_storeu_ps(mem_addr, a) easysimd_mm512_storeu_ps(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_storeu_pd (void * mem_addr, easysimd__m512d a) {
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
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_storeu_pd(mem_addr, a);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_storeu_pd
  #define _mm512_storeu_pd(mem_addr, a) easysimd_mm512_storeu_pd(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_storeu_si512 (void * mem_addr, easysimd__m512i a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    #if 0
      vst1q_s64((int64_t*)mem_addr + 0, a.m128i[0].neon_i64);
      vst1q_s64((int64_t*)mem_addr + 2, a.m128i[1].neon_i64);
      vst1q_s64((int64_t*)mem_addr + 4, a.m128i[2].neon_i64);
      vst1q_s64((int64_t*)mem_addr + 6, a.m128i[3].neon_i64);
    #endif
    int32_t *addr_a = (int32_t *)(&a);
    __asm__ __volatile__ (
      "ldp    q0,  q1, [%[pa]],  #32     \n\t"
      "ldp    q2,  q3, [%[pa]],  #-32    \n\t"
      "stp    q0,  q1, [%[mem]], #32     \n\t"
      "stp    q2,  q3, [%[mem]], #-32    \n\t"
      :[mem] "+r"(mem_addr)
      :[pa] "r"(addr_a)
      :"q0", "q1", "q2", "q3", "memory"
    );
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_storeu_si512(HEDLEY_REINTERPRET_CAST(void*, mem_addr), a);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}

#define easysimd_mm512_storeu_epi8(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
#define easysimd_mm512_storeu_epi16(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
#define easysimd_mm512_storeu_epi32(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
#define easysimd_mm512_storeu_epi64(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_storeu_epi8
  #undef _mm512_storeu_epi16
  #undef _mm512_storeu_epi32
  #undef _mm512_storeu_epi64
  #undef _mm512_storeu_si512
  #define _mm512_storeu_si512(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
  #define _mm512_storeu_epi8(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
  #define _mm512_storeu_epi16(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
  #define _mm512_storeu_epi32(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
  #define _mm512_storeu_epi64(mem_addr, a) easysimd_mm512_storeu_si512(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_stream_si512 (void * mem_addr, easysimd__m512i a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_s64((int64_t*)mem_addr + 0, a.m128i[0].neon_i64);
    vst1q_s64((int64_t*)mem_addr + 2, a.m128i[1].neon_i64);
    vst1q_s64((int64_t*)mem_addr + 4, a.m128i[2].neon_i64);
    vst1q_s64((int64_t*)mem_addr + 6, a.m128i[3].neon_i64);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_stream_si512(HEDLEY_REINTERPRET_CAST(easysimd__m512i *, mem_addr), a);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}

#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_stream_si512
  #define _mm512_stream_si512(mem_addr, a) easysimd_mm512_stream_si512(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_stream_load_si512 (void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_stream_load_si512(HEDLEY_CONST_CAST(void*, mem_addr));   
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, (HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, (HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svld1_s32(pg, (HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svld1_s32(pg, (HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #else
    easysimd__m512i r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512), sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_stream_load_si512
  #define _mm512_stream_load_si512(mem_addr) easysimd_mm512_stream_load_si512(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_storeu_epi8(void * mem_addr, easysimd__mmask32 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    _mm256_mask_storeu_epi8(HEDLEY_REINTERPRET_CAST(void*, mem_addr), k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), HEDLEY_REINTERPRET_CAST(int8_t *, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 3), a.sve_i8[EASYSIMD_SV_INDEX_0]);
    svst1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), HEDLEY_REINTERPRET_CAST(int8_t *, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 3), a.sve_i8[EASYSIMD_SV_INDEX_1]);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a.i8) / sizeof(a.i8[0])) ; i++) {
      if ((k >> i) & 1)
        HEDLEY_STATIC_CAST(int8_t*, mem_addr)[i] = a.i8[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_storeu_epi8
  #define _mm256_mask_storeu_epi8(mem_addr, k, a) easysimd_mm256_mask_storeu_epi8(mem_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_storeu_epi16(void * mem_addr, easysimd__mmask16 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    _mm256_mask_storeu_epi16(HEDLEY_REINTERPRET_CAST(void*, mem_addr), k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), HEDLEY_REINTERPRET_CAST(int16_t *, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 4), a.sve_i16[EASYSIMD_SV_INDEX_0]);
    svst1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), HEDLEY_REINTERPRET_CAST(int16_t *, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 4), a.sve_i16[EASYSIMD_SV_INDEX_1]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint16_t mask_epi16[8] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint16x8_t mask_vec = vld1q_u16(mask_epi16);

    uint16x8_t k_vec = vdupq_n_u16(k & 0xFF);
    a.m128i[0].neon_u16 = vandq_u16(vtstq_u16(k_vec, mask_vec), a.m128i[0].neon_u16);
    vst1q_u16((uint16_t *)mem_addr, a.m128i[0].neon_u16);

    k_vec = vdupq_n_u16((k >> 8) & 0xFF);
    a.m128i[1].neon_u16 = vandq_u16(vtstq_u16(k_vec, mask_vec), a.m128i[1].neon_u16);
    vst1q_u16(((uint16_t *)mem_addr) + 8, a.m128i[1].neon_u16);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a.i16) / sizeof(a.i16[0])) ; i++) {
      if ((k >> i) & 1)
        HEDLEY_STATIC_CAST(int16_t*, mem_addr)[i] = a.i16[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_storeu_epi16
  #define _mm256_mask_storeu_epi16(mem_addr, k, a) easysimd_mm256_mask_storeu_epi16(mem_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_storeu_epi16(void * mem_addr, easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    _mm512_mask_storeu_epi16(HEDLEY_REINTERPRET_CAST(void*, mem_addr), k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), HEDLEY_REINTERPRET_CAST(int16_t *, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 4), a.sve_i16[EASYSIMD_SV_INDEX_0]);
    svst1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), HEDLEY_REINTERPRET_CAST(int16_t *, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 4), a.sve_i16[EASYSIMD_SV_INDEX_1]);
    svst1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), HEDLEY_REINTERPRET_CAST(int16_t *, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 4), a.sve_i16[EASYSIMD_SV_INDEX_2]);
    svst1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), HEDLEY_REINTERPRET_CAST(int16_t *, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 4), a.sve_i16[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint16_t mask_epi16[8] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint16x8_t mask_vec = vld1q_u16(mask_epi16);
    uint16x8_t k_vec = vdupq_n_u16(k & 0xFF);
    a.m128i[0].neon_u16 = vandq_u16(vtstq_u16(k_vec, mask_vec), a.m128i[0].neon_u16);
    vst1q_u16((uint16_t *)mem_addr, a.m128i[0].neon_u16);

    k_vec = vdupq_n_u16((k >> 8) & 0xFF);
    a.m128i[1].neon_u16 = vandq_u16(vtstq_u16(k_vec, mask_vec), a.m128i[1].neon_u16);
    vst1q_u16(((uint16_t *)mem_addr) + 8, a.m128i[1].neon_u16);

    k_vec = vdupq_n_u16((k >> 16) & 0xFF);
    a.m128i[2].neon_u16 = vandq_u16(vtstq_u16(k_vec, mask_vec), a.m128i[2].neon_u16);
    vst1q_u16(((uint16_t *)mem_addr) + 16, a.m128i[2].neon_u16);

    k_vec = vdupq_n_u16((k >> 24) & 0xFF);
    a.m128i[3].neon_u16 = vandq_u16(vtstq_u16(k_vec, mask_vec), a.m128i[3].neon_u16);
    vst1q_u16(((uint16_t *)mem_addr) + 24, a.m128i[3].neon_u16);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a.i16) / sizeof(a.i16[0])) ; i++) {
      if ((k >> i) & 1)
        HEDLEY_STATIC_CAST(int16_t*, mem_addr)[i] = a.i16[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_storeu_epi16
  #define _mm512_mask_storeu_epi16(mem_addr, k, a) easysimd_mm512_mask_storeu_epi16(mem_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_storeu_epi32(void * mem_addr, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    _mm512_mask_storeu_epi32(HEDLEY_REINTERPRET_CAST(void*, mem_addr), k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), HEDLEY_REINTERPRET_CAST(int32_t *, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    svst1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), HEDLEY_REINTERPRET_CAST(int32_t *, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    svst1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), HEDLEY_REINTERPRET_CAST(int32_t *, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    svst1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), HEDLEY_REINTERPRET_CAST(int32_t *, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_3]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    uint32_t mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t mask_vec = vld1q_u32(mask_epi32);
    uint32x4_t k_vec = vdupq_n_u32(k & 0xFFFF);
    a.m128i[0].neon_u32 = vandq_u32(vtstq_u32(k_vec, mask_vec), a.m128i[0].neon_u32);
    vst1q_u32((uint32_t *)mem_addr, a.m128i[0].neon_u32);

    k_vec = vdupq_n_u32((k >> 4) & 0xFFFF);
    a.m128i[1].neon_u32 = vandq_u32(vtstq_u32(k_vec, mask_vec), a.m128i[1].neon_u32);
    vst1q_u32(((uint32_t *)mem_addr) + 4, a.m128i[1].neon_u32);

    k_vec = vdupq_n_u32((k >> 8) & 0xFFFF);
    a.m128i[2].neon_u32 = vandq_u32(vtstq_u32(k_vec, mask_vec), a.m128i[2].neon_u32);
    vst1q_u32(((uint32_t *)mem_addr) + 8, a.m128i[2].neon_u32);

    k_vec = vdupq_n_u32((k >> 12) & 0xFFFF);
    a.m128i[3].neon_u32 = vandq_u32(vtstq_u32(k_vec, mask_vec), a.m128i[3].neon_u32);
    vst1q_u32(((uint32_t *)mem_addr) + 12, a.m128i[3].neon_u32);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a.i32) / sizeof(a.i32[0])) ; i++) {
      if ((k >> i) & 1)
        HEDLEY_STATIC_CAST(int32_t*, mem_addr)[i] = a.i32[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_storeu_epi32
  #define _mm512_mask_storeu_epi32(mem_addr, k, a) easysimd_mm512_mask_storeu_epi32(mem_addr, k, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_STOREU_H) */
