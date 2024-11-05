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

#if !defined(EASYSIMD_X86_AVX512_SRLV_H)
#define EASYSIMD_X86_AVX512_SRLV_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srlv_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_srlv_epi16(a, b);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u16), (b_.u16 < 16)) & (a_.u16 >> b_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = (b_.u16[i] < 16) ? (a_.u16[i] >> b_.u16[i]) : 0;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_srlv_epi16
  #define _mm_srlv_epi16(a, b) easysimd_mm_srlv_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_srlv_epi16(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_srlv_epi16(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_srlv_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_srlv_epi16
  #define _mm_mask_srlv_epi16(src, k, a, b) easysimd_mm_mask_srlv_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_srlv_epi16(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_srlv_epi16(k, a, b);
  #else
    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_srlv_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_srlv_epi16
  #define _mm_maskz_srlv_epi16(k, a, b) easysimd_mm_maskz_srlv_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_srlv_epi32(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_srlv_epi32(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_srlv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_srlv_epi32
  #define _mm_mask_srlv_epi32(src, k, a, b) easysimd_mm_mask_srlv_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_srlv_epi32(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_srlv_epi32(k, a, b);
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_srlv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_srlv_epi32
  #define _mm_maskz_srlv_epi32(k, a, b) easysimd_mm_maskz_srlv_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_srlv_epi64(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_srlv_epi64(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_srlv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_srlv_epi64
  #define _mm_mask_srlv_epi64(src, k, a, b) easysimd_mm_mask_srlv_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_srlv_epi64(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_srlv_epi64(k, a, b);
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_srlv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_srlv_epi64
  #define _mm_maskz_srlv_epi64(k, a, b) easysimd_mm_maskz_srlv_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srlv_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_srlv_epi16(a, b);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b),
      r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_srlv_epi16(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u16), (b_.u16 < 16)) & (a_.u16 >> b_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = (b_.u16[i] < 16) ? (a_.u16[i] >> b_.u16[i]) : 0;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srlv_epi16
  #define _mm256_srlv_epi16(a, b) easysimd_mm256_srlv_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_srlv_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_srlv_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;

    r.m128i[0].neon_u16 = vandq_u16(vshlq_u16(a.m128i[0].neon_u16, vmulq_s16(b.m128i[0].neon_i16, vdupq_n_s16(-1))),
                                    vcltq_u16(b.m128i[0].neon_u16, vdupq_n_u16(16)));
    r.m128i[1].neon_u16 = vandq_u16(vshlq_u16(a.m128i[1].neon_u16, vmulq_s16(b.m128i[1].neon_i16, vdupq_n_s16(-1))),
                                    vcltq_u16(b.m128i[1].neon_u16, vdupq_n_u16(16)));
    r.m128i[2].neon_u16 = vandq_u16(vshlq_u16(a.m128i[2].neon_u16, vmulq_s16(b.m128i[2].neon_i16, vdupq_n_s16(-1))),
                                    vcltq_u16(b.m128i[2].neon_u16, vdupq_n_u16(16)));
    r.m128i[3].neon_u16 = vandq_u16(vshlq_u16(a.m128i[3].neon_u16, vmulq_s16(b.m128i[3].neon_i16, vdupq_n_s16(-1))),
                                    vcltq_u16(b.m128i[3].neon_u16, vdupq_n_u16(16)));
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b),
      r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_srlv_epi16(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u16), (b_.u16 < 16)) & (a_.u16 >> b_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = (b_.u16[i] < 16) ? (a_.u16[i] >> b_.u16[i]) : 0;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_srlv_epi16
  #define _mm512_srlv_epi16(a, b) easysimd_mm512_srlv_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_srlv_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_srlv_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;

    r.m128i[0].neon_u32 = vandq_u32(vshlq_u32(a.m128i[0].neon_u32, vmulq_s32(b.m128i[0].neon_i32, vdupq_n_s32(-1))),
                                    vcltq_u32(b.m128i[0].neon_u32, vdupq_n_u32(32)));
    r.m128i[1].neon_u32 = vandq_u32(vshlq_u32(a.m128i[1].neon_u32, vmulq_s32(b.m128i[1].neon_i32, vdupq_n_s32(-1))),
                                    vcltq_u32(b.m128i[1].neon_u32, vdupq_n_u32(32)));
    r.m128i[2].neon_u32 = vandq_u32(vshlq_u32(a.m128i[2].neon_u32, vmulq_s32(b.m128i[2].neon_i32, vdupq_n_s32(-1))),
                                    vcltq_u32(b.m128i[2].neon_u32, vdupq_n_u32(32)));
    r.m128i[3].neon_u32 = vandq_u32(vshlq_u32(a.m128i[3].neon_u32, vmulq_s32(b.m128i[3].neon_i32, vdupq_n_s32(-1))),
                                    vcltq_u32(b.m128i[3].neon_u32, vdupq_n_u32(32)));
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b),
      r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_srlv_epi32(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), (b_.u32 < 32)) & (a_.u32 >> b_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = (b_.u32[i] < 32) ? (a_.u32[i] >> b_.u32[i]) : 0;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_srlv_epi32
  #define _mm512_srlv_epi32(a, b) easysimd_mm512_srlv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_srlv_epi64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_srlv_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    const int32_t arr[4] = {-1, 1, -1, 1};
    int32x4_t mask = vld1q_s32(arr);
    r.m128i[0].neon_u64 = vandq_u64(a.m128i[0].neon_u64, vcltq_u64(b.m128i[0].neon_u64, vdupq_n_u64(64)));
    r.m128i[1].neon_u64 = vandq_u64(a.m128i[1].neon_u64, vcltq_u64(b.m128i[1].neon_u64, vdupq_n_u64(64)));
    r.m128i[2].neon_u64 = vandq_u64(a.m128i[2].neon_u64, vcltq_u64(b.m128i[2].neon_u64, vdupq_n_u64(64)));
    r.m128i[3].neon_u64 = vandq_u64(a.m128i[3].neon_u64, vcltq_u64(b.m128i[3].neon_u64, vdupq_n_u64(64)));
    b.m128i[0].neon_i32 = vmulq_s32(b.m128i[0].neon_i32, mask);
    b.m128i[1].neon_i32 = vmulq_s32(b.m128i[1].neon_i32, mask);
    b.m128i[2].neon_i32 = vmulq_s32(b.m128i[2].neon_i32, mask);
    b.m128i[3].neon_i32 = vmulq_s32(b.m128i[3].neon_i32, mask);
    r.m128i[0].neon_u64 = vshlq_u64(r.m128i[0].neon_u64, b.m128i[0].neon_i64);
    r.m128i[1].neon_u64 = vshlq_u64(r.m128i[1].neon_u64, b.m128i[1].neon_i64);
    r.m128i[2].neon_u64 = vshlq_u64(r.m128i[2].neon_u64, b.m128i[2].neon_i64);
    r.m128i[3].neon_u64 = vshlq_u64(r.m128i[3].neon_u64, b.m128i[3].neon_i64);

    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b),
      r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_srlv_epi64(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u64), (b_.u64 < 64)) & (a_.u64 >> b_.u64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = (b_.u64[i] < 64) ? (a_.u64[i] >> b_.u64[i]) : 0;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_srlv_epi64
  #define _mm512_srlv_epi64(a, b) easysimd_mm512_srlv_epi64(a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SRLV_H) */
