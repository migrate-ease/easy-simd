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

#if !defined(EASYSIMD_X86_AVX512_SLLV_H)
#define EASYSIMD_X86_AVX512_SLLV_H

#include "types.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sllv_epi16 (easysimd__m512i a, easysimd__m512i b) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b16();
  r.sve_u16[EASYSIMD_SV_INDEX_0] = svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
  r.sve_u16[EASYSIMD_SV_INDEX_1] = svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
  r.sve_u16[EASYSIMD_SV_INDEX_2] = svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]);
  r.sve_u16[EASYSIMD_SV_INDEX_3] = svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]);
  return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u16 = vandq_u16(vshlq_u16(a.m128i[0].neon_u16, b.m128i[0].neon_i16), vcltq_u16(b.m128i[0].neon_u16, vdupq_n_u16(16)));
    r.m128i[1].neon_u16 = vandq_u16(vshlq_u16(a.m128i[1].neon_u16, b.m128i[1].neon_i16), vcltq_u16(b.m128i[1].neon_u16, vdupq_n_u16(16)));
    r.m128i[2].neon_u16 = vandq_u16(vshlq_u16(a.m128i[2].neon_u16, b.m128i[2].neon_i16), vcltq_u16(b.m128i[2].neon_u16, vdupq_n_u16(16)));
    r.m128i[3].neon_u16 = vandq_u16(vshlq_u16(a.m128i[3].neon_u16, b.m128i[3].neon_i16), vcltq_u16(b.m128i[3].neon_u16, vdupq_n_u16(16)));
    return r;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    r_;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.u16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u16), (b_.u16 < 16)) & (a_.u16 << b_.u16);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (b_.u16[i] < 16) ? HEDLEY_STATIC_CAST(uint16_t, (a_.u16[i] << b_.u16[i])) : 0;
    }
  #endif

  return easysimd__m512i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  #define easysimd_mm512_sllv_epi16(a, b) _mm512_sllv_epi16(a, b)
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sllv_epi16
  #define _mm512_sllv_epi16(a, b) easysimd_mm512_sllv_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sllv_epi32 (easysimd__m512i a, easysimd__m512i b) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b32();
  r.sve_u32[EASYSIMD_SV_INDEX_0] = svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
  r.sve_u32[EASYSIMD_SV_INDEX_1] = svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
  r.sve_u32[EASYSIMD_SV_INDEX_2] = svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]);
  r.sve_u32[EASYSIMD_SV_INDEX_3] = svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]);
  return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u32 = vandq_u32(vshlq_u32(a.m128i[0].neon_u32, b.m128i[0].neon_i32), vcltq_u32(b.m128i[0].neon_u32, vdupq_n_u32(32)));
    r.m128i[1].neon_u32 = vandq_u32(vshlq_u32(a.m128i[1].neon_u32, b.m128i[1].neon_i32), vcltq_u32(b.m128i[1].neon_u32, vdupq_n_u32(32)));
    r.m128i[2].neon_u32 = vandq_u32(vshlq_u32(a.m128i[2].neon_u32, b.m128i[2].neon_i32), vcltq_u32(b.m128i[2].neon_u32, vdupq_n_u32(32)));
    r.m128i[3].neon_u32 = vandq_u32(vshlq_u32(a.m128i[3].neon_u32, b.m128i[3].neon_i32), vcltq_u32(b.m128i[3].neon_u32, vdupq_n_u32(32)));
    return r;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    r_;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), (b_.u32 < 32)) & (a_.u32 << b_.u32);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (b_.u32[i] < 32) ? HEDLEY_STATIC_CAST(uint32_t, (a_.u32[i] << b_.u32[i])) : 0;
    }
  #endif

  return easysimd__m512i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_sllv_epi32(a, b) _mm512_sllv_epi32(a, b)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sllv_epi32
  #define _mm512_sllv_epi32(a, b) easysimd_mm512_sllv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sllv_epi64 (easysimd__m512i a, easysimd__m512i b) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b64();
  r.sve_u64[EASYSIMD_SV_INDEX_0] = svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
  r.sve_u64[EASYSIMD_SV_INDEX_1] = svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
  r.sve_u64[EASYSIMD_SV_INDEX_2] = svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]);
  r.sve_u64[EASYSIMD_SV_INDEX_3] = svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]);
  return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u64 = vandq_u64(vshlq_u64(a.m128i[0].neon_u64, b.m128i[0].neon_i64), vcltq_u64(b.m128i[0].neon_u64, vdupq_n_u64(64)));
    r.m128i[1].neon_u64 = vandq_u64(vshlq_u64(a.m128i[1].neon_u64, b.m128i[1].neon_i64), vcltq_u64(b.m128i[1].neon_u64, vdupq_n_u64(64)));
    r.m128i[2].neon_u64 = vandq_u64(vshlq_u64(a.m128i[2].neon_u64, b.m128i[2].neon_i64), vcltq_u64(b.m128i[2].neon_u64, vdupq_n_u64(64)));
    r.m128i[3].neon_u64 = vandq_u64(vshlq_u64(a.m128i[3].neon_u64, b.m128i[3].neon_i64), vcltq_u64(b.m128i[3].neon_u64, vdupq_n_u64(64)));
    return r;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    r_;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u64), (b_.u64 < 64)) & (a_.u64 << b_.u64);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (b_.u64[i] < 64) ? HEDLEY_STATIC_CAST(uint64_t, (a_.u64[i] << b_.u64[i])) : 0;
    }
  #endif

  return easysimd__m512i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_sllv_epi64(a, b) _mm512_sllv_epi64(a, b)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sllv_epi64
  #define _mm512_sllv_epi64(a, b) easysimd_mm512_sllv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_sllv_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i count) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm512_mask_sllv_epi16(src, k, a, count);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b16();
  r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], count.sve_u16[EASYSIMD_SV_INDEX_0]), src.sve_u16[EASYSIMD_SV_INDEX_0]);
  r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], count.sve_u16[EASYSIMD_SV_INDEX_1]), src.sve_u16[EASYSIMD_SV_INDEX_1]);
  r.sve_u16[EASYSIMD_SV_INDEX_2] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], count.sve_u16[EASYSIMD_SV_INDEX_2]), src.sve_u16[EASYSIMD_SV_INDEX_2]);
  r.sve_u16[EASYSIMD_SV_INDEX_3] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], count.sve_u16[EASYSIMD_SV_INDEX_3]), src.sve_u16[EASYSIMD_SV_INDEX_3]);
  return r;
#else

  return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_sllv_epi16(a, count));

#endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sllv_epi16
  #define _mm512_mask_sllv_epi16(src, k, a, b) easysimd_mm512_mask_sllv_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_sllv_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i count) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_sllv_epi32(src, k, a, count);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b32();
  r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], count.sve_u32[EASYSIMD_SV_INDEX_0]), src.sve_u32[EASYSIMD_SV_INDEX_0]);
  r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], count.sve_u32[EASYSIMD_SV_INDEX_1]), src.sve_u32[EASYSIMD_SV_INDEX_1]);
  r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], count.sve_u32[EASYSIMD_SV_INDEX_2]), src.sve_u32[EASYSIMD_SV_INDEX_2]);
  r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], count.sve_u32[EASYSIMD_SV_INDEX_3]), src.sve_u32[EASYSIMD_SV_INDEX_3]);
  return r;
#else

  return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_sllv_epi32(a, count));

#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sllv_epi32
  #define _mm512_mask_sllv_epi32(src, k, a, b) easysimd_mm512_mask_sllv_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_sllv_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i count) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_sllv_epi64(src, k, a, count);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svbool_t pg = svptrue_b64();
  r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], count.sve_u64[EASYSIMD_SV_INDEX_0]), src.sve_u64[EASYSIMD_SV_INDEX_0]);
  r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], count.sve_u64[EASYSIMD_SV_INDEX_1]), src.sve_u64[EASYSIMD_SV_INDEX_1]);
  r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], count.sve_u64[EASYSIMD_SV_INDEX_2]), src.sve_u64[EASYSIMD_SV_INDEX_2]);
  r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svlsl_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], count.sve_u64[EASYSIMD_SV_INDEX_3]), src.sve_u64[EASYSIMD_SV_INDEX_3]);
  return r;
#else

  return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_sllv_epi64(a, count));

#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sllv_epi64
  #define _mm512_mask_sllv_epi64(src, k, a, b) easysimd_mm512_mask_sllv_epi64(src, k, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SLLV_H) */
