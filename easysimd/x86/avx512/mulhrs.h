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

#if !defined(EASYSIMD_X86_AVX512_MULHRS_H)
#define EASYSIMD_X86_AVX512_MULHRS_H

#include "types.h"
#include "mov.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif
#include "../avx2.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mulhrs_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mulhrs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    sveint32_t add = svdup_n_s32(0x00004000);

    sveint32_t
      r0 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r1 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r2 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r3 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r4 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r5 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r6 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r7 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])));
    
    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r0, add), 15));
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r1, add), 15));
    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r2, add), 15));
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r3, add), 15));
    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r4, add), 15));
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r5, add), 15));
    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r6, add), 15));
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), svasr_n_s32_z(pg, svadd_s32_z(pg, r7, add), 15));
    
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m256i[0] = easysimd_mm256_mulhrs_epi16(a.m256i[0], b.m256i[0]);
    r.m256i[1] = easysimd_mm256_mulhrs_epi16(a.m256i[1], b.m256i[1]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, (((HEDLEY_STATIC_CAST(int32_t, a_.i16[i]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[i])) + 0x4000) >> 15));
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mulhrs_epi16
  #define _mm512_mulhrs_epi16(a, b) easysimd_mm512_mulhrs_epi16(a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_MULHRS_H) */
