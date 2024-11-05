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
 *   2020-2021 Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_X86_AVX512_CMPLE_H)
#define EASYSIMD_X86_AVX512_CMPLE_H

#include "types.h"
#include "mov.h"
#include "mov_mask.h"
#include "movm.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_movm_epi8(_mm_cmple_epi8_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u8 = vcleq_s8(a_.neon_i8, b_.neon_i8);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_cmple_epi8_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_cmple_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(pg, a.sve_i8, b.sve_i8), EASYSIMD_SV_INDEX_0);
    return k;
  #else
    return easysimd_mm_movepi8_mask(easysimd_x_mm_cmple_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmple_epi8_mask
  #define _mm_cmple_epi8_mask(a, b) easysimd_mm_cmple_epi8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_mask_cmple_epi8_mask(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_cmple_epi8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmple_s8(pg, a.sve_i8, b.sve_i8), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epi8_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VBW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epi8_mask
  #define _mm_mask_cmple_epi8_mask(k, a, b) easysimd_mm_mask_cmple_epi8_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_cmpnle_epi8_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return ~_mm_cmple_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(svptrue_b8(), a.sve_i8, b.sve_i8), EASYSIMD_SV_INDEX_0);
    return ~(k);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
      r |= (a_.i8[i] > b_.i8[i]) ? (UINT16_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpnle_epi8_mask
  #define _mm_cmpnle_epi8_mask(a, b) easysimd_mm_cmpnle_epi8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm256_movm_epi8(_mm256_cmple_epi8_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi8(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_cmple_epi8_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_cmple_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 k = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi8_mask(easysimd_x_mm256_cmple_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VBW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmple_epi8_mask
  #define _mm256_cmple_epi8_mask(a, b) easysimd_mm256_cmple_epi8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_mask_cmple_epi8_mask(easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)&& defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_cmple_epi8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epi8_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epi8_mask
  #define _mm256_mask_cmple_epi8_mask(k, a, b) easysimd_mm256_mask_cmple_epi8_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm512_movm_epi8(_mm512_cmple_epi8_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u8 = vcleq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_u8 = vcleq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    r.m128i[2].neon_u8 = vcleq_s8(a.m128i[2].neon_i8, b.m128i[2].neon_i8);
    r.m128i[3].neon_u8 = vcleq_s8(a.m128i[3].neon_i8, b.m128i[3].neon_i8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svdup_n_s8_z(svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), ~INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svdup_n_s8_z(svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), ~INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svdup_n_s8_z(svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), ~INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svdup_n_s8_z(svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), ~INT8_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi8(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epi8(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_cmple_epi8_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint8_t g_mask_epi8[16] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t vect_mask = vld1q_u8(g_mask_epi8);
    easysimd__m512i r;
    r.m128i[0].neon_u8 = vandq_u8(vcleq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8), vect_mask);
    r.m128i[1].neon_u8 = vandq_u8(vcleq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8), vect_mask);
    r.m128i[2].neon_u8 = vandq_u8(vcleq_s8(a.m128i[2].neon_i8, b.m128i[2].neon_i8), vect_mask);
    r.m128i[3].neon_u8 = vandq_u8(vcleq_s8(a.m128i[3].neon_i8, b.m128i[3].neon_i8), vect_mask);
    uint64_t r0 = vaddv_u8(vget_low_u8(r.m128i[0].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[0].neon_u8)) << 8);
    uint64_t r1 = vaddv_u8(vget_low_u8(r.m128i[1].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[1].neon_u8)) << 8);
    uint64_t r2 = vaddv_u8(vget_low_u8(r.m128i[2].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[2].neon_u8)) << 8);
    uint64_t r3 = vaddv_u8(vget_low_u8(r.m128i[3].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[3].neon_u8)) << 8);
    easysimd__mmask64 mask = r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
    return mask;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 k = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(k, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmple_epi8_mask(a, b);
  #else
    return easysimd_mm512_movepi8_mask(easysimd_x_mm512_cmple_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epi8_mask
  #define _mm512_cmple_epi8_mask(a, b) easysimd_mm512_cmple_epi8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_mask_cmple_epi8_mask(easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmple_epi8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(rk, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(rk, svcmple_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epi8_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epi8_mask
  #define _mm512_mask_cmple_epi8_mask(k, a, b) easysimd_mm512_mask_cmple_epi8_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_movm_epi8(_mm_cmple_epu8_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u8 = vcleq_u8(a_.neon_u8, b_.neon_u8);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u8), a_.u8 <= b_.u8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        r_.u8[i] = (a_.u8[i] <= b_.u8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_cmple_epu8_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_cmple_epu8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(pg, a.sve_u8, b.sve_u8), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    return easysimd_mm_movepi8_mask(easysimd_x_mm_cmple_epu8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmple_epu8_mask
  #define _mm_cmple_epu8_mask(a, b) easysimd_mm_cmple_epu8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_mask_cmple_epu8_mask(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_cmple_epu8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(pg, a.sve_u8, b.sve_u8), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epu8_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epu8_mask
  #define _mm_mask_cmple_epu8_mask(k, a, b) easysimd_mm_mask_cmple_epu8_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm256_movm_epi8(_mm256_cmple_epu8_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu8(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u8), a_.u8 <= b_.u8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        r_.u8[i] = (a_.u8[i] <= b_.u8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_cmple_epu8_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_cmple_epu8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 k = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(k, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi8_mask(easysimd_x_mm256_cmple_epu8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmple_epu8_mask
  #define _mm256_cmple_epu8_mask(a, b) easysimd_mm256_cmple_epu8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_mask_cmple_epu8_mask(easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_cmple_epu8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epu8_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epu8_mask
  #define _mm256_mask_cmple_epu8_mask(k, a, b) easysimd_mm256_mask_cmple_epu8_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epu8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm512_movm_epi8(_mm512_cmple_epu8_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u8 = vcleq_u8(a.m128i[0].neon_u8, b.m128i[0].neon_u8);
    r.m128i[1].neon_u8 = vcleq_u8(a.m128i[1].neon_u8, b.m128i[1].neon_u8);
    r.m128i[2].neon_u8 = vcleq_u8(a.m128i[2].neon_u8, b.m128i[2].neon_u8);
    r.m128i[3].neon_u8 = vcleq_u8(a.m128i[3].neon_u8, b.m128i[3].neon_u8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svdup_n_u8_z(svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), ~INT8_C(0));
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svdup_n_u8_z(svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), ~INT8_C(0));
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svdup_n_u8_z(svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), ~INT8_C(0));
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svdup_n_u8_z(svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), ~INT8_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu8(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epu8(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u8), a_.u8 <= b_.u8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        r_.u8[i] = (a_.u8[i] <= b_.u8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_cmple_epu8_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmple_epu8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 k = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(k, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(k, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(k, svcmple_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #else
    return easysimd_mm512_movepi8_mask(easysimd_x_mm512_cmple_epu8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epu8_mask
  #define _mm512_cmple_epu8_mask(a, b) easysimd_mm512_cmple_epu8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_mask_cmple_epu8_mask(easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmple_epu8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 rk = 0;
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(rk, svcmple_u8(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epu8_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epu8_mask
  #define _mm512_mask_cmple_epu8_mask(k, a, b) easysimd_mm512_mask_cmple_epu8_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_movm_epi16(_mm_cmple_epi16_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vcleq_s16(a_.neon_i16, b_.neon_i16);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmple_epi16_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_cmple_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(k, svcmple_s16(pg, a.sve_i16, b.sve_i16), EASYSIMD_SV_INDEX_0);
    return k;
  #else
    return easysimd_mm_movepi16_mask(easysimd_x_mm_cmple_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmple_epi16_mask
  #define _mm_cmple_epi16_mask(a, b) easysimd_mm_cmple_epi16_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmple_epi16_mask(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_cmple_epi16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmple_s16(pg, a.sve_i16, b.sve_i16), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epi16_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epi16_mask
  #define _mm_mask_cmple_epi16_mask(k, a, b) easysimd_mm_mask_cmple_epi16_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm256_movm_epi16(_mm256_cmple_epi16_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi16(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_cmple_epi16_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_cmple_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(k, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(k, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi16_mask(easysimd_x_mm256_cmple_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmple_epi16_mask
  #define _mm256_cmple_epi16_mask(a, b) easysimd_mm256_cmple_epi16_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_mask_cmple_epi16_mask(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_cmple_epi16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epi16_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epi16_mask
  #define _mm256_mask_cmple_epi16_mask(k, a, b) easysimd_mm256_mask_cmple_epi16_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm512_movm_epi16(_mm512_cmple_epi16_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u16 = vcleq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_u16 = vcleq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    r.m128i[2].neon_u16 = vcleq_s16(a.m128i[2].neon_i16, b.m128i[2].neon_i16);
    r.m128i[3].neon_u16 = vcleq_s16(a.m128i[3].neon_i16, b.m128i[3].neon_i16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svdup_n_s16_z(svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), ~INT16_C(0));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svdup_n_s16_z(svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), ~INT16_C(0));
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svdup_n_s16_z(svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), ~INT16_C(0));
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svdup_n_s16_z(svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), ~INT16_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi16(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epi16(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_cmple_epi16_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmple_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 k = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(k, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(k, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(k, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(k, svcmple_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #else
    return easysimd_mm512_movepi16_mask(easysimd_x_mm512_cmple_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epi16_mask
  #define _mm512_cmple_epi16_mask(a, b) easysimd_mm512_cmple_epi16_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_mask_cmple_epi16_mask(easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmple_epi16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    EASYSIMD_B16_TO_MASK(rk, svcmple_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmple_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(rk, svcmple_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(rk, svcmple_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epi16_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epi16_mask
  #define _mm512_mask_cmple_epi16_mask(k, a, b) easysimd_mm512_mask_cmple_epi16_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_movm_epi16(_mm_cmple_epu16_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vcleq_u16(a_.neon_u16, b_.neon_u16);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u16), a_.u16 <= b_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
        r_.u16[i] = (a_.u16[i] <= b_.u16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmple_epu16_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_cmple_epu16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(pg, a.sve_u16, b.sve_u16), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    return easysimd_mm_movepi16_mask(easysimd_x_mm_cmple_epu16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmple_epu16_mask
  #define _mm_cmple_epu16_mask(a, b) easysimd_mm_cmple_epu16_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmple_epu16_mask(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_cmple_epu16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(pg, a.sve_u16, b.sve_u16), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epu16_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epu16_mask
  #define _mm_mask_cmple_epu16_mask(k, a, b) easysimd_mm_mask_cmple_epu16_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epu16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm256_movm_epi16(_mm256_cmple_epu16_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu16(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u16), a_.u16 <= b_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
        r_.u16[i] = (a_.u16[i] <= b_.u16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_cmple_epu16_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_cmple_epu16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(k, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(k, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi16_mask(easysimd_x_mm256_cmple_epu16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmple_epu16_mask
  #define _mm256_cmple_epu16_mask(a, b) easysimd_mm256_cmple_epu16_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_mask_cmple_epu16_mask(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_cmple_epu16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epu16_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epu16_mask
  #define _mm256_mask_cmple_epu16_mask(k, a, b) easysimd_mm256_mask_cmple_epu16_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epu16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return easysimd_mm512_movm_epi16(_mm512_cmple_epu16_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u16 = vcleq_u16(a.m128i[0].neon_u16, b.m128i[0].neon_u16);
    r.m128i[1].neon_u16 = vcleq_u16(a.m128i[1].neon_u16, b.m128i[1].neon_u16);
    r.m128i[2].neon_u16 = vcleq_u16(a.m128i[2].neon_u16, b.m128i[2].neon_u16);
    r.m128i[3].neon_u16 = vcleq_u16(a.m128i[3].neon_u16, b.m128i[3].neon_u16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svdup_n_u16_z(svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), ~INT16_C(0));
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svdup_n_u16_z(svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), ~INT16_C(0));
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svdup_n_u16_z(svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), ~INT16_C(0));
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svdup_n_u16_z(svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), ~INT16_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu16(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epu16(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u16), a_.u16 <= b_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
        r_.u16[i] = (a_.u16[i] <= b_.u16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_cmple_epu16_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmple_epu16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 k = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(k, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(k, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(k, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(k, svcmple_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #else
    return easysimd_mm512_movepi16_mask(easysimd_x_mm512_cmple_epu16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epu16_mask
  #define _mm512_cmple_epu16_mask(a, b) easysimd_mm512_cmple_epu16_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_mask_cmple_epu16_mask(easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmple_epu16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(svptrue_b16(), a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(svptrue_b16(), a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(svptrue_b16(), a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(rk, svcmple_u16(svptrue_b16(), a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epu16_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epu16_mask
  #define _mm512_mask_cmple_epu16_mask(k, a, b) easysimd_mm512_mask_cmple_epu16_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm_movm_epi32(_mm_cmple_epi32_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vcleq_s32(a_.neon_i32, b_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmple_epi32_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmple_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(k, svcmple_s32(pg, a.sve_i32, b.sve_i32), EASYSIMD_SV_INDEX_0);
    return k;
  #else
    return easysimd_mm_movepi32_mask(easysimd_x_mm_cmple_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmple_epi32_mask
  #define _mm_cmple_epi32_mask(a, b) easysimd_mm_cmple_epi32_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmple_epi32_mask(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmple_epi32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmple_s32(pg, a.sve_i32, b.sve_i32), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epi32_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epi32_mask
  #define _mm_mask_cmple_epi32_mask(k, a, b) easysimd_mm_mask_cmple_epi32_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm256_movm_epi32(_mm256_cmple_epi32_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi32(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmple_epi32_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmple_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(k, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi32_mask(easysimd_x_mm256_cmple_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmple_epi32_mask
  #define _mm256_cmple_epi32_mask(a, b) easysimd_mm256_cmple_epi32_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmple_epi32_mask(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmple_epi32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epi32_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epi32_mask
  #define _mm256_mask_cmple_epi32_mask(k, a, b) easysimd_mm256_mask_cmple_epi32_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return easysimd_mm512_movm_epi32(_mm512_cmple_epi32_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u32 = vcleq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_u32 = vcleq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    r.m128i[2].neon_u32 = vcleq_s32(a.m128i[2].neon_i32, b.m128i[2].neon_i32);
    r.m128i[3].neon_u32 = vcleq_s32(a.m128i[3].neon_i32, b.m128i[3].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdup_n_s32_z(svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), ~INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32_z(svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), ~INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svdup_n_s32_z(svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), ~INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svdup_n_s32_z(svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), ~INT32_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi32(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epi32(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_cmple_epi32_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(k, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(k, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(k, svcmple_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    easysimd__m512i r;
    r.m128i[0].neon_u32 = vandq_u32(vcleq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32), vect_mask);
    r.m128i[1].neon_u32 = vandq_u32(vcleq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32), vect_mask);
    r.m128i[2].neon_u32 = vandq_u32(vcleq_s32(a.m128i[2].neon_i32, b.m128i[2].neon_i32), vect_mask);
    r.m128i[3].neon_u32 = vandq_u32(vcleq_s32(a.m128i[3].neon_i32, b.m128i[3].neon_i32), vect_mask);
    uint32_t r0 = vaddvq_u32(r.m128i[0].neon_u32);
    uint32_t r1 = vaddvq_u32(r.m128i[1].neon_u32);
    uint32_t r2 = vaddvq_u32(r.m128i[2].neon_u32);
    uint32_t r3 = vaddvq_u32(r.m128i[3].neon_u32);
    easysimd__mmask16 mask = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return mask;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmple_epi32_mask(a, b);
  #else
    return easysimd_mm512_movepi32_mask(easysimd_x_mm512_cmple_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epi32_mask
  #define _mm512_cmple_epi32_mask(a, b) easysimd_mm512_cmple_epi32_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_mask_cmple_epi32_mask(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmple_epi32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    EASYSIMD_B32_TO_MASK(rk, svcmple_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmple_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(rk, svcmple_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(rk, svcmple_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epi32_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epi32_mask
  #define _mm512_mask_cmple_epi32_mask(k, a, b) easysimd_mm512_mask_cmple_epi32_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epu32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm_movm_epi32(_mm_cmple_epu32_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vcleq_u32(a_.neon_u32, b_.neon_u32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), a_.u32 <= b_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
        r_.u32[i] = (a_.u32[i] <= b_.u32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmple_epu32_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmple_epu32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(pg, a.sve_u32, b.sve_u32), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    return easysimd_mm_movepi32_mask(easysimd_x_mm_cmple_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmple_epu32_mask
  #define _mm_cmple_epu32_mask(a, b) easysimd_mm_cmple_epu32_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmple_epu32_mask(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmple_epu32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(pg, a.sve_u32, b.sve_u32), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epu32_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epu32_mask
  #define _mm_mask_cmple_epu32_mask(k, a, b) easysimd_mm_mask_cmple_epu32_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epu32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm256_movm_epi32(_mm256_cmple_epu32_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu32(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), a_.u32 <= b_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
        r_.u32[i] = (a_.u32[i] <= b_.u32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmple_epu32_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmple_epu32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(k, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi32_mask(easysimd_x_mm256_cmple_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epu32_mask
  #define _mm512_cmple_epu32_mask(a, b) easysimd_mm512_cmple_epu32_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmple_epu32_mask(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmple_epu32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epu32_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epu32_mask
  #define _mm256_mask_cmple_epu32_mask(k, a, b) easysimd_mm256_mask_cmple_epu32_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epu32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return easysimd_mm512_movm_epi32(_mm512_cmple_epu32_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u32 = vcleq_u32(a.m128i[0].neon_u32, b.m128i[0].neon_u32);
    r.m128i[1].neon_u32 = vcleq_u32(a.m128i[1].neon_u32, b.m128i[1].neon_u32);
    r.m128i[2].neon_u32 = vcleq_u32(a.m128i[2].neon_u32, b.m128i[2].neon_u32);
    r.m128i[3].neon_u32 = vcleq_u32(a.m128i[3].neon_u32, b.m128i[3].neon_u32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svdup_n_u32_z(svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), ~INT32_C(0));
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svdup_n_u32_z(svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), ~INT32_C(0));
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svdup_n_u32_z(svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), ~INT32_C(0));
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svdup_n_u32_z(svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), ~INT32_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu32(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epu32(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), a_.u32 <= b_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
        r_.u32[i] = (a_.u32[i] <= b_.u32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_cmple_epu32_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmple_epu32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(k, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(k, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(k, svcmple_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #else
    return easysimd_mm512_movepi32_mask(easysimd_x_mm512_cmple_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epu32_mask
  #define _mm512_cmple_epu32_mask(a, b) easysimd_mm512_cmple_epu32_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_mask_cmple_epu32_mask(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmple_epu32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(rk, svcmple_u32(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epu32_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epu32_mask
  #define _mm512_mask_cmple_epu32_mask(k, a, b) easysimd_mm512_mask_cmple_epu32_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm_movm_epi64(_mm_cmple_epi64_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_u64 = vcleq_s64(a_.neon_i64, b_.neon_i64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
        r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmple_epi64_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmple_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(k, svcmple_s64(pg, a.sve_i64, b.sve_i64), EASYSIMD_SV_INDEX_0);
    return k;
  #else
    return easysimd_mm_movepi64_mask(easysimd_x_mm_cmple_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmple_epi64_mask
  #define _mm_cmple_epi64_mask(a, b) easysimd_mm_cmple_epi64_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmple_epi64_mask(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmple_epi64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmple_s64(pg, a.sve_i64, b.sve_i64), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epi64_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epi64_mask
  #define _mm_mask_cmple_epi64_mask(k, a, b) easysimd_mm_mask_cmple_epi64_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm256_movm_epi64(_mm256_cmple_epi64_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi64(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
        r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmple_epi64_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmple_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(k, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi64_mask(easysimd_x_mm256_cmple_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmple_epi64_mask
  #define _mm256_cmple_epi64_mask(a, b) easysimd_mm256_cmple_epi64_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmple_epi64_mask(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmple_epi64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epi64_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epi64_mask
  #define _mm256_mask_cmple_epi64_mask(k, a, b) easysimd_mm256_mask_cmple_epi64_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epi64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return easysimd_mm512_movm_epi64(_mm512_cmple_epi64_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u64 = vcleq_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    r.m128i[1].neon_u64 = vcleq_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    r.m128i[2].neon_u64 = vcleq_s64(a.m128i[2].neon_i64, b.m128i[2].neon_i64);
    r.m128i[3].neon_u64 = vcleq_s64(a.m128i[3].neon_i64, b.m128i[3].neon_i64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdup_n_s64_z(svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), ~INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdup_n_s64_z(svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), ~INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svdup_n_s64_z(svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), ~INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svdup_n_s64_z(svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), ~INT64_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epi64(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epi64(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
        r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_cmple_epi64_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmple_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(k, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(k, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(k, svcmple_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #else
    return easysimd_mm512_movepi64_mask(easysimd_x_mm512_cmple_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epi64_mask
  #define _mm512_cmple_epi64_mask(a, b) easysimd_mm512_cmple_epi64_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_mask_cmple_epi64_mask(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmple_epi64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    EASYSIMD_B64_TO_MASK(rk, svcmple_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmple_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(rk, svcmple_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(rk, svcmple_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epi64_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epi64_mask
  #define _mm512_mask_cmple_epi64_mask(k, a, b) easysimd_mm512_mask_cmple_epi64_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cmple_epu64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm_movm_epi64(_mm_cmple_epu64_mask(a, b));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_u64 = vcleq_u64(a_.neon_u64, b_.neon_u64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u64), a_.u64 <= b_.u64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        r_.u64[i] = (a_.u64[i] <= b_.u64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmple_epu64_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmple_epu64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(pg, a.sve_u64, b.sve_u64), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    return easysimd_mm_movepi64_mask(easysimd_x_mm_cmple_epu64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epu64_mask
  #define _mm512_cmple_epu64_mask(a, b) easysimd_mm512_cmple_epu64_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmple_epu64_mask(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmple_epu64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(pg, a.sve_u64, b.sve_u64), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm_cmple_epu64_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmple_epu64_mask
  #define _mm_mask_cmple_epu64_mask(k, a, b) easysimd_mm_mask_cmple_epu64_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_cmple_epu64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return easysimd_mm256_movm_epi64(_mm256_cmple_epu64_mask(a, b));
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu64(a_.m128i[i], b_.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u64), a_.u64 <= b_.u64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        r_.u64[i] = (a_.u64[i] <= b_.u64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmple_epu64_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmple_epu64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(k, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    return easysimd_mm256_movepi64_mask(easysimd_x_mm256_cmple_epu64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epu64_mask
  #define _mm512_cmple_epu64_mask(a, b) easysimd_mm512_cmple_epu64_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmple_epu64_mask(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmple_epu64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm256_cmple_epu64_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmple_epu64_mask
  #define _mm256_mask_cmple_epu64_mask(k, a, b) easysimd_mm256_mask_cmple_epu64_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_cmple_epu64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return easysimd_mm512_movm_epi64(_mm512_cmple_epu64_mask(a, b));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u64 = vcleq_u64(a.m128i[0].neon_u64, b.m128i[0].neon_u64);
    r.m128i[1].neon_u64 = vcleq_u64(a.m128i[1].neon_u64, b.m128i[1].neon_u64);
    r.m128i[2].neon_u64 = vcleq_u64(a.m128i[2].neon_u64, b.m128i[2].neon_u64);
    r.m128i[3].neon_u64 = vcleq_u64(a.m128i[3].neon_u64, b.m128i[3].neon_u64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svdup_n_u64_z(svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), ~INT64_C(0));
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svdup_n_u64_z(svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), ~INT64_C(0));
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svdup_n_u64_z(svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), ~INT64_C(0));
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svdup_n_u64_z(svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), ~INT64_C(0));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_x_mm_cmple_epu64(a_.m128i[i], b_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_x_mm256_cmple_epu64(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u64), a_.u64 <= b_.u64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        r_.u64[i] = (a_.u64[i] <= b_.u64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_cmple_epu64_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmple_epu64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(k, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(k, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(k, svcmple_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #else
    return easysimd_mm512_movepi64_mask(easysimd_x_mm512_cmple_epu64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmple_epu64_mask
  #define _mm512_cmple_epu64_mask(a, b) easysimd_mm512_cmple_epu64_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_mask_cmple_epu64_mask(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmple_epu64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(rk, svcmple_u64(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k;
    return rk;
  #else
    return k & easysimd_mm512_cmple_epu64_mask(a, b);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmple_epu64_mask
  #define _mm512_mask_cmple_epu64_mask(k, a, b) easysimd_mm512_mask_cmple_epu64_mask((k), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmpnle_epi32_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return ~_mm256_cmple_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    EASYSIMD_B32_TO_MASK(k, svcmpgt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmpgt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= (a_.i32[i] > b_.i32[i]) ? (UINT8_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpnle_epi32_mask
  #define _mm256_cmpnle_epi32_mask(a, b) easysimd_mm256_cmpnle_epi32_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_cmpnle_epi8_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return ~_mm512_cmple_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint8_t g_mask_epi8[16] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t vect_mask = vld1q_u8(g_mask_epi8);
    easysimd__m512i r;
    r.m128i[0].neon_u8 = vandq_u8(vcleq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8), vect_mask);
    r.m128i[1].neon_u8 = vandq_u8(vcleq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8), vect_mask);
    r.m128i[2].neon_u8 = vandq_u8(vcleq_s8(a.m128i[2].neon_i8, b.m128i[2].neon_i8), vect_mask);
    r.m128i[3].neon_u8 = vandq_u8(vcleq_s8(a.m128i[3].neon_i8, b.m128i[3].neon_i8), vect_mask);
    uint64_t r0 = vaddv_u8(vget_low_u8(r.m128i[0].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[0].neon_u8)) << 8);
    uint64_t r1 = vaddv_u8(vget_low_u8(r.m128i[1].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[1].neon_u8)) << 8);
    uint64_t r2 = vaddv_u8(vget_low_u8(r.m128i[2].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[2].neon_u8)) << 8);
    uint64_t r3 = vaddv_u8(vget_low_u8(r.m128i[3].neon_u8)) | (vaddv_u8(vget_high_u8(r.m128i[3].neon_u8)) << 8);
    easysimd__mmask64 mask = r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
    return ~mask;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 k = 0;
    EASYSIMD_B8_TO_MASK(k, svcmpgt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmpgt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(k, svcmpgt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(k, svcmpgt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask64 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
      r |= (a_.i8[i] > b_.i8[i]) ? (UINT64_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpnle_epi8_mask
  #define _mm512_cmpnle_epi8_mask(a, b) easysimd_mm512_cmpnle_epi8_mask((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_cmpnle_epi32_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return ~_mm512_cmple_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    EASYSIMD_B32_TO_MASK(k, svcmpgt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmpgt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(k, svcmpgt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(k, svcmpgt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return k;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    easysimd__m512i r;
    r.m128i[0].neon_u32 = vandq_u32(vcleq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32), vect_mask);
    r.m128i[1].neon_u32 = vandq_u32(vcleq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32), vect_mask);
    r.m128i[2].neon_u32 = vandq_u32(vcleq_s32(a.m128i[2].neon_i32, b.m128i[2].neon_i32), vect_mask);
    r.m128i[3].neon_u32 = vandq_u32(vcleq_s32(a.m128i[3].neon_i32, b.m128i[3].neon_i32), vect_mask);
    uint32_t r0 = vaddvq_u32(r.m128i[0].neon_u32);
    uint32_t r1 = vaddvq_u32(r.m128i[1].neon_u32);
    uint32_t r2 = vaddvq_u32(r.m128i[2].neon_u32);
    uint32_t r3 = vaddvq_u32(r.m128i[3].neon_u32);
    easysimd__mmask16 mask = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return ~mask;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask16 r = 0;
    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= (a_.i32[i] > b_.i32[i]) ? (UINT16_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpnle_epi32_mask
  #define _mm512_cmpnle_epi32_mask(a, b) easysimd_mm512_cmpnle_epi32_mask((a), (b))
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_CMPLE_H) */
