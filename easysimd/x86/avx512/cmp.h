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
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 */

#if !defined(EASYSIMD_X86_AVX512_CMP_H)
#define EASYSIMD_X86_AVX512_CMP_H

#include "types.h"
#include "mov.h"
#include "mov_mask.h"
#include "setzero.h"
#include "setone.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cmp_ps (easysimd__m512 a, easysimd__m512 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128d[0].neon_f32 = vreinterpretq_f32_u32(neonfunlistcmpps[imm8].neoncmpfun_ps(a.m128[0], b.m128[0]));
    r.m128d[1].neon_f32 = vreinterpretq_f32_u32(neonfunlistcmpps[imm8].neoncmpfun_ps(a.m128[1], b.m128[1]));
    r.m128d[2].neon_f32 = vreinterpretq_f32_u32(neonfunlistcmpps[imm8].neoncmpfun_ps(a.m128[2], b.m128[2]));
    r.m128d[3].neon_f32 = vreinterpretq_f32_u32(neonfunlistcmpps[imm8].neoncmpfun_ps(a.m128[3], b.m128[3]));
    return r;
  #else
  easysimd__m512_private
    r_,
    a_ = easysimd__m512_to_private(a),
    b_ = easysimd__m512_to_private(b);
  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 == b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] == b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] < b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] <= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != a_.f32) | (b_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] != a_.f32[i]) || (b_.f32[i] != b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] != b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 == a_.f32) & (b_.f32 == b_.f32) & (a_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i]) & (a_.f32[i] != b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 < b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] < b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 <= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] <= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ((a_.f32 == a_.f32) & (b_.f32 == b_.f32)));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != a_.f32) | (b_.f32 != b_.f32) | (a_.f32 == b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] != a_.f32[i]) | (b_.f32[i] != b_.f32[i]) | (a_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 >= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] >= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 > b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] > b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      r_ = easysimd__m512_to_private(easysimd_mm512_setzero_ps());
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] >= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] > b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      r_ = easysimd__m512_to_private(easysimd_x_mm512_setone_ps());
      break;

    default:
      HEDLEY_UNREACHABLE();
  }
  return easysimd__m512_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_ps
  #define _mm512_cmp_ps(a, b, imm8) easysimd_mm512_cmp_ps((a), (b), (imm8))
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cmp_pd (easysimd__m512d a, easysimd__m512d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = (float64x2_t)neonfunlistcmppd[imm8].neoncmpfun_pd(a.m128d[0], b.m128d[0]);
    r.m128d[1].neon_f64 = (float64x2_t)neonfunlistcmppd[imm8].neoncmpfun_pd(a.m128d[1], b.m128d[1]);
    r.m128d[2].neon_f64 = (float64x2_t)neonfunlistcmppd[imm8].neoncmpfun_pd(a.m128d[2], b.m128d[2]);
    r.m128d[3].neon_f64 = (float64x2_t)neonfunlistcmppd[imm8].neoncmpfun_pd(a.m128d[3], b.m128d[3]);
    return r;
  #else
  easysimd__m512d_private
    r_,
    a_ = easysimd__m512d_to_private(a),
    b_ = easysimd__m512d_to_private(b);
  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] == b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] < b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] <= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != a_.f64) | (b_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] != a_.f64[i]) || (b_.f64[i] != b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] != b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 == a_.f64) & (b_.f64 == b_.f64) & (a_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i]) & (a_.f64[i] != b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 < b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] < b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 <= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] <= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ((a_.f64 == a_.f64) & (b_.f64 == b_.f64)));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != a_.f64) | (b_.f64 != b_.f64) | (a_.f64 == b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] != a_.f64[i]) | (b_.f64[i] != b_.f64[i]) | (a_.f64[i] == b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 >= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] >= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 > b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] > b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      r_ = easysimd__m512d_to_private(easysimd_mm512_setzero_pd());
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] >= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] > b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      r_ = easysimd__m512d_to_private(easysimd_x_mm512_setone_pd());
      break;

    default:
      HEDLEY_UNREACHABLE();
  }
  return easysimd__m512d_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_pd
  #define _mm512_cmp_pd(a, b, imm8) easysimd_mm512_cmp_pd((a), (b), (imm8))
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_cmp_ps_mask (easysimd__m512 a, easysimd__m512 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return easysimd_mm512_movepi32_mask(easysimd_mm512_castps_si512(r));
  #else
  easysimd__m512_private
    r_,
    a_ = easysimd__m512_to_private(a),
    b_ = easysimd__m512_to_private(b);
  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 == b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] == b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] < b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] <= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != a_.f32) | (b_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] != a_.f32[i]) || (b_.f32[i] != b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] != b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 == a_.f32) & (b_.f32 == b_.f32) & (a_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i]) & (a_.f32[i] != b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 < b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] < b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 <= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] <= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ((a_.f32 == a_.f32) & (b_.f32 == b_.f32)));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != a_.f32) | (b_.f32 != b_.f32) | (a_.f32 == b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] != a_.f32[i]) | (b_.f32[i] != b_.f32[i]) | (a_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 >= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] >= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 > b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] > b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      r_ = easysimd__m512_to_private(easysimd_mm512_setzero_ps());
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] >= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] > b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      r_ = easysimd__m512_to_private(easysimd_x_mm512_setone_ps());
      break;

    default:
      HEDLEY_UNREACHABLE();
  }
  return easysimd_mm512_movepi32_mask(easysimd_mm512_castps_si512(easysimd__m512_from_private(r_)));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_cmp_ps_mask(a, b, imm8) _mm512_cmp_ps_mask((a), (b), (imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm512_cmp_ps_mask(a, b, imm8) ({ \
    easysimd__m512 dst = easysimd_mm512_cmp_ps(a, b, imm8); \
    easysimd__mmask16 r = 0; \
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08}; \
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32); \
    uint32_t r0 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.m128[0].neon_f32), vect_mask)); \
    uint32_t r1 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.m128[1].neon_f32), vect_mask)); \
    uint32_t r2 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.m128[2].neon_f32), vect_mask)); \
    uint32_t r3 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.m128[3].neon_f32), vect_mask)); \
    r = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12); \
    r; \
  })
#elif defined(EASYSIMD_STATEMENT_EXPR_) && EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
  #define easysimd_mm512_cmp_ps_mask(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512_private \
      easysimd_mm512_cmp_ps_mask_r_, \
      easysimd_mm512_cmp_ps_mask_a_ = easysimd__m512_to_private((a)), \
      easysimd_mm512_cmp_ps_mask_b_ = easysimd__m512_to_private((b)); \
    \
    for (size_t i = 0 ; i < (sizeof(easysimd_mm512_cmp_ps_mask_r_.m128) / sizeof(easysimd_mm512_cmp_ps_mask_r_.m128[0])) ; i++) { \
      easysimd_mm512_cmp_ps_mask_r_.m128[i] = easysimd_mm_cmp_ps(easysimd_mm512_cmp_ps_mask_a_.m128[i], easysimd_mm512_cmp_ps_mask_b_.m128[i], (imm8)); \
    } \
    \
    easysimd_mm512_movepi32_mask(easysimd_mm512_castps_si512(easysimd__m512_from_private(easysimd_mm512_cmp_ps_mask_r_))); \
  }))
#elif defined(EASYSIMD_STATEMENT_EXPR_) && EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
  #define easysimd_mm512_cmp_ps_mask(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512_private \
      easysimd_mm512_cmp_ps_mask_r_, \
      easysimd_mm512_cmp_ps_mask_a_ = easysimd__m512_to_private((a)), \
      easysimd_mm512_cmp_ps_mask_b_ = easysimd__m512_to_private((b)); \
    \
    for (size_t i = 0 ; i < (sizeof(easysimd_mm512_cmp_ps_mask_r_.m256) / sizeof(easysimd_mm512_cmp_ps_mask_r_.m256[0])) ; i++) { \
      easysimd_mm512_cmp_ps_mask_r_.m256[i] = easysimd_mm256_cmp_ps(easysimd_mm512_cmp_ps_mask_a_.m256[i], easysimd_mm512_cmp_ps_mask_b_.m256[i], (imm8)); \
    } \
    \
    easysimd_mm512_movepi32_mask(easysimd_mm512_castps_si512(easysimd__m512_from_private(easysimd_mm512_cmp_ps_mask_r_))); \
  }))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_ps_mask
  #define _mm512_cmp_ps_mask(a, b, imm8) easysimd_mm512_cmp_ps_mask((a), (b), (imm8))
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_cmp_ps_mask(a, b, imm8) _mm256_cmp_ps_mask((a), (b), (imm8))
#else
  #define easysimd_mm256_cmp_ps_mask(a, b, imm8) easysimd_mm256_movepi32_mask(easysimd_mm256_castps_si256(easysimd_mm256_cmp_ps((a), (b), (imm8))))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_ps_mask
  #define _mm256_cmp_ps_mask(a, b, imm8) easysimd_mm256_cmp_ps_mask((a), (b), (imm8))
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_cmp_ps_mask(a, b, imm8) _mm_cmp_ps_mask((a), (b), (imm8))
#else
  #define easysimd_mm_cmp_ps_mask(a, b, imm8) easysimd_mm_movepi32_mask(easysimd_mm_castps_si128(easysimd_mm_cmp_ps((a), (b), (imm8))))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_ps_mask
  #define _mm_cmp_ps_mask(a, b, imm8) easysimd_mm_cmp_ps_mask((a), (b), (imm8))
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_cmp_pd_mask (easysimd__m512d a, easysimd__m512d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return easysimd_mm512_movepi64_mask(easysimd_mm512_castpd_si512(r));
  #else
  easysimd__m512d_private
    r_,
    a_ = easysimd__m512d_to_private(a),
    b_ = easysimd__m512d_to_private(b);
  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] == b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] < b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] <= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != a_.f64) | (b_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] != a_.f64[i]) || (b_.f64[i] != b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] != b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 == a_.f64) & (b_.f64 == b_.f64) & (a_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i]) & (a_.f64[i] != b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 < b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] < b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 <= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] <= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ((a_.f64 == a_.f64) & (b_.f64 == b_.f64)));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != a_.f64) | (b_.f64 != b_.f64) | (a_.f64 == b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] != a_.f64[i]) | (b_.f64[i] != b_.f64[i]) | (a_.f64[i] == b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 >= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] >= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 > b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] > b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      r_ = easysimd__m512d_to_private(easysimd_mm512_setzero_pd());
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] >= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] > b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      r_ = easysimd__m512d_to_private(easysimd_x_mm512_setone_pd());
      break;

    default:
      HEDLEY_UNREACHABLE();
  }
  return easysimd_mm512_movepi64_mask(easysimd_mm512_castpd_si512(easysimd__m512d_from_private(r_)));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_cmp_pd_mask(a, b, imm8) _mm512_cmp_pd_mask((a), (b), (imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm512_cmp_pd_mask(a, b, imm8) ({ \
    easysimd__m512d dst = easysimd_mm512_cmp_pd(a, b, imm8); \
    easysimd__mmask16 r = 0; \
    uint64_t g_mask_epi64[2] __attribute__((aligned(16))) = {0x01, 0x02}; \
    uint64x2_t vect_mask = vld1q_u64(g_mask_epi64); \
    uint64_t r0 = vaddvq_u64(vandq_u64(vreinterpretq_u64_f64(dst.m128[0].neon_f64), vect_mask)); \
    uint64_t r1 = vaddvq_u64(vandq_u64(vreinterpretq_u64_f64(dst.m128[1].neon_f64), vect_mask)); \
    uint64_t r2 = vaddvq_u64(vandq_u64(vreinterpretq_u64_f64(dst.m128[2].neon_f64), vect_mask)); \
    uint64_t r3 = vaddvq_u64(vandq_u64(vreinterpretq_u64_f64(dst.m128[3].neon_f64), vect_mask)); \
    r = r0 | (r1 << 2) | (r2 << 4) | (r3 << 6); \
    r; \
  })
#elif defined(EASYSIMD_STATEMENT_EXPR_) && EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
  #define easysimd_mm512_cmp_pd_mask(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512d_private \
      easysimd_mm512_cmp_pd_mask_r_, \
      easysimd_mm512_cmp_pd_mask_a_ = easysimd__m512d_to_private((a)), \
      easysimd_mm512_cmp_pd_mask_b_ = easysimd__m512d_to_private((b)); \
    \
    for (size_t easysimd_mm512_cmp_pd_mask_i = 0 ; easysimd_mm512_cmp_pd_mask_i < (sizeof(easysimd_mm512_cmp_pd_mask_r_.m128d) / sizeof(easysimd_mm512_cmp_pd_mask_r_.m128d[0])) ; easysimd_mm512_cmp_pd_mask_i++) { \
      easysimd_mm512_cmp_pd_mask_r_.m128d[easysimd_mm512_cmp_pd_mask_i] = easysimd_mm_cmp_pd(easysimd_mm512_cmp_pd_mask_a_.m128d[easysimd_mm512_cmp_pd_mask_i], easysimd_mm512_cmp_pd_mask_b_.m128d[easysimd_mm512_cmp_pd_mask_i], (imm8)); \
    } \
    \
    easysimd_mm512_movepi64_mask(easysimd_mm512_castpd_si512(easysimd__m512d_from_private(easysimd_mm512_cmp_pd_mask_r_))); \
  }))
#elif defined(EASYSIMD_STATEMENT_EXPR_) && EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
  #define easysimd_mm512_cmp_pd_mask(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512d_private \
      easysimd_mm512_cmp_pd_mask_r_, \
      easysimd_mm512_cmp_pd_mask_a_ = easysimd__m512d_to_private((a)), \
      easysimd_mm512_cmp_pd_mask_b_ = easysimd__m512d_to_private((b)); \
    \
    for (size_t easysimd_mm512_cmp_pd_mask_i = 0 ; easysimd_mm512_cmp_pd_mask_i < (sizeof(easysimd_mm512_cmp_pd_mask_r_.m256d) / sizeof(easysimd_mm512_cmp_pd_mask_r_.m256d[0])) ; easysimd_mm512_cmp_pd_mask_i++) { \
      easysimd_mm512_cmp_pd_mask_r_.m256d[easysimd_mm512_cmp_pd_mask_i] = easysimd_mm256_cmp_pd(easysimd_mm512_cmp_pd_mask_a_.m256d[easysimd_mm512_cmp_pd_mask_i], easysimd_mm512_cmp_pd_mask_b_.m256d[easysimd_mm512_cmp_pd_mask_i], (imm8)); \
    } \
    \
    easysimd_mm512_movepi64_mask(easysimd_mm512_castpd_si512(easysimd__m512d_from_private(easysimd_mm512_cmp_pd_mask_r_))); \
  }))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_pd_mask
  #define _mm512_cmp_pd_mask(a, b, imm8) easysimd_mm512_cmp_pd_mask((a), (b), (imm8))
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_cmp_pd_mask(a, b, imm8) _mm256_cmp_pd_mask((a), (b), (imm8))
#else
  #define easysimd_mm256_cmp_pd_mask(a, b, imm8) easysimd_mm256_movepi64_mask(easysimd_mm256_castpd_si256(easysimd_mm256_cmp_pd((a), (b), (imm8))))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_pd_mask
  #define _mm256_cmp_pd_mask(a, b, imm8) easysimd_mm256_cmp_pd_mask((a), (b), (imm8))
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_cmp_pd_mask(a, b, imm8) _mm_cmp_pd_mask((a), (b), (imm8))
#else
  #define easysimd_mm_cmp_pd_mask(a, b, imm8) easysimd_mm_movepi64_mask(easysimd_mm_castpd_si128(easysimd_mm_cmp_pd((a), (b), (imm8))))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_pd_mask
  #define _mm_cmp_pd_mask(a, b, imm8) easysimd_mm_cmp_pd_mask((a), (b), (imm8))
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_CMP_H) */
