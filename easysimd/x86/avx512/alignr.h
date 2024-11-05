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
 *   2020      Hidayat Khan <huk2209@gmail.com>
 *   2021      Andrew Rodriguez <anrodriguez@linkedin.com>
 */

#if !defined(EASYSIMD_X86_AVX512_ALIGNR_H)
#define EASYSIMD_X86_AVX512_ALIGNR_H

#include "types.h"
#include "mov.h"
#include "../../easysimd-f16.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_alignr_epi64 (easysimd__m512i a, easysimd__m512i b, const int imm8) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    int64_t arr[16];
    vst1q_s64_x4(&arr[0], b.neon_i64x4);
    vst1q_s64_x4(&arr[8], a.neon_i64x4);
    int32_t idx = imm8 & 0x07;
    a.neon_i64x4 = vld1q_s64_x4(arr + idx);
    return a;
  #else
    easysimd__m512i_private r_;
    int64_t arr_[16];
    easysimd_memcpy(&arr_[0], &b, sizeof(b));
    easysimd_memcpy(&arr_[8], &a, sizeof(a));
    int32_t select_idx = imm8 & 0x07;

    EASYSIMD_VECTORIZE
    for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
      r_.i64[i] = arr_[i + select_idx];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_alignr_epi64(a, b, imm8) _mm512_alignr_epi64(a, b, imm8)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm512_alignr_epi64(a, b, imm8) ({easysimd__m512i r; \
    switch((imm8) / 2) { \
      case 0: \
        r.m128i[0].neon_i64 = vextq_s64((b).m128i[0].neon_i64, (b).m128i[1].neon_i64, imm8 & 0x01); \
        r.m128i[1].neon_i64 = vextq_s64((b).m128i[1].neon_i64, (b).m128i[2].neon_i64, imm8 & 0x01); \
        r.m128i[2].neon_i64 = vextq_s64((b).m128i[2].neon_i64, (b).m128i[3].neon_i64, imm8 & 0x01); \
        r.m128i[3].neon_i64 = vextq_s64((b).m128i[3].neon_i64, (a).m128i[0].neon_i64, imm8 & 0x01); \
        break; \
      case 1: \
        r.m128i[0].neon_i64 = vextq_s64((b).m128i[1].neon_i64, (b).m128i[2].neon_i64, imm8 & 0x01); \
        r.m128i[1].neon_i64 = vextq_s64((b).m128i[2].neon_i64, (b).m128i[3].neon_i64, imm8 & 0x01); \
        r.m128i[2].neon_i64 = vextq_s64((b).m128i[3].neon_i64, (a).m128i[0].neon_i64, imm8 & 0x01); \
        r.m128i[3].neon_i64 = vextq_s64((a).m128i[0].neon_i64, (a).m128i[1].neon_i64, imm8 & 0x01); \
        break; \
      case 2:  \
        r.m128i[0].neon_i64 = vextq_s64((b).m128i[2].neon_i64, (b).m128i[3].neon_i64, imm8 & 0x01); \
        r.m128i[1].neon_i64 = vextq_s64((b).m128i[3].neon_i64, (a).m128i[0].neon_i64, imm8 & 0x01); \
        r.m128i[2].neon_i64 = vextq_s64((a).m128i[0].neon_i64, (a).m128i[1].neon_i64, imm8 & 0x01); \
        r.m128i[3].neon_i64 = vextq_s64((a).m128i[1].neon_i64, (a).m128i[2].neon_i64, imm8 & 0x01); \
        break; \
      case 3: \
        r.m128i[0].neon_i64 = vextq_s64((b).m128i[3].neon_i64, (a).m128i[0].neon_i64, imm8 & 0x01); \
        r.m128i[1].neon_i64 = vextq_s64((a).m128i[0].neon_i64, (a).m128i[1].neon_i64, imm8 & 0x01); \
        r.m128i[2].neon_i64 = vextq_s64((a).m128i[1].neon_i64, (a).m128i[2].neon_i64, imm8 & 0x01); \
        r.m128i[3].neon_i64 = vextq_s64((a).m128i[2].neon_i64, (a).m128i[3].neon_i64, imm8 & 0x01); \
        break; \
    } \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_alignr_epi64
  #define _mm512_alignr_epi64(a, b, imm8) easysimd_mm512_alignr_epi64(a, b, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_ALIGNR_H) */
