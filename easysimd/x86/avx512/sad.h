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

#if !defined(EASYSIMD_X86_AVX512_SAD_H)
#define EASYSIMD_X86_AVX512_SAD_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sad_epu8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_sad_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r, temp;
    svbool_t pg   = svptrue_b8();
    svbool_t pglo = svwhilele_b8(1, 8);
    svbool_t pghi = svnot_b_z(pg, pglo);
    temp.sve_u8[EASYSIMD_SV_INDEX_0] = svabd_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svdupq_n_u64(svaddv_u8(pglo, temp.sve_u8[EASYSIMD_SV_INDEX_0]), svaddv_u8(pghi, temp.sve_u8[EASYSIMD_SV_INDEX_0]));

    temp.sve_u8[EASYSIMD_SV_INDEX_1] = svabd_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svdupq_n_u64(svaddv_u8(pglo, temp.sve_u8[EASYSIMD_SV_INDEX_1]), svaddv_u8(pghi, temp.sve_u8[EASYSIMD_SV_INDEX_1]));

    temp.sve_u8[EASYSIMD_SV_INDEX_2] = svabd_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svdupq_n_u64(svaddv_u8(pglo, temp.sve_u8[EASYSIMD_SV_INDEX_2]), svaddv_u8(pghi, temp.sve_u8[EASYSIMD_SV_INDEX_2]));

    temp.sve_u8[EASYSIMD_SV_INDEX_3] = svabd_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svdupq_n_u64(svaddv_u8(pglo, temp.sve_u8[EASYSIMD_SV_INDEX_3]), svaddv_u8(pghi, temp.sve_u8[EASYSIMD_SV_INDEX_3]));

    return r;

  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_sad_epu8(a_.m256i[i], b_.m256i[i]);
        }
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        uint16_t tmp = 0;
        EASYSIMD_VECTORIZE_REDUCTION(+:tmp)
        for (size_t j = 0 ; j < ((sizeof(r_.u8) / sizeof(r_.u8[0])) / 8) ; j++) {
          const size_t e = j + (i * 8);
          tmp += (a_.u8[e] > b_.u8[e]) ? (a_.u8[e] - b_.u8[e]) : (b_.u8[e] - a_.u8[e]);
        }
        r_.i64[i] = tmp;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sad_epu8
  #define _mm512_sad_epu8(a, b) easysimd_mm512_sad_epu8(a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SAD_H) */
