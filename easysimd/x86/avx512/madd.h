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
 *   2020      Ashleigh Newman-Jones <ashnewman-jones@hotmail.co.uk>
 */

#if !defined(EASYSIMD_X86_AVX512_MADD_H)
#define EASYSIMD_X86_AVX512_MADD_H

#include "types.h"
#include "mov.h"
#include "../avx2.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_madd_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_madd_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_s32_z(svptrue_b32(), svmullb_s32(a.sve_i16, b.sve_i16), svmullt_s32(a.sve_i16, b.sve_i16)), src.sve_i32);
    return r;
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_madd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_madd_epi16
  #define _mm_mask_madd_epi16(src, k, a, b) easysimd_mm_mask_madd_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_madd_epi16 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_madd_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_s32_z(svptrue_b32(), svmullb_s32(a.sve_i16, b.sve_i16), svmullt_s32(a.sve_i16, b.sve_i16)), svdup_n_s32(0));
    return r;
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_madd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_madd_epi16
  #define _mm_maskz_madd_epi16(src, k, a, b) easysimd_mm_maskz_madd_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_madd_epi16 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_madd_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_0],b.sve_i16[EASYSIMD_SV_INDEX_0]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0])), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_1],b.sve_i16[EASYSIMD_SV_INDEX_1]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1])), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_madd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_madd_epi16
  #define _mm256_mask_madd_epi16(src, k, a, b) easysimd_mm256_mask_madd_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_madd_epi16 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_madd_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    easysimd__m128i svz;
    svz.sve_i32 = svdup_n_s32(0);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_0],b.sve_i16[EASYSIMD_SV_INDEX_0]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0])), svz.sve_i32);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_1],b.sve_i16[EASYSIMD_SV_INDEX_1]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1])), svz.sve_i32);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_madd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_madd_epi16
  #define _mm256_maskz_madd_epi16(src, k, a, b) easysimd_mm256_maskz_madd_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_madd_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_madd_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_0],b.sve_i16[EASYSIMD_SV_INDEX_0]),
                                              svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_1],b.sve_i16[EASYSIMD_SV_INDEX_1]),
                                              svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]));
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_2],b.sve_i16[EASYSIMD_SV_INDEX_2]),
                                              svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]));
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_3],b.sve_i16[EASYSIMD_SV_INDEX_3]),
                                              svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512i_private r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if 0 && EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) || defined(EASYSIMD_BUG_CLANG_BAD_MADD)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_madd_epi16(a_.m256i[i], b_.m256i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i16[0])) ; i += 2) {
        r_.i32[i / 2] =
          (HEDLEY_STATIC_CAST(int32_t, a_.i16[  i  ]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[  i  ])) +
          (HEDLEY_STATIC_CAST(int32_t, a_.i16[i + 1]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[i + 1]));
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_madd_epi16
  #define _mm512_madd_epi16(src, k, a, b) easysimd_mm512_madd_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_madd_epi16 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_madd_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_0],b.sve_i16[EASYSIMD_SV_INDEX_0]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0])), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_1],b.sve_i16[EASYSIMD_SV_INDEX_1]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1])), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_2],b.sve_i16[EASYSIMD_SV_INDEX_2]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2])), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_3],b.sve_i16[EASYSIMD_SV_INDEX_3]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3])), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_madd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_madd_epi16
  #define _mm512_mask_madd_epi16(src, k, a, b) easysimd_mm512_mask_madd_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_madd_epi16 (easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_madd_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    easysimd__m128i svz;
    svz.sve_i32 = svdup_n_s32(0);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_0],b.sve_i16[EASYSIMD_SV_INDEX_0]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0])), svz.sve_i32);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_1],b.sve_i16[EASYSIMD_SV_INDEX_1]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1])), svz.sve_i32);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_2],b.sve_i16[EASYSIMD_SV_INDEX_2]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2])), svz.sve_i32);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                                            svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_3],b.sve_i16[EASYSIMD_SV_INDEX_3]),
                                            svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3])), svz.sve_i32);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_madd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_madd_epi16
  #define _mm512_maskz_madd_epi16(src, k, a, b) easysimd_mm512_maskz_madd_epi16(src, k, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_MADD_H) */
