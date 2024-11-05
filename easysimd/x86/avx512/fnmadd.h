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
 *   2020      kitegi <kitegi@users.noreply.github.com>
 */

#if !defined(EASYSIMD_X86_AVX512_FNMADD_H)
#define EASYSIMD_X86_AVX512_FNMADD_H

#include "types.h"
#include "mov.h"
#include "../fma.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask3_fnmadd_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c, easysimd__mmask8 k) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask3_fnmadd_ps(a, b, c, k);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32, b.sve_f32)), c.sve_f32), c.sve_f32);
    return r;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? -(a_.f32[i] * b_.f32[i]) + c_.f32[i] : c_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask3_fnmadd_ps
  #define _mm_mask3_fnmadd_ps(a, b, c, k) easysimd_mm_mask3_fnmadd_ps(a, b, c, k)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_fnmadd_ps (easysimd__m128 a, easysimd__mmask8 k, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_fnmadd_ps(a, k, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32, b.sve_f32)), c.sve_f32), a.sve_f32);
    return r;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? -(a_.f32[i] * b_.f32[i]) + c_.f32[i] : a_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fnmadd_ps
  #define _mm_mask_fnmadd_ps(a, k, b, c) easysimd_mm_mask_fnmadd_ps(a, k, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_fnmadd_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_fnmadd_ps(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32, b.sve_f32)), c.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? -(a_.f32[i] * b_.f32[i]) + c_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fnmadd_ps
  #define _mm_maskz_fnmadd_ps(k, a, b, c) easysimd_mm_maskz_fnmadd_ps(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask3_fnmadd_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c, easysimd__mmask8 k) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask3_fnmadd_pd(a, b, c, k);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64, b.sve_f64)), c.sve_f64), c.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? -(a_.f64[i] * b_.f64[i]) + c_.f64[i] : c_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask3_fnmadd_pd
  #define _mm_mask3_fnmadd_pd(a, b, c, k) easysimd_mm_mask3_fnmadd_pd(a, b, c, k)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_fnmadd_pd (easysimd__m128d a, easysimd__mmask8 k, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_fnmadd_pd(a, k, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64, b.sve_f64)), c.sve_f64), a.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? -(a_.f64[i] * b_.f64[i]) + c_.f64[i] : a_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fnmadd_pd
  #define _mm_mask_fnmadd_pd(a, k, b, c) easysimd_mm_mask_fnmadd_pd(a, k, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_fnmadd_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_fnmadd_pd(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64, b.sve_f64)), c.sve_f64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? -(a_.f64[i] * b_.f64[i]) + c_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fnmadd_pd
  #define _mm_maskz_fnmadd_pd(k, a, b, c) easysimd_mm_maskz_fnmadd_pd(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_fnmadd_ps (easysimd__m256 a, easysimd__mmask8 k, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_fnmadd_ps(a, k, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0])), c.sve_f32[EASYSIMD_SV_INDEX_0]), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1])), c.sve_f32[EASYSIMD_SV_INDEX_1]), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      c_ = easysimd__m256_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? -(a_.f32[i] * b_.f32[i]) + c_.f32[i] : a_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_fnmadd_ps
  #define _mm256_mask_fnmadd_ps(a, k, b, c) easysimd_mm256_mask_fnmadd_ps(a, k, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_fnmadd_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_fnmadd_ps(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0])), c.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1])), c.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      c_ = easysimd__m256_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? -(a_.f32[i] * b_.f32[i]) + c_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_fnmadd_ps
  #define _mm256_maskz_fnmadd_ps(k, a, b, c) easysimd_mm256_maskz_fnmadd_ps(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_fnmadd_pd (easysimd__m256d a, easysimd__mmask8 k, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_fnmadd_pd(a, k, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                                            svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0])), c.sve_f64[EASYSIMD_SV_INDEX_0]), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                                            svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1])), c.sve_f64[EASYSIMD_SV_INDEX_1]), a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b),
      c_ = easysimd__m256d_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? -(a_.f64[i] * b_.f64[i]) + c_.f64[i] : a_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_fnmadd_pd
  #define _mm256_mask_fnmadd_pd(a, k, b, c) easysimd_mm256_mask_fnmadd_pd(a, k, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_fnmadd_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_fnmadd_pd(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                                            svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0])), c.sve_f64[EASYSIMD_SV_INDEX_0]), svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                                            svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1])), c.sve_f64[EASYSIMD_SV_INDEX_1]), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b),
      c_ = easysimd__m256d_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? -(a_.f64[i] * b_.f64[i]) + c_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_fnmadd_pd
  #define _mm256_maskz_fnmadd_pd(k, a, b, c) easysimd_mm256_maskz_fnmadd_pd(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_fnmadd_ps (easysimd__m512 a, easysimd__m512 b, easysimd__m512 c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_fnmadd_ps(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0])), c.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1])), c.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2])), c.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3])), c.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b),
      c_ = easysimd__m512_to_private(c);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_fnmadd_ps(a_.m256[i], b_.m256[i], c_.m256[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = -(a_.f32 * b_.f32) + c_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = -(a_.f32[i] * b_.f32[i]) + c_.f32[i];
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_fnmadd_ps
  #define _mm512_fnmadd_ps(a, b, c) easysimd_mm512_fnmadd_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_fnmadd_ps (easysimd__m512 a, easysimd__mmask16 k, easysimd__m512 b, easysimd__m512 c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_fnmadd_ps(a, k, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0])), c.sve_f32[EASYSIMD_SV_INDEX_0]), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1])), c.sve_f32[EASYSIMD_SV_INDEX_1]), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2])), c.sve_f32[EASYSIMD_SV_INDEX_2]), a.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3])), c.sve_f32[EASYSIMD_SV_INDEX_3]), a.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b),
      c_ = easysimd__m512_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? -(a_.f32[i] * b_.f32[i]) + c_.f32[i] : a_.f32[i];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_fnmadd_ps
  #define _mm512_mask_fnmadd_ps(a, k, b, c) easysimd_mm512_mask_fnmadd_ps(a, k, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_fnmadd_ps (easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b, easysimd__m512 c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_fnmadd_ps(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0])), c.sve_f32[EASYSIMD_SV_INDEX_0]), svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1])), c.sve_f32[EASYSIMD_SV_INDEX_1]), svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2])), c.sve_f32[EASYSIMD_SV_INDEX_2]), svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                                            svadd_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3])), c.sve_f32[EASYSIMD_SV_INDEX_3]), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b),
      c_ = easysimd__m512_to_private(c),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? -(a_.f32[i] * b_.f32[i]) + c_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_fnmadd_ps
  #define _mm512_maskz_fnmadd_ps(k, a, b, c) easysimd_mm512_maskz_fnmadd_ps(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_fnmadd_pd (easysimd__m512d a, easysimd__m512d b, easysimd__m512d c) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_fnmadd_pd(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0])), c.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1])), c.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2])), c.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svadd_f64_z(pg, svneg_f64_z(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3])), c.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b),
      c_ = easysimd__m512d_to_private(c);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_fnmadd_pd(a_.m256d[i], b_.m256d[i], c_.m256d[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = -(a_.f64 * b_.f64) + c_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = -(a_.f64[i] * b_.f64[i]) + c_.f64[i];
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_fnmadd_pd
  #define _mm512_fnmadd_pd(a, b, c) easysimd_mm512_fnmadd_pd(a, b, c)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_FNMADD_H) */
