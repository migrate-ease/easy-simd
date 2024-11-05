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
 *   2019      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_X86_FMA_H)
#define EASYSIMD_X86_FMA_H

#include "avx.h"

#if !defined(EASYSIMD_X86_FMA_NATIVE) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
#  define EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES
#endif
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fmadd_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmadd_pd(a, b, c);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vfmaq_f64(c_.neon_f64, b_.neon_f64, a_.neon_f64);
    #elif defined(easysimd_math_fma) && (defined(__FP_FAST_FMA) || defined(FP_FAST_FMA))
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_fma(a_.f64[i], b_.f64[i], c_.f64[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = (a_.f64[i] * b_.f64[i]) + c_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmadd_pd
  #define _mm_fmadd_pd(a, b, c) easysimd_mm_fmadd_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_fmadd_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmadd_pd(a, b, c);
  #else
    return easysimd_mm256_add_pd(easysimd_mm256_mul_pd(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmadd_pd
  #define _mm256_fmadd_pd(a, b, c) easysimd_mm256_fmadd_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fmadd_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmadd_ps(a, b, c);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(__ARM_FEATURE_FMA)
      r_.neon_f32 = vfmaq_f32(c_.neon_f32, b_.neon_f32, a_.neon_f32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vmlaq_f32(c_.neon_f32, b_.neon_f32, a_.neon_f32);
    #elif defined(easysimd_math_fmaf) && (defined(__FP_FAST_FMAF) || defined(FP_FAST_FMAF))
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_fmaf(a_.f32[i], b_.f32[i], c_.f32[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = (a_.f32[i] * b_.f32[i]) + c_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmadd_ps
  #define _mm_fmadd_ps(a, b, c) easysimd_mm_fmadd_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_fmadd_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmadd_ps(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), c.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), c.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      c_ = easysimd__m256_to_private(c),
      r_;

    for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
      r_.m128[i] = easysimd_mm_fmadd_ps(a_.m128[i], b_.m128[i], c_.m128[i]);
    }

    return easysimd__m256_from_private(r_);
  #else
    return easysimd_mm256_add_ps(easysimd_mm256_mul_ps(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmadd_ps
  #define _mm256_fmadd_ps(a, b, c) easysimd_mm256_fmadd_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fmadd_sd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fmadd_sd(a, b, c);
  #else
    return easysimd_mm_add_sd(easysimd_mm_mul_sd(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmadd_sd
  #define _mm_fmadd_sd(a, b, c) easysimd_mm_fmadd_sd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fmadd_ss (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fmadd_ss(a, b, c);
  #else
    return easysimd_mm_add_ss(easysimd_mm_mul_ss(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmadd_ss
  #define _mm_fmadd_ss(a, b, c) easysimd_mm_fmadd_ss(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fmaddsub_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmaddsub_pd(a, b, c);
  #else
    return easysimd_mm_addsub_pd(easysimd_mm_mul_pd(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmaddsub_pd
  #define _mm_fmaddsub_pd(a, b, c) easysimd_mm_fmaddsub_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_fmaddsub_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmaddsub_pd(a, b, c);
  #else
    return easysimd_mm256_addsub_pd(easysimd_mm256_mul_pd(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmaddsub_pd
  #define _mm256_fmaddsub_pd(a, b, c) easysimd_mm256_fmaddsub_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fmaddsub_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmaddsub_ps(a, b, c);
  #else
    return easysimd_mm_addsub_ps(easysimd_mm_mul_ps(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmaddsub_ps
  #define _mm_fmaddsub_ps(a, b, c) easysimd_mm_fmaddsub_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_fmaddsub_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmaddsub_ps(a, b, c);
  #else
    return easysimd_mm256_addsub_ps(easysimd_mm256_mul_ps(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmaddsub_ps
  #define _mm256_fmaddsub_ps(a, b, c) easysimd_mm256_fmaddsub_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fmsub_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmsub_pd(a, b, c);
  #else
    return easysimd_mm_sub_pd(easysimd_mm_mul_pd(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmsub_pd
  #define _mm_fmsub_pd(a, b, c) easysimd_mm_fmsub_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_fmsub_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmsub_pd(a, b, c);
  #else
    return easysimd_mm256_sub_pd(easysimd_mm256_mul_pd(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmsub_pd
  #define _mm256_fmsub_pd(a, b, c) easysimd_mm256_fmsub_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fmsub_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmsub_ps(a, b, c);
  #else
    return easysimd_mm_sub_ps(easysimd_mm_mul_ps(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmsub_ps
  #define _mm_fmsub_ps(a, b, c) easysimd_mm_fmsub_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_fmsub_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmsub_ps(a, b, c);
  #else
    return easysimd_mm256_sub_ps(easysimd_mm256_mul_ps(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmsub_ps
  #define _mm256_fmsub_ps(a, b, c) easysimd_mm256_fmsub_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fmsub_sd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fmsub_sd(a, b, c);
  #else
    return easysimd_mm_sub_sd(easysimd_mm_mul_sd(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmsub_sd
  #define _mm_fmsub_sd(a, b, c) easysimd_mm_fmsub_sd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fmsub_ss (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fmsub_ss(a, b, c);
  #else
    return easysimd_mm_sub_ss(easysimd_mm_mul_ss(a, b), c);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmsub_ss
  #define _mm_fmsub_ss(a, b, c) easysimd_mm_fmsub_ss(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fmsubadd_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmsubadd_pd(a, b, c);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
      r_.f64[  i  ] = (a_.f64[  i  ] * b_.f64[  i  ]) + c_.f64[  i  ];
      r_.f64[i + 1] = (a_.f64[i + 1] * b_.f64[i + 1]) - c_.f64[i + 1];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmsubadd_pd
  #define _mm_fmsubadd_pd(a, b, c) easysimd_mm_fmsubadd_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_fmsubadd_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmsubadd_pd(a, b, c);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b),
      c_ = easysimd__m256d_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
      r_.f64[  i  ] = (a_.f64[  i  ] * b_.f64[  i  ]) + c_.f64[  i  ];
      r_.f64[i + 1] = (a_.f64[i + 1] * b_.f64[i + 1]) - c_.f64[i + 1];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmsubadd_pd
  #define _mm256_fmsubadd_pd(a, b, c) easysimd_mm256_fmsubadd_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fmsubadd_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fmsubadd_ps(a, b, c);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
      r_.f32[  i  ] = (a_.f32[  i  ] * b_.f32[  i  ]) + c_.f32[  i  ];
      r_.f32[i + 1] = (a_.f32[i + 1] * b_.f32[i + 1]) - c_.f32[i + 1];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fmsubadd_ps
  #define _mm_fmsubadd_ps(a, b, c) easysimd_mm_fmsubadd_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_fmsubadd_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fmsubadd_ps(a, b, c);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      c_ = easysimd__m256_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
      r_.f32[  i  ] = (a_.f32[  i  ] * b_.f32[  i  ]) + c_.f32[  i  ];
      r_.f32[i + 1] = (a_.f32[i + 1] * b_.f32[i + 1]) - c_.f32[i + 1];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fmsubadd_ps
  #define _mm256_fmsubadd_ps(a, b, c) easysimd_mm256_fmsubadd_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fnmadd_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fnmadd_pd(a, b, c);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vfmsq_f64(c_.neon_f64, a_.neon_f64, b_.neon_f64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = -(a_.f64[i] * b_.f64[i]) + c_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmadd_pd
  #define _mm_fnmadd_pd(a, b, c) easysimd_mm_fnmadd_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_fnmadd_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fnmadd_pd(a, b, c);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b),
      c_ = easysimd__m256d_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = -(a_.f64[i] * b_.f64[i]) + c_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fnmadd_pd
  #define _mm256_fnmadd_pd(a, b, c) easysimd_mm256_fnmadd_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fnmadd_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fnmadd_ps(a, b, c);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(__ARM_FEATURE_FMA)
      r_.neon_f32 = vfmsq_f32(c_.neon_f32, a_.neon_f32, b_.neon_f32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vmlsq_f32(c_.neon_f32, a_.neon_f32, b_.neon_f32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = -(a_.f32[i] * b_.f32[i]) + c_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmadd_ps
  #define _mm_fnmadd_ps(a, b, c) easysimd_mm_fnmadd_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_fnmadd_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fnmadd_ps(a, b, c);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      c_ = easysimd__m256_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = -(a_.f32[i] * b_.f32[i]) + c_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fnmadd_ps
  #define _mm256_fnmadd_ps(a, b, c) easysimd_mm256_fnmadd_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fnmadd_sd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fnmadd_sd(a, b, c);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c);

    r_ = a_;
    r_.f64[0] = -(a_.f64[0] * b_.f64[0]) + c_.f64[0];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmadd_sd
  #define _mm_fnmadd_sd(a, b, c) easysimd_mm_fnmadd_sd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fnmadd_ss (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fnmadd_ss(a, b, c);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c);

    r_ = a_;
    r_.f32[0] = -(a_.f32[0] * b_.f32[0]) + c_.f32[0];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmadd_ss
  #define _mm_fnmadd_ss(a, b, c) easysimd_mm_fnmadd_ss(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fnmsub_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fnmsub_pd(a, b, c);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = -(a_.f64[i] * b_.f64[i]) - c_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmsub_pd
  #define _mm_fnmsub_pd(a, b, c) easysimd_mm_fnmsub_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_fnmsub_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fnmsub_pd(a, b, c);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b),
      c_ = easysimd__m256d_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = -(a_.f64[i] * b_.f64[i]) - c_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fnmsub_pd
  #define _mm256_fnmsub_pd(a, b, c) easysimd_mm256_fnmsub_pd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fnmsub_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm_fnmsub_ps(a, b, c);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = -(a_.f32[i] * b_.f32[i]) - c_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmsub_ps
  #define _mm_fnmsub_ps(a, b, c) easysimd_mm_fnmsub_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_fnmsub_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE)
    return _mm256_fnmsub_ps(a, b, c);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      c_ = easysimd__m256_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = -(a_.f32[i] * b_.f32[i]) - c_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fnmsub_ps
  #define _mm256_fnmsub_ps(a, b, c) easysimd_mm256_fnmsub_ps(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fnmsub_sd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fnmsub_sd(a, b, c);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      c_ = easysimd__m128d_to_private(c);

    r_ = a_;
    r_.f64[0] = -(a_.f64[0] * b_.f64[0]) - c_.f64[0];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmsub_sd
  #define _mm_fnmsub_sd(a, b, c) easysimd_mm_fnmsub_sd(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fnmsub_ss (easysimd__m128 a, easysimd__m128 b, easysimd__m128 c) {
  #if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_BUG_MCST_LCC_FMA_WRONG_RESULT)
    return _mm_fnmsub_ss(a, b, c);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      c_ = easysimd__m128_to_private(c);

    r_ = easysimd__m128_to_private(a);
    r_.f32[0] = -(a_.f32[0] * b_.f32[0]) - c_.f32[0];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES)
  #undef _mm_fnmsub_ss
  #define _mm_fnmsub_ss(a, b, c) easysimd_mm_fnmsub_ss(a, b, c)
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_FMA_H) */
