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
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 */

#if !defined(EASYSIMD_X86_SVML_H)
#define EASYSIMD_X86_SVML_H

#include "fma.h"
#include "avx2.h"
#include "avx512/abs.h"
#include "avx512/add.h"
#include "avx512/cmp.h"
#include "avx512/copysign.h"
#include "avx512/xorsign.h"
#include "avx512/div.h"
#include "avx512/fmadd.h"
#include "avx512/mov.h"
#include "avx512/mul.h"
#include "avx512/negate.h"
#include "avx512/or.h"
#include "avx512/set1.h"
#include "avx512/setone.h"
#include "avx512/setzero.h"
#include "avx512/sqrt.h"
#include "avx512/sub.h"

#include "../easysimd-complex.h"

#if !defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
#  define EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if !defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
#  define EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_acos_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_acos_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_acosf4_u10(a);
    #else
      return Sleef_acosf4_u35(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_acosf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_acos_ps
  #define _mm_acos_ps(a) easysimd_mm_acos_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_acos_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_acos_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_acosd2_u10(a);
    #else
      return Sleef_acosd2_u35(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_acos(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_acos_pd
  #define _mm_acos_pd(a) easysimd_mm_acos_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_acos_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_acos_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_acosf8_u10(a);
    #else
      return Sleef_acosf8_u35(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_acos_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_acosf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_acos_ps
  #define _mm256_acos_ps(a) easysimd_mm256_acos_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_acos_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_acos_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_acosd4_u10(a);
    #else
      return Sleef_acosd4_u35(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_acos_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_acos(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_acos_pd
  #define _mm256_acos_pd(a) easysimd_mm256_acos_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_acos_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_acos_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_acosf16_u10(a);
    #else
      return Sleef_acosf16_u35(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_acos_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_acosf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_acos_ps
  #define _mm512_acos_ps(a) easysimd_mm512_acos_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_acos_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_acos_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_acosd8_u10(a);
    #else
      return Sleef_acosd8_u35(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_acos_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_acos(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_acos_pd
  #define _mm512_acos_pd(a) easysimd_mm512_acos_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_acos_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_acos_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_acos_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_acos_ps
  #define _mm512_mask_acos_ps(src, k, a) easysimd_mm512_mask_acos_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_acos_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_acos_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_acos_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_acos_pd
  #define _mm512_mask_acos_pd(src, k, a) easysimd_mm512_mask_acos_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_acosh_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_acosh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_acoshf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_acoshf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_acosh_ps
  #define _mm_acosh_ps(a) easysimd_mm_acosh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_acosh_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_acosh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_acoshd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_acosh(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_acosh_pd
  #define _mm_acosh_pd(a) easysimd_mm_acosh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_acosh_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_acosh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_acoshf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_acosh_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_acoshf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_acosh_ps
  #define _mm256_acosh_ps(a) easysimd_mm256_acosh_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_acosh_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_acosh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_acoshd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_acosh_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_acosh(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_acosh_pd
  #define _mm256_acosh_pd(a) easysimd_mm256_acosh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_acosh_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_acosh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_acoshf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_acosh_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_acoshf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_acosh_ps
  #define _mm512_acosh_ps(a) easysimd_mm512_acosh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_acosh_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_acosh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_acoshd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_acosh_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_acosh(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_acosh_pd
  #define _mm512_acosh_pd(a) easysimd_mm512_acosh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_acosh_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_acosh_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_acosh_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_acosh_ps
  #define _mm512_mask_acosh_ps(src, k, a) easysimd_mm512_mask_acosh_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_acosh_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_acosh_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_acosh_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_acosh_pd
  #define _mm512_mask_acosh_pd(src, k, a) easysimd_mm512_mask_acosh_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_asin_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_asin_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_asinf4_u10(a);
    #else
      return Sleef_asinf4_u35(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_asinf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_asin_ps
  #define _mm_asin_ps(a) easysimd_mm_asin_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_asin_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_asin_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_asind2_u10(a);
    #else
      return Sleef_asind2_u35(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_asin(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_asin_pd
  #define _mm_asin_pd(a) easysimd_mm_asin_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_asin_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_asin_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_asinf8_u10(a);
    #else
      return Sleef_asinf8_u35(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_asin_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_asinf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_asin_ps
  #define _mm256_asin_ps(a) easysimd_mm256_asin_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_asin_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_asin_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_asind4_u10(a);
    #else
      return Sleef_asind4_u35(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_asin_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_asin(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_asin_pd
  #define _mm256_asin_pd(a) easysimd_mm256_asin_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_asin_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_asin_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_asinf16_u10(a);
    #else
      return Sleef_asinf16_u35(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_asin_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_asinf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_asin_ps
  #define _mm512_asin_ps(a) easysimd_mm512_asin_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_asin_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_asin_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_asind8_u10(a);
    #else
      return Sleef_asind8_u35(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_asin_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_asin(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_asin_pd
  #define _mm512_asin_pd(a) easysimd_mm512_asin_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_asin_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_asin_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_asin_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_asin_ps
  #define _mm512_mask_asin_ps(src, k, a) easysimd_mm512_mask_asin_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_asin_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_asin_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_asin_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_asin_pd
  #define _mm512_mask_asin_pd(src, k, a) easysimd_mm512_mask_asin_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_asinh_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_asinh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_asinhf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_asinhf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_asinh_ps
  #define _mm_asinh_ps(a) easysimd_mm_asinh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_asinh_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_asinh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_asinhd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_asinh(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_asinh_pd
  #define _mm_asinh_pd(a) easysimd_mm_asinh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_asinh_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_asinh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_asinhf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_asinh_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_asinhf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_asinh_ps
  #define _mm256_asinh_ps(a) easysimd_mm256_asinh_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_asinh_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_asinh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_asinhd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_asinh_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_asinh(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_asinh_pd
  #define _mm256_asinh_pd(a) easysimd_mm256_asinh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_asinh_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_asinh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_asinhf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_asinh_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_asinhf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_asinh_ps
  #define _mm512_asinh_ps(a) easysimd_mm512_asinh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_asinh_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_asinh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_asinhd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_asinh_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_asinh(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_asinh_pd
  #define _mm512_asinh_pd(a) easysimd_mm512_asinh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_asinh_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_asinh_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_asinh_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_asinh_ps
  #define _mm512_mask_asinh_ps(src, k, a) easysimd_mm512_mask_asinh_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_asinh_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_asinh_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_asinh_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_asinh_pd
  #define _mm512_mask_asinh_pd(src, k, a) easysimd_mm512_mask_asinh_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_atan_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_atan_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atanf4_u10(a);
    #else
      return Sleef_atanf4_u35(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_atanf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_atan_ps
  #define _mm_atan_ps(a) easysimd_mm_atan_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_atan_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_atan_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atand2_u10(a);
    #else
      return Sleef_atand2_u35(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_atan(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_atan_pd
  #define _mm_atan_pd(a) easysimd_mm_atan_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_atan_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_atan_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atanf8_u10(a);
    #else
      return Sleef_atanf8_u35(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_atan_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_atanf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_atan_ps
  #define _mm256_atan_ps(a) easysimd_mm256_atan_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_atan_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_atan_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atand4_u10(a);
    #else
      return Sleef_atand4_u35(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_atan_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_atan(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_atan_pd
  #define _mm256_atan_pd(a) easysimd_mm256_atan_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_atan_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_atan_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atanf16_u10(a);
    #else
      return Sleef_atanf16_u35(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_atan_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_atanf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_atan_ps
  #define _mm512_atan_ps(a) easysimd_mm512_atan_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_atan_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_atan_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atand8_u10(a);
    #else
      return Sleef_atand8_u35(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_atan_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_atan(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_atan_pd
  #define _mm512_atan_pd(a) easysimd_mm512_atan_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_atan_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_atan_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_atan_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_atan_ps
  #define _mm512_mask_atan_ps(src, k, a) easysimd_mm512_mask_atan_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_atan_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_atan_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_atan_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_atan_pd
  #define _mm512_mask_atan_pd(src, k, a) easysimd_mm512_mask_atan_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_atan2_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_atan2_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atan2f4_u10(a, b);
    #else
      return Sleef_atan2f4_u35(a, b);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_atan2f(a_.f32[i], b_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_atan2_ps
  #define _mm_atan2_ps(a, b) easysimd_mm_atan2_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_atan2_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_atan2_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atan2d2_u10(a, b);
    #else
      return Sleef_atan2d2_u35(a, b);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_atan2(a_.f64[i], b_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_atan2_pd
  #define _mm_atan2_pd(a, b) easysimd_mm_atan2_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_atan2_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_atan2_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atan2f8_u10(a, b);
    #else
      return Sleef_atan2f8_u35(a, b);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_atan2_ps(a_.m128[i], b_.m128[i]);
    }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_atan2f(a_.f32[i], b_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_atan2_ps
  #define _mm256_atan2_ps(a, b) easysimd_mm256_atan2_ps(a, b)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_atan2_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_atan2_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atan2d4_u10(a, b);
    #else
      return Sleef_atan2d4_u35(a, b);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_atan2_pd(a_.m128d[i], b_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_atan2(a_.f64[i], b_.f64[i]);
      }
  #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_atan2_pd
  #define _mm256_atan2_pd(a, b) easysimd_mm256_atan2_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_atan2_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_atan2_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atan2f16_u10(a, b);
    #else
      return Sleef_atan2f16_u35(a, b);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_atan2_ps(a_.m256[i], b_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_atan2f(a_.f32[i], b_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_atan2_ps
  #define _mm512_atan2_ps(a, b) easysimd_mm512_atan2_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_atan2_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_atan2_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_atan2d8_u10(a, b);
    #else
      return Sleef_atan2d8_u35(a, b);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_atan2_pd(a_.m256d[i], b_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_atan2(a_.f64[i], b_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_atan2_pd
  #define _mm512_atan2_pd(a, b) easysimd_mm512_atan2_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_atan2_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_atan2_ps(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_atan2_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_atan2_ps
  #define _mm512_mask_atan2_ps(src, k, a, b) easysimd_mm512_mask_atan2_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_atan2_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_atan2_pd(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_atan2_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_atan2_pd
  #define _mm512_mask_atan2_pd(src, k, a, b) easysimd_mm512_mask_atan2_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_atanh_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_atanh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_atanhf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_atanhf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_atanh_ps
  #define _mm_atanh_ps(a) easysimd_mm_atanh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_atanh_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_atanh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_atanhd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_atanh(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_atanh_pd
  #define _mm_atanh_pd(a) easysimd_mm_atanh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_atanh_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_atanh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_atanhf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_atanh_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_atanhf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_atanh_ps
  #define _mm256_atanh_ps(a) easysimd_mm256_atanh_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_atanh_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_atanh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_atanhd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_atanh_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_atanh(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_atanh_pd
  #define _mm256_atanh_pd(a) easysimd_mm256_atanh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_atanh_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_atanh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_atanhf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_atanh_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_atanhf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_atanh_ps
  #define _mm512_atanh_ps(a) easysimd_mm512_atanh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_atanh_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_atanh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_atanhd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_atanh_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_atanh(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_atanh_pd
  #define _mm512_atanh_pd(a) easysimd_mm512_atanh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_atanh_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_atanh_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_atanh_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_atanh_ps
  #define _mm512_mask_atanh_ps(src, k, a) easysimd_mm512_mask_atanh_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_atanh_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_atanh_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_atanh_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_atanh_pd
  #define _mm512_mask_atanh_pd(src, k, a) easysimd_mm512_mask_atanh_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cbrt_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cbrt_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_cbrtf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_cbrtf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cbrt_ps
  #define _mm_cbrt_ps(a) easysimd_mm_cbrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cbrt_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cbrt_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_cbrtd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_cbrt(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cbrt_pd
  #define _mm_cbrt_pd(a) easysimd_mm_cbrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cbrt_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cbrt_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_cbrtf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_cbrt_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cbrtf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cbrt_ps
  #define _mm256_cbrt_ps(a) easysimd_mm256_cbrt_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cbrt_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cbrt_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_cbrtd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_cbrt_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cbrt(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cbrt_pd
  #define _mm256_cbrt_pd(a) easysimd_mm256_cbrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cbrt_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cbrt_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_cbrtf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_cbrt_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cbrtf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cbrt_ps
  #define _mm512_cbrt_ps(a) easysimd_mm512_cbrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cbrt_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cbrt_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_cbrtd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_cbrt_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cbrt(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cbrt_pd
  #define _mm512_cbrt_pd(a) easysimd_mm512_cbrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_cbrt_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cbrt_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_cbrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cbrt_ps
  #define _mm512_mask_cbrt_ps(src, k, a) easysimd_mm512_mask_cbrt_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_cbrt_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cbrt_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_cbrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cbrt_pd
  #define _mm512_mask_cbrt_pd(src, k, a) easysimd_mm512_mask_cbrt_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cexp_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cexp_ps(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i+=2) {
      easysimd_cfloat32 val = easysimd_math_cexpf(EASYSIMD_MATH_CMPLXF(a_.f32[i], a_.f32[i+1]));
      r_.f32[  i  ] = easysimd_math_crealf(val);
      r_.f32[i + 1] = easysimd_math_cimagf(val);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cexp_ps
  #define _mm_cexp_ps(a) easysimd_mm_cexp_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cexp_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cexp_ps(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i+=2) {
      easysimd_cfloat32 val = easysimd_math_cexpf(EASYSIMD_MATH_CMPLXF(a_.f32[i], a_.f32[i+1]));
      r_.f32[  i  ] = easysimd_math_crealf(val);
      r_.f32[i + 1] = easysimd_math_cimagf(val);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cexp_ps
  #define _mm256_cexp_ps(a) easysimd_mm256_cexp_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cos_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cos_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosf4_u10(a);
    #else
      return Sleef_cosf4_u35(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_cosf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cos_ps
  #define _mm_cos_ps(a) easysimd_mm_cos_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cos_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cos_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosd2_u10(a);
    #else
      return Sleef_cosd2_u35(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_cos(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cos_pd
  #define _mm_cos_pd(a) easysimd_mm_cos_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cos_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cos_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosf8_u10(a);
    #else
      return Sleef_cosf8_u35(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_cos_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cosf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cos_ps
  #define _mm256_cos_ps(a) easysimd_mm256_cos_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cos_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cos_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosd4_u10(a);
    #else
      return Sleef_cosd4_u35(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_cos_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cos(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cos_pd
  #define _mm256_cos_pd(a) easysimd_mm256_cos_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cos_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cos_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosf16_u10(a);
    #else
      return Sleef_cosf16_u35(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_cos_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cosf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cos_ps
  #define _mm512_cos_ps(a) easysimd_mm512_cos_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cos_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cos_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosd8_u10(a);
    #else
      return Sleef_cosd8_u35(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_cos_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cos(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cos_pd
  #define _mm512_cos_pd(a) easysimd_mm512_cos_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_cos_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cos_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_cos_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cos_ps
  #define _mm512_mask_cos_ps(src, k, a) easysimd_mm512_mask_cos_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_cos_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cos_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_cos_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cos_pd
  #define _mm512_mask_cos_pd(src, k, a) easysimd_mm512_mask_cos_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_deg2rad_ps(easysimd__m128 a) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_mm_mul_ps(a, easysimd_mm_set1_ps(EASYSIMD_MATH_PI_OVER_180F));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vmulq_n_f32(a_.neon_i32, EASYSIMD_MATH_PI_OVER_180F);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_53784)
      r_.f32 = a_.f32 * EASYSIMD_MATH_PI_OVER_180F;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    const __typeof__(r_.f32) tmp = { EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F };
      r_.f32 = a_.f32 * tmp;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_deg2radf(a_.f32[i]);
      }

    #endif
    return easysimd__m128_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_deg2rad_pd(easysimd__m128d a) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
      return easysimd_mm_mul_pd(a, easysimd_mm_set1_pd(EASYSIMD_MATH_PI_OVER_180));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vmulq_n_f64(a_.neon_i64, EASYSIMD_MATH_PI_OVER_180);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_53784)
    r_.f64 = a_.f64 * EASYSIMD_MATH_PI_OVER_180;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    const __typeof__(r_.f64) tmp = { EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180 };
      r_.f64 = a_.f64 * tmp;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_deg2rad(a_.f64[i]);
      }

    #endif
    return easysimd__m128d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_deg2rad_ps(easysimd__m256 a) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE_GE(256)
    return easysimd_mm256_mul_ps(a, easysimd_mm256_set1_ps(EASYSIMD_MATH_PI_OVER_180F));
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_x_mm_deg2rad_ps(a_.m128[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_53784)
    r_.f32 = a_.f32 * EASYSIMD_MATH_PI_OVER_180F;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    const __typeof__(r_.f32) tmp = {
        EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F,
        EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F
      };
      r_.f32 = a_.f32 * tmp;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_deg2radf(a_.f32[i]);
      }

    #endif
    return easysimd__m256_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_deg2rad_pd(easysimd__m256d a) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE_GE(256)
    return easysimd_mm256_mul_pd(a, easysimd_mm256_set1_pd(EASYSIMD_MATH_PI_OVER_180));
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_x_mm_deg2rad_pd(a_.m128d[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_53784)
    r_.f64 = a_.f64 * EASYSIMD_MATH_PI_OVER_180;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    const __typeof__(r_.f64) tmp = { EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180 };
      r_.f64 = a_.f64 * tmp;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_deg2rad(a_.f64[i]);
      }

    #endif
    return easysimd__m256d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_x_mm512_deg2rad_ps(easysimd__m512 a) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE_GE(512)
      return easysimd_mm512_mul_ps(a, easysimd_mm512_set1_ps(EASYSIMD_MATH_PI_OVER_180F));
  #else
    easysimd__m512_private
        r_,
        a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_x_mm256_deg2rad_ps(a_.m256[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_53784)
    r_.f32 = a_.f32 * EASYSIMD_MATH_PI_OVER_180F;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    const __typeof__(r_.f32) tmp = {
        EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F,
        EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F,
        EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F,
        EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F, EASYSIMD_MATH_PI_OVER_180F
      };
      r_.f32 = a_.f32 * tmp;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_deg2radf(a_.f32[i]);
      }

    #endif
    return easysimd__m512_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_x_mm512_deg2rad_pd(easysimd__m512d a) {
  #if EASYSIMD_NATURAL_VECTOR_SIZE_GE(512)
      return easysimd_mm512_mul_pd(a, easysimd_mm512_set1_pd(EASYSIMD_MATH_PI_OVER_180));
  #else
    easysimd__m512d_private
        r_,
        a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_x_mm256_deg2rad_pd(a_.m256d[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_53784)
    r_.f64 = a_.f64 * EASYSIMD_MATH_PI_OVER_180;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    const __typeof__(r_.f64) tmp = {
        EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180,
        EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180, EASYSIMD_MATH_PI_OVER_180
      };
      r_.f64 = a_.f64 * tmp;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_deg2rad(a_.f64[i]);
      }

    #endif
    return easysimd__m512d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cosd_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cosd_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosf4_u10(easysimd_x_mm_deg2rad_ps(a));
    #else
      return Sleef_cosf4_u35(easysimd_x_mm_deg2rad_ps(a));
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_cosf(easysimd_math_deg2radf(a_.f32[i]));
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cosd_ps
  #define _mm_cosd_ps(a) easysimd_mm_cosd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cosd_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cosd_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosd2_u10(easysimd_x_mm_deg2rad_pd(a));
    #else
      return Sleef_cosd2_u35(easysimd_x_mm_deg2rad_pd(a));
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_cos(easysimd_math_deg2rad(a_.f64[i]));
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cosd_pd
  #define _mm_cosd_pd(a) easysimd_mm_cosd_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cosd_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cosd_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosf8_u10(easysimd_x_mm256_deg2rad_ps(a));
    #else
      return Sleef_cosf8_u35(easysimd_x_mm256_deg2rad_ps(a));
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_cosd_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cosf(easysimd_math_deg2radf(a_.f32[i]));
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cosd_ps
  #define _mm256_cosd_ps(a) easysimd_mm256_cosd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cosd_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cosd_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosd4_u10(easysimd_x_mm256_deg2rad_pd(a));
    #else
      return Sleef_cosd4_u35(easysimd_x_mm256_deg2rad_pd(a));
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_cosd_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cos(easysimd_math_deg2rad(a_.f64[i]));
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cosd_pd
  #define _mm256_cosd_pd(a) easysimd_mm256_cosd_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cosd_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cosd_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosf16_u10(easysimd_x_mm512_deg2rad_ps(a));
    #else
      return Sleef_cosf16_u35(easysimd_x_mm512_deg2rad_ps(a));
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_cosd_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cosf(easysimd_math_deg2radf(a_.f32[i]));
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cosd_ps
  #define _mm512_cosd_ps(a) easysimd_mm512_cosd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cosd_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cosd_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_cosd8_u10(easysimd_x_mm512_deg2rad_pd(a));
    #else
      return Sleef_cosd8_u35(easysimd_x_mm512_deg2rad_pd(a));
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_cosd_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cos(easysimd_math_deg2rad(a_.f64[i]));
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cosd_pd
  #define _mm512_cosd_pd(a) easysimd_mm512_cosd_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_cosd_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cosd_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_cosd_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cosd_ps
  #define _mm512_mask_cosd_ps(src, k, a) easysimd_mm512_mask_cosd_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_cosd_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cosd_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_cosd_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cosd_pd
  #define _mm512_mask_cosd_pd(src, k, a) easysimd_mm512_mask_cosd_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cosh_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cosh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_coshf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_coshf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cosh_ps
  #define _mm_cosh_ps(a) easysimd_mm_cosh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cosh_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cosh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_coshd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_cosh(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cosh_pd
  #define _mm_cosh_pd(a) easysimd_mm_cosh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cosh_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cosh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_coshf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_cosh_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_coshf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cosh_ps
  #define _mm256_cosh_ps(a) easysimd_mm256_cosh_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cosh_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cosh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_coshd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_cosh_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cosh(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cosh_pd
  #define _mm256_cosh_pd(a) easysimd_mm256_cosh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cosh_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cosh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_coshf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_cosh_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_coshf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cosh_ps
  #define _mm512_cosh_ps(a) easysimd_mm512_cosh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cosh_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cosh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_coshd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_cosh_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cosh(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cosh_pd
  #define _mm512_cosh_pd(a) easysimd_mm512_cosh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_cosh_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cosh_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_cosh_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cosh_ps
  #define _mm512_mask_cosh_ps(src, k, a) easysimd_mm512_mask_cosh_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_cosh_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cosh_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_cosh_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cosh_pd
  #define _mm512_mask_cosh_pd(src, k, a) easysimd_mm512_mask_cosh_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svint32_t sva, svb;
    svbool_t pg = svptrue_b32();
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]), svdiv_s32_x(pg, sva, svb));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = a_.i8 / b_.i8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] / b_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epi8
  #define _mm_div_epi8(a, b) easysimd_mm_div_epi8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    svint32_t sva, svb;
    sva = svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svb = svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svb = svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]), svdiv_s32_x(pg, sva, svb));

    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 / b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] / b_.i16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epi16
  #define _mm_div_epi16(a, b) easysimd_mm_div_epi16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdiv_s32_x(svptrue_b32(), a.sve_i32, b.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 / b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] / b_.i32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_mm_idiv_epi32(a, b) easysimd_mm_div_epi32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epi32
  #define _mm_div_epi32(a, b) easysimd_mm_div_epi32(a, b)
  #undef _mm_idiv_epi32
  #define _mm_idiv_epi32(a, b) easysimd_mm_div_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdiv_s64_x(svptrue_b64(), a.sve_i64, b.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 / b_.i64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] / b_.i64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epi64
  #define _mm_div_epi64(a, b) easysimd_mm_div_epi64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    svuint32_t sva, svb;
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]), svdiv_u32_x(pg, sva, svb));

    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u8 = a_.u8 / b_.u8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = a_.u8[i] / b_.u8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epu8
  #define _mm_div_epu8(a, b) easysimd_mm_div_epu8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    svuint32_t sva, svb;
    sva = svld1uh_u32(pg, &(a.u16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svb = svld1uh_u32(pg, &(b.u16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svst1h_u32(pg, &(r.u16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1uh_u32(pg, &(a.u16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svb = svld1uh_u32(pg, &(b.u16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svst1h_u32(pg, &(r.u16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]), svdiv_u32_x(pg, sva, svb));

    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u16 = a_.u16 / b_.u16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = a_.u16[i] / b_.u16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epu16
  #define _mm_div_epu16(a, b) easysimd_mm_div_epu16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epu32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u32 = svdiv_u32_x(svptrue_b32(), a.sve_u32, b.sve_u32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = a_.u32 / b_.u32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] / b_.u32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_mm_udiv_epi32(a, b) easysimd_mm_div_epu32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epu32
  #define _mm_div_epu32(a, b) easysimd_mm_div_epu32(a, b)
  #undef _mm_udiv_epi32
  #define _mm_udiv_epi32(a, b) easysimd_mm_div_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_div_epu64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_epu64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svdiv_u64_x(svptrue_b64(), a.sve_u64, b.sve_u64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u64 = a_.u64 / b_.u64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] / b_.u64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_div_epu64
  #define _mm_div_epu64(a, b) easysimd_mm_div_epu64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    svint32_t sva, svb;
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]), svdiv_s32_x(pg, sva, svb));

    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(1, 0)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(1, 0)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(1, 0)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(1, 1)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(1, 1)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(1, 1)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(1, 2)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(1, 2)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(1, 2)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sb_s32(pg, &(a.i8[EASYSIMD_SV_B8_B32_INDEX(1, 3)]));
    svb = svld1sb_s32(pg, &(b.i8[EASYSIMD_SV_B8_B32_INDEX(1, 3)]));
    svst1b_s32(pg, &(r.i8[EASYSIMD_SV_B8_B32_INDEX(1, 3)]), svdiv_s32_x(pg, sva, svb));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined (EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = a_.i8 / b_.i8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epi8(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
          r_.i8[i] = a_.i8[i] / b_.i8[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epi8
  #define _mm256_div_epi8(a, b) easysimd_mm256_div_epi8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    svint32_t sva, svb;
    sva = svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svb = svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svb = svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]), svdiv_s32_x(pg, sva, svb));

    sva = svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(1, 0)]));
    svb = svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(1, 0)]));
    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(1, 0)]), svdiv_s32_x(pg, sva, svb));
    sva = svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(1, 1)]));
    svb = svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(1, 1)]));
    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(1, 1)]), svdiv_s32_x(pg, sva, svb));

    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 / b_.i16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epi16(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = a_.i16[i] / b_.i16[i];
        }
       #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epi16
  #define _mm256_div_epi16(a, b) easysimd_mm256_div_epi16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdiv_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdiv_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res = easysimd_mm256_setzero_si256();
    int32_t ptr_a[8], ptr_b[8], ptr_r[8];
    easysimd_mm256_convert_to_int32(ptr_a, a);
    easysimd_mm256_convert_to_int32(ptr_b, b);
    ptr_r[0] = ptr_a[0] / ptr_b[0];
    ptr_r[1] = ptr_a[1] / ptr_b[1];
    ptr_r[2] = ptr_a[2] / ptr_b[2];
    ptr_r[3] = ptr_a[3] / ptr_b[3];
    ptr_r[4] = ptr_a[4] / ptr_b[4];
    ptr_r[5] = ptr_a[5] / ptr_b[5];
    ptr_r[6] = ptr_a[6] / ptr_b[6];
    ptr_r[7] = ptr_a[7] / ptr_b[7];
    res.m128i[0].neon_i32 = vsetq_lane_s32(ptr_r[0], res.m128i[0].neon_i32, 0);
    res.m128i[0].neon_i32 = vsetq_lane_s32(ptr_r[1], res.m128i[0].neon_i32, 1);
    res.m128i[0].neon_i32 = vsetq_lane_s32(ptr_r[2], res.m128i[0].neon_i32, 2);
    res.m128i[0].neon_i32 = vsetq_lane_s32(ptr_r[3], res.m128i[0].neon_i32, 3);
    res.m128i[1].neon_i32 = vsetq_lane_s32(ptr_r[4], res.m128i[1].neon_i32, 0);
    res.m128i[1].neon_i32 = vsetq_lane_s32(ptr_r[5], res.m128i[1].neon_i32, 1);
    res.m128i[1].neon_i32 = vsetq_lane_s32(ptr_r[6], res.m128i[1].neon_i32, 2);
    res.m128i[1].neon_i32 = vsetq_lane_s32(ptr_r[7], res.m128i[1].neon_i32, 3);
    return res;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 / b_.i32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epi32(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = a_.i32[i] / b_.i32[i];
        }
       #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#define easysimd_mm256_idiv_epi32(a, b) easysimd_mm256_div_epi32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epi32
  #define _mm256_div_epi32(a, b) easysimd_mm256_div_epi32(a, b)
  #undef _mm256_idiv_epi32
  #define _mm256_idiv_epi32(a, b) easysimd_mm256_div_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdiv_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdiv_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res = easysimd_mm256_setzero_si256();
    int64_t ptr_a[4], ptr_b[4], ptr_r[4];
    easysimd_mm256_convert_to_int64(ptr_a, a);
    easysimd_mm256_convert_to_int64(ptr_b, b);
    ptr_r[0] = ptr_a[0] / ptr_b[0];
    ptr_r[1] = ptr_a[1] / ptr_b[1];
    ptr_r[2] = ptr_a[2] / ptr_b[2];
    ptr_r[3] = ptr_a[3] / ptr_b[3];
    res.m128i[0].neon_i64 = vsetq_lane_s64(ptr_r[0], res.m128i[0].neon_i64, 0);
    res.m128i[0].neon_i64 = vsetq_lane_s64(ptr_r[1], res.m128i[0].neon_i64, 1);
    res.m128i[1].neon_i64 = vsetq_lane_s64(ptr_r[2], res.m128i[1].neon_i64, 0);
    res.m128i[1].neon_i64 = vsetq_lane_s64(ptr_r[3], res.m128i[1].neon_i64, 1);
    return res;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 / b_.i64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epi64(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.i64[i] = a_.i64[i] / b_.i64[i];
        }
        #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epi64
  #define _mm256_div_epi64(a, b) easysimd_mm256_div_epi64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    svuint32_t sva, svb;
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 0)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 1)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 2)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(0, 3)]), svdiv_u32_x(pg, sva, svb));

    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(1, 0)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(1, 0)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(1, 0)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(1, 1)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(1, 1)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(1, 1)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(1, 2)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(1, 2)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(1, 2)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1ub_u32(pg, &(a.u8[EASYSIMD_SV_B8_B32_INDEX(1, 3)]));
    svb = svld1ub_u32(pg, &(b.u8[EASYSIMD_SV_B8_B32_INDEX(1, 3)]));
    svst1b_u32(pg, &(r.u8[EASYSIMD_SV_B8_B32_INDEX(1, 3)]), svdiv_u32_x(pg, sva, svb));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u8 = a_.u8 / b_.u8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epu8(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
          r_.u8[i] = a_.u8[i] / b_.u8[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epu8
  #define _mm256_div_epu8(a, b) easysimd_mm256_div_epu8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epu16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    svuint32_t sva, svb;
    sva = svld1uh_u32(pg, &(a.u16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svb = svld1uh_u32(pg, &(b.u16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]));
    svst1h_u32(pg, &(r.u16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1uh_u32(pg, &(a.u16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svb = svld1uh_u32(pg, &(b.u16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]));
    svst1h_u32(pg, &(r.u16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]), svdiv_u32_x(pg, sva, svb));

    sva = svld1uh_u32(pg, &(a.u16[EASYSIMD_SV_B16_B32_INDEX(1, 0)]));
    svb = svld1uh_u32(pg, &(b.u16[EASYSIMD_SV_B16_B32_INDEX(1, 0)]));
    svst1h_u32(pg, &(r.u16[EASYSIMD_SV_B16_B32_INDEX(1, 0)]), svdiv_u32_x(pg, sva, svb));
    sva = svld1uh_u32(pg, &(a.u16[EASYSIMD_SV_B16_B32_INDEX(1, 1)]));
    svb = svld1uh_u32(pg, &(b.u16[EASYSIMD_SV_B16_B32_INDEX(1, 1)]));
    svst1h_u32(pg, &(r.u16[EASYSIMD_SV_B16_B32_INDEX(1, 1)]), svdiv_u32_x(pg, sva, svb));

    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u16 = a_.u16 / b_.u16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epu16(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
          r_.u16[i] = a_.u16[i] / b_.u16[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epu16
  #define _mm256_div_epu16(a, b) easysimd_mm256_div_epu16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epu32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svdiv_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svdiv_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res = easysimd_mm256_setzero_si256();
    uint32_t ptr_a[8], ptr_b[8], ptr_r[8];
    easysimd_mm256_convert_to_uint32(ptr_a, a);
    easysimd_mm256_convert_to_uint32(ptr_b, b);
    ptr_r[0] = ptr_a[0] / ptr_b[0];
    ptr_r[1] = ptr_a[1] / ptr_b[1];
    ptr_r[2] = ptr_a[2] / ptr_b[2];
    ptr_r[3] = ptr_a[3] / ptr_b[3];
    ptr_r[4] = ptr_a[4] / ptr_b[4];
    ptr_r[5] = ptr_a[5] / ptr_b[5];
    ptr_r[6] = ptr_a[6] / ptr_b[6];
    ptr_r[7] = ptr_a[7] / ptr_b[7];
    res.m128i[0].neon_u32 = vsetq_lane_u32(ptr_r[0], res.m128i[0].neon_u32, 0);
    res.m128i[0].neon_u32 = vsetq_lane_u32(ptr_r[1], res.m128i[0].neon_u32, 1);
    res.m128i[0].neon_u32 = vsetq_lane_u32(ptr_r[2], res.m128i[0].neon_u32, 2);
    res.m128i[0].neon_u32 = vsetq_lane_u32(ptr_r[3], res.m128i[0].neon_u32, 3);
    res.m128i[1].neon_u32 = vsetq_lane_u32(ptr_r[4], res.m128i[1].neon_u32, 0);
    res.m128i[1].neon_u32 = vsetq_lane_u32(ptr_r[5], res.m128i[1].neon_u32, 1);
    res.m128i[1].neon_u32 = vsetq_lane_u32(ptr_r[6], res.m128i[1].neon_u32, 2);
    res.m128i[1].neon_u32 = vsetq_lane_u32(ptr_r[7], res.m128i[1].neon_u32, 3);
    return res;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = a_.u32 / b_.u32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epu32(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = a_.u32[i] / b_.u32[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#define easysimd_mm256_udiv_epi32(a, b) easysimd_mm256_div_epu32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epu32
  #define _mm256_div_epu32(a, b) easysimd_mm256_div_epu32(a, b)
  #undef _mm256_udiv_epi32
  #define _mm256_udiv_epi32(a, b) easysimd_mm256_div_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_div_epu64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_epu64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svdiv_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    svuint64_t svr = svdiv_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svr;
//    r.sve_u64[EASYSIMD_SV_INDEX_1] = svdiv_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res = easysimd_mm256_setzero_si256();
    uint64_t ptr_a[4], ptr_b[4], ptr_r[4];
    easysimd_mm256_convert_to_uint64(ptr_a, a);
    easysimd_mm256_convert_to_uint64(ptr_b, b);
    ptr_r[0] = ptr_a[0] / ptr_b[0];
    ptr_r[1] = ptr_a[1] / ptr_b[1];
    ptr_r[2] = ptr_a[2] / ptr_b[2];
    ptr_r[3] = ptr_a[3] / ptr_b[3];
    res.m128i[0].neon_u64 = vsetq_lane_u64(ptr_r[0], res.m128i[0].neon_u64, 0);
    res.m128i[0].neon_u64 = vsetq_lane_u64(ptr_r[1], res.m128i[0].neon_u64, 1);
    res.m128i[1].neon_u64 = vsetq_lane_u64(ptr_r[2], res.m128i[1].neon_u64, 0);
    res.m128i[1].neon_u64 = vsetq_lane_u64(ptr_r[3], res.m128i[1].neon_u64, 1);
    return res;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u64 = a_.u64 / b_.u64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_div_epu64(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = a_.u64[i] / b_.u64[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_epu64
  #define _mm256_div_epu64(a, b) easysimd_mm256_div_epu64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();

    sveint32_t
      r0 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[4 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[4 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r1 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r2 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_0 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_0 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r3 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_0 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_0 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r4 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[4 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[4 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r5 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r6 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_1 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_1 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r7 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_1 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_1 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r8 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[4 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[4 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r9 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r10 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_2 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_2 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r11 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_2 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_2 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r12 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[4 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[4 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r13 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r14 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_3 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_3 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r15 = svdiv_s32_z(pg, svld1sb_s32(pg, &(a.i8[(4 * EASYSIMD_SV_INDEX_3 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sb_s32(pg, &(b.i8[(4 * EASYSIMD_SV_INDEX_3 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])));

    svst1b_s32(pg, &(r.i8[4 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]), r0);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r1);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_0 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r2);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_0 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r3);
    svst1b_s32(pg, &(r.i8[4 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]), r4);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r5);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_1 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r6);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_1 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r7);
    svst1b_s32(pg, &(r.i8[4 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]), r8);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r9);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_2 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r10);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_2 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r11);
    svst1b_s32(pg, &(r.i8[4 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]), r12);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r13);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_3 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r14);
    svst1b_s32(pg, &(r.i8[(4 * EASYSIMD_SV_INDEX_3 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r15);

    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0] = easysimd_mm_div_epi8(a.m128i[0], b.m128i[0]);
    r.m128i[1] = easysimd_mm_div_epi8(a.m128i[1], b.m128i[1]);
    r.m128i[2] = easysimd_mm_div_epi8(a.m128i[2], b.m128i[2]);
    r.m128i[3] = easysimd_mm_div_epi8(a.m128i[3], b.m128i[3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = a_.i8 / b_.i8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epi8(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
          r_.i8[i] = a_.i8[i] / b_.i8[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epi8
  #define _mm512_div_epi8(a, b) easysimd_mm512_div_epi8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();

    sveint32_t
      r0 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r1 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r2 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r3 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r4 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r5 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r6 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r7 = svdiv_s32_z(pg, svld1sh_s32(pg, &(a.i16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1sh_s32(pg, &(b.i16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])));

    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]), r0);
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r1);
    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]), r2);
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r3);
    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]), r4);
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r5);
    svst1h_s32(pg, &(r.i16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]), r6);
    svst1h_s32(pg, &(r.i16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r7);

    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0] = easysimd_mm_div_epi16(a.m128i[0], b.m128i[0]);
    r.m128i[1] = easysimd_mm_div_epi16(a.m128i[1], b.m128i[1]);
    r.m128i[2] = easysimd_mm_div_epi16(a.m128i[2], b.m128i[2]);
    r.m128i[3] = easysimd_mm_div_epi16(a.m128i[3], b.m128i[3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 / b_.i16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epi16(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = a_.i16[i] / b_.i16[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epi16
  #define _mm512_div_epi16(a, b) easysimd_mm512_div_epi16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdiv_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdiv_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svdiv_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svdiv_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m256i[0] = easysimd_mm256_div_epi32(a.m256i[0], b.m256i[0]);
    r.m256i[1] = easysimd_mm256_div_epi32(a.m256i[1], b.m256i[1]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 / b_.i32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epi32(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = a_.i32[i] / b_.i32[i];
        }
        #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epi32
  #define _mm512_div_epi32(a, b) easysimd_mm512_div_epi32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_div_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_div_epi32(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_div_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_div_epi32
  #define _mm512_mask_div_epi32(src, k, a, b) easysimd_mm512_mask_div_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epi64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdiv_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdiv_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svdiv_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svdiv_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m256i[0] = easysimd_mm256_div_epi64(a.m256i[0], b.m256i[0]);
    r.m256i[1] = easysimd_mm256_div_epi64(a.m256i[1], b.m256i[1]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 / b_.i64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epi64(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.i64[i] = a_.i64[i] / b_.i64[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epi64
  #define _mm512_div_epi64(a, b) easysimd_mm512_div_epi64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epu8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();

    sveuint32_t
      r0 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[4 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[4 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r1 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r2 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_0 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_0 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r3 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_0 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_0 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r4 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[4 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[4 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r5 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r6 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_1 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_1 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r7 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_1 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_1 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r8 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[4 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[4 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r9 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r10 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_2 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_2 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r11 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_2 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_2 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r12 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[4 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[4 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r13 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r14 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_3 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_3 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r15 = svdiv_u32_z(pg, svld1ub_u32(pg, &(a.u8[(4 * EASYSIMD_SV_INDEX_3 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1ub_u32(pg, &(b.u8[(4 * EASYSIMD_SV_INDEX_3 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)])));

    svst1b_u32(pg, &(r.u8[4 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]), r0);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r1);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_0 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r2);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_0 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r3);
    svst1b_u32(pg, &(r.u8[4 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]), r4);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r5);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_1 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r6);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_1 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r7);
    svst1b_u32(pg, &(r.u8[4 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]), r8);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r9);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_2 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r10);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_2 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r11);
    svst1b_u32(pg, &(r.u8[4 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]), r12);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r13);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_3 + 2) * (__ARM_FEATURE_SVE_BITS >> 5)]), r14);
    svst1b_u32(pg, &(r.u8[(4 * EASYSIMD_SV_INDEX_3 + 3) * (__ARM_FEATURE_SVE_BITS >> 5)]), r15);

    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0] = easysimd_mm_div_epu8(a.m128i[0], b.m128i[0]);
    r.m128i[1] = easysimd_mm_div_epu8(a.m128i[1], b.m128i[1]);
    r.m128i[2] = easysimd_mm_div_epu8(a.m128i[2], b.m128i[2]);
    r.m128i[3] = easysimd_mm_div_epu8(a.m128i[3], b.m128i[3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u8 = a_.u8 / b_.u8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epu8(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
          r_.u8[i] = a_.u8[i] / b_.u8[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epu8
  #define _mm512_div_epu8(a, b) easysimd_mm512_div_epu8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epu16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();

    sveuint32_t
      r0 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r1 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r2 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r3 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r4 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r5 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r6 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]))),
      r7 = svdiv_u32_z(pg, svld1uh_u32(pg, &(a.u16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])), svld1uh_u32(pg, &(b.u16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)])));

    svst1h_u32(pg, &(r.u16[2 * EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]), r0);
    svst1h_u32(pg, &(r.u16[(2 * EASYSIMD_SV_INDEX_0 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r1);
    svst1h_u32(pg, &(r.u16[2 * EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]), r2);
    svst1h_u32(pg, &(r.u16[(2 * EASYSIMD_SV_INDEX_1 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r3);
    svst1h_u32(pg, &(r.u16[2 * EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)]), r4);
    svst1h_u32(pg, &(r.u16[(2 * EASYSIMD_SV_INDEX_2 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r5);
    svst1h_u32(pg, &(r.u16[2 * EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)]), r6);
    svst1h_u32(pg, &(r.u16[(2 * EASYSIMD_SV_INDEX_3 + 1) * (__ARM_FEATURE_SVE_BITS >> 5)]), r7);

    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0] = easysimd_mm_div_epu16(a.m128i[0], b.m128i[0]);
    r.m128i[1] = easysimd_mm_div_epu16(a.m128i[1], b.m128i[1]);
    r.m128i[2] = easysimd_mm_div_epu16(a.m128i[2], b.m128i[2]);
    r.m128i[3] = easysimd_mm_div_epu16(a.m128i[3], b.m128i[3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u16 = a_.u16 / b_.u16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epu16(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
          r_.u16[i] = a_.u16[i] / b_.u16[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epu16
  #define _mm512_div_epu16(a, b) easysimd_mm512_div_epu16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epu32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svdiv_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svdiv_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svdiv_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svdiv_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m256i[0] = easysimd_mm256_div_epu32(a.m256i[0], b.m256i[0]);
    r.m256i[1] = easysimd_mm256_div_epu32(a.m256i[1], b.m256i[1]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = a_.u32 / b_.u32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epu32(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = a_.u32[i] / b_.u32[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epu32
  #define _mm512_div_epu32(a, b) easysimd_mm512_div_epu32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_div_epu32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_div_epu32(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_div_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_div_epu32
  #define _mm512_mask_div_epu32(src, k, a, b) easysimd_mm512_mask_div_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_div_epu64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_epu64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svdiv_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svdiv_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svdiv_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svdiv_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m256i[0] = easysimd_mm256_div_epu64(a.m256i[0], b.m256i[0]);
    r.m256i[1] = easysimd_mm256_div_epu64(a.m256i[1], b.m256i[1]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u64 = a_.u64 / b_.u64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_div_epu64(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = a_.u64[i] / b_.u64[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_epu64
  #define _mm512_div_epu64(a, b) easysimd_mm512_div_epu64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_erf_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erf_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_erff4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_erff(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erf_ps
  #define _mm_erf_ps(a) easysimd_mm_erf_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_erf_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erf_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_erfd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_erf(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erf_pd
  #define _mm_erf_pd(a) easysimd_mm_erf_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_erf_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erf_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_erff8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_erf_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_erff(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erf_ps
  #define _mm256_erf_ps(a) easysimd_mm256_erf_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_erf_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erf_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_erfd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_erf_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_erf(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erf_pd
  #define _mm256_erf_pd(a) easysimd_mm256_erf_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_erf_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erf_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_erff16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_erf_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_erff(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erf_ps
  #define _mm512_erf_ps(a) easysimd_mm512_erf_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_erf_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erf_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_erfd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_erf_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_erf(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erf_pd
  #define _mm512_erf_pd(a) easysimd_mm512_erf_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_erf_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erf_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_erf_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erf_ps
  #define _mm512_mask_erf_ps(src, k, a) easysimd_mm512_mask_erf_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_erf_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erf_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_erf_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erf_pd
  #define _mm512_mask_erf_pd(src, k, a) easysimd_mm512_mask_erf_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_erfc_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erfc_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_erfcf4_u15(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_erfcf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erfc_ps
  #define _mm_erfc_ps(a) easysimd_mm_erfc_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_erfc_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erfc_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_erfcd2_u15(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_erfc(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erfc_pd
  #define _mm_erfc_pd(a) easysimd_mm_erfc_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_erfc_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erfc_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_erfcf8_u15(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_erfc_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_erfcf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erfc_ps
  #define _mm256_erfc_ps(a) easysimd_mm256_erfc_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_erfc_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erfc_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_erfcd4_u15(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_erfc_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_erfc(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erfc_pd
  #define _mm256_erfc_pd(a) easysimd_mm256_erfc_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_erfc_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erfc_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_erfcf16_u15(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_erfc_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_erfcf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erfc_ps
  #define _mm512_erfc_ps(a) easysimd_mm512_erfc_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_erfc_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erfc_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_erfcd8_u15(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_erfc_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_erfc(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erfc_pd
  #define _mm512_erfc_pd(a) easysimd_mm512_erfc_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_erfc_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erfc_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_erfc_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erfc_ps
  #define _mm512_mask_erfc_ps(src, k, a) easysimd_mm512_mask_erfc_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_erfc_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erfc_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_erfc_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erfc_pd
  #define _mm512_mask_erfc_pd(src, k, a) easysimd_mm512_mask_erfc_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_exp_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_exp_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_expf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_expf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_exp_ps
  #define _mm_exp_ps(a) easysimd_mm_exp_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_exp_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_exp_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_expd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_exp(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_exp_pd
  #define _mm_exp_pd(a) easysimd_mm_exp_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_exp_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_exp_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_expf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_exp_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_expf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_exp_ps
  #define _mm256_exp_ps(a) easysimd_mm256_exp_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_exp_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_exp_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_expd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_exp_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_exp(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_exp_pd
  #define _mm256_exp_pd(a) easysimd_mm256_exp_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_exp_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_exp_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_expf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_exp_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_expf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_exp_ps
  #define _mm512_exp_ps(a) easysimd_mm512_exp_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_exp_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_exp_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_expd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_exp_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_exp(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_exp_pd
  #define _mm512_exp_pd(a) easysimd_mm512_exp_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_exp_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_exp_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_exp_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_exp_ps
  #define _mm512_mask_exp_ps(src, k, a) easysimd_mm512_mask_exp_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_exp_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_exp_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_exp_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_exp_pd
  #define _mm512_mask_exp_pd(src, k, a) easysimd_mm512_mask_exp_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_expm1_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_expm1_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_expm1f4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_expm1f(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_expm1_ps
  #define _mm_expm1_ps(a) easysimd_mm_expm1_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_expm1_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_expm1_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_expm1d2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_expm1(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_expm1_pd
  #define _mm_expm1_pd(a) easysimd_mm_expm1_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_expm1_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_expm1_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_expm1f8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_expm1_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_expm1f(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_expm1_ps
  #define _mm256_expm1_ps(a) easysimd_mm256_expm1_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_expm1_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_expm1_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_expm1d4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_expm1_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_expm1(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_expm1_pd
  #define _mm256_expm1_pd(a) easysimd_mm256_expm1_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_expm1_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_expm1_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_expm1f16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_expm1_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_expm1f(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_expm1_ps
  #define _mm512_expm1_ps(a) easysimd_mm512_expm1_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_expm1_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_expm1_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_expm1d8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_expm1_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_expm1(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_expm1_pd
  #define _mm512_expm1_pd(a) easysimd_mm512_expm1_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_expm1_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_expm1_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_expm1_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_expm1_ps
  #define _mm512_mask_expm1_ps(src, k, a) easysimd_mm512_mask_expm1_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_expm1_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_expm1_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_expm1_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_expm1_pd
  #define _mm512_mask_expm1_pd(src, k, a) easysimd_mm512_mask_expm1_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_exp2_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_exp2_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_exp2f4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_exp2f(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_exp2_ps
  #define _mm_exp2_ps(a) easysimd_mm_exp2_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_exp2_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_exp2_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_exp2d2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_exp2(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_exp2_pd
  #define _mm_exp2_pd(a) easysimd_mm_exp2_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_exp2_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_exp2_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_exp2f8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_exp2_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_exp2f(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_exp2_ps
  #define _mm256_exp2_ps(a) easysimd_mm256_exp2_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_exp2_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_exp2_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_exp2d4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_exp2_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_exp2(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_exp2_pd
  #define _mm256_exp2_pd(a) easysimd_mm256_exp2_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_exp2_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_exp2_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_exp2f16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_exp2_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_exp2f(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_exp2_ps
  #define _mm512_exp2_ps(a) easysimd_mm512_exp2_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_exp2_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_exp2_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_exp2d8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_exp2_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_exp2(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_exp2_pd
  #define _mm512_exp2_pd(a) easysimd_mm512_exp2_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_exp2_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_exp2_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_exp2_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_exp2_ps
  #define _mm512_mask_exp2_ps(src, k, a) easysimd_mm512_mask_exp2_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_exp2_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_exp2_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_exp2_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_exp2_pd
  #define _mm512_mask_exp2_pd(src, k, a) easysimd_mm512_mask_exp2_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_exp10_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_exp10_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_exp10f4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_exp10f(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_exp10_ps
  #define _mm_exp10_ps(a) easysimd_mm_exp10_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_exp10_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_exp10_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_exp10d2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_exp10(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_exp10_pd
  #define _mm_exp10_pd(a) easysimd_mm_exp10_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_exp10_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_exp10_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_exp10f8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_exp10_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_exp10f(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_exp10_ps
  #define _mm256_exp10_ps(a) easysimd_mm256_exp10_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_exp10_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_exp10_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_exp10d4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_exp10_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_exp10(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_exp10_pd
  #define _mm256_exp10_pd(a) easysimd_mm256_exp10_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_exp10_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_exp10_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_exp10f16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_exp10_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_exp10f(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_exp10_ps
  #define _mm512_exp10_ps(a) easysimd_mm512_exp10_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_exp10_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_exp10_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_exp10d8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_exp10_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_exp10(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_exp10_pd
  #define _mm512_exp10_pd(a) easysimd_mm512_exp10_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_exp10_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_exp10_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_exp10_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_exp10_ps
  #define _mm512_mask_exp10_ps(src, k, a) easysimd_mm512_mask_exp10_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_exp10_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_exp10_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_exp10_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_exp10_pd
  #define _mm512_mask_exp10_pd(src, k, a) easysimd_mm512_mask_exp10_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cdfnorm_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cdfnorm_ps(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    /* https://www.johndcook.com/blog/cpp_phi/ */
    const easysimd__m128 a1 = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.254829592));
    const easysimd__m128 a2 = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-0.284496736));
    const easysimd__m128 a3 = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.421413741));
    const easysimd__m128 a4 = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-1.453152027));
    const easysimd__m128 a5 = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.061405429));
    const easysimd__m128 p = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.3275911));
    const easysimd__m128 one = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0));

    /* easysimd_math_fabsf(x) / sqrtf(2.0) */
    const easysimd__m128 x = easysimd_mm_div_ps(easysimd_x_mm_abs_ps(a), easysimd_mm_sqrt_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(2.0))));

    /* 1.0 / (1.0 + p * x) */
    const easysimd__m128 t = easysimd_mm_div_ps(one, easysimd_mm_add_ps(one, easysimd_mm_mul_ps(p, x)));

    /* 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x) */
    easysimd__m128 y = easysimd_mm_mul_ps(a5, t);
    y = easysimd_mm_add_ps(y, a4);
    y = easysimd_mm_mul_ps(y, t);
    y = easysimd_mm_add_ps(y, a3);
    y = easysimd_mm_mul_ps(y, t);
    y = easysimd_mm_add_ps(y, a2);
    y = easysimd_mm_mul_ps(y, t);
    y = easysimd_mm_add_ps(y, a1);
    y = easysimd_mm_mul_ps(y, t);
    y = easysimd_mm_mul_ps(y, easysimd_mm_exp_ps(easysimd_mm_mul_ps(x, easysimd_x_mm_negate_ps(x))));
    y = easysimd_mm_sub_ps(one, y);

    /* 0.5 * (1.0 + ((a < 0.0) ? -y : y)) */
    return easysimd_mm_mul_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.5)), easysimd_mm_add_ps(one, easysimd_x_mm_xorsign_ps(y, a)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_cdfnormf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cdfnorm_ps
  #define _mm_cdfnorm_ps(a) easysimd_mm_cdfnorm_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cdfnorm_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cdfnorm_pd(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    /* https://www.johndcook.com/blog/cpp_phi/ */
    const easysimd__m128d a1 = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.254829592));
    const easysimd__m128d a2 = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-0.284496736));
    const easysimd__m128d a3 = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.421413741));
    const easysimd__m128d a4 = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-1.453152027));
    const easysimd__m128d a5 = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.061405429));
    const easysimd__m128d p = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.6475911));
    const easysimd__m128d one = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0));

    /* easysimd_math_fabs(x) / sqrt(2.0) */
    const easysimd__m128d x = easysimd_mm_div_pd(easysimd_x_mm_abs_pd(a), easysimd_mm_sqrt_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(2.0))));

    /* 1.0 / (1.0 + p * x) */
    const easysimd__m128d t = easysimd_mm_div_pd(one, easysimd_mm_add_pd(one, easysimd_mm_mul_pd(p, x)));

    /* 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x) */
    easysimd__m128d y = easysimd_mm_mul_pd(a5, t);
    y = easysimd_mm_add_pd(y, a4);
    y = easysimd_mm_mul_pd(y, t);
    y = easysimd_mm_add_pd(y, a3);
    y = easysimd_mm_mul_pd(y, t);
    y = easysimd_mm_add_pd(y, a2);
    y = easysimd_mm_mul_pd(y, t);
    y = easysimd_mm_add_pd(y, a1);
    y = easysimd_mm_mul_pd(y, t);
    y = easysimd_mm_mul_pd(y, easysimd_mm_exp_pd(easysimd_mm_mul_pd(x, easysimd_x_mm_negate_pd(x))));
    y = easysimd_mm_sub_pd(one, y);

    /* 0.5 * (1.0 + ((a < 0.0) ? -y : y)) */
    return easysimd_mm_mul_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.5)), easysimd_mm_add_pd(one, easysimd_x_mm_xorsign_pd(y, a)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_cdfnorm(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cdfnorm_pd
  #define _mm_cdfnorm_pd(a) easysimd_mm_cdfnorm_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cdfnorm_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cdfnorm_ps(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    /* https://www.johndcook.com/blog/cpp_phi/ */
    const easysimd__m256 a1 = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.254829592));
    const easysimd__m256 a2 = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-0.284496736));
    const easysimd__m256 a3 = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.421413741));
    const easysimd__m256 a4 = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-1.453152027));
    const easysimd__m256 a5 = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.061405429));
    const easysimd__m256 p = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.3275911));
    const easysimd__m256 one = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0));

    /* easysimd_math_fabsf(x) / sqrtf(2.0) */
    const easysimd__m256 x = easysimd_mm256_div_ps(easysimd_x_mm256_abs_ps(a), easysimd_mm256_sqrt_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(2.0))));

    /* 1.0 / (1.0 + p * x) */
    const easysimd__m256 t = easysimd_mm256_div_ps(one, easysimd_mm256_add_ps(one, easysimd_mm256_mul_ps(p, x)));

    /* 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x) */
    easysimd__m256 y = easysimd_mm256_mul_ps(a5, t);
    y = easysimd_mm256_add_ps(y, a4);
    y = easysimd_mm256_mul_ps(y, t);
    y = easysimd_mm256_add_ps(y, a3);
    y = easysimd_mm256_mul_ps(y, t);
    y = easysimd_mm256_add_ps(y, a2);
    y = easysimd_mm256_mul_ps(y, t);
    y = easysimd_mm256_add_ps(y, a1);
    y = easysimd_mm256_mul_ps(y, t);
    y = easysimd_mm256_mul_ps(y, easysimd_mm256_exp_ps(easysimd_mm256_mul_ps(x, easysimd_x_mm256_negate_ps(x))));
    y = easysimd_mm256_sub_ps(one, y);

    /* 0.5 * (1.0 + ((a < 0.0) ? -y : y)) */
    return easysimd_mm256_mul_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.5)), easysimd_mm256_add_ps(one, easysimd_x_mm256_xorsign_ps(y, a)));
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_cdfnorm_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cdfnormf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cdfnorm_ps
  #define _mm256_cdfnorm_ps(a) easysimd_mm256_cdfnorm_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cdfnorm_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cdfnorm_pd(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    /* https://www.johndcook.com/blog/cpp_phi/ */
    const easysimd__m256d a1 = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.254829592));
    const easysimd__m256d a2 = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-0.284496736));
    const easysimd__m256d a3 = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.421413741));
    const easysimd__m256d a4 = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-1.453152027));
    const easysimd__m256d a5 = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.061405429));
    const easysimd__m256d p = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.6475911));
    const easysimd__m256d one = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0));

    /* easysimd_math_fabs(x) / sqrt(2.0) */
    const easysimd__m256d x = easysimd_mm256_div_pd(easysimd_x_mm256_abs_pd(a), easysimd_mm256_sqrt_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(2.0))));

    /* 1.0 / (1.0 + p * x) */
    const easysimd__m256d t = easysimd_mm256_div_pd(one, easysimd_mm256_add_pd(one, easysimd_mm256_mul_pd(p, x)));

    /* 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x) */
    easysimd__m256d y = easysimd_mm256_mul_pd(a5, t);
    y = easysimd_mm256_add_pd(y, a4);
    y = easysimd_mm256_mul_pd(y, t);
    y = easysimd_mm256_add_pd(y, a3);
    y = easysimd_mm256_mul_pd(y, t);
    y = easysimd_mm256_add_pd(y, a2);
    y = easysimd_mm256_mul_pd(y, t);
    y = easysimd_mm256_add_pd(y, a1);
    y = easysimd_mm256_mul_pd(y, t);
    y = easysimd_mm256_mul_pd(y, easysimd_mm256_exp_pd(easysimd_mm256_mul_pd(x, easysimd_x_mm256_negate_pd(x))));
    y = easysimd_mm256_sub_pd(one, y);

    /* 0.5 * (1.0 + ((a < 0.0) ? -y : y)) */
    return easysimd_mm256_mul_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.5)), easysimd_mm256_add_pd(one, easysimd_x_mm256_xorsign_pd(y, a)));
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_cdfnorm_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cdfnorm(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cdfnorm_pd
  #define _mm256_cdfnorm_pd(a) easysimd_mm256_cdfnorm_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cdfnorm_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cdfnorm_ps(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    /* https://www.johndcook.com/blog/cpp_phi/ */
    const easysimd__m512 a1 = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.254829592));
    const easysimd__m512 a2 = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-0.284496736));
    const easysimd__m512 a3 = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.421413741));
    const easysimd__m512 a4 = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-1.453152027));
    const easysimd__m512 a5 = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.061405429));
    const easysimd__m512 p = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.3275911));
    const easysimd__m512 one = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0));

    /* easysimd_math_fabsf(x) / sqrtf(2.0) */
    const easysimd__m512 x = easysimd_mm512_div_ps(easysimd_mm512_abs_ps(a), easysimd_mm512_sqrt_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(2.0))));

    /* 1.0 / (1.0 + p * x) */
    const easysimd__m512 t = easysimd_mm512_div_ps(one, easysimd_mm512_add_ps(one, easysimd_mm512_mul_ps(p, x)));

    /* 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x) */
    easysimd__m512 y = easysimd_mm512_mul_ps(a5, t);
    y = easysimd_mm512_add_ps(y, a4);
    y = easysimd_mm512_mul_ps(y, t);
    y = easysimd_mm512_add_ps(y, a3);
    y = easysimd_mm512_mul_ps(y, t);
    y = easysimd_mm512_add_ps(y, a2);
    y = easysimd_mm512_mul_ps(y, t);
    y = easysimd_mm512_add_ps(y, a1);
    y = easysimd_mm512_mul_ps(y, t);
    y = easysimd_mm512_mul_ps(y, easysimd_mm512_exp_ps(easysimd_mm512_mul_ps(x, easysimd_x_mm512_negate_ps(x))));
    y = easysimd_mm512_sub_ps(one, y);

    /* 0.5 * (1.0 + ((a < 0.0) ? -y : y)) */
    return easysimd_mm512_mul_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.5)), easysimd_mm512_add_ps(one, easysimd_x_mm512_xorsign_ps(y, a)));
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_cdfnorm_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cdfnormf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cdfnorm_ps
  #define _mm512_cdfnorm_ps(a) easysimd_mm512_cdfnorm_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cdfnorm_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cdfnorm_pd(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    /* https://www.johndcook.com/blog/cpp_phi/ */
    const easysimd__m512d a1 = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.254829592));
    const easysimd__m512d a2 = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-0.284496736));
    const easysimd__m512d a3 = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.421413741));
    const easysimd__m512d a4 = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-1.453152027));
    const easysimd__m512d a5 = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.061405429));
    const easysimd__m512d p = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.6475911));
    const easysimd__m512d one = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0));

    /* easysimd_math_fabs(x) / sqrt(2.0) */
    const easysimd__m512d x = easysimd_mm512_div_pd(easysimd_mm512_abs_pd(a), easysimd_mm512_sqrt_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(2.0))));

    /* 1.0 / (1.0 + p * x) */
    const easysimd__m512d t = easysimd_mm512_div_pd(one, easysimd_mm512_add_pd(one, easysimd_mm512_mul_pd(p, x)));

    /* 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x) */
    easysimd__m512d y = easysimd_mm512_mul_pd(a5, t);
    y = easysimd_mm512_add_pd(y, a4);
    y = easysimd_mm512_mul_pd(y, t);
    y = easysimd_mm512_add_pd(y, a3);
    y = easysimd_mm512_mul_pd(y, t);
    y = easysimd_mm512_add_pd(y, a2);
    y = easysimd_mm512_mul_pd(y, t);
    y = easysimd_mm512_add_pd(y, a1);
    y = easysimd_mm512_mul_pd(y, t);
    y = easysimd_mm512_mul_pd(y, easysimd_mm512_exp_pd(easysimd_mm512_mul_pd(x, easysimd_x_mm512_negate_pd(x))));
    y = easysimd_mm512_sub_pd(one, y);

    /* 0.5 * (1.0 + ((a < 0.0) ? -y : y)) */
    return easysimd_mm512_mul_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.5)), easysimd_mm512_add_pd(one, easysimd_x_mm512_xorsign_pd(y, a)));
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_cdfnorm_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cdfnorm(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cdfnorm_pd
  #define _mm512_cdfnorm_pd(a) easysimd_mm512_cdfnorm_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_cdfnorm_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cdfnorm_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_cdfnorm_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cdfnorm_ps
  #define _mm512_mask_cdfnorm_ps(src, k, a) easysimd_mm512_mask_cdfnorm_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_cdfnorm_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cdfnorm_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_cdfnorm_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cdfnorm_pd
  #define _mm512_mask_cdfnorm_pd(src, k, a) easysimd_mm512_mask_cdfnorm_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_idivrem_epi32 (easysimd__m128i* mem_addr, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_idivrem_epi32(HEDLEY_REINTERPRET_CAST(__m128i*, mem_addr), a, b);
  #else
    easysimd__m128i r;

    r = easysimd_mm_div_epi32(a, b);
    *mem_addr = easysimd_mm_sub_epi32(a, easysimd_mm_mullo_epi32(r, b));

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_idivrem_epi32
  #define _mm_idivrem_epi32(mem_addr, a, b) easysimd_mm_idivrem_epi32((mem_addr),(a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_idivrem_epi32 (easysimd__m256i* mem_addr, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_idivrem_epi32(HEDLEY_REINTERPRET_CAST(__m256i*, mem_addr), a, b);
  #else
    easysimd__m256i r;

    r = easysimd_mm256_div_epi32(a, b);
    *mem_addr = easysimd_mm256_sub_epi32(a, easysimd_mm256_mullo_epi32(r, b));

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_idivrem_epi32
  #define _mm256_idivrem_epi32(mem_addr, a, b) easysimd_mm256_idivrem_epi32((mem_addr),(a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_hypot_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_hypot_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_hypotf4_u05(a, b);
    #else
      return Sleef_hypotf4_u35(a, b);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_hypotf(a_.f32[i], b_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_hypot_ps
  #define _mm_hypot_ps(a, b) easysimd_mm_hypot_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_hypot_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_hypot_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_hypotd2_u05(a, b);
    #else
      return Sleef_hypotd2_u35(a, b);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_hypot(a_.f64[i], b_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_hypot_pd
  #define _mm_hypot_pd(a, b) easysimd_mm_hypot_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_hypot_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_hypot_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_hypotf8_u05(a, b);
    #else
      return Sleef_hypotf8_u35(a, b);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_hypot_ps(a_.m128[i], b_.m128[i]);
    }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_hypotf(a_.f32[i], b_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hypot_ps
  #define _mm256_hypot_ps(a, b) easysimd_mm256_hypot_ps(a, b)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_hypot_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_hypot_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_hypotd4_u05(a, b);
    #else
      return Sleef_hypotd4_u35(a, b);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_hypot_pd(a_.m128d[i], b_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_hypot(a_.f64[i], b_.f64[i]);
      }
  #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hypot_pd
  #define _mm256_hypot_pd(a, b) easysimd_mm256_hypot_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_hypot_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_hypot_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_hypotf16_u05(a, b);
    #else
      return Sleef_hypotf16_u35(a, b);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_hypot_ps(a_.m256[i], b_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_hypotf(a_.f32[i], b_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_hypot_ps
  #define _mm512_hypot_ps(a, b) easysimd_mm512_hypot_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_hypot_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_hypot_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_hypotd8_u05(a, b);
    #else
      return Sleef_hypotd8_u35(a, b);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_hypot_pd(a_.m256d[i], b_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_hypot(a_.f64[i], b_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_hypot_pd
  #define _mm512_hypot_pd(a, b) easysimd_mm512_hypot_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_hypot_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_hypot_ps(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_hypot_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_hypot_ps
  #define _mm512_mask_hypot_ps(src, k, a, b) easysimd_mm512_mask_hypot_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_hypot_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_hypot_pd(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_hypot_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_hypot_pd
  #define _mm512_mask_hypot_pd(src, k, a, b) easysimd_mm512_mask_hypot_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_invcbrt_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_invcbrt_ps(a);
  #else
    return easysimd_mm_rcp_ps(easysimd_mm_cbrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_invcbrt_ps
  #define _mm_invcbrt_ps(a) easysimd_mm_invcbrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_invcbrt_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_invcbrt_pd(a);
  #else
    return easysimd_mm_div_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)), easysimd_mm_cbrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_invcbrt_pd
  #define _mm_invcbrt_pd(a) easysimd_mm_invcbrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_invcbrt_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_invcbrt_ps(a);
  #else
    return easysimd_mm256_rcp_ps(easysimd_mm256_cbrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_invcbrt_ps
  #define _mm256_invcbrt_ps(a) easysimd_mm256_invcbrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_invcbrt_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_invcbrt_pd(a);
  #else
    return easysimd_mm256_div_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)), easysimd_mm256_cbrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_invcbrt_pd
  #define _mm256_invcbrt_pd(a) easysimd_mm256_invcbrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_invsqrt_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_invsqrt_ps(a);
  #else
    return easysimd_mm_rcp_ps(easysimd_mm_sqrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_invsqrt_ps
  #define _mm_invsqrt_ps(a) easysimd_mm_invsqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_invsqrt_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_invsqrt_pd(a);
  #else
    return easysimd_mm_div_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)), easysimd_mm_sqrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_invsqrt_pd
  #define _mm_invsqrt_pd(a) easysimd_mm_invsqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_invsqrt_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_invsqrt_ps(a);
  #else
    return easysimd_mm256_rcp_ps(easysimd_mm256_sqrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_invsqrt_ps
  #define _mm256_invsqrt_ps(a) easysimd_mm256_invsqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_invsqrt_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_invsqrt_pd(a);
  #else
    return easysimd_mm256_div_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)), easysimd_mm256_sqrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_invsqrt_pd
  #define _mm256_invsqrt_pd(a) easysimd_mm256_invsqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_invsqrt_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_invsqrt_ps(a);
  #else
    return easysimd_mm512_div_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)), easysimd_mm512_sqrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_invsqrt_ps
  #define _mm512_invsqrt_ps(a) easysimd_mm512_invsqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_invsqrt_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_invsqrt_pd(a);
  #else
    return easysimd_mm512_div_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)), easysimd_mm512_sqrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_invsqrt_pd
  #define _mm512_invsqrt_pd(a) easysimd_mm512_invsqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_invsqrt_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_invsqrt_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_invsqrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_invsqrt_ps
  #define _mm512_mask_invsqrt_ps(src, k, a) easysimd_mm512_mask_invsqrt_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_invsqrt_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_invsqrt_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_invsqrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_invsqrt_pd
  #define _mm512_mask_invsqrt_pd(src, k, a) easysimd_mm512_mask_invsqrt_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_log_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_logf4_u10(a);
    #else
      return Sleef_logf4_u35(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_logf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log_ps
  #define _mm_log_ps(a) easysimd_mm_log_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_log_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_logd2_u10(a);
    #else
      return Sleef_logd2_u35(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_log(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log_pd
  #define _mm_log_pd(a) easysimd_mm_log_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_log_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_logf8_u10(a);
    #else
      return Sleef_logf8_u35(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_log_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_logf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log_ps
  #define _mm256_log_ps(a) easysimd_mm256_log_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_log_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_logd4_u10(a);
    #else
      return Sleef_logd4_u35(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_log_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log_pd
  #define _mm256_log_pd(a) easysimd_mm256_log_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_log_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_logf16_u10(a);
    #else
      return Sleef_logf16_u35(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_log_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_logf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log_ps
  #define _mm512_log_ps(a) easysimd_mm512_log_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_log_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_logd8_u10(a);
    #else
      return Sleef_logd8_u35(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_log_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log_pd
  #define _mm512_log_pd(a) easysimd_mm512_log_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_log_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_log_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log_ps
  #define _mm512_mask_log_ps(src, k, a) easysimd_mm512_mask_log_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_log_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_log_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log_pd
  #define _mm512_mask_log_pd(src, k, a) easysimd_mm512_mask_log_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cdfnorminv_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cdfnorminv_ps(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    easysimd__m128 matched, retval = easysimd_mm_setzero_ps();

    { /* if (a < 0 || a > 1) */
      matched = easysimd_mm_or_ps(easysimd_mm_cmplt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.0))), easysimd_mm_cmpgt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0))));

      /* We don't actually need to do anything here since we initialize
       * retval to 0.0. */
    }

    { /* else if (a == 0) */
      easysimd__m128 mask = easysimd_mm_cmpeq_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.0)));
      mask = easysimd_mm_andnot_ps(matched, mask);
      matched = easysimd_mm_or_ps(matched, mask);

      easysimd__m128 res = easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITYF);

      retval = easysimd_mm_or_ps(retval, easysimd_mm_and_ps(mask, res));
    }

    { /* else if (a == 1) */
      easysimd__m128 mask = easysimd_mm_cmpeq_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0)));
      mask = easysimd_mm_andnot_ps(matched, mask);
      matched = easysimd_mm_or_ps(matched, mask);

      easysimd__m128 res = easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITYF);

      retval = easysimd_mm_or_ps(retval, easysimd_mm_and_ps(mask, res));
    }

    { /* Remaining conditions.
       *
       * Including the else case in this complicates things a lot, but
       * we're using cheap operations to get rid of expensive multiply
       * and add functions.  This should be a small improvement on SSE
       * prior to 4.1.  On SSE 4.1 we can use _mm_blendv_ps which is
       * very fast and this becomes a huge win.  NEON, AltiVec, and
       * WASM also have blend operations, so this should be a big win
       * there, too. */

      /* else if (a < 0.02425) */
      easysimd__m128 mask_lo = easysimd_mm_cmplt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.02425)));
      /* else if (a > 0.97575) */
      easysimd__m128 mask_hi = easysimd_mm_cmpgt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.97575)));

      easysimd__m128 mask = easysimd_mm_or_ps(mask_lo, mask_hi);
      matched = easysimd_mm_or_ps(matched, mask);

      /* else */
      easysimd__m128 mask_el = easysimd_x_mm_not_ps(matched);
      mask = easysimd_mm_or_ps(mask, mask_el);

      /* r = a - 0.5f */
      easysimd__m128 r = easysimd_mm_sub_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.5)));

      /* lo: q = a
       * hi: q = (1.0 - a) */
      easysimd__m128 q = easysimd_mm_and_ps(mask_lo, a);
      q = easysimd_mm_or_ps(q, easysimd_mm_and_ps(mask_hi, easysimd_mm_sub_ps(easysimd_mm_set1_ps(1.0f), a)));

      /* q = easysimd_math_sqrtf(-2.0f * easysimd_math_logf(q)) */
      q = easysimd_mm_log_ps(q);
      q = easysimd_mm_mul_ps(q, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-2.0)));
      q = easysimd_mm_sqrt_ps(q);

      /* el: q = r * r */
      q = easysimd_x_mm_select_ps(q, easysimd_mm_mul_ps(r, r), mask_el);

      /* lo: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) *  1.0f); */
      /* hi: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) * -1.0f); */
      /* el: float numerator = ((((((c_a[0] * q + c_a[1]) * q + c_a[2]) * q + c_a[3]) * q + c_a[4]) * q + c_a[5]) *  r); */
      easysimd__m128 numerator = easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-7.784894002430293e-03)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-3.969683028665376e+01)), mask_el);
      numerator = easysimd_mm_fmadd_ps(numerator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-3.223964580411365e-01)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 2.209460984245205e+02)), mask_el));
      numerator = easysimd_mm_fmadd_ps(numerator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-2.400758277161838e+00)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-2.759285104469687e+02)), mask_el));
      numerator = easysimd_mm_fmadd_ps(numerator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-2.549732539343734e+00)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.383577518672690e+02)), mask_el));
      numerator = easysimd_mm_fmadd_ps(numerator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 4.374664141464968e+00)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-3.066479806614716e+01)), mask_el));
      numerator = easysimd_mm_fmadd_ps(numerator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 2.938163982698783e+00)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 2.506628277459239e+00)), mask_el));
      {
        easysimd__m128 multiplier;
        multiplier =                            easysimd_mm_and_ps(mask_lo, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.0)));
        multiplier = easysimd_mm_or_ps(multiplier, easysimd_mm_and_ps(mask_hi, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-1.0))));
        multiplier = easysimd_mm_or_ps(multiplier, easysimd_mm_and_ps(mask_el, r));
        numerator = easysimd_mm_mul_ps(numerator, multiplier);
      }

      /* lo/hi: float denominator = (((((c_d[0] * q + c_d[1]) * q + c_d[2]) * q + c_d[3]) * 1 +   0.0f) * q + 1); */
      /*    el: float denominator = (((((c_b[0] * q + c_b[1]) * q + c_b[2]) * q + c_b[3]) * q + c_b[4]) * q + 1); */
      easysimd__m128 denominator = easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 7.784695709041462e-03)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-5.447609879822406e+01)), mask_el);
      denominator = easysimd_mm_fmadd_ps(denominator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 3.224671290700398e-01)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.615858368580409e+02)), mask_el));
      denominator = easysimd_mm_fmadd_ps(denominator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 2.445134137142996e+00)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-1.556989798598866e+02)), mask_el));
      denominator = easysimd_mm_fmadd_ps(denominator, q, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 3.754408661907416e+00)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 6.680131188771972e+01)), mask_el));
      denominator = easysimd_mm_fmadd_ps(denominator, easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.0)), q, mask_el),
                                                   easysimd_x_mm_select_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.0)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-1.328068155288572e+01)), mask_el));
      denominator = easysimd_mm_fmadd_ps(denominator, q, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0)));

      /* res = numerator / denominator; */
      easysimd__m128 res = easysimd_mm_div_ps(numerator, denominator);

      retval = easysimd_mm_or_ps(retval, easysimd_mm_and_ps(mask, res));
    }

    return retval;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_cdfnorminvf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cdfnorminv_ps
  #define _mm_cdfnorminv_ps(a) easysimd_mm_cdfnorminv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cdfnorminv_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cdfnorminv_pd(a);
   #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    easysimd__m128d matched, retval = easysimd_mm_setzero_pd();

    { /* if (a < 0 || a > 1) */
      matched = easysimd_mm_or_pd(easysimd_mm_cmplt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.0))), easysimd_mm_cmpgt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0))));

      /* We don't actually need to do anything here since we initialize
       * retval to 0.0. */
    }

    { /* else if (a == 0) */
      easysimd__m128d mask = easysimd_mm_cmpeq_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.0)));
      mask = easysimd_mm_andnot_pd(matched, mask);
      matched = easysimd_mm_or_pd(matched, mask);

      easysimd__m128d res = easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY);

      retval = easysimd_mm_or_pd(retval, easysimd_mm_and_pd(mask, res));
    }

    { /* else if (a == 1) */
      easysimd__m128d mask = easysimd_mm_cmpeq_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)));
      mask = easysimd_mm_andnot_pd(matched, mask);
      matched = easysimd_mm_or_pd(matched, mask);

      easysimd__m128d res = easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY);

      retval = easysimd_mm_or_pd(retval, easysimd_mm_and_pd(mask, res));
    }

    { /* Remaining conditions.
       *
       * Including the else case in this complicates things a lot, but
       * we're using cheap operations to get rid of expensive multiply
       * and add functions.  This should be a small improvement on SSE
       * prior to 4.1.  On SSE 4.1 we can use _mm_blendv_pd which is
       * very fast and this becomes a huge win.  NEON, AltiVec, and
       * WASM also have blend operations, so this should be a big win
       * there, too. */

      /* else if (a < 0.02425) */
      easysimd__m128d mask_lo = easysimd_mm_cmplt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.02425)));
      /* else if (a > 0.97575) */
      easysimd__m128d mask_hi = easysimd_mm_cmpgt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.97575)));

      easysimd__m128d mask = easysimd_mm_or_pd(mask_lo, mask_hi);
      matched = easysimd_mm_or_pd(matched, mask);

      /* else */
      easysimd__m128d mask_el = easysimd_x_mm_not_pd(matched);
      mask = easysimd_mm_or_pd(mask, mask_el);

      /* r = a - 0.5 */
      easysimd__m128d r = easysimd_mm_sub_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.5)));

      /* lo: q = a
       * hi: q = (1.0 - a) */
      easysimd__m128d q = easysimd_mm_and_pd(mask_lo, a);
      q = easysimd_mm_or_pd(q, easysimd_mm_and_pd(mask_hi, easysimd_mm_sub_pd(easysimd_mm_set1_pd(1.0), a)));

      /* q = easysimd_math_sqrt(-2.0 * easysimd_math_log(q)) */
      q = easysimd_mm_log_pd(q);
      q = easysimd_mm_mul_pd(q, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-2.0)));
      q = easysimd_mm_sqrt_pd(q);

      /* el: q = r * r */
      q = easysimd_x_mm_select_pd(q, easysimd_mm_mul_pd(r, r), mask_el);

      /* lo: double numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) *  1.0); */
      /* hi: double numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) * -1.0); */
      /* el: double numerator = ((((((c_a[0] * q + c_a[1]) * q + c_a[2]) * q + c_a[3]) * q + c_a[4]) * q + c_a[5]) *  r); */
      easysimd__m128d numerator = easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-7.784894002430293e-03)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-3.969683028665376e+01)), mask_el);
      numerator = easysimd_mm_fmadd_pd(numerator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-3.223964580411365e-01)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 2.209460984245205e+02)), mask_el));
      numerator = easysimd_mm_fmadd_pd(numerator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-2.400758277161838e+00)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-2.759285104469687e+02)), mask_el));
      numerator = easysimd_mm_fmadd_pd(numerator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-2.549732539343734e+00)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.383577518672690e+02)), mask_el));
      numerator = easysimd_mm_fmadd_pd(numerator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 4.374664141464968e+00)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-3.066479806614716e+01)), mask_el));
      numerator = easysimd_mm_fmadd_pd(numerator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 2.938163982698783e+00)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 2.506628277459239e+00)), mask_el));
      {
        easysimd__m128d multiplier;
        multiplier =                            easysimd_mm_and_pd(mask_lo, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.0)));
        multiplier = easysimd_mm_or_pd(multiplier, easysimd_mm_and_pd(mask_hi, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-1.0))));
        multiplier = easysimd_mm_or_pd(multiplier, easysimd_mm_and_pd(mask_el, r));
        numerator = easysimd_mm_mul_pd(numerator, multiplier);
      }

      /* lo/hi: double denominator = (((((c_d[0] * q + c_d[1]) * q + c_d[2]) * q + c_d[3]) * 1 +   0.0f) * q + 1); */
      /*    el: double denominator = (((((c_b[0] * q + c_b[1]) * q + c_b[2]) * q + c_b[3]) * q + c_b[4]) * q + 1); */
      easysimd__m128d denominator = easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 7.784695709041462e-03)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-5.447609879822406e+01)), mask_el);
      denominator = easysimd_mm_fmadd_pd(denominator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 3.224671290700398e-01)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.615858368580409e+02)), mask_el));
      denominator = easysimd_mm_fmadd_pd(denominator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 2.445134137142996e+00)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-1.556989798598866e+02)), mask_el));
      denominator = easysimd_mm_fmadd_pd(denominator, q, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 3.754408661907416e+00)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 6.680131188771972e+01)), mask_el));
      denominator = easysimd_mm_fmadd_pd(denominator, easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.0)), q, mask_el),
                                                   easysimd_x_mm_select_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.0)), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-1.328068155288572e+01)), mask_el));
      denominator = easysimd_mm_fmadd_pd(denominator, q, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)));

      /* res = numerator / denominator; */
      easysimd__m128d res = easysimd_mm_div_pd(numerator, denominator);

      retval = easysimd_mm_or_pd(retval, easysimd_mm_and_pd(mask, res));
    }

    return retval;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_cdfnorminv(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_cdfnorminv_pd
  #define _mm_cdfnorminv_pd(a) easysimd_mm_cdfnorminv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cdfnorminv_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cdfnorminv_ps(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(256)
    easysimd__m256 matched, retval = easysimd_mm256_setzero_ps();

    { /* if (a < 0 || a > 1) */
      matched = easysimd_mm256_or_ps(easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_LT_OQ), easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0)), EASYSIMD_CMP_GT_OQ));

      /* We don't actually need to do anything here since we initialize
       * retval to 0.0. */
    }

    { /* else if (a == 0) */
      easysimd__m256 mask = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      mask = easysimd_mm256_andnot_ps(matched, mask);
      matched = easysimd_mm256_or_ps(matched, mask);

      easysimd__m256 res = easysimd_mm256_set1_ps(-EASYSIMD_MATH_INFINITYF);

      retval = easysimd_mm256_or_ps(retval, easysimd_mm256_and_ps(mask, res));
    }

    { /* else if (a == 1) */
      easysimd__m256 mask = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0)), EASYSIMD_CMP_EQ_OQ);
      mask = easysimd_mm256_andnot_ps(matched, mask);
      matched = easysimd_mm256_or_ps(matched, mask);

      easysimd__m256 res = easysimd_mm256_set1_ps(EASYSIMD_MATH_INFINITYF);

      retval = easysimd_mm256_or_ps(retval, easysimd_mm256_and_ps(mask, res));
    }

    { /* Remaining conditions.
       *
       * Including the else case in this complicates things a lot, but
       * we're using cheap operations to get rid of expensive multiply
       * and add functions.  This should be a small improvement on SSE
       * prior to 4.1.  On SSE 4.1 we can use _mm256_blendv_ps which is
       * very fast and this becomes a huge win.  NEON, AltiVec, and
       * WASM also have blend operations, so this should be a big win
       * there, too. */

      /* else if (a < 0.02425) */
      easysimd__m256 mask_lo = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.02425)), EASYSIMD_CMP_LT_OQ);
      /* else if (a > 0.97575) */
      easysimd__m256 mask_hi = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.97575)), EASYSIMD_CMP_GT_OQ);

      easysimd__m256 mask = easysimd_mm256_or_ps(mask_lo, mask_hi);
      matched = easysimd_mm256_or_ps(matched, mask);

      /* else */
      easysimd__m256 mask_el = easysimd_x_mm256_not_ps(matched);
      mask = easysimd_mm256_or_ps(mask, mask_el);

      /* r = a - 0.5f */
      easysimd__m256 r = easysimd_mm256_sub_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.5)));

      /* lo: q = a
       * hi: q = (1.0 - a) */
      easysimd__m256 q = easysimd_mm256_and_ps(mask_lo, a);
      q = easysimd_mm256_or_ps(q, easysimd_mm256_and_ps(mask_hi, easysimd_mm256_sub_ps(easysimd_mm256_set1_ps(1.0f), a)));

      /* q = easysimd_math_sqrtf(-2.0f * easysimd_math_logf(q)) */
      q = easysimd_mm256_log_ps(q);
      q = easysimd_mm256_mul_ps(q, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-2.0)));
      q = easysimd_mm256_sqrt_ps(q);

      /* el: q = r * r */
      q = easysimd_x_mm256_select_ps(q, easysimd_mm256_mul_ps(r, r), mask_el);

      /* lo: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) *  1.0f); */
      /* hi: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) * -1.0f); */
      /* el: float numerator = ((((((c_a[0] * q + c_a[1]) * q + c_a[2]) * q + c_a[3]) * q + c_a[4]) * q + c_a[5]) *  r); */
      easysimd__m256 numerator = easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-7.784894002430293e-03)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-3.969683028665376e+01)), mask_el);
      numerator = easysimd_mm256_fmadd_ps(numerator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-3.223964580411365e-01)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 2.209460984245205e+02)), mask_el));
      numerator = easysimd_mm256_fmadd_ps(numerator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-2.400758277161838e+00)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-2.759285104469687e+02)), mask_el));
      numerator = easysimd_mm256_fmadd_ps(numerator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-2.549732539343734e+00)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.383577518672690e+02)), mask_el));
      numerator = easysimd_mm256_fmadd_ps(numerator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 4.374664141464968e+00)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-3.066479806614716e+01)), mask_el));
      numerator = easysimd_mm256_fmadd_ps(numerator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 2.938163982698783e+00)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 2.506628277459239e+00)), mask_el));
      {
        easysimd__m256 multiplier;
        multiplier =                            easysimd_mm256_and_ps(mask_lo, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.0)));
        multiplier = easysimd_mm256_or_ps(multiplier, easysimd_mm256_and_ps(mask_hi, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-1.0))));
        multiplier = easysimd_mm256_or_ps(multiplier, easysimd_mm256_and_ps(mask_el, r));
        numerator = easysimd_mm256_mul_ps(numerator, multiplier);
      }

      /* lo/hi: float denominator = (((((c_d[0] * q + c_d[1]) * q + c_d[2]) * q + c_d[3]) * 1 +   0.0f) * q + 1); */
      /*    el: float denominator = (((((c_b[0] * q + c_b[1]) * q + c_b[2]) * q + c_b[3]) * q + c_b[4]) * q + 1); */
      easysimd__m256 denominator = easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 7.784695709041462e-03)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-5.447609879822406e+01)), mask_el);
      denominator = easysimd_mm256_fmadd_ps(denominator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 3.224671290700398e-01)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.615858368580409e+02)), mask_el));
      denominator = easysimd_mm256_fmadd_ps(denominator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 2.445134137142996e+00)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-1.556989798598866e+02)), mask_el));
      denominator = easysimd_mm256_fmadd_ps(denominator, q, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 3.754408661907416e+00)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 6.680131188771972e+01)), mask_el));
      denominator = easysimd_mm256_fmadd_ps(denominator, easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.0)), q, mask_el),
                                                   easysimd_x_mm256_select_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.0)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-1.328068155288572e+01)), mask_el));
      denominator = easysimd_mm256_fmadd_ps(denominator, q, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0)));

      /* res = numerator / denominator; */
      easysimd__m256 res = easysimd_mm256_div_ps(numerator, denominator);

      retval = easysimd_mm256_or_ps(retval, easysimd_mm256_and_ps(mask, res));
    }

    return retval;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_cdfnorminv_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_cdfnorminvf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cdfnorminv_ps
  #define _mm256_cdfnorminv_ps(a) easysimd_mm256_cdfnorminv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cdfnorminv_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cdfnorminv_pd(a);
   #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(256)
    easysimd__m256d matched, retval = easysimd_mm256_setzero_pd();

    { /* if (a < 0 || a > 1) */
      matched = easysimd_mm256_or_pd(easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_LT_OQ), easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)), EASYSIMD_CMP_GT_OQ));

      /* We don't actually need to do anything here since we initialize
       * retval to 0.0. */
    }

    { /* else if (a == 0) */
      easysimd__m256d mask = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      mask = easysimd_mm256_andnot_pd(matched, mask);
      matched = easysimd_mm256_or_pd(matched, mask);

      easysimd__m256d res = easysimd_mm256_set1_pd(-EASYSIMD_MATH_INFINITY);

      retval = easysimd_mm256_or_pd(retval, easysimd_mm256_and_pd(mask, res));
    }

    { /* else if (a == 1) */
      easysimd__m256d mask = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)), EASYSIMD_CMP_EQ_OQ);
      mask = easysimd_mm256_andnot_pd(matched, mask);
      matched = easysimd_mm256_or_pd(matched, mask);

      easysimd__m256d res = easysimd_mm256_set1_pd(EASYSIMD_MATH_INFINITY);

      retval = easysimd_mm256_or_pd(retval, easysimd_mm256_and_pd(mask, res));
    }

    { /* Remaining conditions.
       *
       * Including the else case in this complicates things a lot, but
       * we're using cheap operations to get rid of expensive multiply
       * and add functions.  This should be a small improvement on SSE
       * prior to 4.1.  On SSE 4.1 we can use _mm256_blendv_pd which is
       * very fast and this becomes a huge win.  NEON, AltiVec, and
       * WASM also have blend operations, so this should be a big win
       * there, too. */

      /* else if (a < 0.02425) */
      easysimd__m256d mask_lo = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.02425)), EASYSIMD_CMP_LT_OQ);
      /* else if (a > 0.97575) */
      easysimd__m256d mask_hi = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.97575)), EASYSIMD_CMP_GT_OQ);

      easysimd__m256d mask = easysimd_mm256_or_pd(mask_lo, mask_hi);
      matched = easysimd_mm256_or_pd(matched, mask);

      /* else */
      easysimd__m256d mask_el = easysimd_x_mm256_not_pd(matched);
      mask = easysimd_mm256_or_pd(mask, mask_el);

      /* r = a - 0.5 */
      easysimd__m256d r = easysimd_mm256_sub_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.5)));

      /* lo: q = a
       * hi: q = (1.0 - a) */
      easysimd__m256d q = easysimd_mm256_and_pd(mask_lo, a);
      q = easysimd_mm256_or_pd(q, easysimd_mm256_and_pd(mask_hi, easysimd_mm256_sub_pd(easysimd_mm256_set1_pd(1.0), a)));

      /* q = easysimd_math_sqrt(-2.0 * easysimd_math_log(q)) */
      q = easysimd_mm256_log_pd(q);
      q = easysimd_mm256_mul_pd(q, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-2.0)));
      q = easysimd_mm256_sqrt_pd(q);

      /* el: q = r * r */
      q = easysimd_x_mm256_select_pd(q, easysimd_mm256_mul_pd(r, r), mask_el);

      /* lo: double numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) *  1.0); */
      /* hi: double numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) * -1.0); */
      /* el: double numerator = ((((((c_a[0] * q + c_a[1]) * q + c_a[2]) * q + c_a[3]) * q + c_a[4]) * q + c_a[5]) *  r); */
      easysimd__m256d numerator = easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-7.784894002430293e-03)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-3.969683028665376e+01)), mask_el);
      numerator = easysimd_mm256_fmadd_pd(numerator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-3.223964580411365e-01)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 2.209460984245205e+02)), mask_el));
      numerator = easysimd_mm256_fmadd_pd(numerator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-2.400758277161838e+00)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-2.759285104469687e+02)), mask_el));
      numerator = easysimd_mm256_fmadd_pd(numerator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-2.549732539343734e+00)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.383577518672690e+02)), mask_el));
      numerator = easysimd_mm256_fmadd_pd(numerator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 4.374664141464968e+00)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-3.066479806614716e+01)), mask_el));
      numerator = easysimd_mm256_fmadd_pd(numerator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 2.938163982698783e+00)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 2.506628277459239e+00)), mask_el));
      {
        easysimd__m256d multiplier;
        multiplier =                            easysimd_mm256_and_pd(mask_lo, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.0)));
        multiplier = easysimd_mm256_or_pd(multiplier, easysimd_mm256_and_pd(mask_hi, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-1.0))));
        multiplier = easysimd_mm256_or_pd(multiplier, easysimd_mm256_and_pd(mask_el, r));
        numerator = easysimd_mm256_mul_pd(numerator, multiplier);
      }

      /* lo/hi: double denominator = (((((c_d[0] * q + c_d[1]) * q + c_d[2]) * q + c_d[3]) * 1 +   0.0f) * q + 1); */
      /*    el: double denominator = (((((c_b[0] * q + c_b[1]) * q + c_b[2]) * q + c_b[3]) * q + c_b[4]) * q + 1); */
      easysimd__m256d denominator = easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 7.784695709041462e-03)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-5.447609879822406e+01)), mask_el);
      denominator = easysimd_mm256_fmadd_pd(denominator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 3.224671290700398e-01)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.615858368580409e+02)), mask_el));
      denominator = easysimd_mm256_fmadd_pd(denominator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 2.445134137142996e+00)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-1.556989798598866e+02)), mask_el));
      denominator = easysimd_mm256_fmadd_pd(denominator, q, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 3.754408661907416e+00)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 6.680131188771972e+01)), mask_el));
      denominator = easysimd_mm256_fmadd_pd(denominator, easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.0)), q, mask_el),
                                                   easysimd_x_mm256_select_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.0)), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-1.328068155288572e+01)), mask_el));
      denominator = easysimd_mm256_fmadd_pd(denominator, q, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)));

      /* res = numerator / denominator; */
      easysimd__m256d res = easysimd_mm256_div_pd(numerator, denominator);

      retval = easysimd_mm256_or_pd(retval, easysimd_mm256_and_pd(mask, res));
    }

    return retval;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_cdfnorminv_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_cdfnorminv(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cdfnorminv_pd
  #define _mm256_cdfnorminv_pd(a) easysimd_mm256_cdfnorminv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cdfnorminv_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cdfnorminv_ps(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
      r_.m256[i] = easysimd_mm256_cdfnorminv_ps(a_.m256[i]);
    }

    return easysimd__m512_from_private(r_);
  #else

    easysimd__m512 retval = easysimd_mm512_setzero_ps();
    easysimd__mmask16 matched;

    { /* if (a < 0 || a > 1) */
      matched  = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_LT_OQ);
      matched |= easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)), EASYSIMD_CMP_GT_OQ);

      /* We don't actually need to do anything here since we initialize
       * retval to 0.0. */
    }

    { /* else if (a == 0) */
      easysimd__mmask16 mask = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      matched |= mask;

      retval = easysimd_mm512_mask_mov_ps(retval, mask, easysimd_mm512_set1_ps(-EASYSIMD_MATH_INFINITYF));
    }

    { /* else if (a == 1) */
      easysimd__mmask16 mask = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      matched |= mask;

      retval = easysimd_mm512_mask_mov_ps(retval, mask, easysimd_mm512_set1_ps(EASYSIMD_MATH_INFINITYF));
    }

    { /* else if (a < 0.02425) */
      easysimd__mmask16 mask_lo = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.02425)), EASYSIMD_CMP_LT_OQ);
      /* else if (a > 0.97575) */
      easysimd__mmask16 mask_hi = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.97575)), EASYSIMD_CMP_GT_OQ);

      easysimd__mmask16 mask = mask_lo | mask_hi;
      matched = matched | mask;

      /* else */
      easysimd__mmask16 mask_el = ~matched;

      /* r = a - 0.5f */
      easysimd__m512 r = easysimd_mm512_sub_ps(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.5)));

      /* lo: q = a
       * hi: q = (1.0 - a) */
      easysimd__m512 q = easysimd_mm512_maskz_mov_ps(mask_lo, a);
      q = easysimd_mm512_mask_sub_ps(q, mask_hi, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)), a);

      /* q = easysimd_math_sqrtf(-2.0f * easysimd_math_logf(q)) */
      q = easysimd_mm512_log_ps(q);
      q = easysimd_mm512_mul_ps(q, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-2.0)));
      q = easysimd_mm512_sqrt_ps(q);

      /* el: q = r * r */
      q = easysimd_mm512_mask_mul_ps(q, mask_el, r, r);

      /* lo: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) *  1.0f); */
      /* hi: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) * -1.0f); */
      /* el: float numerator = ((((((c_a[0] * q + c_a[1]) * q + c_a[2]) * q + c_a[3]) * q + c_a[4]) * q + c_a[5]) *  r); */
      easysimd__m512 numerator = easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-7.784894002430293e-03)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-3.969683028665376e+01)));
      numerator = easysimd_mm512_fmadd_ps(numerator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-3.223964580411365e-01)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 2.209460984245205e+02))));
      numerator = easysimd_mm512_fmadd_ps(numerator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-2.400758277161838e+00)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-2.759285104469687e+02))));
      numerator = easysimd_mm512_fmadd_ps(numerator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-2.549732539343734e+00)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.383577518672690e+02))));
      numerator = easysimd_mm512_fmadd_ps(numerator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 4.374664141464968e+00)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-3.066479806614716e+01))));
      numerator = easysimd_mm512_fmadd_ps(numerator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 2.938163982698783e+00)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 2.506628277459239e+00))));
      {
        easysimd__m512 multiplier;
        multiplier =                                              easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.0));
        multiplier = easysimd_mm512_mask_mov_ps(multiplier, mask_hi, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-1.0)));
        multiplier = easysimd_mm512_mask_mov_ps(multiplier, mask_el, r);
        numerator = easysimd_mm512_mul_ps(numerator, multiplier);
      }

      /* lo/hi: float denominator = (((((c_d[0] * q + c_d[1]) * q + c_d[2]) * q + c_d[3]) * 1 +   0.0f) * q + 1); */
      /*    el: float denominator = (((((c_b[0] * q + c_b[1]) * q + c_b[2]) * q + c_b[3]) * q + c_b[4]) * q + 1); */
      easysimd__m512 denominator = easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 7.784695709041462e-03)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-5.447609879822406e+01)));
      denominator = easysimd_mm512_fmadd_ps(denominator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 3.224671290700398e-01)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.615858368580409e+02))));
      denominator = easysimd_mm512_fmadd_ps(denominator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 2.445134137142996e+00)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-1.556989798598866e+02))));
      denominator = easysimd_mm512_fmadd_ps(denominator, q, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 3.754408661907416e+00)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 6.680131188771972e+01))));
      denominator = easysimd_mm512_fmadd_ps(denominator, easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.0)), mask_el, q),
                                                      easysimd_mm512_mask_mov_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.0)), mask_el, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-1.328068155288572e+01))));
      denominator = easysimd_mm512_fmadd_ps(denominator, q, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)));

      /* res = numerator / denominator; */
      retval = easysimd_mm512_mask_div_ps(retval, mask_lo | mask_hi | mask_el, numerator, denominator);
    }

    return retval;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cdfnorminv_ps
  #define _mm512_cdfnorminv_ps(a) easysimd_mm512_cdfnorminv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cdfnorminv_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cdfnorminv_pd(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
      r_.m256d[i] = easysimd_mm256_cdfnorminv_pd(a_.m256d[i]);
    }

    return easysimd__m512d_from_private(r_);
  #else

    easysimd__m512d retval = easysimd_mm512_setzero_pd();
    easysimd__mmask8 matched;

    { /* if (a < 0 || a > 1) */
      matched  = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_LT_OQ);
      matched |= easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)), EASYSIMD_CMP_GT_OQ);

      /* We don't actually need to do anything here since we initialize
       * retval to 0.0. */
    }

    { /* else if (a == 0) */
      easysimd__mmask8 mask = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      matched |= mask;

      retval = easysimd_mm512_mask_mov_pd(retval, mask, easysimd_mm512_set1_pd(-EASYSIMD_MATH_INFINITY));
    }

    { /* else if (a == 1) */
      easysimd__mmask8 mask = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      matched |= mask;

      retval = easysimd_mm512_mask_mov_pd(retval, mask, easysimd_mm512_set1_pd(EASYSIMD_MATH_INFINITY));
    }

    { /* else if (a < 0.02425) */
      easysimd__mmask8 mask_lo = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.02425)), EASYSIMD_CMP_LT_OQ);
      /* else if (a > 0.97575) */
      easysimd__mmask8 mask_hi = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.97575)), EASYSIMD_CMP_GT_OQ);

      easysimd__mmask8 mask = mask_lo | mask_hi;
      matched = matched | mask;

      /* else */
      easysimd__mmask8 mask_el = ~matched;

      /* r = a - 0.5f */
      easysimd__m512d r = easysimd_mm512_sub_pd(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.5)));

      /* lo: q = a
       * hi: q = (1.0 - a) */
      easysimd__m512d q = a;
      q = easysimd_mm512_mask_sub_pd(q, mask_hi, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)), a);

      /* q = easysimd_math_sqrtf(-2.0f * easysimd_math_logf(q)) */
      q = easysimd_mm512_log_pd(q);
      q = easysimd_mm512_mul_pd(q, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-2.0)));
      q = easysimd_mm512_sqrt_pd(q);

      /* el: q = r * r */
      q = easysimd_mm512_mask_mul_pd(q, mask_el, r, r);

      /* lo: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) *  1.0f); */
      /* hi: float numerator = ((((((c_c[0] * q + c_c[1]) * q + c_c[2]) * q + c_c[3]) * q + c_c[4]) * q + c_c[5]) * -1.0f); */
      /* el: float numerator = ((((((c_a[0] * q + c_a[1]) * q + c_a[2]) * q + c_a[3]) * q + c_a[4]) * q + c_a[5]) *  r); */
      easysimd__m512d numerator = easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-7.784894002430293e-03)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-3.969683028665376e+01)));
      numerator = easysimd_mm512_fmadd_pd(numerator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-3.223964580411365e-01)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 2.209460984245205e+02))));
      numerator = easysimd_mm512_fmadd_pd(numerator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-2.400758277161838e+00)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-2.759285104469687e+02))));
      numerator = easysimd_mm512_fmadd_pd(numerator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-2.549732539343734e+00)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.383577518672690e+02))));
      numerator = easysimd_mm512_fmadd_pd(numerator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 4.374664141464968e+00)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-3.066479806614716e+01))));
      numerator = easysimd_mm512_fmadd_pd(numerator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 2.938163982698783e+00)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 2.506628277459239e+00))));
      {
        easysimd__m512d multiplier;
        multiplier =                                              easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.0));
        multiplier = easysimd_mm512_mask_mov_pd(multiplier, mask_hi, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-1.0)));
        multiplier = easysimd_mm512_mask_mov_pd(multiplier, mask_el, r);
        numerator = easysimd_mm512_mul_pd(numerator, multiplier);
      }

      /* lo/hi: float denominator = (((((c_d[0] * q + c_d[1]) * q + c_d[2]) * q + c_d[3]) * 1 +   0.0f) * q + 1); */
      /*    el: float denominator = (((((c_b[0] * q + c_b[1]) * q + c_b[2]) * q + c_b[3]) * q + c_b[4]) * q + 1); */
      easysimd__m512d denominator = easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 7.784695709041462e-03)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-5.447609879822406e+01)));
      denominator = easysimd_mm512_fmadd_pd(denominator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 3.224671290700398e-01)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.615858368580409e+02))));
      denominator = easysimd_mm512_fmadd_pd(denominator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 2.445134137142996e+00)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-1.556989798598866e+02))));
      denominator = easysimd_mm512_fmadd_pd(denominator, q, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 3.754408661907416e+00)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 6.680131188771972e+01))));
      denominator = easysimd_mm512_fmadd_pd(denominator, easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.0)), mask_el, q),
                                                      easysimd_mm512_mask_mov_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.0)), mask_el, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-1.328068155288572e+01))));
      denominator = easysimd_mm512_fmadd_pd(denominator, q, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)));

      /* res = numerator / denominator; */
      retval = easysimd_mm512_mask_div_pd(retval, mask_lo | mask_hi | mask_el, numerator, denominator);
    }

    return retval;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cdfnorminv_pd
  #define _mm512_cdfnorminv_pd(a) easysimd_mm512_cdfnorminv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_cdfnorminv_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cdfnorminv_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_cdfnorminv_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cdfnorminv_ps
  #define _mm512_mask_cdfnorminv_ps(src, k, a) easysimd_mm512_mask_cdfnorminv_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_cdfnorminv_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cdfnorminv_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_cdfnorminv_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cdfnorminv_pd
  #define _mm512_mask_cdfnorminv_pd(src, k, a) easysimd_mm512_mask_cdfnorminv_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_erfinv_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erfinv_ps(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    /* https://stackoverflow.com/questions/27229371/inverse-error-function-in-c */
    easysimd__m128 one = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0));

    easysimd__m128 lnx = easysimd_mm_log_ps(easysimd_mm_mul_ps(easysimd_mm_sub_ps(one, a), easysimd_mm_add_ps(one, a)));

    easysimd__m128 tt1 = easysimd_mm_mul_ps(easysimd_mm_set1_ps(HEDLEY_STATIC_CAST(easysimd_float32, EASYSIMD_MATH_PI)), easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.147)));
    tt1 = easysimd_mm_div_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(2.0)), tt1);
    tt1 = easysimd_mm_add_ps(tt1, easysimd_mm_mul_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.5)), lnx));

    easysimd__m128 tt2 = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0) / EASYSIMD_FLOAT32_C(0.147));
    tt2 = easysimd_mm_mul_ps(tt2, lnx);

    easysimd__m128 r = easysimd_mm_mul_ps(tt1, tt1);
    r = easysimd_mm_sub_ps(r, tt2);
    r = easysimd_mm_sqrt_ps(r);
    r = easysimd_mm_add_ps(easysimd_x_mm_negate_ps(tt1), r);
    r = easysimd_mm_sqrt_ps(r);

    return easysimd_x_mm_xorsign_ps(r, a);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_erfinvf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erfinv_ps
  #define _mm_erfinv_ps(a) easysimd_mm_erfinv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_erfinv_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erfinv_pd(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    easysimd__m128d one = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0));

    easysimd__m128d lnx = easysimd_mm_log_pd(easysimd_mm_mul_pd(easysimd_mm_sub_pd(one, a), easysimd_mm_add_pd(one, a)));

    easysimd__m128d tt1 = easysimd_mm_mul_pd(easysimd_mm_set1_pd(EASYSIMD_MATH_PI), easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.147)));
    tt1 = easysimd_mm_div_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(2.0)), tt1);
    tt1 = easysimd_mm_add_pd(tt1, easysimd_mm_mul_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.5)), lnx));

    easysimd__m128d tt2 = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0) / EASYSIMD_FLOAT64_C(0.147));
    tt2 = easysimd_mm_mul_pd(tt2, lnx);

    easysimd__m128d r = easysimd_mm_mul_pd(tt1, tt1);
    r = easysimd_mm_sub_pd(r, tt2);
    r = easysimd_mm_sqrt_pd(r);
    r = easysimd_mm_add_pd(easysimd_x_mm_negate_pd(tt1), r);
    r = easysimd_mm_sqrt_pd(r);

    return easysimd_x_mm_xorsign_pd(r, a);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_erfinv(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erfinv_pd
  #define _mm_erfinv_pd(a) easysimd_mm_erfinv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_erfinv_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erfinv_ps(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    easysimd__m256 one = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0));
    easysimd__m256 sgn = easysimd_x_mm256_copysign_ps(one, a);

    a = easysimd_mm256_mul_ps(easysimd_mm256_sub_ps(one, a), easysimd_mm256_add_ps(one, a));
    easysimd__m256 lnx = easysimd_mm256_log_ps(a);

    easysimd__m256 tt1 = easysimd_mm256_mul_ps(easysimd_mm256_set1_ps(HEDLEY_STATIC_CAST(easysimd_float32, EASYSIMD_MATH_PI)), easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.147)));
    tt1 = easysimd_mm256_div_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(2.0)), tt1);
    tt1 = easysimd_mm256_add_ps(tt1, easysimd_mm256_mul_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.5)), lnx));

    easysimd__m256 tt2 = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0) / EASYSIMD_FLOAT32_C(0.147));
    tt2 = easysimd_mm256_mul_ps(tt2, lnx);

    easysimd__m256 r = easysimd_mm256_mul_ps(tt1, tt1);
    r = easysimd_mm256_sub_ps(r, tt2);
    r = easysimd_mm256_sqrt_ps(r);
    r = easysimd_mm256_add_ps(easysimd_x_mm256_negate_ps(tt1), r);
    r = easysimd_mm256_sqrt_ps(r);

    return easysimd_mm256_mul_ps(sgn, r);
  #else
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_erfinvf(a_.f32[i]);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erfinv_ps
  #define _mm256_erfinv_ps(a) easysimd_mm256_erfinv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_erfinv_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erfinv_pd(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    easysimd__m256d one = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0));
    easysimd__m256d sgn = easysimd_x_mm256_copysign_pd(one, a);

    a = easysimd_mm256_mul_pd(easysimd_mm256_sub_pd(one, a), easysimd_mm256_add_pd(one, a));
    easysimd__m256d lnx = easysimd_mm256_log_pd(a);

    easysimd__m256d tt1 = easysimd_mm256_mul_pd(easysimd_mm256_set1_pd(EASYSIMD_MATH_PI), easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.147)));
    tt1 = easysimd_mm256_div_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(2.0)), tt1);
    tt1 = easysimd_mm256_add_pd(tt1, easysimd_mm256_mul_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.5)), lnx));

    easysimd__m256d tt2 = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0) / EASYSIMD_FLOAT64_C(0.147));
    tt2 = easysimd_mm256_mul_pd(tt2, lnx);

    easysimd__m256d r = easysimd_mm256_mul_pd(tt1, tt1);
    r = easysimd_mm256_sub_pd(r, tt2);
    r = easysimd_mm256_sqrt_pd(r);
    r = easysimd_mm256_add_pd(easysimd_x_mm256_negate_pd(tt1), r);
    r = easysimd_mm256_sqrt_pd(r);

    return easysimd_mm256_mul_pd(sgn, r);
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_erfinv(a_.f64[i]);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erfinv_pd
  #define _mm256_erfinv_pd(a) easysimd_mm256_erfinv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_erfinv_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erfinv_ps(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    easysimd__m512 one = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0));
    easysimd__m512 sgn = easysimd_x_mm512_copysign_ps(one, a);

    a = easysimd_mm512_mul_ps(easysimd_mm512_sub_ps(one, a), easysimd_mm512_add_ps(one, a));
    easysimd__m512 lnx = easysimd_mm512_log_ps(a);

    easysimd__m512 tt1 = easysimd_mm512_mul_ps(easysimd_mm512_set1_ps(HEDLEY_STATIC_CAST(easysimd_float32, EASYSIMD_MATH_PI)), easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.147)));
    tt1 = easysimd_mm512_div_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(2.0)), tt1);
    tt1 = easysimd_mm512_add_ps(tt1, easysimd_mm512_mul_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.5)), lnx));

    easysimd__m512 tt2 = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0) / EASYSIMD_FLOAT32_C(0.147));
    tt2 = easysimd_mm512_mul_ps(tt2, lnx);

    easysimd__m512 r = easysimd_mm512_mul_ps(tt1, tt1);
    r = easysimd_mm512_sub_ps(r, tt2);
    r = easysimd_mm512_sqrt_ps(r);
    r = easysimd_mm512_add_ps(easysimd_x_mm512_negate_ps(tt1), r);
    r = easysimd_mm512_sqrt_ps(r);

    return easysimd_mm512_mul_ps(sgn, r);
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_erfinvf(a_.f32[i]);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erfinv_ps
  #define _mm512_erfinv_ps(a) easysimd_mm512_erfinv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_erfinv_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erfinv_pd(a);
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    easysimd__m512d one = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0));
    easysimd__m512d sgn = easysimd_x_mm512_copysign_pd(one, a);

    a = easysimd_mm512_mul_pd(easysimd_mm512_sub_pd(one, a), easysimd_mm512_add_pd(one, a));
    easysimd__m512d lnx = easysimd_mm512_log_pd(a);

    easysimd__m512d tt1 = easysimd_mm512_mul_pd(easysimd_mm512_set1_pd(EASYSIMD_MATH_PI), easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.147)));
    tt1 = easysimd_mm512_div_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(2.0)), tt1);
    tt1 = easysimd_mm512_add_pd(tt1, easysimd_mm512_mul_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.5)), lnx));

    easysimd__m512d tt2 = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0) / EASYSIMD_FLOAT64_C(0.147));
    tt2 = easysimd_mm512_mul_pd(tt2, lnx);

    easysimd__m512d r = easysimd_mm512_mul_pd(tt1, tt1);
    r = easysimd_mm512_sub_pd(r, tt2);
    r = easysimd_mm512_sqrt_pd(r);
    r = easysimd_mm512_add_pd(easysimd_x_mm512_negate_pd(tt1), r);
    r = easysimd_mm512_sqrt_pd(r);

    return easysimd_mm512_mul_pd(sgn, r);
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_erfinv(a_.f64[i]);
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erfinv_pd
  #define _mm512_erfinv_pd(a) easysimd_mm512_erfinv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_erfinv_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erfinv_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_erfinv_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erfinv_ps
  #define _mm512_mask_erfinv_ps(src, k, a) easysimd_mm512_mask_erfinv_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_erfinv_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erfinv_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_erfinv_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erfinv_pd
  #define _mm512_mask_erfinv_pd(src, k, a) easysimd_mm512_mask_erfinv_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_erfcinv_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erfcinv_ps(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    easysimd__m128 matched, retval = easysimd_mm_setzero_ps();

    { /* if (a < 2.0f && a > 0.0625f) */
      matched = easysimd_mm_cmplt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(2.0)));
      matched = easysimd_mm_and_ps(matched, easysimd_mm_cmpgt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.0625))));

      if (!easysimd_mm_test_all_zeros(easysimd_mm_castps_si128(matched), easysimd_x_mm_setone_si128())) {
        retval = easysimd_mm_erfinv_ps(easysimd_mm_sub_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0)), a));
      }

      if (easysimd_mm_test_all_ones(easysimd_mm_castps_si128(matched))) {
        return retval;
      }
    }

    { /* else if (a < 0.0625f && a > 0.0f) */
      easysimd__m128 mask = easysimd_mm_cmplt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.0625)));
      mask = easysimd_mm_and_ps(mask, easysimd_mm_cmpgt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.0))));
      mask = easysimd_mm_andnot_ps(matched, mask);

      if (!easysimd_mm_test_all_zeros(easysimd_mm_castps_si128(mask), easysimd_x_mm_setone_si128())) {
        matched = easysimd_mm_or_ps(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m128 t = easysimd_x_mm_negate_ps(easysimd_mm_log_ps(a));
        t = easysimd_mm_sqrt_ps(t);
        t = easysimd_mm_div_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0)), t);

        const easysimd__m128 p[] = {
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.1550470003116)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.382719649631)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.690969348887)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-1.128081391617)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.680544246825)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-0.164441567910))
        };

        const easysimd__m128 q[] = {
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.155024849822)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.385228141995)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.000000000000))
        };

        /* float numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) */
        easysimd__m128 numerator = easysimd_mm_fmadd_ps(p[5], t, p[4]);
        numerator = easysimd_mm_fmadd_ps(numerator, t, p[3]);
        numerator = easysimd_mm_fmadd_ps(numerator, t, p[2]);
        numerator = easysimd_mm_fmadd_ps(numerator, t, p[1]);
        numerator = easysimd_mm_add_ps(numerator, easysimd_mm_div_ps(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m128 denominator = easysimd_mm_fmadd_ps(q[2], t, q[1]);
        denominator = easysimd_mm_fmadd_ps(denominator, t, q[0]);

        easysimd__m128 res = easysimd_mm_div_ps(numerator, denominator);

        retval = easysimd_mm_or_ps(retval, easysimd_mm_and_ps(mask, res));
      }
    }

    { /* else if (a < 0.0f) */
      easysimd__m128 mask = easysimd_mm_cmplt_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.0)));
      mask = easysimd_mm_andnot_ps(matched, mask);

      if (!easysimd_mm_test_all_zeros(easysimd_mm_castps_si128(mask), easysimd_x_mm_setone_si128())) {
        matched = easysimd_mm_or_ps(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m128 t = easysimd_x_mm_negate_ps(easysimd_mm_log_ps(a));
        t = easysimd_mm_sqrt_ps(t);
        t = easysimd_mm_div_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0)), t);

        const easysimd__m128 p[] = {
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.00980456202915)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.36366788917100)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.97302949837000)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( -0.5374947401000))
        };

        const easysimd__m128 q[] = {
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.00980451277802)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 0.36369997154400)),
          easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C( 1.00000000000000))
        };

        /* float numerator = (p[0] / t + p[1] + t * (p[2] + t * p[3])) */
        easysimd__m128 numerator = easysimd_mm_fmadd_ps(p[3], t, p[2]);
        numerator = easysimd_mm_fmadd_ps(numerator, t, p[1]);
        numerator = easysimd_mm_add_ps(numerator, easysimd_mm_div_ps(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m128 denominator = easysimd_mm_fmadd_ps(q[2], t, q[1]);
        denominator = easysimd_mm_fmadd_ps(denominator, t, q[0]);

        easysimd__m128 res = easysimd_mm_div_ps(numerator, denominator);

        retval = easysimd_mm_or_ps(retval, easysimd_mm_and_ps(mask, res));

        if (easysimd_mm_test_all_ones(easysimd_mm_castps_si128(matched))) {
          return retval;
        }
      }
    }

    { /* else if (a == 0.0f) */
      easysimd__m128 mask = easysimd_mm_cmpeq_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(0.0)));
      mask = easysimd_mm_andnot_ps(matched, mask);
      matched = easysimd_mm_or_ps(matched, mask);

      easysimd__m128 res = easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITYF);

      retval = easysimd_mm_or_ps(retval, easysimd_mm_and_ps(mask, res));
    }

    { /* else */
      /* (a >= 2.0f) */
      retval = easysimd_mm_or_ps(retval, easysimd_mm_andnot_ps(matched, easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITYF)));
    }

    return retval;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_erfcinvf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erfcinv_ps
  #define _mm_erfcinv_ps(a) easysimd_mm_erfcinv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_erfcinv_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_erfcinv_pd(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    easysimd__m128d matched, retval = easysimd_mm_setzero_pd();

    { /* if (a < 2.0 && a > 0.0625) */
      matched = easysimd_mm_cmplt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(2.0)));
      matched = easysimd_mm_and_pd(matched, easysimd_mm_cmpgt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.0625))));

      if (!easysimd_mm_test_all_zeros(easysimd_mm_castpd_si128(matched), easysimd_x_mm_setone_si128())) {
        retval = easysimd_mm_erfinv_pd(easysimd_mm_sub_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)), a));
      }

      if (easysimd_mm_test_all_ones(easysimd_mm_castpd_si128(matched))) {
        return retval;
      }
    }

    { /* else if (a < 0.0625 && a > 0.0) */
      easysimd__m128d mask = easysimd_mm_cmplt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.0625)));
      mask = easysimd_mm_and_pd(mask, easysimd_mm_cmpgt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.0))));
      mask = easysimd_mm_andnot_pd(matched, mask);

      if (!easysimd_mm_test_all_zeros(easysimd_mm_castpd_si128(mask), easysimd_x_mm_setone_si128())) {
        matched = easysimd_mm_or_pd(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m128d t = easysimd_x_mm_negate_pd(easysimd_mm_log_pd(a));
        t = easysimd_mm_sqrt_pd(t);
        t = easysimd_mm_div_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)), t);

        const easysimd__m128d p[] = {
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.1550470003116)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.382719649631)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.690969348887)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-1.128081391617)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.680544246825)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-0.164441567910))
        };

        const easysimd__m128d q[] = {
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.155024849822)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.385228141995)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.000000000000))
        };

        /* float numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) */
        easysimd__m128d numerator = easysimd_mm_fmadd_pd(p[5], t, p[4]);
        numerator = easysimd_mm_fmadd_pd(numerator, t, p[3]);
        numerator = easysimd_mm_fmadd_pd(numerator, t, p[2]);
        numerator = easysimd_mm_fmadd_pd(numerator, t, p[1]);
        numerator = easysimd_mm_add_pd(numerator, easysimd_mm_div_pd(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m128d denominator = easysimd_mm_fmadd_pd(q[2], t, q[1]);
        denominator = easysimd_mm_fmadd_pd(denominator, t, q[0]);

        easysimd__m128d res = easysimd_mm_div_pd(numerator, denominator);

        retval = easysimd_mm_or_pd(retval, easysimd_mm_and_pd(mask, res));
      }
    }

    { /* else if (a < 0.0) */
      easysimd__m128d mask = easysimd_mm_cmplt_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.0)));
      mask = easysimd_mm_andnot_pd(matched, mask);

      if (!easysimd_mm_test_all_zeros(easysimd_mm_castpd_si128(mask), easysimd_x_mm_setone_si128())) {
        matched = easysimd_mm_or_pd(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m128d t = easysimd_x_mm_negate_pd(easysimd_mm_log_pd(a));
        t = easysimd_mm_sqrt_pd(t);
        t = easysimd_mm_div_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)), t);

        const easysimd__m128d p[] = {
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.00980456202915)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.36366788917100)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.97302949837000)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( -0.5374947401000))
        };

        const easysimd__m128d q[] = {
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.00980451277802)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 0.36369997154400)),
          easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C( 1.00000000000000))
        };

        /* float numerator = (p[0] / t + p[1] + t * (p[2] + t * p[3])) */
        easysimd__m128d numerator = easysimd_mm_fmadd_pd(p[3], t, p[2]);
        numerator = easysimd_mm_fmadd_pd(numerator, t, p[1]);
        numerator = easysimd_mm_add_pd(numerator, easysimd_mm_div_pd(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m128d denominator = easysimd_mm_fmadd_pd(q[2], t, q[1]);
        denominator = easysimd_mm_fmadd_pd(denominator, t, q[0]);

        easysimd__m128d res = easysimd_mm_div_pd(numerator, denominator);

        retval = easysimd_mm_or_pd(retval, easysimd_mm_and_pd(mask, res));

        if (easysimd_mm_test_all_ones(easysimd_mm_castpd_si128(matched))) {
          return retval;
        }
      }
    }

    { /* else if (a == 0.0) */
      easysimd__m128d mask = easysimd_mm_cmpeq_pd(a, easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(0.0)));
      mask = easysimd_mm_andnot_pd(matched, mask);
      matched = easysimd_mm_or_pd(matched, mask);

      easysimd__m128d res = easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY);

      retval = easysimd_mm_or_pd(retval, easysimd_mm_and_pd(mask, res));
    }

    { /* else */
      /* (a >= 2.0) */
      retval = easysimd_mm_or_pd(retval, easysimd_mm_andnot_pd(matched, easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY)));
    }

    return retval;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_erfcinv(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_erfcinv_pd
  #define _mm_erfcinv_pd(a) easysimd_mm_erfcinv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_erfcinv_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erfcinv_ps(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(256)
    easysimd__m256 matched, retval = easysimd_mm256_setzero_ps();

    { /* if (a < 2.0f && a > 0.0625f) */
      matched = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(2.0)), EASYSIMD_CMP_LT_OQ);
      matched = easysimd_mm256_and_ps(matched, easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0625)), EASYSIMD_CMP_GT_OQ));

      if (!easysimd_mm256_testz_ps(matched, matched)) {
        retval = easysimd_mm256_erfinv_ps(easysimd_mm256_sub_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0)), a));
      }

      if (easysimd_x_mm256_test_all_ones(easysimd_mm256_castps_si256(matched))) {
        return retval;
      }
    }

    { /* else if (a < 0.0625f && a > 0.0f) */
      easysimd__m256 mask = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0625)), EASYSIMD_CMP_LT_OQ);
      mask = easysimd_mm256_and_ps(mask, easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_GT_OQ));
      mask = easysimd_mm256_andnot_ps(matched, mask);

      if (!easysimd_mm256_testz_ps(mask, mask)) {
        matched = easysimd_mm256_or_ps(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m256 t = easysimd_x_mm256_negate_ps(easysimd_mm256_log_ps(a));
        t = easysimd_mm256_sqrt_ps(t);
        t = easysimd_mm256_div_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0)), t);

        const easysimd__m256 p[] = {
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.1550470003116)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.382719649631)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.690969348887)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-1.128081391617)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.680544246825)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-0.16444156791))
        };

        const easysimd__m256 q[] = {
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.155024849822)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.385228141995)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.000000000000))
        };

        /* float numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) */
        easysimd__m256 numerator = easysimd_mm256_fmadd_ps(p[5], t, p[4]);
        numerator = easysimd_mm256_fmadd_ps(numerator, t, p[3]);
        numerator = easysimd_mm256_fmadd_ps(numerator, t, p[2]);
        numerator = easysimd_mm256_fmadd_ps(numerator, t, p[1]);
        numerator = easysimd_mm256_add_ps(numerator, easysimd_mm256_div_ps(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m256 denominator = easysimd_mm256_fmadd_ps(q[2], t, q[1]);
        denominator = easysimd_mm256_fmadd_ps(denominator, t, q[0]);

        easysimd__m256 res = easysimd_mm256_div_ps(numerator, denominator);

        retval = easysimd_mm256_or_ps(retval, easysimd_mm256_and_ps(mask, res));
      }
    }

    { /* else if (a < 0.0f) */
      easysimd__m256 mask = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_LT_OQ);
      mask = easysimd_mm256_andnot_ps(matched, mask);

      if (!easysimd_mm256_testz_ps(mask, mask)) {
        matched = easysimd_mm256_or_ps(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m256 t = easysimd_x_mm256_negate_ps(easysimd_mm256_log_ps(a));
        t = easysimd_mm256_sqrt_ps(t);
        t = easysimd_mm256_div_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0)), t);

        const easysimd__m256 p[] = {
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.00980456202915)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.36366788917100)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.97302949837000)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-0.5374947401000))
        };

        const easysimd__m256 q[] = {
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.00980451277802)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 0.36369997154400)),
          easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C( 1.00000000000000))
        };

        /* float numerator = (p[0] / t + p[1] + t * (p[2] + t * p[3])) */
        easysimd__m256 numerator = easysimd_mm256_fmadd_ps(p[3], t, p[2]);
        numerator = easysimd_mm256_fmadd_ps(numerator, t, p[1]);
        numerator = easysimd_mm256_add_ps(numerator, easysimd_mm256_div_ps(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m256 denominator = easysimd_mm256_fmadd_ps(q[2], t, q[1]);
        denominator = easysimd_mm256_fmadd_ps(denominator, t, q[0]);

        easysimd__m256 res = easysimd_mm256_div_ps(numerator, denominator);

        retval = easysimd_mm256_or_ps(retval, easysimd_mm256_and_ps(mask, res));

        if (easysimd_x_mm256_test_all_ones(easysimd_mm256_castps_si256(matched))) {
          return retval;
        }
      }
    }

    { /* else if (a == 0.0f) */
      easysimd__m256 mask = easysimd_mm256_cmp_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      mask = easysimd_mm256_andnot_ps(matched, mask);
      matched = easysimd_mm256_or_ps(matched, mask);

      easysimd__m256 res = easysimd_mm256_set1_ps(EASYSIMD_MATH_INFINITYF);

      retval = easysimd_mm256_or_ps(retval, easysimd_mm256_and_ps(mask, res));
    }

    { /* else */
      /* (a >= 2.0f) */
      retval = easysimd_mm256_or_ps(retval, easysimd_mm256_andnot_ps(matched, easysimd_mm256_set1_ps(-EASYSIMD_MATH_INFINITYF)));
    }

    return retval;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_erfcinv_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_erfcinvf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erfcinv_ps
  #define _mm256_erfcinv_ps(a) easysimd_mm256_erfcinv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_erfcinv_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_erfcinv_pd(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(256)
    easysimd__m256d matched, retval = easysimd_mm256_setzero_pd();

    { /* if (a < 2.0 && a > 0.0625) */
      matched = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(2.0)), EASYSIMD_CMP_LT_OQ);
      matched = easysimd_mm256_and_pd(matched, easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0625)), EASYSIMD_CMP_GT_OQ));

      if (!easysimd_mm256_testz_pd(matched, matched)) {
        retval = easysimd_mm256_erfinv_pd(easysimd_mm256_sub_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)), a));
      }

      if (easysimd_x_mm256_test_all_ones(easysimd_mm256_castpd_si256(matched))) {
        return retval;
      }
    }

    { /* else if (a < 0.0625 && a > 0.0) */
      easysimd__m256d mask = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0625)), EASYSIMD_CMP_LT_OQ);
      mask = easysimd_mm256_and_pd(mask, easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_GT_OQ));
      mask = easysimd_mm256_andnot_pd(matched, mask);

      if (!easysimd_mm256_testz_pd(mask, mask)) {
        matched = easysimd_mm256_or_pd(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m256d t = easysimd_x_mm256_negate_pd(easysimd_mm256_log_pd(a));
        t = easysimd_mm256_sqrt_pd(t);
        t = easysimd_mm256_div_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)), t);

        const easysimd__m256d p[] = {
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.1550470003116)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.382719649631)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.690969348887)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-1.128081391617)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.680544246825)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-0.16444156791))
        };

        const easysimd__m256d q[] = {
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.155024849822)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.385228141995)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.000000000000))
        };

        /* float numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) */
        easysimd__m256d numerator = easysimd_mm256_fmadd_pd(p[5], t, p[4]);
        numerator = easysimd_mm256_fmadd_pd(numerator, t, p[3]);
        numerator = easysimd_mm256_fmadd_pd(numerator, t, p[2]);
        numerator = easysimd_mm256_fmadd_pd(numerator, t, p[1]);
        numerator = easysimd_mm256_add_pd(numerator, easysimd_mm256_div_pd(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m256d denominator = easysimd_mm256_fmadd_pd(q[2], t, q[1]);
        denominator = easysimd_mm256_fmadd_pd(denominator, t, q[0]);

        easysimd__m256d res = easysimd_mm256_div_pd(numerator, denominator);

        retval = easysimd_mm256_or_pd(retval, easysimd_mm256_and_pd(mask, res));
      }
    }

    { /* else if (a < 0.0) */
      easysimd__m256d mask = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_LT_OQ);
      mask = easysimd_mm256_andnot_pd(matched, mask);

      if (!easysimd_mm256_testz_pd(mask, mask)) {
        matched = easysimd_mm256_or_pd(matched, mask);

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m256d t = easysimd_x_mm256_negate_pd(easysimd_mm256_log_pd(a));
        t = easysimd_mm256_sqrt_pd(t);
        t = easysimd_mm256_div_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)), t);

        const easysimd__m256d p[] = {
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.00980456202915)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.36366788917100)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.97302949837000)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-0.5374947401000))
        };

        const easysimd__m256d q[] = {
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.00980451277802)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 0.36369997154400)),
          easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C( 1.00000000000000))
        };

        /* float numerator = (p[0] / t + p[1] + t * (p[2] + t * p[3])) */
        easysimd__m256d numerator = easysimd_mm256_fmadd_pd(p[3], t, p[2]);
        numerator = easysimd_mm256_fmadd_pd(numerator, t, p[1]);
        numerator = easysimd_mm256_add_pd(numerator, easysimd_mm256_div_pd(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m256d denominator = easysimd_mm256_fmadd_pd(q[2], t, q[1]);
        denominator = easysimd_mm256_fmadd_pd(denominator, t, q[0]);

        easysimd__m256d res = easysimd_mm256_div_pd(numerator, denominator);

        retval = easysimd_mm256_or_pd(retval, easysimd_mm256_and_pd(mask, res));

        if (easysimd_x_mm256_test_all_ones(easysimd_mm256_castpd_si256(matched))) {
          return retval;
        }
      }
    }

    { /* else if (a == 0.0) */
      easysimd__m256d mask = easysimd_mm256_cmp_pd(a, easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      mask = easysimd_mm256_andnot_pd(matched, mask);
      matched = easysimd_mm256_or_pd(matched, mask);

      easysimd__m256d res = easysimd_mm256_set1_pd(EASYSIMD_MATH_INFINITY);

      retval = easysimd_mm256_or_pd(retval, easysimd_mm256_and_pd(mask, res));
    }

    { /* else */
      /* (a >= 2.0) */
      retval = easysimd_mm256_or_pd(retval, easysimd_mm256_andnot_pd(matched, easysimd_mm256_set1_pd(-EASYSIMD_MATH_INFINITY)));
    }

    return retval;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_erfcinv_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_erfcinv(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_erfcinv_pd
  #define _mm256_erfcinv_pd(a) easysimd_mm256_erfcinv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_erfcinv_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erfcinv_ps(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && (!defined(EASYSIMD_ARCH_ARM) || defined(EASYSIMD_ARCH_AARCH64))
    /* The results on Arm are *slightly* off, which causes problems for
     * the edge cases; for example, if you pass 2.0 sqrt will be called
     * with a value of -0.0 instead of 0.0, resulting in a NaN. */
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
      r_.m256[i] = easysimd_mm256_erfcinv_ps(a_.m256[i]);
    }
    return easysimd__m512_from_private(r_);
  #else
    easysimd__m512 retval = easysimd_mm512_setzero_ps();
    easysimd__mmask16 matched;

    { /* if (a < 2.0f && a > 0.0625f) */
      matched =  easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(2.0)), EASYSIMD_CMP_LT_OQ);
      matched &= easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0625)), EASYSIMD_CMP_GT_OQ);

      if (matched != 0) {
        retval = easysimd_mm512_erfinv_ps(easysimd_mm512_sub_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)), a));
      }

      if (matched == 1) {
        return retval;
      }
    }

    { /* else if (a < 0.0625f && a > 0.0f) */
      easysimd__mmask16 mask = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0625)), EASYSIMD_CMP_LT_OQ);
      mask &= easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_GT_OQ);
      mask = ~matched & mask;

      if (mask != 0) {
        matched = matched | mask;

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m512 t = easysimd_x_mm512_negate_ps(easysimd_mm512_log_ps(a));
        t = easysimd_mm512_sqrt_ps(t);
        t = easysimd_mm512_div_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)), t);

        const easysimd__m512 p[] = {
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.1550470003116)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.382719649631)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.690969348887)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-1.128081391617)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.680544246825)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-0.16444156791))
        };

        const easysimd__m512 q[] = {
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.155024849822)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.385228141995)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.000000000000))
        };

        /* float numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) */
        easysimd__m512 numerator = easysimd_mm512_fmadd_ps(p[5], t, p[4]);
        numerator = easysimd_mm512_fmadd_ps(numerator, t, p[3]);
        numerator = easysimd_mm512_fmadd_ps(numerator, t, p[2]);
        numerator = easysimd_mm512_fmadd_ps(numerator, t, p[1]);
        numerator = easysimd_mm512_add_ps(numerator, easysimd_mm512_div_ps(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m512 denominator = easysimd_mm512_fmadd_ps(q[2], t, q[1]);
        denominator = easysimd_mm512_fmadd_ps(denominator, t, q[0]);

        easysimd__m512 res = easysimd_mm512_div_ps(numerator, denominator);

        retval = easysimd_mm512_or_ps(retval, easysimd_mm512_maskz_mov_ps(mask, res));
      }
    }

    { /* else if (a < 0.0f) */
      easysimd__mmask16 mask = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_LT_OQ);
      mask = ~matched & mask;

      if (mask != 0) {
        matched = matched | mask;

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m512 t = easysimd_x_mm512_negate_ps(easysimd_mm512_log_ps(a));
        t = easysimd_mm512_sqrt_ps(t);
        t = easysimd_mm512_div_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)), t);

        const easysimd__m512 p[] = {
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.00980456202915)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.36366788917100)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.97302949837000)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( -0.5374947401000))
        };

        const easysimd__m512 q[] = {
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.00980451277802)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 0.36369997154400)),
          easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C( 1.00000000000000))
        };

        /* float numerator = (p[0] / t + p[1] + t * (p[2] + t * p[3])) */
        easysimd__m512 numerator = easysimd_mm512_fmadd_ps(p[3], t, p[2]);
        numerator = easysimd_mm512_fmadd_ps(numerator, t, p[1]);
        numerator = easysimd_mm512_add_ps(numerator, easysimd_mm512_div_ps(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m512 denominator = easysimd_mm512_fmadd_ps(q[2], t, q[1]);
        denominator = easysimd_mm512_fmadd_ps(denominator, t, q[0]);

        easysimd__m512 res = easysimd_mm512_div_ps(numerator, denominator);

        retval = easysimd_mm512_or_ps(retval, easysimd_mm512_maskz_mov_ps(mask, res));

        if (matched == 1) {
          return retval;
        }
      }
    }

    { /* else if (a == 0.0f) */
      easysimd__mmask16 mask = easysimd_mm512_cmp_ps_mask(a, easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      mask = ~matched & mask;
      matched = matched | mask;

      easysimd__m512 res = easysimd_mm512_set1_ps(EASYSIMD_MATH_INFINITYF);

      retval = easysimd_mm512_or_ps(retval, easysimd_mm512_maskz_mov_ps(mask, res));
    }

    { /* else */
      /* (a >= 2.0f) */
      retval = easysimd_mm512_or_ps(retval, easysimd_mm512_maskz_mov_ps(~matched, easysimd_mm512_set1_ps(-EASYSIMD_MATH_INFINITYF)));
    }

    return retval;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erfcinv_ps
  #define _mm512_erfcinv_ps(a) easysimd_mm512_erfcinv_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_erfcinv_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_erfcinv_pd(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
      r_.m256d[i] = easysimd_mm256_erfcinv_pd(a_.m256d[i]);
    }
    return easysimd__m512d_from_private(r_);
  #else
    easysimd__m512d retval = easysimd_mm512_setzero_pd();
    easysimd__mmask8 matched;

    { /* if (a < 2.0f && a > 0.0625f) */
      matched =  easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(2.0)), EASYSIMD_CMP_LT_OQ);
      matched &= easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0625)), EASYSIMD_CMP_GT_OQ);

      if (matched != 0) {
        retval = easysimd_mm512_erfinv_pd(easysimd_mm512_sub_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)), a));
      }

      if (matched == 1) {
        return retval;
      }
    }

    { /* else if (a < 0.0625f && a > 0.0f) */
      easysimd__mmask8 mask = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0625)), EASYSIMD_CMP_LT_OQ);
      mask &= easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_GT_OQ);
      mask = ~matched & mask;

      if (mask != 0) {
        matched = matched | mask;

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m512d t = easysimd_x_mm512_negate_pd(easysimd_mm512_log_pd(a));
        t = easysimd_mm512_sqrt_pd(t);
        t = easysimd_mm512_div_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)), t);

        const easysimd__m512d p[] = {
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.1550470003116)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.382719649631)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.690969348887)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-1.128081391617)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.680544246825)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-0.16444156791))
        };

        const easysimd__m512d q[] = {
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.155024849822)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.385228141995)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.000000000000))
        };

        /* float numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) */
        easysimd__m512d numerator = easysimd_mm512_fmadd_pd(p[5], t, p[4]);
        numerator = easysimd_mm512_fmadd_pd(numerator, t, p[3]);
        numerator = easysimd_mm512_fmadd_pd(numerator, t, p[2]);
        numerator = easysimd_mm512_fmadd_pd(numerator, t, p[1]);
        numerator = easysimd_mm512_add_pd(numerator, easysimd_mm512_div_pd(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m512d denominator = easysimd_mm512_fmadd_pd(q[2], t, q[1]);
        denominator = easysimd_mm512_fmadd_pd(denominator, t, q[0]);

        easysimd__m512d res = easysimd_mm512_div_pd(numerator, denominator);

        retval = easysimd_mm512_or_pd(retval, easysimd_mm512_maskz_mov_pd(mask, res));
      }
    }

    { /* else if (a < 0.0f) */
      easysimd__mmask8 mask = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_LT_OQ);
      mask = ~matched & mask;

      if (mask != 0) {
        matched = matched | mask;

        /* t =  1/(sqrt(-log(a))) */
        easysimd__m512d t = easysimd_x_mm512_negate_pd(easysimd_mm512_log_pd(a));
        t = easysimd_mm512_sqrt_pd(t);
        t = easysimd_mm512_div_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)), t);

        const easysimd__m512d p[] = {
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.00980456202915)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.36366788917100)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.97302949837000)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( -0.5374947401000))
        };

        const easysimd__m512d q[] = {
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.00980451277802)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 0.36369997154400)),
          easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C( 1.00000000000000))
        };

        /* float numerator = (p[0] / t + p[1] + t * (p[2] + t * p[3])) */
        easysimd__m512d numerator = easysimd_mm512_fmadd_pd(p[3], t, p[2]);
        numerator = easysimd_mm512_fmadd_pd(numerator, t, p[1]);
        numerator = easysimd_mm512_add_pd(numerator, easysimd_mm512_div_pd(p[0], t));

        /* float denominator = (q[0] + t * (q[1] + t * (q[2]))) */
        easysimd__m512d denominator = easysimd_mm512_fmadd_pd(q[2], t, q[1]);
        denominator = easysimd_mm512_fmadd_pd(denominator, t, q[0]);

        easysimd__m512d res = easysimd_mm512_div_pd(numerator, denominator);

        retval = easysimd_mm512_or_pd(retval, easysimd_mm512_maskz_mov_pd(mask, res));

        if (matched == 1) {
          return retval;
        }
      }
    }

    { /* else if (a == 0.0f) */
      easysimd__mmask8 mask = easysimd_mm512_cmp_pd_mask(a, easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), EASYSIMD_CMP_EQ_OQ);
      mask = ~matched & mask;
      matched = matched | mask;

      easysimd__m512d res = easysimd_mm512_set1_pd(EASYSIMD_MATH_INFINITY);

      retval = easysimd_mm512_or_pd(retval, easysimd_mm512_maskz_mov_pd(mask, res));
    }

    { /* else */
      /* (a >= 2.0f) */
      retval = easysimd_mm512_or_pd(retval, easysimd_mm512_maskz_mov_pd(~matched, easysimd_mm512_set1_pd(-EASYSIMD_MATH_INFINITY)));
    }

    return retval;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_erfcinv_pd
  #define _mm512_erfcinv_pd(a) easysimd_mm512_erfcinv_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_erfcinv_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erfcinv_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_erfcinv_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erfcinv_ps
  #define _mm512_mask_erfcinv_ps(src, k, a) easysimd_mm512_mask_erfcinv_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_erfcinv_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_erfcinv_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_erfcinv_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_erfcinv_pd
  #define _mm512_mask_erfcinv_pd(src, k, a) easysimd_mm512_mask_erfcinv_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_logb_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_logb_ps(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_logbf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_logb_ps
  #define _mm_logb_ps(a) easysimd_mm_logb_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_logb_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_logb_pd(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_logb(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_logb_pd
  #define _mm_logb_pd(a) easysimd_mm_logb_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_logb_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_logb_ps(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_logb_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_logbf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_logb_ps
  #define _mm256_logb_ps(a) easysimd_mm256_logb_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_logb_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_logb_pd(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_logb_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_logb(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_logb_pd
  #define _mm256_logb_pd(a) easysimd_mm256_logb_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_logb_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_logb_ps(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_logb_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_logbf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_logb_ps
  #define _mm512_logb_ps(a) easysimd_mm512_logb_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_logb_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_logb_pd(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_logb_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_logb(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_logb_pd
  #define _mm512_logb_pd(a) easysimd_mm512_logb_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_logb_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_logb_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_logb_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_logb_ps
  #define _mm512_mask_logb_ps(src, k, a) easysimd_mm512_mask_logb_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_logb_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_logb_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_logb_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_logb_pd
  #define _mm512_mask_logb_pd(src, k, a) easysimd_mm512_mask_logb_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_log2_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log2_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_MATH_SLEEF_VERSION_CHECK(3,4,0) && (EASYSIMD_ACCURACY_PREFERENCE <= 1)
      return Sleef_log2f4_u35(a);
    #else
      return Sleef_log2f4_u10(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_log2f(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log2_ps
  #define _mm_log2_ps(a) easysimd_mm_log2_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_log2_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log2_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_MATH_SLEEF_VERSION_CHECK(3,4,0) && (EASYSIMD_ACCURACY_PREFERENCE <= 1)
      return Sleef_log2d2_u35(a);
    #else
      return Sleef_log2d2_u10(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_log2(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log2_pd
  #define _mm_log2_pd(a) easysimd_mm_log2_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_log2_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log2_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_MATH_SLEEF_VERSION_CHECK(3,4,0) && (EASYSIMD_ACCURACY_PREFERENCE <= 1)
      return Sleef_log2f8_u35(a);
    #else
      return Sleef_log2f8_u10(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_log2_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_log2f(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log2_ps
  #define _mm256_log2_ps(a) easysimd_mm256_log2_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_log2_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log2_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_MATH_SLEEF_VERSION_CHECK(3,4,0) && (EASYSIMD_ACCURACY_PREFERENCE <= 1)
      return Sleef_log2d4_u35(a);
    #else
      return Sleef_log2d4_u10(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_log2_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log2(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log2_pd
  #define _mm256_log2_pd(a) easysimd_mm256_log2_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_log2_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log2_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_MATH_SLEEF_VERSION_CHECK(3,4,0) && (EASYSIMD_ACCURACY_PREFERENCE <= 1)
      return Sleef_log2f16_u35(a);
    #else
      return Sleef_log2f16_u10(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_log2_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_log2f(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log2_ps
  #define _mm512_log2_ps(a) easysimd_mm512_log2_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_log2_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log2_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_MATH_SLEEF_VERSION_CHECK(3,4,0) && (EASYSIMD_ACCURACY_PREFERENCE <= 1)
      return Sleef_log2d8_u35(a);
    #else
      return Sleef_log2d8_u10(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_log2_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log2(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log2_pd
  #define _mm512_log2_pd(a) easysimd_mm512_log2_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_log2_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log2_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_log2_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log2_ps
  #define _mm512_mask_log2_ps(src, k, a) easysimd_mm512_mask_log2_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_log2_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log2_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_log2_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log2_pd
  #define _mm512_mask_log2_pd(src, k, a) easysimd_mm512_mask_log2_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_log1p_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log1p_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_log1pf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_log1pf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log1p_ps
  #define _mm_log1p_ps(a) easysimd_mm_log1p_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_log1p_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log1p_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_log1pd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_log1p(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log1p_pd
  #define _mm_log1p_pd(a) easysimd_mm_log1p_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_log1p_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log1p_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_log1pf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_log1p_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_log1pf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log1p_ps
  #define _mm256_log1p_ps(a) easysimd_mm256_log1p_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_log1p_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log1p_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_log1pd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_log1p_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log1p(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log1p_pd
  #define _mm256_log1p_pd(a) easysimd_mm256_log1p_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_log1p_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log1p_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_log1pf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_log1p_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_log1pf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log1p_ps
  #define _mm512_log1p_ps(a) easysimd_mm512_log1p_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_log1p_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log1p_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_log1pd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_log1p_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log1p(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log1p_pd
  #define _mm512_log1p_pd(a) easysimd_mm512_log1p_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_log1p_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log1p_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_log1p_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log1p_ps
  #define _mm512_mask_log1p_ps(src, k, a) easysimd_mm512_mask_log1p_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_log1p_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log1p_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_log1p_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log1p_pd
  #define _mm512_mask_log1p_pd(src, k, a) easysimd_mm512_mask_log1p_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_log10_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log10_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_log10f4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_log10f(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log10_ps
  #define _mm_log10_ps(a) easysimd_mm_log10_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_log10_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_log10_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_log10d2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_log10(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_log10_pd
  #define _mm_log10_pd(a) easysimd_mm_log10_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_log10_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log10_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_log10f8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_log10_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_log10f(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log10_ps
  #define _mm256_log10_ps(a) easysimd_mm256_log10_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_log10_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_log10_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_log10d4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_log10_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log10(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_log10_pd
  #define _mm256_log10_pd(a) easysimd_mm256_log10_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_log10_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log10_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_log10f16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_log10_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_log10f(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log10_ps
  #define _mm512_log10_ps(a) easysimd_mm512_log10_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_log10_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_log10_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_log10d8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_log10_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_log10(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_log10_pd
  #define _mm512_log10_pd(a) easysimd_mm512_log10_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_log10_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log10_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_log10_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log10_ps
  #define _mm512_mask_log10_ps(src, k, a) easysimd_mm512_mask_log10_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_log10_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_log10_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_log10_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_log10_pd
  #define _mm512_mask_log10_pd(src, k, a) easysimd_mm512_mask_log10_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_nearbyint_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_nearbyint_ps(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_nearbyintf(a_.f32[i]);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_nearbyint_ps
  #define _mm512_nearbyint_ps(a) easysimd_mm512_nearbyint_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_nearbyint_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_nearbyint_pd(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_nearbyint(a_.f64[i]);
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_nearbyint_pd
  #define _mm512_nearbyint_pd(a) easysimd_mm512_nearbyint_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_nearbyint_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_nearbyint_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_nearbyint_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_nearbyint_ps
  #define _mm512_mask_nearbyint_ps(src, k, a) easysimd_mm512_mask_nearbyint_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_nearbyint_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_nearbyint_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_nearbyint_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_nearbyint_pd
  #define _mm512_mask_nearbyint_pd(src, k, a) easysimd_mm512_mask_nearbyint_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_pow_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_pow_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_powf4_u10(a, b);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_powf(a_.f32[i], b_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_pow_ps
  #define _mm_pow_ps(a, b) easysimd_mm_pow_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_pow_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_pow_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_powd2_u10(a, b);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_pow(a_.f64[i], b_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_pow_pd
  #define _mm_pow_pd(a, b) easysimd_mm_pow_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_pow_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_pow_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_powf8_u10(a, b);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_powf(a_.f32[i], b_.f32[i]);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_pow_ps
  #define _mm256_pow_ps(a, b) easysimd_mm256_pow_ps(a, b)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_pow_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_pow_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_powd4_u10(a, b);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_pow(a_.f64[i], b_.f64[i]);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_pow_pd
  #define _mm256_pow_pd(a, b) easysimd_mm256_pow_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_pow_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_pow_ps(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_powf16_u10(a, b);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_powf(a_.f32[i], b_.f32[i]);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_pow_ps
  #define _mm512_pow_ps(a, b) easysimd_mm512_pow_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_pow_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_pow_pd(a, b);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_powd8_u10(a, b);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_pow(a_.f64[i], b_.f64[i]);
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_pow_pd
  #define _mm512_pow_pd(a, b) easysimd_mm512_pow_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_pow_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_pow_ps(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_pow_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_pow_ps
  #define _mm512_mask_pow_ps(src, k, a, b) easysimd_mm512_mask_pow_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_pow_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_pow_pd(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_pow_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_pow_pd
  #define _mm512_mask_pow_pd(src, k, a, b) easysimd_mm512_mask_pow_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_clog_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_clog_ps(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    easysimd__m128_private pow_res_ = easysimd__m128_to_private(easysimd_mm_pow_ps(a, easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(2.0))));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
      r_.f32[  i  ] = easysimd_math_logf(easysimd_math_sqrtf(pow_res_.f32[i] + pow_res_.f32[i+1]));
      r_.f32[i + 1] = easysimd_math_atan2f(a_.f32[i + 1], a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_clog_ps
  #define _mm_clog_ps(a) easysimd_mm_clog_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_clog_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm256_clog_ps(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    easysimd__m256_private pow_res_ = easysimd__m256_to_private(easysimd_mm256_pow_ps(a, easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(2.0))));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
      r_.f32[  i  ] = easysimd_math_logf(easysimd_math_sqrtf(pow_res_.f32[i] + pow_res_.f32[i + 1]));
      r_.f32[i + 1] = easysimd_math_atan2f(a_.f32[i + 1], a_.f32[i]);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_clog_ps
  #define _mm256_clog_ps(a) easysimd_mm256_clog_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_csqrt_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_csqrt_ps(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    easysimd__m128 pow_res= easysimd_mm_pow_ps(a,easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(2.0)));
    easysimd__m128_private pow_res_=easysimd__m128_to_private(pow_res);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i+=2) {
      easysimd_float32 sign = easysimd_math_copysignf(EASYSIMD_FLOAT32_C(1.0), a_.f32[i + 1]);
      easysimd_float32 temp = easysimd_math_sqrtf(pow_res_.f32[i] + pow_res_.f32[i+1]);

      r_.f32[  i  ] =       easysimd_math_sqrtf(( a_.f32[i] + temp) / EASYSIMD_FLOAT32_C(2.0));
      r_.f32[i + 1] = sign * easysimd_math_sqrtf((-a_.f32[i] + temp) / EASYSIMD_FLOAT32_C(2.0));
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_csqrt_ps
  #define _mm_csqrt_ps(a) easysimd_mm_csqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_csqrt_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm256_csqrt_ps(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    easysimd__m256 pow_res= easysimd_mm256_pow_ps(a,easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(2.0)));
    easysimd__m256_private pow_res_=easysimd__m256_to_private(pow_res);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i+=2) {
      easysimd_float32 sign = easysimd_math_copysignf(EASYSIMD_FLOAT32_C(1.0), a_.f32[i + 1]);
      easysimd_float32 temp = easysimd_math_sqrtf(pow_res_.f32[i] + pow_res_.f32[i+1]);

      r_.f32[  i  ] =       easysimd_math_sqrtf(( a_.f32[i] + temp) / EASYSIMD_FLOAT32_C(2.0));
      r_.f32[i + 1] = sign * easysimd_math_sqrtf((-a_.f32[i] + temp) / EASYSIMD_FLOAT32_C(2.0));
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_csqrt_ps
  #define _mm256_csqrt_ps(a) easysimd_mm256_csqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epi8(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i8 = a_.i8 % b_.i8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] % b_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epi8
  #define _mm_rem_epi8(a, b) easysimd_mm_rem_epi8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epi16(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i16 = a_.i16 % b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] % b_.i16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epi16
  #define _mm_rem_epi16(a, b) easysimd_mm_rem_epi16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epi32(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i32 = a_.i32 % b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] % b_.i32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_mm_irem_epi32(a, b) easysimd_mm_rem_epi32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epi32
  #define _mm_rem_epi32(a, b) easysimd_mm_rem_epi32(a, b)
  #undef _mm_irem_epi32
  #define _mm_irem_epi32(a, b) easysimd_mm_rem_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epi64(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i64 = a_.i64 % b_.i64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] % b_.i64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epi64
  #define _mm_rem_epi64(a, b) easysimd_mm_rem_epi64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epu8(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u8 = a_.u8 % b_.u8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = a_.u8[i] % b_.u8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epu8
  #define _mm_rem_epu8(a, b) easysimd_mm_rem_epu8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epu16(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u16 = a_.u16 % b_.u16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = a_.u16[i] % b_.u16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epu16
  #define _mm_rem_epu16(a, b) easysimd_mm_rem_epu16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epu32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epu32(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u32 = a_.u32 % b_.u32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] % b_.u32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_mm_urem_epi32(a, b) easysimd_mm_rem_epu32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epu32
  #define _mm_rem_epu32(a, b) easysimd_mm_rem_epu32(a, b)
  #undef _mm_urem_epi32
  #define _mm_urem_epi32(a, b) easysimd_mm_rem_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rem_epu64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_rem_epu64(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u64 = a_.u64 % b_.u64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] % b_.u64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_rem_epu64
  #define _mm_rem_epu64(a, b) easysimd_mm_rem_epu64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epi8(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i8 = a_.i8 % b_.i8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epi8(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
          r_.i8[i] = a_.i8[i] % b_.i8[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epi8
  #define _mm256_rem_epi8(a, b) easysimd_mm256_rem_epi8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epi16(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i16 = a_.i16 % b_.i16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epi16(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = a_.i16[i] % b_.i16[i];
        }
       #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epi16
  #define _mm256_rem_epi16(a, b) easysimd_mm256_rem_epi16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epi32(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i32 = a_.i32 % b_.i32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epi32(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = a_.i32[i] % b_.i32[i];
        }
       #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#define easysimd_mm256_irem_epi32(a, b) easysimd_mm256_rem_epi32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epi32
  #define _mm256_rem_epi32(a, b) easysimd_mm256_rem_epi32(a, b)
  #undef _mm256_irem_epi32
  #define _mm256_irem_epi32(a, b) easysimd_mm256_rem_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epi64(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i64 = a_.i64 % b_.i64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epi64(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.i64[i] = a_.i64[i] % b_.i64[i];
        }
        #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epi64
  #define _mm256_rem_epi64(a, b) easysimd_mm256_rem_epi64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epu8(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u8 = a_.u8 % b_.u8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epu8(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
          r_.u8[i] = a_.u8[i] % b_.u8[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epu8
  #define _mm256_rem_epu8(a, b) easysimd_mm256_rem_epu8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epu16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epu16(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u16 = a_.u16 % b_.u16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epu16(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
          r_.u16[i] = a_.u16[i] % b_.u16[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epu16
  #define _mm256_rem_epu16(a, b) easysimd_mm256_rem_epu16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epu32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epu32(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u32 = a_.u32 % b_.u32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epu32(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = a_.u32[i] % b_.u32[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#define easysimd_mm256_urem_epi32(a, b) easysimd_mm256_rem_epu32(a, b)
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epu32
  #define _mm256_rem_epu32(a, b) easysimd_mm256_rem_epu32(a, b)
  #undef _mm256_urem_epi32
  #define _mm256_urem_epi32(a, b) easysimd_mm256_rem_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rem_epu64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rem_epu64(a, b);
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u64 = a_.u64 % b_.u64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
          r_.m128i[i] = easysimd_mm_rem_epu64(a_.m128i[i], b_.m128i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = a_.u64[i] % b_.u64[i];
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rem_epu64
  #define _mm256_rem_epu64(a, b) easysimd_mm256_rem_epu64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epi8(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i8 = a_.i8 % b_.i8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epi8(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
          r_.i8[i] = a_.i8[i] % b_.i8[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epi8
  #define _mm512_rem_epi8(a, b) easysimd_mm512_rem_epi8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epi16(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i16 = a_.i16 % b_.i16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epi16(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = a_.i16[i] % b_.i16[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epi16
  #define _mm512_rem_epi16(a, b) easysimd_mm512_rem_epi16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epi32(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i32 = a_.i32 % b_.i32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epi32(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = a_.i32[i] % b_.i32[i];
        }
        #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epi32
  #define _mm512_rem_epi32(a, b) easysimd_mm512_rem_epi32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_rem_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_rem_epi32(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_rem_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rem_epi32
  #define _mm512_mask_rem_epi32(src, k, a, b) easysimd_mm512_mask_rem_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epi64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epi64(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.i64 = a_.i64 % b_.i64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epi64(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.i64[i] = a_.i64[i] % b_.i64[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epi64
  #define _mm512_rem_epi64(a, b) easysimd_mm512_rem_epi64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epu8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epu8(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u8 = a_.u8 % b_.u8;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epu8(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
          r_.u8[i] = a_.u8[i] % b_.u8[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epu8
  #define _mm512_rem_epu8(a, b) easysimd_mm512_rem_epu8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epu16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epu16(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u16 = a_.u16 % b_.u16;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epu16(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
          r_.u16[i] = a_.u16[i] % b_.u16[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epu16
  #define _mm512_rem_epu16(a, b) easysimd_mm512_rem_epu16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epu32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epu32(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u32 = a_.u32 % b_.u32;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epu32(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = a_.u32[i] % b_.u32[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epu32
  #define _mm512_rem_epu32(a, b) easysimd_mm512_rem_epu32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_rem_epu32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_rem_epu32(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_rem_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rem_epu32
  #define _mm512_mask_rem_epu32(src, k, a, b) easysimd_mm512_mask_rem_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rem_epu64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rem_epu64(a, b);
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
      r_.u64 = a_.u64 % b_.u64;
    #else
      #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_rem_epu64(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = a_.u64[i] % b_.u64[i];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rem_epu64
  #define _mm512_rem_epu64(a, b) easysimd_mm512_rem_epu64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_recip_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_recip_ps(a);
  #else
    return easysimd_mm512_div_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)), a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_recip_ps
  #define _mm512_recip_ps(a) easysimd_mm512_recip_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_recip_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_recip_pd(a);
  #else
    return easysimd_mm512_div_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)), a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_recip_pd
  #define _mm512_recip_pd(a) easysimd_mm512_recip_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_recip_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_recip_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_recip_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_recip_ps
  #define _mm512_mask_recip_ps(src, k, a) easysimd_mm512_mask_recip_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_recip_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_recip_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_recip_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_recip_pd
  #define _mm512_mask_recip_pd(src, k, a) easysimd_mm512_mask_recip_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_rint_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rint_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_rintf16(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_rintf(a_.f32[i]);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rint_ps
  #define _mm512_rint_ps(a) easysimd_mm512_rint_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_rint_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rint_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_rintd8(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_rint(a_.f64[i]);
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rint_pd
  #define _mm512_rint_pd(a) easysimd_mm512_rint_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_rint_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_rint_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_rint_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rint_ps
  #define _mm512_mask_rint_ps(src, k, a) easysimd_mm512_mask_rint_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_rint_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_rint_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_rint_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rint_pd
  #define _mm512_mask_rint_pd(src, k, a) easysimd_mm512_mask_rint_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sin_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sin_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sinf4_u10(a);
    #else
      return Sleef_sinf4_u35(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_sinf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sin_ps
  #define _mm_sin_ps(a) easysimd_mm_sin_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sin_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sin_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sind2_u10(a);
    #else
      return Sleef_sind2_u35(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_sin(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sin_pd
  #define _mm_sin_pd(a) easysimd_mm_sin_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_sin_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sin_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sinf8_u10(a);
    #else
      return Sleef_sinf8_u35(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_sin_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sinf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sin_ps
  #define _mm256_sin_ps(a) easysimd_mm256_sin_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_sin_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sin_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sind4_u10(a);
    #else
      return Sleef_sind4_u35(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_sin_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sin(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sin_pd
  #define _mm256_sin_pd(a) easysimd_mm256_sin_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_sin_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sin_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sinf16_u10(a);
    #else
      return Sleef_sinf16_u35(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_sin_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sinf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sin_ps
  #define _mm512_sin_ps(a) easysimd_mm512_sin_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_sin_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sin_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sind8_u10(a);
    #else
      return Sleef_sind8_u35(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_sin_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sin(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sin_pd
  #define _mm512_sin_pd(a) easysimd_mm512_sin_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_sin_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sin_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_sin_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sin_ps
  #define _mm512_mask_sin_ps(src, k, a) easysimd_mm512_mask_sin_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_sin_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sin_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_sin_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sin_pd
  #define _mm512_mask_sin_pd(src, k, a) easysimd_mm512_mask_sin_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sincos_ps (easysimd__m128* mem_addr, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sincos_ps(HEDLEY_REINTERPRET_CAST(__m128*, mem_addr), a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    Sleef___m128_2 temp;

    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      temp = Sleef_sincosf4_u10(a);
    #else
      temp = Sleef_sincosf4_u35(a);
    #endif

    *mem_addr = temp.y;
    return temp.x;
  #else
    easysimd__m128 r;

    r = easysimd_mm_sin_ps(a);
    *mem_addr = easysimd_mm_cos_ps(a);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sincos_ps
  #define _mm_sincos_ps(mem_addr, a) easysimd_mm_sincos_ps((mem_addr),(a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sincos_pd (easysimd__m128d* mem_addr, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sincos_pd(HEDLEY_REINTERPRET_CAST(__m128d*, mem_addr), a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    Sleef___m128d_2 temp;

    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      temp = Sleef_sincosd2_u10(a);
    #else
      temp = Sleef_sincosd2_u35(a);
    #endif

    *mem_addr = temp.y;
    return temp.x;
  #else
    easysimd__m128d r;

    r = easysimd_mm_sin_pd(a);
    *mem_addr = easysimd_mm_cos_pd(a);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sincos_pd
  #define _mm_sincos_pd(mem_addr, a) easysimd_mm_sincos_pd((mem_addr),(a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_sincos_ps (easysimd__m256* mem_addr, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sincos_ps(HEDLEY_REINTERPRET_CAST(__m256*, mem_addr), a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    Sleef___m256_2 temp;

    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      temp = Sleef_sincosf8_u10(a);
    #else
      temp = Sleef_sincosf8_u35(a);
    #endif

    *mem_addr = temp.y;
    return temp.x;
  #else
    easysimd__m256 r;

    r = easysimd_mm256_sin_ps(a);
    *mem_addr = easysimd_mm256_cos_ps(a);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sincos_ps
  #define _mm256_sincos_ps(mem_addr, a) easysimd_mm256_sincos_ps((mem_addr),(a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_sincos_pd (easysimd__m256d* mem_addr, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sincos_pd(HEDLEY_REINTERPRET_CAST(__m256d*, mem_addr), a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    Sleef___m256d_2 temp;

    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      temp = Sleef_sincosd4_u10(a);
    #else
      temp = Sleef_sincosd4_u35(a);
    #endif

    *mem_addr = temp.y;
    return temp.x;
  #else
    easysimd__m256d r;

    r = easysimd_mm256_sin_pd(a);
    *mem_addr = easysimd_mm256_cos_pd(a);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sincos_pd
  #define _mm256_sincos_pd(mem_addr, a) easysimd_mm256_sincos_pd((mem_addr),(a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_sincos_ps (easysimd__m512* mem_addr, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sincos_ps(HEDLEY_REINTERPRET_CAST(__m512*, mem_addr), a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    Sleef___m512_2 temp;

    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      temp = Sleef_sincosf16_u10(a);
    #else
      temp = Sleef_sincosf16_u35(a);
    #endif

    *mem_addr = temp.y;
    return temp.x;
  #else
    easysimd__m512 r;

    r = easysimd_mm512_sin_ps(a);
    *mem_addr = easysimd_mm512_cos_ps(a);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sincos_ps
  #define _mm512_sincos_ps(mem_addr, a) easysimd_mm512_sincos_ps((mem_addr),(a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_sincos_pd (easysimd__m512d* mem_addr, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sincos_pd(HEDLEY_REINTERPRET_CAST(__m512d*, mem_addr), a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    Sleef___m512d_2 temp;

    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      temp = Sleef_sincosd8_u10(a);
    #else
      temp = Sleef_sincosd8_u35(a);
    #endif

    *mem_addr = temp.y;
    return temp.x;
  #else
    easysimd__m512d r;

    r = easysimd_mm512_sin_pd(a);
    *mem_addr = easysimd_mm512_cos_pd(a);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sincos_pd
  #define _mm512_sincos_pd(mem_addr, a) easysimd_mm512_sincos_pd((mem_addr),(a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_sincos_ps(easysimd__m512* mem_addr, easysimd__m512 sin_src, easysimd__m512 cos_src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sincos_ps(mem_addr, sin_src, cos_src, k, a);
  #else
    easysimd__m512 cos_res, sin_res;
    sin_res = easysimd_mm512_sincos_ps(&cos_res, a);
    *mem_addr = easysimd_mm512_mask_mov_ps(cos_src, k, cos_res);
    return easysimd_mm512_mask_mov_ps(sin_src, k, sin_res);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sincos_ps
  #define _mm512_mask_sincos_ps(mem_addr, sin_src, cos_src, k, a) easysimd_mm512_mask_sincos_ps(mem_addr, sin_src, cos_src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_sincos_pd(easysimd__m512d* mem_addr, easysimd__m512d sin_src, easysimd__m512d cos_src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sincos_pd(mem_addr, sin_src, cos_src, k, a);
  #else
    easysimd__m512d cos_res, sin_res;
    sin_res = easysimd_mm512_sincos_pd(&cos_res, a);
    *mem_addr = easysimd_mm512_mask_mov_pd(cos_src, k, cos_res);
    return easysimd_mm512_mask_mov_pd(sin_src, k, sin_res);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sincos_pd
  #define _mm512_mask_sincos_pd(mem_addr, sin_src, cos_src, k, a) easysimd_mm512_mask_sincos_pd(mem_addr, sin_src, cos_src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sind_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sind_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sinf4_u10(easysimd_x_mm_deg2rad_ps(a));
    #else
      return Sleef_sinf4_u35(easysimd_x_mm_deg2rad_ps(a));
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_sinf(easysimd_math_deg2radf(a_.f32[i]));
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sind_ps
  #define _mm_sind_ps(a) easysimd_mm_sind_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sind_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sind_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sind2_u10(easysimd_x_mm_deg2rad_pd(a));
    #else
      return Sleef_sind2_u35(easysimd_x_mm_deg2rad_pd(a));
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_sin(easysimd_math_deg2rad(a_.f64[i]));
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sind_pd
  #define _mm_sind_pd(a) easysimd_mm_sind_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_sind_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sind_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sinf8_u10(easysimd_x_mm256_deg2rad_ps(a));
    #else
      return Sleef_sinf8_u35(easysimd_x_mm256_deg2rad_ps(a));
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_sind_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sinf(easysimd_math_deg2radf(a_.f32[i]));
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sind_ps
  #define _mm256_sind_ps(a) easysimd_mm256_sind_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_sind_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sind_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sind4_u10(easysimd_x_mm256_deg2rad_pd(a));
    #else
      return Sleef_sind4_u35(easysimd_x_mm256_deg2rad_pd(a));
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_sind_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sin(easysimd_math_deg2rad(a_.f64[i]));
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sind_pd
  #define _mm256_sind_pd(a) easysimd_mm256_sind_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_sind_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sind_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sinf16_u10(easysimd_x_mm512_deg2rad_ps(a));
    #else
      return Sleef_sinf16_u35(easysimd_x_mm512_deg2rad_ps(a));
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_sind_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sinf(easysimd_math_deg2radf(a_.f32[i]));
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sind_ps
  #define _mm512_sind_ps(a) easysimd_mm512_sind_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_sind_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sind_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_sind8_u10(easysimd_x_mm512_deg2rad_pd(a));
    #else
      return Sleef_sind8_u35(easysimd_x_mm512_deg2rad_pd(a));
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_sind_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sin(easysimd_math_deg2rad(a_.f64[i]));
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sind_pd
  #define _mm512_sind_pd(a) easysimd_mm512_sind_pd(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_sind_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sind_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_sind_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sind_ps
  #define _mm512_mask_sind_ps(src, k, a) easysimd_mm512_mask_sind_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_sind_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sind_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_sind_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sind_pd
  #define _mm512_mask_sind_pd(src, k, a) easysimd_mm512_mask_sind_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sinh_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sinh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_sinhf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_sinhf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sinh_ps
  #define _mm_sinh_ps(a) easysimd_mm_sinh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sinh_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sinh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_sinhd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_sinh(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_sinh_pd
  #define _mm_sinh_pd(a) easysimd_mm_sinh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_sinh_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sinh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_sinhf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_sinh_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sinhf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sinh_ps
  #define _mm256_sinh_ps(a) easysimd_mm256_sinh_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_sinh_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sinh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_sinhd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_sinh_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sinh(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sinh_pd
  #define _mm256_sinh_pd(a) easysimd_mm256_sinh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_sinh_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sinh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_sinhf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_sinh_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sinhf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sinh_ps
  #define _mm512_sinh_ps(a) easysimd_mm512_sinh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_sinh_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sinh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_sinhd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_sinh_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sinh(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sinh_pd
  #define _mm512_sinh_pd(a) easysimd_mm512_sinh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_sinh_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sinh_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_sinh_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sinh_ps
  #define _mm512_mask_sinh_ps(src, k, a) easysimd_mm512_mask_sinh_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_sinh_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sinh_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_sinh_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sinh_pd
  #define _mm512_mask_sinh_pd(src, k, a) easysimd_mm512_mask_sinh_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_svml_ceil_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_ceil_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_ceilf4(a);
  #else
    return easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_TO_POS_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_ceil_ps
  #define _mm_svml_ceil_ps(a) easysimd_mm_svml_ceil_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_svml_ceil_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_ceil_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_ceild2(a);
  #else
    return easysimd_mm_round_pd(a, EASYSIMD_MM_FROUND_TO_POS_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_ceil_pd
  #define _mm_svml_ceil_pd(a) easysimd_mm_svml_ceil_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_svml_ceil_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_ceil_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_ceilf8(a);
  #else
    return easysimd_mm256_round_ps(a, EASYSIMD_MM_FROUND_TO_POS_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_ceil_ps
  #define _mm256_svml_ceil_ps(a) easysimd_mm256_svml_ceil_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_svml_ceil_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_ceil_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_ceild4(a);
  #else
    return easysimd_mm256_round_pd(a, EASYSIMD_MM_FROUND_TO_POS_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_ceil_pd
  #define _mm256_svml_ceil_pd(a) easysimd_mm256_svml_ceil_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_ceil_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_ceil_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_ceilf16(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_ceil_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_ceilf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_ceil_ps
  #define _mm512_ceil_ps(a) easysimd_mm512_ceil_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_ceil_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_ceil_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_ceild8(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_ceil_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_ceil(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_ceil_pd
  #define _mm512_ceil_pd(a) easysimd_mm512_ceil_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_ceil_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_ceil_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_ceil_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_ceil_ps
  #define _mm512_mask_ceil_ps(src, k, a) easysimd_mm512_mask_ceil_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_ceil_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_ceil_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_ceil_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_ceil_pd
  #define _mm512_mask_ceil_pd(src, k, a) easysimd_mm512_mask_ceil_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_svml_floor_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_floor_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_floorf4(a);
  #else
    return easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_floor_ps
  #define _mm_svml_floor_ps(a) easysimd_mm_svml_floor_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_svml_floor_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_floor_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_floord2(a);
  #else
    return easysimd_mm_round_pd(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_floor_pd
  #define _mm_svml_floor_pd(a) easysimd_mm_svml_floor_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_svml_floor_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_floor_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_floorf8(a);
  #else
    return easysimd_mm256_round_ps(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_floor_ps
  #define _mm256_svml_floor_ps(a) easysimd_mm256_svml_floor_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_svml_floor_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_floor_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_floord4(a);
  #else
    return easysimd_mm256_round_pd(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_floor_pd
  #define _mm256_svml_floor_pd(a) easysimd_mm256_svml_floor_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_floor_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_floor_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_floorf16(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_floor_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_floorf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_floor_ps
  #define _mm512_floor_ps(a) easysimd_mm512_floor_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_floor_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_floor_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_floord8(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_floor_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_floor(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_floor_pd
  #define _mm512_floor_pd(a) easysimd_mm512_floor_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_floor_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_floor_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_floor_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_floor_ps
  #define _mm512_mask_floor_ps(src, k, a) easysimd_mm512_mask_floor_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_floor_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_floor_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_floor_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_floor_pd
  #define _mm512_mask_floor_pd(src, k, a) easysimd_mm512_mask_floor_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_svml_round_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_round_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_roundf4(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_roundf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_round_ps
  #define _mm_svml_round_ps(a) easysimd_mm_svml_round_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_svml_round_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_round_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_roundd2(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_round(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_round_pd
  #define _mm_svml_round_pd(a) easysimd_mm_svml_round_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_svml_round_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_round_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_roundf8(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_svml_round_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_roundf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_round_ps
  #define _mm256_svml_round_ps(a) easysimd_mm256_svml_round_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_svml_round_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_round_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_roundd4(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_svml_round_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_round(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_round_pd
  #define _mm256_svml_round_pd(a) easysimd_mm256_svml_round_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_svml_round_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_svml_round_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_roundd8(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_svml_round_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_round(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_svml_round_pd
  #define _mm512_svml_round_pd(a) easysimd_mm512_svml_round_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_svml_round_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_svml_round_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_svml_round_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_svml_round_pd
  #define _mm512_mask_svml_round_pd(src, k, a) easysimd_mm512_mask_svml_round_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_svml_sqrt_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_sqrt_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_sqrtf4(a);
  #else
    return easysimd_mm_sqrt_ps(a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_sqrt_ps
  #define _mm_svml_sqrt_ps(a) easysimd_mm_svml_sqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_svml_sqrt_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_svml_sqrt_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_sqrtd2(a);
  #else
    return easysimd_mm_sqrt_pd(a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_svml_sqrt_pd
  #define _mm_svml_sqrt_pd(a) easysimd_mm_svml_sqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_svml_sqrt_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_sqrt_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_sqrtf8(a);
  #else
    return easysimd_mm256_sqrt_ps(a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_sqrt_ps
  #define _mm256_svml_sqrt_ps(a) easysimd_mm256_svml_sqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_svml_sqrt_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_svml_sqrt_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_sqrtd4(a);
  #else
    return easysimd_mm256_sqrt_pd(a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_svml_sqrt_pd
  #define _mm256_svml_sqrt_pd(a) easysimd_mm256_svml_sqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_svml_sqrt_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_svml_sqrt_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_sqrtf16(a);
  #else
    return easysimd_mm512_sqrt_ps(a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_svml_sqrt_ps
  #define _mm512_svml_sqrt_ps(a) easysimd_mm512_svml_sqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_svml_sqrt_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_svml_sqrt_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_sqrtd8(a);
  #else
    return easysimd_mm512_sqrt_pd(a);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_svml_sqrt_pd
  #define _mm512_svml_sqrt_pd(a) easysimd_mm512_svml_sqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_tan_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_tan_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tanf4_u10(a);
    #else
      return Sleef_tanf4_u35(a);
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_tanf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_tan_ps
  #define _mm_tan_ps(a) easysimd_mm_tan_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_tan_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_tan_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tand2_u10(a);
    #else
      return Sleef_tand2_u35(a);
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_tan(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_tan_pd
  #define _mm_tan_pd(a) easysimd_mm_tan_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_tan_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_tan_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tanf8_u10(a);
    #else
      return Sleef_tanf8_u35(a);
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_tan_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_tanf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_tan_ps
  #define _mm256_tan_ps(a) easysimd_mm256_tan_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_tan_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_tan_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tand4_u10(a);
    #else
      return Sleef_tand4_u35(a);
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_tan_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_tan(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_tan_pd
  #define _mm256_tan_pd(a) easysimd_mm256_tan_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_tan_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_tan_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tanf16_u10(a);
    #else
      return Sleef_tanf16_u35(a);
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_tan_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_tanf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_tan_ps
  #define _mm512_tan_ps(a) easysimd_mm512_tan_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_tan_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_tan_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tand8_u10(a);
    #else
      return Sleef_tand8_u35(a);
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_tan_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_tan(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_tan_pd
  #define _mm512_tan_pd(a) easysimd_mm512_tan_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_tan_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_tan_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_tan_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_tan_ps
  #define _mm512_mask_tan_ps(src, k, a) easysimd_mm512_mask_tan_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_tan_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_tan_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_tan_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_tan_pd
  #define _mm512_mask_tan_pd(src, k, a) easysimd_mm512_mask_tan_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_tand_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_tand_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tanf4_u10(easysimd_x_mm_deg2rad_ps(a));
    #else
      return Sleef_tanf4_u35(easysimd_x_mm_deg2rad_ps(a));
    #endif
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_tanf(easysimd_math_deg2radf(a_.f32[i]));
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_tand_ps
  #define _mm_tand_ps(a) easysimd_mm_tand_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_tand_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_tand_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tand2_u10(easysimd_x_mm_deg2rad_pd(a));
    #else
      return Sleef_tand2_u35(easysimd_x_mm_deg2rad_pd(a));
    #endif
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_tan(easysimd_math_deg2rad(a_.f64[i]));
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_tand_pd
  #define _mm_tand_pd(a) easysimd_mm_tand_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_tand_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_tand_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tanf8_u10(easysimd_x_mm256_deg2rad_ps(a));
    #else
      return Sleef_tanf8_u35(easysimd_x_mm256_deg2rad_ps(a));
    #endif
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_tand_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_tanf(easysimd_math_deg2radf(a_.f32[i]));
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_tand_ps
  #define _mm256_tand_ps(a) easysimd_mm256_tand_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_tand_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_tand_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tand4_u10(easysimd_x_mm256_deg2rad_pd(a));
    #else
      return Sleef_tand4_u35(easysimd_x_mm256_deg2rad_pd(a));
    #endif
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_tand_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_tan(easysimd_math_deg2rad(a_.f64[i]));
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_tand_pd
  #define _mm256_tand_pd(a) easysimd_mm256_tand_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_tand_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_tand_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tanf16_u10(easysimd_x_mm512_deg2rad_ps(a));
    #else
      return Sleef_tanf16_u35(easysimd_x_mm512_deg2rad_ps(a));
    #endif
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_tand_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_tanf(easysimd_math_deg2radf(a_.f32[i]));
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_tand_ps
  #define _mm512_tand_ps(a) easysimd_mm512_tand_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_tand_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_tand_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if EASYSIMD_ACCURACY_PREFERENCE > 1
      return Sleef_tand8_u10(easysimd_x_mm512_deg2rad_pd(a));
    #else
      return Sleef_tand8_u35(easysimd_x_mm512_deg2rad_pd(a));
    #endif
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_tand_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_tan(easysimd_math_deg2rad(a_.f64[i]));
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_tand_pd
  #define _mm512_tand_pd(a) easysimd_mm512_tand_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_tand_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_tand_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_tand_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_tand_ps
  #define _mm512_mask_tand_ps(src, k, a) easysimd_mm512_mask_tand_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_tand_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_tand_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_tand_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_tand_pd
  #define _mm512_mask_tand_pd(src, k, a) easysimd_mm512_mask_tand_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_tanh_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_tanh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_tanhf4_u10(a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_tanhf(a_.f32[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_tanh_ps
  #define _mm_tanh_ps(a) easysimd_mm_tanh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_tanh_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_tanh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_tanhd2_u10(a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_tanh(a_.f64[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_tanh_pd
  #define _mm_tanh_pd(a) easysimd_mm_tanh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_tanh_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_tanh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_tanhf8_u10(a);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = easysimd_mm_tanh_ps(a_.m128[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_tanhf(a_.f32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_tanh_ps
  #define _mm256_tanh_ps(a) easysimd_mm256_tanh_ps(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_tanh_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_tanh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_tanhd4_u10(a);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        r_.m128d[i] = easysimd_mm_tanh_pd(a_.m128d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_tanh(a_.f64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_tanh_pd
  #define _mm256_tanh_pd(a) easysimd_mm256_tanh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_tanh_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_tanh_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_tanhf16_u10(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_tanh_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_tanhf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_tanh_ps
  #define _mm512_tanh_ps(a) easysimd_mm512_tanh_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_tanh_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_tanh_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_tanhd8_u10(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_tanh_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_tanh(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_tanh_pd
  #define _mm512_tanh_pd(a) easysimd_mm512_tanh_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_tanh_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_tanh_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_tanh_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_tanh_ps
  #define _mm512_mask_tanh_ps(src, k, a) easysimd_mm512_mask_tanh_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_tanh_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_tanh_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_tanh_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_tanh_pd
  #define _mm512_mask_tanh_pd(src, k, a) easysimd_mm512_mask_tanh_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_trunc_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_trunc_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_truncf4(a);
  #else
    return easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_TO_ZERO);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_trunc_ps
  #define _mm_trunc_ps(a) easysimd_mm_trunc_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_trunc_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_trunc_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_SSE_NATIVE)
    return Sleef_truncd2(a);
  #else
    return easysimd_mm_round_pd(a, EASYSIMD_MM_FROUND_TO_ZERO);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_trunc_pd
  #define _mm_trunc_pd(a) easysimd_mm_trunc_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_trunc_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_trunc_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_truncf8(a);
  #else
    return easysimd_mm256_round_ps(a, EASYSIMD_MM_FROUND_TO_ZERO);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_trunc_ps
  #define _mm256_trunc_ps(a) easysimd_mm256_trunc_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_trunc_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_trunc_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return Sleef_truncd4(a);
  #else
    return easysimd_mm256_round_pd(a, EASYSIMD_MM_FROUND_TO_ZERO);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_trunc_pd
  #define _mm256_trunc_pd(a) easysimd_mm256_trunc_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_trunc_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_trunc_ps(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_truncf16(a);
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_trunc_ps(a_.m256[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_truncf(a_.f32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_trunc_ps
  #define _mm512_trunc_ps(a) easysimd_mm512_trunc_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_trunc_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_trunc_pd(a);
  #elif defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return Sleef_truncd8(a);
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_trunc_pd(a_.m256d[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_trunc(a_.f64[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_trunc_pd
  #define _mm512_trunc_pd(a) easysimd_mm512_trunc_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_trunc_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_trunc_ps(src, k, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_trunc_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_trunc_ps
  #define _mm512_mask_trunc_ps(src, k, a) easysimd_mm512_mask_trunc_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_trunc_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_trunc_pd(src, k, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_trunc_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_trunc_pd
  #define _mm512_mask_trunc_pd(src, k, a) easysimd_mm512_mask_trunc_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_udivrem_epi32 (easysimd__m128i * mem_addr, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_udivrem_epi32(mem_addr, a, b);
  #else
    easysimd__m128i r;

    r = easysimd_mm_div_epu32(a, b);
    *mem_addr = easysimd_x_mm_sub_epu32(a, easysimd_x_mm_mullo_epu32(r, b));

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm_udivrem_epi32
  #define _mm_udivrem_epi32(mem_addr, a, b) easysimd_mm_udivrem_epi32((mem_addr),(a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_udivrem_epi32 (easysimd__m256i* mem_addr, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_SVML_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_udivrem_epi32(HEDLEY_REINTERPRET_CAST(__m256i*, mem_addr), a, b);
  #else
    easysimd__m256i r;

    r = easysimd_mm256_div_epu32(a, b);
    *mem_addr = easysimd_x_mm256_sub_epu32(a, easysimd_x_mm256_mullo_epu32(r, b));

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SVML_ENABLE_NATIVE_ALIASES)
  #undef _mm256_udivrem_epi32
  #define _mm256_udivrem_epi32(mem_addr, a, b) easysimd_mm256_udivrem_epi32((mem_addr),(a), (b))
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_SVML_H) */
