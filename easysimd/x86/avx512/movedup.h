#if !defined(EASYSIMD_X86_AVX512_MOVEDUP_H)
#define EASYSIMD_X86_AVX512_MOVEDUP_H

#include "types.h"
#include "cast.h"
#include "set.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_movedup_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_movedup_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_f64(a.f64[0]), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      r_;

    r_.f64[0] = a_.f64[0];
    r_.f64[1] = a_.f64[0];
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_movedup_pd
  #define _mm_mask_movedup_pd(src, k, a) easysimd_mm_mask_movedup_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_movedup_pd (easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_movedup_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_f64(a.f64[0]), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      r_;

    r_.f64[0] = a_.f64[0];
    r_.f64[1] = a_.f64[0];
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_movedup_pd
  #define _mm_maskz_movedup_pd(k, a) easysimd_mm_maskz_movedup_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_movedup_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_movedup_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_f64(a.f64[0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svdup_n_f64(a.f64[2]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      src_ = easysimd__m256d_to_private(src),
      a_ = easysimd__m256d_to_private(a),
      r_;

    r_.f64[0] = a_.f64[0];
    r_.f64[1] = a_.f64[0];
    r_.f64[2] = a_.f64[2];
    r_.f64[3] = a_.f64[2];
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : src_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_movedup_pd
  #define _mm256_mask_movedup_pd(src, k, a) easysimd_mm256_mask_movedup_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_movedup_pd (easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_movedup_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_f64(a.f64[0]), svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svdup_n_f64(a.f64[2]), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      r_;

    r_.f64[0] = a_.f64[0];
    r_.f64[1] = a_.f64[0];
    r_.f64[2] = a_.f64[2];
    r_.f64[3] = a_.f64[2];
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_movedup_pd
  #define _mm256_maskz_movedup_pd(k, a) easysimd_mm256_maskz_movedup_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_movedup_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_movedup_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdup_n_f64(a.f64[0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(a.f64[2]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svdup_n_f64(a.f64[4]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svdup_n_f64(a.f64[6]);
    return r;
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
      r_.f64[i] = r_.f64[i + 1] = a_.f64[i];
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
  #undef _mm512_movedup_pd
  #define _mm512_movedup_pd(a) easysimd_mm512_movedup_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_movedup_pd (easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_movedup_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_f64(a.f64[0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svdup_n_f64(a.f64[2]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svdup_n_f64(a.f64[4]), src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svdup_n_f64(a.f64[6]), src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d_private
      src_ = easysimd__m512d_to_private(src),
      a_ = easysimd__m512d_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
      r_.f64[i] = r_.f64[i + 1] = a_.f64[i];
    }

    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : src_.f64[i];
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_movedup_pd
  #define _mm512_mask_movedup_pd(src, k, a) easysimd_mm512_mask_movedup_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_movedup_pd (easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_movedup_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_f64(a.f64[0]), svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svdup_n_f64(a.f64[2]), svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svdup_n_f64(a.f64[4]), svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svdup_n_f64(a.f64[6]), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
      r_.f64[i] = r_.f64[i + 1] = a_.f64[i];
    }

    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_movedup_pd
  #define _mm512_maskz_movedup_pd(k, a) easysimd_mm512_maskz_movedup_pd(k, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_MOVEDUP_H) */