#if !defined(EASYSIMD_X86_AVX512_RCP_H)
#define EASYSIMD_X86_AVX512_RCP_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_rcp14_ps(easysimd__m128 a)
{
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_rcp14_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svrecpe_f32(a.sve_f32);
    return r;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      r_;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f32[0])); i++){
        r_.f32[i] = 1.0 / a_.f32[i];
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_rcp14_ps
  #define _mm_rcp14_ps(a) easysimd_mm_rcp14_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_rcp14_ps(easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a)
{
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_rcp14_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svrecpe_f32(a.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      r_;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f32[0])); i++){
        r_.f32[i] = ((k >> i) & 1) ? (1.0 / a_.f32[i]) : src_.f32[i];
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_rcp14_ps
  #define _mm_mask_rcp14_ps(src, k, a) easysimd_mm_mask_rcp14_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_rcp14_ps(easysimd__mmask8 k, easysimd__m128 a)
{
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_rcp14_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svrecpe_f32(a.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      r_;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f32[0])); i++){
        r_.f32[i] = ((k >> i) & 1) ? (1.0 / a_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_rcp14_ps
  #define _mm_maskz_rcp14_ps(k, a) easysimd_mm_maskz_rcp14_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_rcp14_pd(easysimd__m128d a)
{
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_rcp14_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svrecpe_f64(a.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      r_;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f64[0])); i++){
        r_.f64[i] = 1.0 / a_.f64[i];
    }
    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_rcp14_pd
  #define _mm_rcp14_pd(a) easysimd_mm_rcp14_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_rcp14_pd(easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a)
{
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_rcp14_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svrecpe_f64(a.sve_f64), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      r_;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f64[0])); i++){
        r_.f64[i] = ((k >> i) & 1) ? (1.0 / a_.f64[i]) : src_.f64[i];
    }
    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_rcp14_pd
  #define _mm_mask_rcp14_pd(src, k, a) easysimd_mm_mask_rcp14_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_rcp14_pd(easysimd__mmask8 k, easysimd__m128d a)
{
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_rcp14_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svrecpe_f64(a.sve_f64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      r_;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f64[0])); i++){
        r_.f64[i] = ((k >> i) & 1) ? (1.0 / a_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
    }
    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_rcp14_pd
  #define _mm_maskz_rcp14_pd(k, a) easysimd_mm_maskz_rcp14_pd(k, a)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_RCP_H) */