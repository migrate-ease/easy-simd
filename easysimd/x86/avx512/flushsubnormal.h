#if !defined(EASYSIMD_X86_AVX512_FLUSHSUBNORMAL_H)
#define EASYSIMD_X86_AVX512_FLUSHSUBNORMAL_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_flushsubnormal_ps (easysimd__m128 a) {
  easysimd__m128_private a_ = easysimd__m128_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
    a_.f32[i] = easysimd_math_issubnormalf(a_.f32[i]) ? 0 : a_.f32[i];
  }

  return easysimd__m128_from_private(a_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_flushsubnormal_ps (easysimd__m256 a) {
  easysimd__m256_private a_ = easysimd__m256_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
    a_.f32[i] = easysimd_math_issubnormalf(a_.f32[i]) ? 0 : a_.f32[i];
  }

  return easysimd__m256_from_private(a_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_x_mm512_flushsubnormal_ps (easysimd__m512 a) {
  easysimd__m512_private a_ = easysimd__m512_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
    a_.f32[i] = easysimd_math_issubnormalf(a_.f32[i]) ? 0 : a_.f32[i];
  }

  return easysimd__m512_from_private(a_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_flushsubnormal_pd (easysimd__m128d a) {
  easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
    a_.f64[i] = easysimd_math_issubnormal(a_.f64[i]) ? 0 : a_.f64[i];
  }

  return easysimd__m128d_from_private(a_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_flushsubnormal_pd (easysimd__m256d a) {
  easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
    a_.f64[i] = easysimd_math_issubnormal(a_.f64[i]) ? 0 : a_.f64[i];
  }

  return easysimd__m256d_from_private(a_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_x_mm512_flushsubnormal_pd (easysimd__m512d a) {
  easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
    a_.f64[i] = easysimd_math_issubnormal(a_.f64[i]) ? 0 : a_.f64[i];
  }

  return easysimd__m512d_from_private(a_);
}

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_FLUSHSUBNORMAL_H) */
