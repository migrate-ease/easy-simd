#if !defined(EASYSIMD_X86_AVX512_SCALEF_H)
#define EASYSIMD_X86_AVX512_SCALEF_H

#include "types.h"
#include "flushsubnormal.h"
#include "../svml.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_scalef_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_scalef_ps(a, b);
  #else
    return easysimd_mm_mul_ps(easysimd_x_mm_flushsubnormal_ps(a), easysimd_mm_exp2_ps(easysimd_mm_floor_ps(easysimd_x_mm_flushsubnormal_ps(b))));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_scalef_ps
  #define _mm_scalef_ps(a, b) easysimd_mm_scalef_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_scalef_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_scalef_ps(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_ps(src, k, easysimd_mm_scalef_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_scalef_ps
  #define _mm_mask_scalef_ps(src, k, a, b) easysimd_mm_mask_scalef_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_scalef_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_scalef_ps(k, a, b);
  #else
    return easysimd_mm_maskz_mov_ps(k, easysimd_mm_scalef_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_scalef_ps
  #define _mm_maskz_scalef_ps(k, a, b) easysimd_mm_maskz_scalef_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_scalef_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_scalef_ps(a, b);
  #else
    return easysimd_mm256_mul_ps(easysimd_x_mm256_flushsubnormal_ps(a), easysimd_mm256_exp2_ps(easysimd_mm256_floor_ps(easysimd_x_mm256_flushsubnormal_ps(b))));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_scalef_ps
  #define _mm256_scalef_ps(a, b) easysimd_mm256_scalef_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_scalef_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_scalef_ps(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_scalef_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_scalef_ps
  #define _mm256_mask_scalef_ps(src, k, a, b) easysimd_mm256_mask_scalef_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_scalef_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_scalef_ps(k, a, b);
  #else
    return easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_scalef_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_scalef_ps
  #define _mm256_maskz_scalef_ps(k, a, b) easysimd_mm256_maskz_scalef_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_scalef_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_scalef_ps(a, b);
  #else
    return easysimd_mm512_mul_ps(easysimd_x_mm512_flushsubnormal_ps(a), easysimd_mm512_exp2_ps(easysimd_mm512_floor_ps(easysimd_x_mm512_flushsubnormal_ps(b))));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_scalef_ps
  #define _mm512_scalef_ps(a, b) easysimd_mm512_scalef_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_scalef_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_scalef_ps(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_scalef_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_scalef_ps
  #define _mm512_mask_scalef_ps(src, k, a, b) easysimd_mm512_mask_scalef_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_scalef_ps (easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_scalef_ps(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_scalef_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_scalef_ps
  #define _mm512_maskz_scalef_ps(k, a, b) easysimd_mm512_maskz_scalef_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_scalef_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_scalef_pd(a, b);
  #else
    return easysimd_mm_mul_pd(easysimd_x_mm_flushsubnormal_pd(a), easysimd_mm_exp2_pd(easysimd_mm_floor_pd(easysimd_x_mm_flushsubnormal_pd(b))));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_scalef_pd
  #define _mm_scalef_pd(a, b) easysimd_mm_scalef_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_scalef_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_scalef_pd(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_pd(src, k, easysimd_mm_scalef_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_scalef_pd
  #define _mm_mask_scalef_pd(src, k, a, b) easysimd_mm_mask_scalef_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_scalef_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_scalef_pd(k, a, b);
  #else
    return easysimd_mm_maskz_mov_pd(k, easysimd_mm_scalef_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_scalef_pd
  #define _mm_maskz_scalef_pd(k, a, b) easysimd_mm_maskz_scalef_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_scalef_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_scalef_pd(a, b);
  #else
    return easysimd_mm256_mul_pd(easysimd_x_mm256_flushsubnormal_pd(a), easysimd_mm256_exp2_pd(easysimd_mm256_floor_pd(easysimd_x_mm256_flushsubnormal_pd(b))));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_scalef_pd
  #define _mm256_scalef_pd(a, b) easysimd_mm256_scalef_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_scalef_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_scalef_pd(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_pd(src, k, easysimd_mm256_scalef_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_scalef_pd
  #define _mm256_mask_scalef_pd(src, k, a, b) easysimd_mm256_mask_scalef_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_scalef_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_scalef_pd(k, a, b);
  #else
    return easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_scalef_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_scalef_pd
  #define _mm256_maskz_scalef_pd(k, a, b) easysimd_mm256_maskz_scalef_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_scalef_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_scalef_pd(a, b);
  #else
    return easysimd_mm512_mul_pd(easysimd_x_mm512_flushsubnormal_pd(a), easysimd_mm512_exp2_pd(easysimd_mm512_floor_pd(easysimd_x_mm512_flushsubnormal_pd(b))));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_scalef_pd
  #define _mm512_scalef_pd(a, b) easysimd_mm512_scalef_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_scalef_pd (easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_scalef_pd(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_scalef_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_scalef_pd
  #define _mm512_mask_scalef_pd(src, k, a, b) easysimd_mm512_mask_scalef_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_scalef_pd (easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_scalef_pd(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_scalef_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_scalef_pd
  #define _mm512_maskz_scalef_pd(k, a, b) easysimd_mm512_maskz_scalef_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_scalef_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_scalef_ss(a, b);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    a_.f32[0] = (easysimd_math_issubnormalf(a_.f32[0]) ? 0 : a_.f32[0]) * easysimd_math_exp2f(easysimd_math_floorf((easysimd_math_issubnormalf(b_.f32[0]) ? 0 : b_.f32[0])));

    return easysimd__m128_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_scalef_ss
  #define _mm_scalef_ss(a, b) easysimd_mm_scalef_ss(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_scalef_ss (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(HEDLEY_GCC_VERSION)
    return _mm_mask_scalef_round_ss(src, k, a, b, _MM_FROUND_CUR_DIRECTION);
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    a_.f32[0] = ((k & 1) ? ((easysimd_math_issubnormalf(a_.f32[0]) ? 0 : a_.f32[0]) * easysimd_math_exp2f(easysimd_math_floorf((easysimd_math_issubnormalf(b_.f32[0]) ? 0 : b_.f32[0])))) : src_.f32[0]);

    return easysimd__m128_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_scalef_ss
  #define _mm_mask_scalef_ss(src, k, a, b) easysimd_mm_mask_scalef_ss(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_scalef_ss (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_GCC_105339)
    return _mm_maskz_scalef_ss(k, a, b);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    a_.f32[0] = ((k & 1) ? ((easysimd_math_issubnormalf(a_.f32[0]) ? 0 : a_.f32[0]) * easysimd_math_exp2f(easysimd_math_floorf((easysimd_math_issubnormalf(b_.f32[0]) ? 0 : b_.f32[0])))) : EASYSIMD_FLOAT32_C(0.0));

    return easysimd__m128_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_scalef_ss
  #define _mm_maskz_scalef_ss(k, a, b) easysimd_mm_maskz_scalef_ss(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_scalef_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_scalef_sd(a, b);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    a_.f64[0] = (easysimd_math_issubnormal(a_.f64[0]) ? 0 : a_.f64[0]) * easysimd_math_exp2(easysimd_math_floor((easysimd_math_issubnormal(b_.f64[0]) ? 0 : b_.f64[0])));

    return easysimd__m128d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_scalef_sd
  #define _mm_scalef_sd(a, b) easysimd_mm_scalef_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_scalef_sd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_GCC_105339)
    return _mm_mask_scalef_sd(src, k, a, b);
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    a_.f64[0] = ((k & 1) ? ((easysimd_math_issubnormal(a_.f64[0]) ? 0 : a_.f64[0]) * easysimd_math_exp2(easysimd_math_floor((easysimd_math_issubnormal(b_.f64[0]) ? 0 : b_.f64[0])))) : src_.f64[0]);

    return easysimd__m128d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_scalef_sd
  #define _mm_mask_scalef_sd(src, k, a, b) easysimd_mm_mask_scalef_sd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_scalef_sd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_GCC_105339)
    return _mm_maskz_scalef_sd(k, a, b);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    a_.f64[0] = ((k & 1) ? ((easysimd_math_issubnormal(a_.f64[0]) ? 0 : a_.f64[0]) * easysimd_math_exp2(easysimd_math_floor(easysimd_math_issubnormal(b_.f64[0]) ? 0 : b_.f64[0]))) : EASYSIMD_FLOAT64_C(0.0));

    return easysimd__m128d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_scalef_sd
  #define _mm_maskz_scalef_sd(k, a, b) easysimd_mm_maskz_scalef_sd(k, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SCALEF_H) */
