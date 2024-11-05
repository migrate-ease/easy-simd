#if !defined(EASYSIMD_X86_AVX512_FIXUPIMM_H)
#define EASYSIMD_X86_AVX512_FIXUPIMM_H

#include "types.h"
#include "flushsubnormal.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fixupimm_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128_to_private(b),
    s_ = easysimd__m128_to_private(easysimd_x_mm_flushsubnormal_ps(b));
  easysimd__m128i_private c_ = easysimd__m128i_to_private(c);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    int32_t select = 1;
    switch (easysimd_math_fpclassifyf(s_.f32[i])) {
      case EASYSIMD_MATH_FP_NORMAL:
        select = (s_.f32[i] < EASYSIMD_FLOAT32_C(0.0)) ? 6 : (fabsf(s_.f32[i] - EASYSIMD_FLOAT32_C(1.0)) < 1e-9f) ? 3 : 7;
        break;
      case EASYSIMD_MATH_FP_ZERO:
        select = 2;
        break;
      case EASYSIMD_MATH_FP_NAN:
        select = 0;
        break;
      case EASYSIMD_MATH_FP_INFINITE:
        select = ((s_.f32[i] > EASYSIMD_FLOAT32_C(0.0)) ? 5 : 4);
        break;
    }

    switch (((c_.i32[i] >> (select << 2)) & 15)) {
      case 0:
        r_.f32[i] =  a_.f32[i];
        break;
      case 1:
        r_.f32[i] =  b_.f32[i];
        break;
      case 2:
        r_.f32[i] =  EASYSIMD_MATH_NANF;
        break;
      case 3:
        r_.f32[i] = -EASYSIMD_MATH_NANF;
        break;
      case 4:
        r_.f32[i] = -EASYSIMD_MATH_INFINITYF;
        break;
      case 5:
        r_.f32[i] =  EASYSIMD_MATH_INFINITYF;
        break;
      case 6:
        r_.f32[i] =  s_.f32[i] < EASYSIMD_FLOAT32_C(0.0) ? -EASYSIMD_MATH_INFINITYF : EASYSIMD_MATH_INFINITYF;
        break;
      case 7:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(-0.0);
        break;
      case 8:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(0.0);
        break;
      case 9:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(-1.0);
        break;
      case 10:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(1.0);
        break;
      case 11:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(0.5);
        break;
      case 12:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(90.0);
        break;
      case 13:
        r_.f32[i] =  EASYSIMD_MATH_PIF / 2;
        break;
      case 14:
        r_.f32[i] =  EASYSIMD_MATH_FLT_MAX;
        break;
      case 15:
        r_.f32[i] = -EASYSIMD_MATH_FLT_MAX;
        break;
    }
  }

  return easysimd__m128_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_fixupimm_ps(a, b, c, imm8) _mm_fixupimm_ps(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_fixupimm_ps
  #define _mm_fixupimm_ps(a, b, c, imm8) easysimd_mm_fixupimm_ps(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_fixupimm_ps(a, k, b, c, imm8) _mm_mask_fixupimm_ps(a, k, b, c, imm8)
#else
  #define easysimd_mm_mask_fixupimm_ps(a, k, b, c, imm8) easysimd_mm_mask_mov_ps(a, k, easysimd_mm_fixupimm_ps(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fixupimm_ps
  #define _mm_mask_fixupimm_ps(a, k, b, c, imm8) easysimd_mm_mask_fixupimm_ps(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_fixupimm_ps(k, a, b, c, imm8) _mm_maskz_fixupimm_ps(k, a, b, c, imm8)
#else
  #define easysimd_mm_maskz_fixupimm_ps(k, a, b, c, imm8) easysimd_mm_maskz_mov_ps(k, easysimd_mm_fixupimm_ps(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fixupimm_ps
  #define _mm_maskz_fixupimm_ps(k, a, b, c, imm8) easysimd_mm_maskz_fixupimm_ps(k, a, b, c, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_fixupimm_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a),
    b_ = easysimd__m256_to_private(b),
    s_ = easysimd__m256_to_private(easysimd_x_mm256_flushsubnormal_ps(b));
  easysimd__m256i_private c_ = easysimd__m256i_to_private(c);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    int32_t select = 1;
    switch (easysimd_math_fpclassifyf(s_.f32[i])) {
      case EASYSIMD_MATH_FP_NORMAL:
        select = (s_.f32[i] < EASYSIMD_FLOAT32_C(0.0)) ? 6 : (fabsf(s_.f32[i] - EASYSIMD_FLOAT32_C(1.0)) < 1e-9f) ? 3 : 7;
        break;
      case EASYSIMD_MATH_FP_ZERO:
        select = 2;
        break;
      case EASYSIMD_MATH_FP_NAN:
        select = 0;
        break;
      case EASYSIMD_MATH_FP_INFINITE:
        select = ((s_.f32[i] > EASYSIMD_FLOAT32_C(0.0)) ? 5 : 4);
        break;
    }

    switch (((c_.i32[i] >> (select << 2)) & 15)) {
      case 0:
        r_.f32[i] =  a_.f32[i];
        break;
      case 1:
        r_.f32[i] =  b_.f32[i];
        break;
      case 2:
        r_.f32[i] =  EASYSIMD_MATH_NANF;
        break;
      case 3:
        r_.f32[i] = -EASYSIMD_MATH_NANF;
        break;
      case 4:
        r_.f32[i] = -EASYSIMD_MATH_INFINITYF;
        break;
      case 5:
        r_.f32[i] =  EASYSIMD_MATH_INFINITYF;
        break;
      case 6:
        r_.f32[i] =  s_.f32[i] < EASYSIMD_FLOAT32_C(0.0) ? -EASYSIMD_MATH_INFINITYF : EASYSIMD_MATH_INFINITYF;
        break;
      case 7:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(-0.0);
        break;
      case 8:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(0.0);
        break;
      case 9:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(-1.0);
        break;
      case 10:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(1.0);
        break;
      case 11:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(0.5);
        break;
      case 12:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(90.0);
        break;
      case 13:
        r_.f32[i] =  EASYSIMD_MATH_PIF / 2;
        break;
      case 14:
        r_.f32[i] =  EASYSIMD_MATH_FLT_MAX;
        break;
      case 15:
        r_.f32[i] = -EASYSIMD_MATH_FLT_MAX;
        break;
    }
  }

  return easysimd__m256_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_fixupimm_ps(a, b, c, imm8) _mm256_fixupimm_ps(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fixupimm_ps
  #define _mm256_fixupimm_ps(a, b, c, imm8) easysimd_mm256_fixupimm_ps(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_fixupimm_ps(a, k, b, c, imm8) _mm256_mask_fixupimm_ps(a, k, b, c, imm8)
#else
  #define easysimd_mm256_mask_fixupimm_ps(a, k, b, c, imm8) easysimd_mm256_mask_mov_ps(a, k, easysimd_mm256_fixupimm_ps(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_fixupimm_ps
  #define _mm256_mask_fixupimm_ps(a, k, b, c, imm8) easysimd_mm256_mask_fixupimm_ps(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_fixupimm_ps(k, a, b, c, imm8) _mm256_maskz_fixupimm_ps(k, a, b, c, imm8)
#else
  #define easysimd_mm256_maskz_fixupimm_ps(k, a, b, c, imm8) easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_fixupimm_ps(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_fixupimm_ps
  #define _mm256_maskz_fixupimm_ps(k, a, b, c, imm8) easysimd_mm256_maskz_fixupimm_ps(k, a, b, c, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_fixupimm_ps (easysimd__m512 a, easysimd__m512 b, easysimd__m512i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m512_private
    r_,
    a_ = easysimd__m512_to_private(a),
    b_ = easysimd__m512_to_private(b),
    s_ = easysimd__m512_to_private(easysimd_x_mm512_flushsubnormal_ps(b));
  easysimd__m512i_private c_ = easysimd__m512i_to_private(c);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    int32_t select = 1;
    switch (easysimd_math_fpclassifyf(s_.f32[i])) {
      case EASYSIMD_MATH_FP_NORMAL:
        select = (s_.f32[i] < EASYSIMD_FLOAT32_C(0.0)) ? 6 : (fabsf(s_.f32[i] - EASYSIMD_FLOAT32_C(1.0)) < 1e-9f) ? 3 : 7;
        break;
      case EASYSIMD_MATH_FP_ZERO:
        select = 2;
        break;
      case EASYSIMD_MATH_FP_NAN:
        select = 0;
        break;
      case EASYSIMD_MATH_FP_INFINITE:
        select = ((s_.f32[i] > EASYSIMD_FLOAT32_C(0.0)) ? 5 : 4);
        break;
    }

    switch (((c_.i32[i] >> (select << 2)) & 15)) {
      case 0:
        r_.f32[i] =  a_.f32[i];
        break;
      case 1:
        r_.f32[i] =  b_.f32[i];
        break;
      case 2:
        r_.f32[i] =  EASYSIMD_MATH_NANF;
        break;
      case 3:
        r_.f32[i] = -EASYSIMD_MATH_NANF;
        break;
      case 4:
        r_.f32[i] = -EASYSIMD_MATH_INFINITYF;
        break;
      case 5:
        r_.f32[i] =  EASYSIMD_MATH_INFINITYF;
        break;
      case 6:
        r_.f32[i] =  s_.f32[i] < EASYSIMD_FLOAT32_C(0.0) ? -EASYSIMD_MATH_INFINITYF : EASYSIMD_MATH_INFINITYF;
        break;
      case 7:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(-0.0);
        break;
      case 8:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(0.0);
        break;
      case 9:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(-1.0);
        break;
      case 10:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(1.0);
        break;
      case 11:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(0.5);
        break;
      case 12:
        r_.f32[i] =  EASYSIMD_FLOAT32_C(90.0);
        break;
      case 13:
        r_.f32[i] =  EASYSIMD_MATH_PIF / 2;
        break;
      case 14:
        r_.f32[i] =  EASYSIMD_MATH_FLT_MAX;
        break;
      case 15:
        r_.f32[i] = -EASYSIMD_MATH_FLT_MAX;
        break;
    }
  }

  return easysimd__m512_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_fixupimm_ps(a, b, c, imm8) _mm512_fixupimm_ps(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_fixupimm_ps
  #define _mm512_fixupimm_ps(a, b, c, imm8) easysimd_mm512_fixupimm_ps(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8) _mm512_mask_fixupimm_ps(a, k, b, c, imm8)
#else
  #define easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8) easysimd_mm512_mask_mov_ps(a, k, easysimd_mm512_fixupimm_ps(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_fixupimm_ps
  #define _mm512_mask_fixupimm_ps(a, k, b, c, imm8) easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8) _mm512_maskz_fixupimm_ps(k, a, b, c, imm8)
#else
  #define easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8) easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_fixupimm_ps(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_fixupimm_ps
  #define _mm512_maskz_fixupimm_ps(k, a, b, c, imm8) easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_fixupimm_ss (easysimd__m128 a, easysimd__m128 b, easysimd__m128i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m128_private
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128_to_private(b),
    s_ = easysimd__m128_to_private(easysimd_x_mm_flushsubnormal_ps(b));
  easysimd__m128i_private c_ = easysimd__m128i_to_private(c);

  int32_t select = 1;
  switch (easysimd_math_fpclassifyf(s_.f32[0])) {
    case EASYSIMD_MATH_FP_NORMAL:
      select = (s_.f32[0] < EASYSIMD_FLOAT32_C(0.0)) ? 6 : (fabsf(s_.f32[0] - EASYSIMD_FLOAT32_C(1.0)) < 1e-9f) ? 3 : 7;
      break;
    case EASYSIMD_MATH_FP_ZERO:
      select = 2;
      break;
    case EASYSIMD_MATH_FP_NAN:
      select = 0;
      break;
    case EASYSIMD_MATH_FP_INFINITE:
      select = ((s_.f32[0] > EASYSIMD_FLOAT32_C(0.0)) ? 5 : 4);
      break;
  }

  switch (((c_.i32[0] >> (select << 2)) & 15)) {
    case 0:
      b_.f32[0] =  a_.f32[0];
      break;
    case 2:
      b_.f32[0] =  EASYSIMD_MATH_NANF;
      break;
    case 3:
      b_.f32[0] = -EASYSIMD_MATH_NANF;
      break;
    case 4:
      b_.f32[0] = -EASYSIMD_MATH_INFINITYF;
      break;
    case 5:
      b_.f32[0] =  EASYSIMD_MATH_INFINITYF;
      break;
    case 6:
      b_.f32[0] =  s_.f32[0] < EASYSIMD_FLOAT32_C(0.0) ? -EASYSIMD_MATH_INFINITYF : EASYSIMD_MATH_INFINITYF;
      break;
    case 7:
      b_.f32[0] =  EASYSIMD_FLOAT32_C(-0.0);
      break;
    case 8:
      b_.f32[0] =  EASYSIMD_FLOAT32_C(0.0);
      break;
    case 9:
      b_.f32[0] =  EASYSIMD_FLOAT32_C(-1.0);
      break;
    case 10:
      b_.f32[0] =  EASYSIMD_FLOAT32_C(1.0);
      break;
    case 11:
      b_.f32[0] =  EASYSIMD_FLOAT32_C(0.5);
      break;
    case 12:
      b_.f32[0] =  EASYSIMD_FLOAT32_C(90.0);
      break;
    case 13:
      b_.f32[0] =  EASYSIMD_MATH_PIF / 2;
      break;
    case 14:
      b_.f32[0] =  EASYSIMD_MATH_FLT_MAX;
      break;
    case 15:
      b_.f32[0] = -EASYSIMD_MATH_FLT_MAX;
      break;
  }

  return easysimd__m128_from_private(b_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_fixupimm_ss(a, b, c, imm8) _mm_fixupimm_ss(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_fixupimm_ss
  #define _mm_fixupimm_ss(a, b, c, imm8) easysimd_mm_fixupimm_ss(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8) _mm_mask_fixupimm_ss(a, k, b, c, imm8)
#else
  #define easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8) easysimd_mm_mask_mov_ps(a, ((k) | 14), easysimd_mm_fixupimm_ss(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fixupimm_ss
  #define _mm_mask_fixupimm_ss(a, k, b, c, imm8) easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8) _mm_maskz_fixupimm_ss(k, a, b, c, imm8)
#else
  #define easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8) easysimd_mm_maskz_mov_ps(((k) | 14), easysimd_mm_fixupimm_ss(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fixupimm_ss
  #define _mm_maskz_fixupimm_ss(k, a, b, c, imm8) easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fixupimm_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m128d_private
    r_,
    a_ = easysimd__m128d_to_private(a),
    b_ = easysimd__m128d_to_private(b),
    s_ = easysimd__m128d_to_private(easysimd_x_mm_flushsubnormal_pd(b));
  easysimd__m128i_private c_ = easysimd__m128i_to_private(c);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    int32_t select = 1;
    switch (easysimd_math_fpclassify(s_.f64[i])) {
      case EASYSIMD_MATH_FP_NORMAL:
        select = (s_.f64[i] < EASYSIMD_FLOAT64_C(0.0)) ? 6 : (fabs(s_.f64[i] - EASYSIMD_FLOAT64_C(1.0)) < 1e-9f) ? 3 : 7;
        break;
      case EASYSIMD_MATH_FP_ZERO:
        select = 2;
        break;
      case EASYSIMD_MATH_FP_NAN:
        select = 0;
        break;
      case EASYSIMD_MATH_FP_INFINITE:
        select = ((s_.f64[i] > EASYSIMD_FLOAT64_C(0.0)) ? 5 : 4);
        break;
    }

    switch (((c_.i64[i] >> (select << 2)) & 15)) {
      case 0:
        r_.f64[i] =  a_.f64[i];
        break;
      case 1:
        r_.f64[i] =  b_.f64[i];
        break;
      case 2:
        r_.f64[i] =  EASYSIMD_MATH_NAN;
        break;
      case 3:
        r_.f64[i] = -EASYSIMD_MATH_NAN;
        break;
      case 4:
        r_.f64[i] = -EASYSIMD_MATH_INFINITY;
        break;
      case 5:
        r_.f64[i] =  EASYSIMD_MATH_INFINITY;
        break;
      case 6:
        r_.f64[i] =  s_.f64[i] < EASYSIMD_FLOAT64_C(0.0) ? -EASYSIMD_MATH_INFINITY : EASYSIMD_MATH_INFINITY;
        break;
      case 7:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(-0.0);
        break;
      case 8:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(0.0);
        break;
      case 9:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(-1.0);
        break;
      case 10:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(1.0);
        break;
      case 11:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(0.5);
        break;
      case 12:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(90.0);
        break;
      case 13:
        r_.f64[i] =  EASYSIMD_MATH_PI / 2;
        break;
      case 14:
        r_.f64[i] =  EASYSIMD_MATH_DBL_MAX;
        break;
      case 15:
        r_.f64[i] = -EASYSIMD_MATH_DBL_MAX;
        break;
    }
  }

  return easysimd__m128d_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_fixupimm_pd(a, b, c, imm8) _mm_fixupimm_pd(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_fixupimm_pd
  #define _mm_fixupimm_pd(a, b, c, imm8) easysimd_mm_fixupimm_pd(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_fixupimm_pd(a, k, b, c, imm8) _mm_mask_fixupimm_pd(a, k, b, c, imm8)
#else
  #define easysimd_mm_mask_fixupimm_pd(a, k, b, c, imm8) easysimd_mm_mask_mov_pd(a, k, easysimd_mm_fixupimm_pd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fixupimm_pd
  #define _mm_mask_fixupimm_pd(a, k, b, c, imm8) easysimd_mm_mask_fixupimm_pd(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_fixupimm_pd(k, a, b, c, imm8) _mm_maskz_fixupimm_pd(k, a, b, c, imm8)
#else
  #define easysimd_mm_maskz_fixupimm_pd(k, a, b, c, imm8) easysimd_mm_maskz_mov_pd(k, easysimd_mm_fixupimm_pd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fixupimm_pd
  #define _mm_maskz_fixupimm_pd(k, a, b, c, imm8) easysimd_mm_maskz_fixupimm_pd(k, a, b, c, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_fixupimm_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a),
    b_ = easysimd__m256d_to_private(b),
    s_ = easysimd__m256d_to_private(easysimd_x_mm256_flushsubnormal_pd(b));
  easysimd__m256i_private c_ = easysimd__m256i_to_private(c);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    int32_t select = 1;
    switch (easysimd_math_fpclassify(s_.f64[i])) {
      case EASYSIMD_MATH_FP_NORMAL:
        select = (s_.f64[i] < EASYSIMD_FLOAT64_C(0.0)) ? 6 : (fabs(s_.f64[i] - EASYSIMD_FLOAT64_C(1.0)) < 1e-9f) ? 3 : 7;
        break;
      case EASYSIMD_MATH_FP_ZERO:
        select = 2;
        break;
      case EASYSIMD_MATH_FP_NAN:
        select = 0;
        break;
      case EASYSIMD_MATH_FP_INFINITE:
        select = ((s_.f64[i] > EASYSIMD_FLOAT64_C(0.0)) ? 5 : 4);
        break;
    }

    switch (((c_.i64[i] >> (select << 2)) & 15)) {
      case 0:
        r_.f64[i] =  a_.f64[i];
        break;
      case 1:
        r_.f64[i] =  b_.f64[i];
        break;
      case 2:
        r_.f64[i] =  EASYSIMD_MATH_NAN;
        break;
      case 3:
        r_.f64[i] = -EASYSIMD_MATH_NAN;
        break;
      case 4:
        r_.f64[i] = -EASYSIMD_MATH_INFINITY;
        break;
      case 5:
        r_.f64[i] =  EASYSIMD_MATH_INFINITY;
        break;
      case 6:
        r_.f64[i] =  s_.f64[i] < EASYSIMD_FLOAT64_C(0.0) ? -EASYSIMD_MATH_INFINITY : EASYSIMD_MATH_INFINITY;
        break;
      case 7:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(-0.0);
        break;
      case 8:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(0.0);
        break;
      case 9:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(-1.0);
        break;
      case 10:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(1.0);
        break;
      case 11:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(0.5);
        break;
      case 12:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(90.0);
        break;
      case 13:
        r_.f64[i] =  EASYSIMD_MATH_PI / 2;
        break;
      case 14:
        r_.f64[i] =  EASYSIMD_MATH_DBL_MAX;
        break;
      case 15:
        r_.f64[i] = -EASYSIMD_MATH_DBL_MAX;
        break;
    }
  }

  return easysimd__m256d_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_fixupimm_pd(a, b, c, imm8) _mm256_fixupimm_pd(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_fixupimm_pd
  #define _mm256_fixupimm_pd(a, b, c, imm8) easysimd_mm256_fixupimm_pd(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_fixupimm_pd(a, k, b, c, imm8) _mm256_mask_fixupimm_pd(a, k, b, c, imm8)
#else
  #define easysimd_mm256_mask_fixupimm_pd(a, k, b, c, imm8) easysimd_mm256_mask_mov_pd(a, k, easysimd_mm256_fixupimm_pd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_fixupimm_pd
  #define _mm256_mask_fixupimm_pd(a, k, b, c, imm8) easysimd_mm256_mask_fixupimm_pd(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_fixupimm_pd(k, a, b, c, imm8) _mm256_maskz_fixupimm_pd(k, a, b, c, imm8)
#else
  #define easysimd_mm256_maskz_fixupimm_pd(k, a, b, c, imm8) easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_fixupimm_pd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_fixupimm_pd
  #define _mm256_maskz_fixupimm_pd(k, a, b, c, imm8) easysimd_mm256_maskz_fixupimm_pd(k, a, b, c, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_fixupimm_pd (easysimd__m512d a, easysimd__m512d b, easysimd__m512i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m512d_private
    r_,
    a_ = easysimd__m512d_to_private(a),
    b_ = easysimd__m512d_to_private(b),
    s_ = easysimd__m512d_to_private(easysimd_x_mm512_flushsubnormal_pd(b));
  easysimd__m512i_private c_ = easysimd__m512i_to_private(c);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    int32_t select = 1;
    switch (easysimd_math_fpclassify(s_.f64[i])) {
      case EASYSIMD_MATH_FP_NORMAL:
        select = (s_.f64[i] < EASYSIMD_FLOAT64_C(0.0)) ? 6 : (fabs(s_.f64[i] - EASYSIMD_FLOAT64_C(1.0)) < 1e-9f) ? 3 : 7;
        break;
      case EASYSIMD_MATH_FP_ZERO:
        select = 2;
        break;
      case EASYSIMD_MATH_FP_NAN:
        select = 0;
        break;
      case EASYSIMD_MATH_FP_INFINITE:
        select = ((s_.f64[i] > EASYSIMD_FLOAT64_C(0.0)) ? 5 : 4);
        break;
    }

    switch (((c_.i64[i] >> (select << 2)) & 15)) {
      case 0:
        r_.f64[i] =  a_.f64[i];
        break;
      case 1:
        r_.f64[i] =  b_.f64[i];
        break;
      case 2:
        r_.f64[i] =  EASYSIMD_MATH_NAN;
        break;
      case 3:
        r_.f64[i] = -EASYSIMD_MATH_NAN;
        break;
      case 4:
        r_.f64[i] = -EASYSIMD_MATH_INFINITY;
        break;
      case 5:
        r_.f64[i] =  EASYSIMD_MATH_INFINITY;
        break;
      case 6:
        r_.f64[i] =  s_.f64[i] < EASYSIMD_FLOAT64_C(0.0) ? -EASYSIMD_MATH_INFINITY : EASYSIMD_MATH_INFINITY;
        break;
      case 7:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(-0.0);
        break;
      case 8:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(0.0);
        break;
      case 9:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(-1.0);
        break;
      case 10:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(1.0);
        break;
      case 11:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(0.5);
        break;
      case 12:
        r_.f64[i] =  EASYSIMD_FLOAT64_C(90.0);
        break;
      case 13:
        r_.f64[i] =  EASYSIMD_MATH_PI / 2;
        break;
      case 14:
        r_.f64[i] =  EASYSIMD_MATH_DBL_MAX;
        break;
      case 15:
        r_.f64[i] = -EASYSIMD_MATH_DBL_MAX;
        break;
    }
  }

  return easysimd__m512d_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_fixupimm_pd(a, b, c, imm8) _mm512_fixupimm_pd(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_fixupimm_pd
  #define _mm512_fixupimm_pd(a, b, c, imm8) easysimd_mm512_fixupimm_pd(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8) _mm512_mask_fixupimm_pd(a, k, b, c, imm8)
#else
  #define easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8) easysimd_mm512_mask_mov_pd(a, k, easysimd_mm512_fixupimm_pd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_fixupimm_pd
  #define _mm512_mask_fixupimm_pd(a, k, b, c, imm8) easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8) _mm512_maskz_fixupimm_pd(k, a, b, c, imm8)
#else
  #define easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8) easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_fixupimm_pd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_fixupimm_pd
  #define _mm512_maskz_fixupimm_pd(k, a, b, c, imm8) easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_fixupimm_sd (easysimd__m128d a, easysimd__m128d b, easysimd__m128i c, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  HEDLEY_STATIC_CAST(void, imm8);
  easysimd__m128d_private
    a_ = easysimd__m128d_to_private(a),
    b_ = easysimd__m128d_to_private(b),
    s_ = easysimd__m128d_to_private(easysimd_x_mm_flushsubnormal_pd(b));
  easysimd__m128i_private c_ = easysimd__m128i_to_private(c);

  int32_t select = 1;
  switch (easysimd_math_fpclassify(s_.f64[0])) {
    case EASYSIMD_MATH_FP_NORMAL:
      select = (s_.f64[0] < EASYSIMD_FLOAT64_C(0.0)) ? 6 : (fabs(s_.f64[0] - EASYSIMD_FLOAT64_C(1.0)) < 1e-9f) ? 3 : 7;
      break;
    case EASYSIMD_MATH_FP_ZERO:
      select = 2;
      break;
    case EASYSIMD_MATH_FP_NAN:
      select = 0;
      break;
    case EASYSIMD_MATH_FP_INFINITE:
      select = ((s_.f64[0] > EASYSIMD_FLOAT64_C(0.0)) ? 5 : 4);
      break;
  }

  switch (((c_.i64[0] >> (select << 2)) & 15)) {
    case 0:
      b_.f64[0] =  a_.f64[0];
      break;
    case 2:
      b_.f64[0] =  EASYSIMD_MATH_NAN;
      break;
    case 3:
      b_.f64[0] = -EASYSIMD_MATH_NAN;
      break;
    case 4:
      b_.f64[0] = -EASYSIMD_MATH_INFINITY;
      break;
    case 5:
      b_.f64[0] =  EASYSIMD_MATH_INFINITY;
      break;
    case 6:
      b_.f64[0] =  s_.f64[0] < EASYSIMD_FLOAT64_C(0.0) ? -EASYSIMD_MATH_INFINITY : EASYSIMD_MATH_INFINITY;
      break;
    case 7:
      b_.f64[0] =  EASYSIMD_FLOAT64_C(-0.0);
      break;
    case 8:
      b_.f64[0] =  EASYSIMD_FLOAT64_C(0.0);
      break;
    case 9:
      b_.f64[0] =  EASYSIMD_FLOAT64_C(-1.0);
      break;
    case 10:
      b_.f64[0] =  EASYSIMD_FLOAT64_C(1.0);
      break;
    case 11:
      b_.f64[0] =  EASYSIMD_FLOAT64_C(0.5);
      break;
    case 12:
      b_.f64[0] =  EASYSIMD_FLOAT64_C(90.0);
      break;
    case 13:
      b_.f64[0] =  EASYSIMD_MATH_PI / 2;
      break;
    case 14:
      b_.f64[0] =  EASYSIMD_MATH_DBL_MAX;
      break;
    case 15:
      b_.f64[0] = -EASYSIMD_MATH_DBL_MAX;
      break;
  }

  return easysimd__m128d_from_private(b_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_fixupimm_sd(a, b, c, imm8) _mm_fixupimm_sd(a, b, c, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_fixupimm_sd
  #define _mm_fixupimm_sd(a, b, c, imm8) easysimd_mm_fixupimm_sd(a, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8) _mm_mask_fixupimm_sd(a, k, b, c, imm8)
#else
  #define easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8) easysimd_mm_mask_mov_pd(a, ((k) | 2), easysimd_mm_fixupimm_sd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fixupimm_sd
  #define _mm_mask_fixupimm_sd(a, k, b, c, imm8) easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8) _mm_maskz_fixupimm_sd(k, a, b, c, imm8)
#else
  #define easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8) easysimd_mm_maskz_mov_pd(((k) | 2), easysimd_mm_fixupimm_sd(a, b, c, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fixupimm_sd
  #define _mm_maskz_fixupimm_sd(k, a, b, c, imm8) easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_FIXUPIMM_H) */
