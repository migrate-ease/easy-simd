#if !defined(EASYSIMD_X86_AVX512_ROUND_H)
#define EASYSIMD_X86_AVX512_ROUND_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_x_mm512_round_ps(a, rounding) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512_private \
      easysimd_x_mm512_round_ps_r_, \
      easysimd_x_mm512_round_ps_a_ = easysimd__m512_to_private(a); \
    \
    for (size_t easysimd_x_mm512_round_ps_i = 0 ; easysimd_x_mm512_round_ps_i < (sizeof(easysimd_x_mm512_round_ps_r_.m256) / sizeof(easysimd_x_mm512_round_ps_r_.m256[0])) ; easysimd_x_mm512_round_ps_i++) { \
      easysimd_x_mm512_round_ps_r_.m256[easysimd_x_mm512_round_ps_i] = easysimd_mm256_round_ps(easysimd_x_mm512_round_ps_a_.m256[easysimd_x_mm512_round_ps_i], rounding); \
    } \
    \
    easysimd__m512_from_private(easysimd_x_mm512_round_ps_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_x_mm512_round_ps (easysimd__m512 a, int rounding)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(rounding, 0, 15) {
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    /* For architectures which lack a current direction SIMD instruction.
    *
    * Note that NEON actually has a current rounding mode instruction,
    * but in ARMv8+ the rounding mode is ignored and nearest is always
    * used, so we treat ARMv7 as having a rounding mode but ARMv8 as
    * not. */
    #if defined(EASYSIMD_ARM_NEON_A32V8)
      if ((rounding & 7) == EASYSIMD_MM_FROUND_CUR_DIRECTION)
        rounding = HEDLEY_STATIC_CAST(int, EASYSIMD_MM_GET_ROUNDING_MODE()) << 13;
    #endif

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && !defined(EASYSIMD_BUG_GCC_95399)
          for (size_t i = 0 ; i < (sizeof(r_.m128_private) / sizeof(r_.m128_private[0])) ; i++) {
            r_.m128_private[i].neon_f32 = vrndiq_f32(a_.m128_private[i].neon_f32);
          }
        #elif defined(easysimd_math_nearbyintf)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
            r_.f32[i] = easysimd_math_nearbyintf(a_.f32[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128_private) / sizeof(r_.m128_private[0])) ; i++) {
            r_.m128_private[i].neon_f32 = vrndnq_f32(a_.m128_private[i].neon_f32);
          }
        #elif defined(easysimd_math_roundevenf)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
            r_.f32[i] = easysimd_math_roundevenf(a_.f32[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128_private) / sizeof(r_.m128_private[0])) ; i++) {
            r_.m128_private[i].neon_f32 = vrndmq_f32(a_.m128_private[i].neon_f32);
          }
        #elif defined(easysimd_math_floorf)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
            r_.f32[i] = easysimd_math_floorf(a_.f32[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_POS_INF:
        #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128_private) / sizeof(r_.m128_private[0])) ; i++) {
            r_.m128_private[i].neon_f32 = vrndpq_f32(a_.m128_private[i].neon_f32);
          }
        #elif defined(easysimd_math_ceilf)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
            r_.f32[i] = easysimd_math_ceilf(a_.f32[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_ZERO:
        #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128_private) / sizeof(r_.m128_private[0])) ; i++) {
            r_.m128_private[i].neon_f32 = vrndq_f32(a_.m128_private[i].neon_f32);
          }
        #elif defined(easysimd_math_truncf)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
            r_.f32[i] = easysimd_math_truncf(a_.f32[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        #endif
        break;

      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
    }

    return easysimd__m512_from_private(r_);
  }
#endif

#if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_x_mm512_round_pd(a, rounding) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512d_private \
      easysimd_x_mm512_round_pd_r_, \
      easysimd_x_mm512_round_pd_a_ = easysimd__m512d_to_private(a); \
    \
    for (size_t easysimd_x_mm512_round_pd_i = 0 ; easysimd_x_mm512_round_pd_i < (sizeof(easysimd_x_mm512_round_pd_r_.m256d) / sizeof(easysimd_x_mm512_round_pd_r_.m256d[0])) ; easysimd_x_mm512_round_pd_i++) { \
      easysimd_x_mm512_round_pd_r_.m256d[easysimd_x_mm512_round_pd_i] = easysimd_mm256_round_pd(easysimd_x_mm512_round_pd_a_.m256d[easysimd_x_mm512_round_pd_i], rounding); \
    } \
    \
    easysimd__m512d_from_private(easysimd_x_mm512_round_pd_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_x_mm512_round_pd (easysimd__m512d a, int rounding)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(rounding, 0, 15) {
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128d_private) / sizeof(r_.m128d_private[0])) ; i++) {
            r_.m128d_private[i].neon_f64 = vrndiq_f64(a_.m128d_private[i].neon_f64);
          }
        #elif defined(easysimd_math_nearbyint)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
            r_.f64[i] = easysimd_math_nearbyint(a_.f64[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128d_private) / sizeof(r_.m128d_private[0])) ; i++) {
            r_.m128d_private[i].neon_f64 = vrndaq_f64(a_.m128d_private[i].neon_f64);
          }
        #elif defined(easysimd_math_roundeven)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
            r_.f64[i] = easysimd_math_roundeven(a_.f64[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128d_private) / sizeof(r_.m128d_private[0])) ; i++) {
            r_.m128d_private[i].neon_f64 = vrndmq_f64(a_.m128d_private[i].neon_f64);
          }
        #elif defined(easysimd_math_floor)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
            r_.f64[i] = easysimd_math_floor(a_.f64[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_POS_INF:
        #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128d_private) / sizeof(r_.m128d_private[0])) ; i++) {
            r_.m128d_private[i].neon_f64 = vrndpq_f64(a_.m128d_private[i].neon_f64);
          }
        #elif defined(easysimd_math_ceil)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
            r_.f64[i] = easysimd_math_ceil(a_.f64[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        #endif
        break;

      case EASYSIMD_MM_FROUND_TO_ZERO:
        #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
          for (size_t i = 0 ; i < (sizeof(r_.m128d_private) / sizeof(r_.m128d_private[0])) ; i++) {
            r_.m128d_private[i].neon_f64 = vrndq_f64(a_.m128d_private[i].neon_f64);
          }
        #elif defined(easysimd_math_trunc)
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
            r_.f64[i] = easysimd_math_trunc(a_.f64[i]);
          }
        #else
          HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        #endif
        break;

      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
    }

    return easysimd__m512d_from_private(r_);
  }
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_ROUND_H) */
