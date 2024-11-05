#if !defined(EASYSIMD_X86_AVX512_ROUNDSCALE_H)
#define EASYSIMD_X86_AVX512_ROUNDSCALE_H

#include "types.h"
#include "andnot.h"
#include "set1.h"
#include "mul.h"
#include "round.h"
#include "cmpeq.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_roundscale_ps(a, imm8) _mm_roundscale_ps((a), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_roundscale_ps_internal_ (easysimd__m128 result, easysimd__m128 a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m128 r, clear_sign;

    clear_sign = easysimd_mm_andnot_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), result);
    r = easysimd_x_mm_select_ps(result, a, easysimd_mm_cmpeq_ps(clear_sign, easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITYF)));

    return r;
  }
  #define easysimd_mm_roundscale_ps(a, imm8) \
    easysimd_mm_roundscale_ps_internal_( \
      easysimd_mm_mul_ps( \
        easysimd_mm_round_ps( \
          easysimd_mm_mul_ps( \
            a, \
            easysimd_mm_set1_ps(easysimd_math_exp2f(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm_set1_ps(easysimd_math_exp2f(-((imm8 >> 4) & 15))) \
      ), \
      (a), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_roundscale_ps
  #define _mm_roundscale_ps(a, imm8) easysimd_mm_roundscale_ps(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_roundscale_ps(src, k, a, imm8) _mm_mask_roundscale_ps(src, k, a, imm8)
#else
  #define easysimd_mm_mask_roundscale_ps(src, k, a, imm8) easysimd_mm_mask_mov_ps(src, k, easysimd_mm_roundscale_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_roundscale_ps
  #define _mm_mask_roundscale_ps(src, k, a, imm8) easysimd_mm_mask_roundscale_ps(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_roundscale_ps(k, a, imm8) _mm_maskz_roundscale_ps(k, a, imm8)
#else
  #define easysimd_mm_maskz_roundscale_ps(k, a, imm8) easysimd_mm_maskz_mov_ps(k, easysimd_mm_roundscale_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_roundscale_ps
  #define _mm_maskz_roundscale_ps(k, a, imm8) easysimd_mm_maskz_roundscale_ps(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm256_roundscale_ps(a, imm8) _mm256_roundscale_ps((a), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm256_roundscale_ps(a, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m256_private \
      easysimd_mm256_roundscale_ps_r_, \
      easysimd_mm256_roundscale_ps_a_ = easysimd__m256_to_private(a); \
    \
    for (size_t easysimd_mm256_roundscale_ps_i = 0 ; easysimd_mm256_roundscale_ps_i < (sizeof(easysimd_mm256_roundscale_ps_r_.m128) / sizeof(easysimd_mm256_roundscale_ps_r_.m128[0])) ; easysimd_mm256_roundscale_ps_i++) { \
      easysimd_mm256_roundscale_ps_r_.m128[easysimd_mm256_roundscale_ps_i] = easysimd_mm_roundscale_ps(easysimd_mm256_roundscale_ps_a_.m128[easysimd_mm256_roundscale_ps_i], imm8); \
    } \
    \
    easysimd__m256_from_private(easysimd_mm256_roundscale_ps_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m256
  easysimd_mm256_roundscale_ps_internal_ (easysimd__m256 result, easysimd__m256 a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m256 r, clear_sign;

    clear_sign = easysimd_mm256_andnot_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), result);
    r = easysimd_x_mm256_select_ps(result, a, easysimd_mm256_castsi256_ps(easysimd_mm256_cmpeq_epi32(easysimd_mm256_castps_si256(clear_sign), easysimd_mm256_castps_si256(easysimd_mm256_set1_ps(EASYSIMD_MATH_INFINITYF)))));

    return r;
  }
  #define easysimd_mm256_roundscale_ps(a, imm8) \
    easysimd_mm256_roundscale_ps_internal_( \
      easysimd_mm256_mul_ps( \
        easysimd_mm256_round_ps( \
          easysimd_mm256_mul_ps( \
            a, \
            easysimd_mm256_set1_ps(easysimd_math_exp2f(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm256_set1_ps(easysimd_math_exp2f(-((imm8 >> 4) & 15))) \
      ), \
      (a), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_roundscale_ps
  #define _mm256_roundscale_ps(a, imm8) easysimd_mm256_roundscale_ps(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_roundscale_ps(src, k, a, imm8) _mm256_mask_roundscale_ps(src, k, a, imm8)
#else
  #define easysimd_mm256_mask_roundscale_ps(src, k, a, imm8) easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_roundscale_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_roundscale_ps
  #define _mm256_mask_roundscale_ps(src, k, a, imm8) easysimd_mm256_mask_roundscale_ps(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_roundscale_ps(k, a, imm8) _mm256_maskz_roundscale_ps(k, a, imm8)
#else
  #define easysimd_mm256_maskz_roundscale_ps(k, a, imm8) easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_roundscale_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_roundscale_ps
  #define _mm256_maskz_roundscale_ps(k, a, imm8) easysimd_mm256_maskz_roundscale_ps(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_roundscale_ps(a, imm8) _mm512_roundscale_ps((a), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_roundscale_ps(a, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512_private \
      easysimd_mm512_roundscale_ps_r_, \
      easysimd_mm512_roundscale_ps_a_ = easysimd__m512_to_private(a); \
    \
    for (size_t easysimd_mm512_roundscale_ps_i = 0 ; easysimd_mm512_roundscale_ps_i < (sizeof(easysimd_mm512_roundscale_ps_r_.m256) / sizeof(easysimd_mm512_roundscale_ps_r_.m256[0])) ; easysimd_mm512_roundscale_ps_i++) { \
      easysimd_mm512_roundscale_ps_r_.m256[easysimd_mm512_roundscale_ps_i] = easysimd_mm256_roundscale_ps(easysimd_mm512_roundscale_ps_a_.m256[easysimd_mm512_roundscale_ps_i], imm8); \
    } \
    \
    easysimd__m512_from_private(easysimd_mm512_roundscale_ps_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_mm512_roundscale_ps_internal_ (easysimd__m512 result, easysimd__m512 a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m512 r, clear_sign;

    clear_sign = easysimd_mm512_andnot_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), result);
    r = easysimd_mm512_mask_mov_ps(result, easysimd_mm512_cmpeq_epi32_mask(easysimd_mm512_castps_si512(clear_sign), easysimd_mm512_castps_si512(easysimd_mm512_set1_ps(EASYSIMD_MATH_INFINITYF))), a);

    return r;
  }
  #define easysimd_mm512_roundscale_ps(a, imm8) \
    easysimd_mm512_roundscale_ps_internal_( \
      easysimd_mm512_mul_ps( \
        easysimd_x_mm512_round_ps( \
          easysimd_mm512_mul_ps( \
            a, \
            easysimd_mm512_set1_ps(easysimd_math_exp2f(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm512_set1_ps(easysimd_math_exp2f(-((imm8 >> 4) & 15))) \
      ), \
      (a), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_roundscale_ps
  #define _mm512_roundscale_ps(a, imm8) easysimd_mm512_roundscale_ps(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_roundscale_ps(src, k, a, imm8) _mm512_mask_roundscale_ps(src, k, a, imm8)
#else
  #define easysimd_mm512_mask_roundscale_ps(src, k, a, imm8) easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_roundscale_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_roundscale_ps
  #define _mm512_mask_roundscale_ps(src, k, a, imm8) easysimd_mm512_mask_roundscale_ps(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_roundscale_ps(k, a, imm8) _mm512_maskz_roundscale_ps(k, a, imm8)
#else
  #define easysimd_mm512_maskz_roundscale_ps(k, a, imm8) easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_roundscale_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_roundscale_ps
  #define _mm512_maskz_roundscale_ps(k, a, imm8) easysimd_mm512_maskz_roundscale_ps(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_roundscale_pd(a, imm8) _mm_roundscale_pd((a), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_roundscale_pd_internal_ (easysimd__m128d result, easysimd__m128d a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m128d r, clear_sign;

    clear_sign = easysimd_mm_andnot_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), result);
    r = easysimd_x_mm_select_pd(result, a, easysimd_mm_cmpeq_pd(clear_sign, easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY)));

    return r;
  }
  #define easysimd_mm_roundscale_pd(a, imm8) \
    easysimd_mm_roundscale_pd_internal_( \
      easysimd_mm_mul_pd( \
        easysimd_mm_round_pd( \
          easysimd_mm_mul_pd( \
            a, \
            easysimd_mm_set1_pd(easysimd_math_exp2(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm_set1_pd(easysimd_math_exp2(-((imm8 >> 4) & 15))) \
      ), \
      (a), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_roundscale_pd
  #define _mm_roundscale_pd(a, imm8) easysimd_mm_roundscale_pd(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_roundscale_pd(src, k, a, imm8) _mm_mask_roundscale_pd(src, k, a, imm8)
#else
  #define easysimd_mm_mask_roundscale_pd(src, k, a, imm8) easysimd_mm_mask_mov_pd(src, k, easysimd_mm_roundscale_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_roundscale_pd
  #define _mm_mask_roundscale_pd(src, k, a, imm8) easysimd_mm_mask_roundscale_pd(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_roundscale_pd(k, a, imm8) _mm_maskz_roundscale_pd(k, a, imm8)
#else
  #define easysimd_mm_maskz_roundscale_pd(k, a, imm8) easysimd_mm_maskz_mov_pd(k, easysimd_mm_roundscale_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_roundscale_pd
  #define _mm_maskz_roundscale_pd(k, a, imm8) easysimd_mm_maskz_roundscale_pd(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm256_roundscale_pd(a, imm8) _mm256_roundscale_pd((a), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm256_roundscale_pd(a, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m256d_private \
      easysimd_mm256_roundscale_pd_r_, \
      easysimd_mm256_roundscale_pd_a_ = easysimd__m256d_to_private(a); \
    \
    for (size_t easysimd_mm256_roundscale_pd_i = 0 ; easysimd_mm256_roundscale_pd_i < (sizeof(easysimd_mm256_roundscale_pd_r_.m128d) / sizeof(easysimd_mm256_roundscale_pd_r_.m128d[0])) ; easysimd_mm256_roundscale_pd_i++) { \
      easysimd_mm256_roundscale_pd_r_.m128d[easysimd_mm256_roundscale_pd_i] = easysimd_mm_roundscale_pd(easysimd_mm256_roundscale_pd_a_.m128d[easysimd_mm256_roundscale_pd_i], imm8); \
    } \
    \
    easysimd__m256d_from_private(easysimd_mm256_roundscale_pd_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m256d
  easysimd_mm256_roundscale_pd_internal_ (easysimd__m256d result, easysimd__m256d a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m256d r, clear_sign;

    clear_sign = easysimd_mm256_andnot_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), result);
    r = easysimd_x_mm256_select_pd(result, a, easysimd_mm256_castsi256_pd(easysimd_mm256_cmpeq_epi64(easysimd_mm256_castpd_si256(clear_sign), easysimd_mm256_castpd_si256(easysimd_mm256_set1_pd(EASYSIMD_MATH_INFINITY)))));

    return r;
  }
  #define easysimd_mm256_roundscale_pd(a, imm8) \
    easysimd_mm256_roundscale_pd_internal_( \
      easysimd_mm256_mul_pd( \
        easysimd_mm256_round_pd( \
          easysimd_mm256_mul_pd( \
            a, \
            easysimd_mm256_set1_pd(easysimd_math_exp2(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm256_set1_pd(easysimd_math_exp2(-((imm8 >> 4) & 15))) \
      ), \
      (a), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_roundscale_pd
  #define _mm256_roundscale_pd(a, imm8) easysimd_mm256_roundscale_pd(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_roundscale_pd(src, k, a, imm8) _mm256_mask_roundscale_pd(src, k, a, imm8)
#else
  #define easysimd_mm256_mask_roundscale_pd(src, k, a, imm8) easysimd_mm256_mask_mov_pd(src, k, easysimd_mm256_roundscale_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_roundscale_pd
  #define _mm256_mask_roundscale_pd(src, k, a, imm8) easysimd_mm256_mask_roundscale_pd(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_roundscale_pd(k, a, imm8) _mm256_maskz_roundscale_pd(k, a, imm8)
#else
  #define easysimd_mm256_maskz_roundscale_pd(k, a, imm8) easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_roundscale_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_roundscale_pd
  #define _mm256_maskz_roundscale_pd(k, a, imm8) easysimd_mm256_maskz_roundscale_pd(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_roundscale_pd(a, imm8) _mm512_roundscale_pd((a), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_roundscale_pd(a, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512d_private \
      easysimd_mm512_roundscale_pd_r_, \
      easysimd_mm512_roundscale_pd_a_ = easysimd__m512d_to_private(a); \
    \
    for (size_t easysimd_mm512_roundscale_pd_i = 0 ; easysimd_mm512_roundscale_pd_i < (sizeof(easysimd_mm512_roundscale_pd_r_.m256d) / sizeof(easysimd_mm512_roundscale_pd_r_.m256d[0])) ; easysimd_mm512_roundscale_pd_i++) { \
      easysimd_mm512_roundscale_pd_r_.m256d[easysimd_mm512_roundscale_pd_i] = easysimd_mm256_roundscale_pd(easysimd_mm512_roundscale_pd_a_.m256d[easysimd_mm512_roundscale_pd_i], imm8); \
    } \
    \
    easysimd__m512d_from_private(easysimd_mm512_roundscale_pd_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_mm512_roundscale_pd_internal_ (easysimd__m512d result, easysimd__m512d a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m512d r, clear_sign;

    clear_sign = easysimd_mm512_andnot_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), result);
    r = easysimd_mm512_mask_mov_pd(result, easysimd_mm512_cmpeq_epi64_mask(easysimd_mm512_castpd_si512(clear_sign), easysimd_mm512_castpd_si512(easysimd_mm512_set1_pd(EASYSIMD_MATH_INFINITY))), a);

    return r;
  }
  #define easysimd_mm512_roundscale_pd(a, imm8) \
    easysimd_mm512_roundscale_pd_internal_( \
      easysimd_mm512_mul_pd( \
        easysimd_x_mm512_round_pd( \
          easysimd_mm512_mul_pd( \
            a, \
            easysimd_mm512_set1_pd(easysimd_math_exp2(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm512_set1_pd(easysimd_math_exp2(-((imm8 >> 4) & 15))) \
      ), \
      (a), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_roundscale_pd
  #define _mm512_roundscale_pd(a, imm8) easysimd_mm512_roundscale_pd(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_roundscale_pd(src, k, a, imm8) _mm512_mask_roundscale_pd(src, k, a, imm8)
#else
  #define easysimd_mm512_mask_roundscale_pd(src, k, a, imm8) easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_roundscale_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_roundscale_pd
  #define _mm512_mask_roundscale_pd(src, k, a, imm8) easysimd_mm512_mask_roundscale_pd(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_roundscale_pd(k, a, imm8) _mm512_maskz_roundscale_pd(k, a, imm8)
#else
  #define easysimd_mm512_maskz_roundscale_pd(k, a, imm8) easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_roundscale_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_roundscale_pd
  #define _mm512_maskz_roundscale_pd(k, a, imm8) easysimd_mm512_maskz_roundscale_pd(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_roundscale_ss(a, b, imm8) _mm_roundscale_ss((a), (b), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_roundscale_ss_internal_ (easysimd__m128 result, easysimd__m128 b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m128_private
      r_ = easysimd__m128_to_private(result),
      b_ = easysimd__m128_to_private(b);

    if(easysimd_math_isinff(r_.f32[0]))
      r_.f32[0] = b_.f32[0];

    return easysimd__m128_from_private(r_);
  }
  #define easysimd_mm_roundscale_ss(a, b, imm8) \
    easysimd_mm_roundscale_ss_internal_( \
      easysimd_mm_mul_ss( \
        easysimd_mm_round_ss( \
          a, \
          easysimd_mm_mul_ss( \
            b, \
            easysimd_mm_set1_ps(easysimd_math_exp2f(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm_set1_ps(easysimd_math_exp2f(-((imm8 >> 4) & 15))) \
      ), \
      (b), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_roundscale_ss
  #define _mm_roundscale_ss(a, b, imm8) easysimd_mm_roundscale_ss(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_92035)
  #define easysimd_mm_mask_roundscale_ss(src, k, a, b, imm8) _mm_mask_roundscale_ss((src), (k), (a), (b), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_mask_roundscale_ss_internal_ (easysimd__m128 a, easysimd__m128 b, easysimd__mmask8 k) {
    easysimd__m128 r;

    if(k & 1)
      r = a;
    else
      r = b;

    return r;
  }
  #define easysimd_mm_mask_roundscale_ss(src, k, a, b, imm8) \
    easysimd_mm_mask_roundscale_ss_internal_( \
      easysimd_mm_roundscale_ss( \
        a, \
        b, \
        imm8 \
      ), \
      easysimd_mm_move_ss( \
        (a), \
        (src) \
      ), \
      (k) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_roundscale_ss
  #define _mm_mask_roundscale_ss(src, k, a, b, imm8) easysimd_mm_mask_roundscale_ss(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_92035)
  #define easysimd_mm_maskz_roundscale_ss(k, a, b, imm8) _mm_maskz_roundscale_ss((k), (a), (b), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_maskz_roundscale_ss_internal_ (easysimd__m128 a, easysimd__m128 b, easysimd__mmask8 k) {
    easysimd__m128 r;

    if(k & 1)
      r = a;
    else
      r = b;

    return r;
  }
  #define easysimd_mm_maskz_roundscale_ss(k, a, b, imm8) \
    easysimd_mm_maskz_roundscale_ss_internal_( \
      easysimd_mm_roundscale_ss( \
        a, \
        b, \
        imm8 \
      ), \
      easysimd_mm_move_ss( \
        (a), \
        easysimd_mm_setzero_ps() \
      ), \
      (k) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_roundscale_ss
  #define _mm_maskz_roundscale_ss(k, a, b, imm8) easysimd_mm_maskz_roundscale_ss(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_roundscale_sd(a, b, imm8) _mm_roundscale_sd((a), (b), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_roundscale_sd_internal_ (easysimd__m128d result, easysimd__m128d b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
    HEDLEY_STATIC_CAST(void, imm8);

    easysimd__m128d_private
      r_ = easysimd__m128d_to_private(result),
      b_ = easysimd__m128d_to_private(b);

    if(easysimd_math_isinf(r_.f64[0]))
      r_.f64[0] = b_.f64[0];

    return easysimd__m128d_from_private(r_);
  }
  #define easysimd_mm_roundscale_sd(a, b, imm8) \
    easysimd_mm_roundscale_sd_internal_( \
      easysimd_mm_mul_sd( \
        easysimd_mm_round_sd( \
          a, \
          easysimd_mm_mul_sd( \
            b, \
            easysimd_mm_set1_pd(easysimd_math_exp2(((imm8 >> 4) & 15)))), \
          ((imm8) & 15) \
        ), \
        easysimd_mm_set1_pd(easysimd_math_exp2(-((imm8 >> 4) & 15))) \
      ), \
      (b), \
      (imm8) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_roundscale_sd
  #define _mm_roundscale_sd(a, b, imm8) easysimd_mm_roundscale_sd(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_92035)
  #define easysimd_mm_mask_roundscale_sd(src, k, a, b, imm8) _mm_mask_roundscale_sd((src), (k), (a), (b), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_mask_roundscale_sd_internal_ (easysimd__m128d a, easysimd__m128d b, easysimd__mmask8 k) {
    easysimd__m128d r;

    if(k & 1)
      r = a;
    else
      r = b;

    return r;
  }
  #define easysimd_mm_mask_roundscale_sd(src, k, a, b, imm8) \
    easysimd_mm_mask_roundscale_sd_internal_( \
      easysimd_mm_roundscale_sd( \
        a, \
        b, \
        imm8 \
      ), \
      easysimd_mm_move_sd( \
        (a), \
        (src) \
      ), \
      (k) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_roundscale_sd
  #define _mm_mask_roundscale_sd(src, k, a, b, imm8) easysimd_mm_mask_roundscale_sd(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_92035)
  #define easysimd_mm_maskz_roundscale_sd(k, a, b, imm8) _mm_maskz_roundscale_sd((k), (a), (b), (imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_maskz_roundscale_sd_internal_ (easysimd__m128d a, easysimd__m128d b, easysimd__mmask8 k) {
    easysimd__m128d r;

    if(k & 1)
      r = a;
    else
      r = b;

    return r;
  }
  #define easysimd_mm_maskz_roundscale_sd(k, a, b, imm8) \
    easysimd_mm_maskz_roundscale_sd_internal_( \
      easysimd_mm_roundscale_sd( \
        a, \
        b, \
        imm8 \
      ), \
      easysimd_mm_move_sd( \
        (a), \
        easysimd_mm_setzero_pd() \
      ), \
      (k) \
    )
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_roundscale_sd
  #define _mm_maskz_roundscale_sd(k, a, b, imm8) easysimd_mm_maskz_roundscale_sd(k, a, b, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_ROUNDSCALE_H) */
