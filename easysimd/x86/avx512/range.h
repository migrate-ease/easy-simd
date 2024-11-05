#if !defined(EASYSIMD_X86_AVX512_RANGE_H)
#define EASYSIMD_X86_AVX512_RANGE_H

#include "types.h"
#include "max.h"
#include "min.h"
#include "set1.h"
#include "copysign.h"
#include "abs.h"
#include "setzero.h"
#include "cmp.h"
#include "or.h"
#include "andnot.h"
#include "insert.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_range_ps (easysimd__m128 a, easysimd__m128 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  easysimd__m128 r;

  switch (imm8 & 3) {
    case 0:
      r = easysimd_mm_min_ps(a, b);
      break;
    case 1:
      r = easysimd_mm_max_ps(a, b);
      break;
    case 2:
      r = easysimd_x_mm_select_ps(b, a, easysimd_mm_cmple_ps(easysimd_x_mm_abs_ps(a), easysimd_x_mm_abs_ps(b)));
      break;
    case 3:
      r = easysimd_x_mm_select_ps(b, a, easysimd_mm_cmpge_ps(easysimd_x_mm_abs_ps(a), easysimd_x_mm_abs_ps(b)));
      break;
    default:
      break;
  }

  switch (imm8 & 12) {
    case 0:
      r = easysimd_x_mm_copysign_ps(r, a);
      break;
    case 8:
      r = easysimd_mm_andnot_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), r);
      break;
    case 12:
      r = easysimd_mm_or_ps(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), r);
      break;
    default:
      break;
  }

  return r;
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_range_ps(a, b, imm8) _mm_range_ps((a), (b), (imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_range_ps
  #define _mm_range_ps(a, b, imm8) easysimd_mm_range_ps(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_range_ps(src, k, a, b, imm8) _mm_mask_range_ps(src, k, a, b, imm8)
#else
  #define easysimd_mm_mask_range_ps(src, k, a, b, imm8) easysimd_mm_mask_mov_ps(src, k, easysimd_mm_range_ps(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_range_ps
  #define _mm_mask_range_ps(src, k, a, b, imm8) easysimd_mm_mask_range_ps(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_range_ps(k, a, b, imm8) _mm_maskz_range_ps(k, a, b, imm8)
#else
  #define easysimd_mm_maskz_range_ps(k, a, b, imm8) easysimd_mm_maskz_mov_ps(k, easysimd_mm_range_ps(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_range_ps
  #define _mm_maskz_range_ps(k, a, b, imm8) easysimd_mm_maskz_range_ps(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_range_ps (easysimd__m256 a, easysimd__m256 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  easysimd__m256 r;

  switch (imm8 & 3) {
    case 0:
      r = easysimd_mm256_min_ps(a, b);
      break;
    case 1:
      r = easysimd_mm256_max_ps(a, b);
      break;
    case 2:
      r = easysimd_x_mm256_select_ps(b, a, easysimd_mm256_cmp_ps(easysimd_x_mm256_abs_ps(a), easysimd_x_mm256_abs_ps(b), EASYSIMD_CMP_LE_OQ));
      break;
    case 3:
      r = easysimd_x_mm256_select_ps(b, a, easysimd_mm256_cmp_ps(easysimd_x_mm256_abs_ps(a), easysimd_x_mm256_abs_ps(b), EASYSIMD_CMP_GE_OQ));
      break;
    default:
      break;
  }

  switch (imm8 & 12) {
    case 0:
      r = easysimd_x_mm256_copysign_ps(r, a);
      break;
    case 8:
      r = easysimd_mm256_andnot_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), r);
      break;
    case 12:
      r = easysimd_mm256_or_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), r);
      break;
    default:
      break;
  }

  return r;
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_range_ps(a, b, imm8) _mm256_range_ps((a), (b), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm256_range_ps(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m256_private \
        easysimd_mm256_range_ps_r_, \
        easysimd_mm256_range_ps_a_ = easysimd__m256_to_private(a), \
        easysimd_mm256_range_ps_b_ = easysimd__m256_to_private(b); \
      \
      for (size_t easysimd_mm256_range_ps_i = 0 ; easysimd_mm256_range_ps_i < (sizeof(easysimd_mm256_range_ps_r_.m128) / sizeof(easysimd_mm256_range_ps_r_.m128[0])) ; easysimd_mm256_range_ps_i++) { \
        easysimd_mm256_range_ps_r_.m128[easysimd_mm256_range_ps_i] = easysimd_mm_range_ps(easysimd_mm256_range_ps_a_.m128[easysimd_mm256_range_ps_i], easysimd_mm256_range_ps_b_.m128[easysimd_mm256_range_ps_i], imm8); \
      } \
      \
      easysimd__m256_from_private(easysimd_mm256_range_ps_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_range_ps
  #define _mm256_range_ps(a, b, imm8) easysimd_mm256_range_ps(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_range_ps(src, k, a, b, imm8) _mm256_mask_range_ps(src, k, a, b, imm8)
#else
  #define easysimd_mm256_mask_range_ps(src, k, a, b, imm8) easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_range_ps(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_range_ps
  #define _mm256_mask_range_ps(src, k, a, b, imm8) easysimd_mm256_mask_range_ps(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_range_ps(k, a, b, imm8) _mm256_maskz_range_ps(k, a, b, imm8)
#else
  #define easysimd_mm256_maskz_range_ps(k, a, b, imm8) easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_range_ps(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_range_ps
  #define _mm256_maskz_range_ps(k, a, b, imm8) easysimd_mm256_maskz_range_ps(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_range_ps (easysimd__m512 a, easysimd__m512 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  easysimd__m512 r;

  switch (imm8 & 3) {
    case 0:
      r = easysimd_mm512_min_ps(a, b);
      break;
    case 1:
      r = easysimd_mm512_max_ps(a, b);
      break;
    case 2:
      r = easysimd_mm512_mask_mov_ps(b, easysimd_mm512_cmp_ps_mask(easysimd_mm512_abs_ps(a), easysimd_mm512_abs_ps(b), EASYSIMD_CMP_LE_OS), a);
      break;
    case 3:
      r = easysimd_mm512_mask_mov_ps(a, easysimd_mm512_cmp_ps_mask(easysimd_mm512_abs_ps(b), easysimd_mm512_abs_ps(a), EASYSIMD_CMP_GE_OS), b);
      break;
    default:
      break;
  }

  switch (imm8 & 12) {
    case 0:
      r = easysimd_x_mm512_copysign_ps(r, a);
      break;
    case 8:
      r = easysimd_mm512_andnot_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), r);
      break;
    case 12:
      r = easysimd_mm512_or_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(-0.0)), r);
      break;
    default:
      break;
  }

  return r;
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm512_range_ps(a, b, imm8) _mm512_range_ps((a), (b), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_range_ps(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512_private \
        easysimd_mm512_range_ps_r_, \
        easysimd_mm512_range_ps_a_ = easysimd__m512_to_private(a), \
        easysimd_mm512_range_ps_b_ = easysimd__m512_to_private(b); \
      \
      for (size_t easysimd_mm512_range_ps_i = 0 ; easysimd_mm512_range_ps_i < (sizeof(easysimd_mm512_range_ps_r_.m128) / sizeof(easysimd_mm512_range_ps_r_.m128[0])) ; easysimd_mm512_range_ps_i++) { \
        easysimd_mm512_range_ps_r_.m128[easysimd_mm512_range_ps_i] = easysimd_mm_range_ps(easysimd_mm512_range_ps_a_.m128[easysimd_mm512_range_ps_i], easysimd_mm512_range_ps_b_.m128[easysimd_mm512_range_ps_i], imm8); \
      } \
      \
      easysimd__m512_from_private(easysimd_mm512_range_ps_r_); \
    }))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_range_ps(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512_private \
        easysimd_mm512_range_ps_r_, \
        easysimd_mm512_range_ps_a_ = easysimd__m512_to_private(a), \
        easysimd_mm512_range_ps_b_ = easysimd__m512_to_private(b); \
      \
      for (size_t easysimd_mm512_range_ps_i = 0 ; easysimd_mm512_range_ps_i < (sizeof(easysimd_mm512_range_ps_r_.m256) / sizeof(easysimd_mm512_range_ps_r_.m256[0])) ; easysimd_mm512_range_ps_i++) { \
        easysimd_mm512_range_ps_r_.m256[easysimd_mm512_range_ps_i] = easysimd_mm256_range_ps(easysimd_mm512_range_ps_a_.m256[easysimd_mm512_range_ps_i], easysimd_mm512_range_ps_b_.m256[easysimd_mm512_range_ps_i], imm8); \
      } \
      \
      easysimd__m512_from_private(easysimd_mm512_range_ps_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_range_ps
  #define _mm512_range_ps(a, b, imm8) easysimd_mm512_range_ps(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_mask_range_ps(src, k, a, b, imm8) _mm512_mask_range_ps(src, k, a, b, imm8)
#else
  #define easysimd_mm512_mask_range_ps(src, k, a, b, imm8) easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_range_ps(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_range_ps
  #define _mm512_mask_range_ps(src, k, a, b, imm8) easysimd_mm512_mask_range_ps(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_maskz_range_ps(k, a, b, imm8) _mm512_maskz_range_ps(k, a, b, imm8)
#else
  #define easysimd_mm512_maskz_range_ps(k, a, b, imm8) easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_range_ps(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_range_ps
  #define _mm512_maskz_range_ps(k, a, b, imm8) easysimd_mm512_maskz_range_ps(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_range_pd (easysimd__m128d a, easysimd__m128d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  easysimd__m128d r;

  switch (imm8 & 3) {
    case 0:
      r = easysimd_mm_min_pd(a, b);
      break;
    case 1:
      r = easysimd_mm_max_pd(a, b);
      break;
    case 2:
      r = easysimd_x_mm_select_pd(b, a, easysimd_mm_cmple_pd(easysimd_x_mm_abs_pd(a), easysimd_x_mm_abs_pd(b)));
      break;
    case 3:
      r = easysimd_x_mm_select_pd(b, a, easysimd_mm_cmpge_pd(easysimd_x_mm_abs_pd(a), easysimd_x_mm_abs_pd(b)));
      break;
    default:
      break;
  }

  switch (imm8 & 12) {
    case 0:
      r = easysimd_x_mm_copysign_pd(r, a);
      break;
    case 8:
      r = easysimd_mm_andnot_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), r);
      break;
    case 12:
      r = easysimd_mm_or_pd(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), r);
      break;
    default:
      break;
  }

  return r;
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_range_pd(a, b, imm8) _mm_range_pd((a), (b), (imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_range_pd
  #define _mm_range_pd(a, b, imm8) easysimd_mm_range_pd(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_range_pd(src, k, a, b, imm8) _mm_mask_range_pd(src, k, a, b, imm8)
#else
  #define easysimd_mm_mask_range_pd(src, k, a, b, imm8) easysimd_mm_mask_mov_pd(src, k, easysimd_mm_range_pd(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_range_pd
  #define _mm_mask_range_pd(src, k, a, b, imm8) easysimd_mm_mask_range_pd(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_range_pd(k, a, b, imm8) _mm_maskz_range_pd(k, a, b, imm8)
#else
  #define easysimd_mm_maskz_range_pd(k, a, b, imm8) easysimd_mm_maskz_mov_pd(k, easysimd_mm_range_pd(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_range_pd
  #define _mm_maskz_range_pd(k, a, b, imm8) easysimd_mm_maskz_range_pd(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_range_pd (easysimd__m256d a, easysimd__m256d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  easysimd__m256d r;

  switch (imm8 & 3) {
    case 0:
      r = easysimd_mm256_min_pd(a, b);
      break;
    case 1:
      r = easysimd_mm256_max_pd(a, b);
      break;
    case 2:
      r = easysimd_x_mm256_select_pd(b, a, easysimd_mm256_cmp_pd(easysimd_x_mm256_abs_pd(a), easysimd_x_mm256_abs_pd(b), EASYSIMD_CMP_LE_OQ));
      break;
    case 3:
      r = easysimd_x_mm256_select_pd(b, a, easysimd_mm256_cmp_pd(easysimd_x_mm256_abs_pd(a), easysimd_x_mm256_abs_pd(b), EASYSIMD_CMP_GE_OQ));
      break;
    default:
      break;
  }

  switch (imm8 & 12) {
    case 0:
      r = easysimd_x_mm256_copysign_pd(r, a);
      break;
    case 8:
      r = easysimd_mm256_andnot_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), r);
      break;
    case 12:
      r = easysimd_mm256_or_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), r);
      break;
    default:
      break;
  }

  return r;
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_range_pd(a, b, imm8) _mm256_range_pd((a), (b), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm256_range_pd(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m256d_private \
        easysimd_mm256_range_pd_r_, \
        easysimd_mm256_range_pd_a_ = easysimd__m256d_to_private(a), \
        easysimd_mm256_range_pd_b_ = easysimd__m256d_to_private(b); \
      \
      for (size_t easysimd_mm256_range_pd_i = 0 ; easysimd_mm256_range_pd_i < (sizeof(easysimd_mm256_range_pd_r_.m128d) / sizeof(easysimd_mm256_range_pd_r_.m128d[0])) ; easysimd_mm256_range_pd_i++) { \
        easysimd_mm256_range_pd_r_.m128d[easysimd_mm256_range_pd_i] = easysimd_mm_range_pd(easysimd_mm256_range_pd_a_.m128d[easysimd_mm256_range_pd_i], easysimd_mm256_range_pd_b_.m128d[easysimd_mm256_range_pd_i], imm8); \
      } \
      \
      easysimd__m256d_from_private(easysimd_mm256_range_pd_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_range_pd
  #define _mm256_range_pd(a, b, imm8) easysimd_mm256_range_pd(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_range_pd(src, k, a, b, imm8) _mm256_mask_range_pd(src, k, a, b, imm8)
#else
  #define easysimd_mm256_mask_range_pd(src, k, a, b, imm8) easysimd_mm256_mask_mov_pd(src, k, easysimd_mm256_range_pd(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_range_pd
  #define _mm256_mask_range_pd(src, k, a, b, imm8) easysimd_mm256_mask_range_pd(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_range_pd(k, a, b, imm8) _mm256_maskz_range_pd(k, a, b, imm8)
#else
  #define easysimd_mm256_maskz_range_pd(k, a, b, imm8) easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_range_pd(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_range_pd
  #define _mm256_maskz_range_pd(k, a, b, imm8) easysimd_mm256_maskz_range_pd(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_range_pd (easysimd__m512d a, easysimd__m512d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  easysimd__m512d r;

  switch (imm8 & 3) {
    case 0:
      r = easysimd_mm512_min_pd(a, b);
      break;
    case 1:
      r = easysimd_mm512_max_pd(a, b);
      break;
    case 2:
      r = easysimd_mm512_mask_mov_pd(b, easysimd_mm512_cmp_pd_mask(easysimd_mm512_abs_pd(a), easysimd_mm512_abs_pd(b), EASYSIMD_CMP_LE_OS), a);
      break;
    case 3:
      r = easysimd_mm512_mask_mov_pd(a, easysimd_mm512_cmp_pd_mask(easysimd_mm512_abs_pd(b), easysimd_mm512_abs_pd(a), EASYSIMD_CMP_GE_OS), b);
      break;
    default:
      break;
  }

  switch (imm8 & 12) {
    case 0:
      r = easysimd_x_mm512_copysign_pd(r, a);
      break;
    case 8:
      r = easysimd_mm512_andnot_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), r);
      break;
    case 12:
      r = easysimd_mm512_or_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(-0.0)), r);
      break;
    default:
      break;
  }

  return r;
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm512_range_pd(a, b, imm8) _mm512_range_pd((a), (b), (imm8))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_range_pd(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d_private \
        easysimd_mm512_range_pd_r_, \
        easysimd_mm512_range_pd_a_ = easysimd__m512d_to_private(a), \
        easysimd_mm512_range_pd_b_ = easysimd__m512d_to_private(b); \
      \
      for (size_t easysimd_mm512_range_pd_i = 0 ; easysimd_mm512_range_pd_i < (sizeof(easysimd_mm512_range_pd_r_.m128d) / sizeof(easysimd_mm512_range_pd_r_.m128d[0])) ; easysimd_mm512_range_pd_i++) { \
        easysimd_mm512_range_pd_r_.m128d[easysimd_mm512_range_pd_i] = easysimd_mm_range_pd(easysimd_mm512_range_pd_a_.m128d[easysimd_mm512_range_pd_i], easysimd_mm512_range_pd_b_.m128d[easysimd_mm512_range_pd_i], imm8); \
      } \
      \
      easysimd__m512d_from_private(easysimd_mm512_range_pd_r_); \
    }))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_range_pd(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d_private \
        easysimd_mm512_range_pd_r_, \
        easysimd_mm512_range_pd_a_ = easysimd__m512d_to_private(a), \
        easysimd_mm512_range_pd_b_ = easysimd__m512d_to_private(b); \
      \
      for (size_t easysimd_mm512_range_pd_i = 0 ; easysimd_mm512_range_pd_i < (sizeof(easysimd_mm512_range_pd_r_.m256d) / sizeof(easysimd_mm512_range_pd_r_.m256d[0])) ; easysimd_mm512_range_pd_i++) { \
        easysimd_mm512_range_pd_r_.m256d[easysimd_mm512_range_pd_i] = easysimd_mm256_range_pd(easysimd_mm512_range_pd_a_.m256d[easysimd_mm512_range_pd_i], easysimd_mm512_range_pd_b_.m256d[easysimd_mm512_range_pd_i], imm8); \
      } \
      \
      easysimd__m512d_from_private(easysimd_mm512_range_pd_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_range_pd
  #define _mm512_range_pd(a, b, imm8) easysimd_mm512_range_pd(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_mask_range_pd(src, k, a, b, imm8) _mm512_mask_range_pd(src, k, a, b, imm8)
#else
  #define easysimd_mm512_mask_range_pd(src, k, a, b, imm8) easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_range_pd(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_range_pd
  #define _mm512_mask_range_pd(src, k, a, b, imm8) easysimd_mm512_mask_range_pd(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_maskz_range_pd(k, a, b, imm8) _mm512_maskz_range_pd(k, a, b, imm8)
#else
  #define easysimd_mm512_maskz_range_pd(k, a, b, imm8) easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_range_pd(a, b, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_range_pd
  #define _mm512_maskz_range_pd(k, a, b, imm8) easysimd_mm512_maskz_range_pd(k, a, b, imm8)
#endif

#if (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_x_mm_range_ss(a, b, imm8) easysimd_mm_move_ss(a, easysimd_mm_range_ps(a, b, imm8))
#elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
  #define easysimd_x_mm_range_ss(a, b, imm8) easysimd_mm_move_ss(a, easysimd_mm_range_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b), imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_x_mm_range_ss (easysimd__m128 a, easysimd__m128 b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
    easysimd__m128_private
      r_ = easysimd__m128_to_private(a),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);
    easysimd_float32 abs_a = easysimd_uint32_as_float32(a_.u32[0] & UINT32_C(2147483647));
    easysimd_float32 abs_b = easysimd_uint32_as_float32(b_.u32[0] & UINT32_C(2147483647));

    switch (imm8 & 3) {
      case 0:
        r_ = easysimd__m128_to_private(easysimd_mm_min_ss(a, b));
        break;
      case 1:
        r_ = easysimd__m128_to_private(easysimd_mm_max_ss(a, b));
        break;
      case 2:
        r_.f32[0] = abs_a <= abs_b ? a_.f32[0] : b_.f32[0];
        break;
      case 3:
        r_.f32[0] = abs_b >= abs_a ? b_.f32[0] : a_.f32[0];
        break;
      default:
        break;
    }

    switch (imm8 & 12) {
      case 0:
        r_.f32[0] = easysimd_uint32_as_float32((a_.u32[0] & UINT32_C(2147483648)) ^ (r_.u32[0] & UINT32_C(2147483647)));
        break;
      case 8:
        r_.f32[0] = easysimd_uint32_as_float32(r_.u32[0] & UINT32_C(2147483647));
        break;
      case 12:
        r_.f32[0] = easysimd_uint32_as_float32(r_.u32[0] | UINT32_C(2147483648));
        break;
      default:
        break;
    }

    return easysimd__m128_from_private(r_);
  }
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_mask_range_ss(src, k, a, b, imm8) _mm_mask_range_ss(src, k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_mask_range_ss(src, k, a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m128_private  \
      easysimd_mm_mask_range_ss_r_ = easysimd__m128_to_private(a), \
      easysimd_mm_mask_range_ss_src_ = easysimd__m128_to_private(src); \
    \
    if (k & 1) \
      easysimd_mm_mask_range_ss_r_ = easysimd__m128_to_private(easysimd_x_mm_range_ss(a, b, imm8)); \
    else \
      easysimd_mm_mask_range_ss_r_.f32[0] = easysimd_mm_mask_range_ss_src_.f32[0]; \
    \
    easysimd__m128_from_private(easysimd_mm_mask_range_ss_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_mask_range_ss (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
    easysimd__m128_private
      r_ = easysimd__m128_to_private(a),
      src_ = easysimd__m128_to_private(src);

    if (k & 1)
      r_ = easysimd__m128_to_private(easysimd_x_mm_range_ss(a, b, imm8));
    else
      r_.f32[0] = src_.f32[0];

    return easysimd__m128_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_range_ss
  #define _mm_mask_range_ss(src, k, a, b, imm8) easysimd_mm_mask_range_ss(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_maskz_range_ss(k, a, b, imm8) _mm_maskz_range_ss(k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_maskz_range_ss(k, a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m128_private easysimd_mm_maskz_range_ss_r_ = easysimd__m128_to_private(a); \
    \
    if (k & 1) \
      easysimd_mm_maskz_range_ss_r_ = easysimd__m128_to_private(easysimd_x_mm_range_ss(a, b, imm8)); \
    else \
      easysimd_mm_maskz_range_ss_r_.f32[0] = EASYSIMD_FLOAT32_C(0.0); \
    \
    easysimd__m128_from_private(easysimd_mm_maskz_range_ss_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_maskz_range_ss (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
    easysimd__m128_private r_ = easysimd__m128_to_private(a);

    if (k & 1)
      r_ = easysimd__m128_to_private(easysimd_x_mm_range_ss(a, b, imm8));
    else
      r_.f32[0] = EASYSIMD_FLOAT32_C(0.0);

    return easysimd__m128_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_range_ss
  #define _mm_maskz_range_ss(k, a, b, imm8) easysimd_mm_mask_range_ss(k, a, b, imm8)
#endif

#if (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_x_mm_range_sd(a, b, imm8) easysimd_mm_move_sd(a, easysimd_mm_range_pd(a, b, imm8))
#elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
  #define easysimd_x_mm_range_sd(a, b, imm8) easysimd_mm_move_sd(a, easysimd_mm_range_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b), imm8))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_x_mm_range_sd (easysimd__m128d a, easysimd__m128d b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
    easysimd__m128d_private
      r_ = easysimd__m128d_to_private(a),
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    easysimd_float64 abs_a = easysimd_uint64_as_float64(a_.u64[0] & UINT64_C(9223372036854775807));
    easysimd_float64 abs_b = easysimd_uint64_as_float64(b_.u64[0] & UINT64_C(9223372036854775807));

    switch (imm8 & 3) {
      case 0:
        r_ = easysimd__m128d_to_private(easysimd_mm_min_sd(a, b));
        break;
      case 1:
        r_ = easysimd__m128d_to_private(easysimd_mm_max_sd(a, b));
        break;
      case 2:
        r_.f64[0] = abs_a <= abs_b ? a_.f64[0] : b_.f64[0];
        break;
      case 3:
        r_.f64[0] = abs_b >= abs_a ? b_.f64[0] : a_.f64[0];
        break;
      default:
        break;
    }

    switch (imm8 & 12) {
      case 0:
        r_.f64[0] = easysimd_uint64_as_float64((a_.u64[0] & UINT64_C(9223372036854775808)) ^ (r_.u64[0] & UINT64_C(9223372036854775807)));
        break;
      case 8:
        r_.f64[0] = easysimd_uint64_as_float64(r_.u64[0] & UINT64_C(9223372036854775807));
        break;
      case 12:
        r_.f64[0] = easysimd_uint64_as_float64(r_.u64[0] | UINT64_C(9223372036854775808));
        break;
      default:
        break;
    }

    return easysimd__m128d_from_private(r_);
  }
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_mask_range_sd(src, k, a, b, imm8) _mm_mask_range_sd(src, k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_mask_range_sd(src, k, a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m128d_private  \
      easysimd_mm_mask_range_sd_r_ = easysimd__m128d_to_private(a), \
      easysimd_mm_mask_range_sd_src_ = easysimd__m128d_to_private(src); \
    \
    if (k & 1) \
      easysimd_mm_mask_range_sd_r_ = easysimd__m128d_to_private(easysimd_x_mm_range_sd(a, b, imm8)); \
    else \
      easysimd_mm_mask_range_sd_r_.f64[0] = easysimd_mm_mask_range_sd_src_.f64[0]; \
    \
    easysimd__m128d_from_private(easysimd_mm_mask_range_sd_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_mask_range_sd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
    easysimd__m128d_private
      r_ = easysimd__m128d_to_private(a),
      src_ = easysimd__m128d_to_private(src);

    if (k & 1)
      r_ = easysimd__m128d_to_private(easysimd_x_mm_range_sd(a, b, imm8));
    else
      r_.f64[0] = src_.f64[0];

    return easysimd__m128d_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_range_sd
  #define _mm_mask_range_sd(src, k, a, b, imm8) easysimd_mm_mask_range_sd(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_maskz_range_sd(k, a, b, imm8) _mm_maskz_range_sd(k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_maskz_range_sd(k, a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m128d_private easysimd_mm_maskz_range_sd_r_ = easysimd__m128d_to_private(a); \
    \
    if (k & 1) \
      easysimd_mm_maskz_range_sd_r_ = easysimd__m128d_to_private(easysimd_x_mm_range_sd(a, b, imm8)); \
    else \
      easysimd_mm_maskz_range_sd_r_.f64[0] = EASYSIMD_FLOAT64_C(0.0); \
    \
    easysimd__m128d_from_private(easysimd_mm_maskz_range_sd_r_); \
  }))
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_maskz_range_sd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
    easysimd__m128d_private r_ = easysimd__m128d_to_private(a);

    if (k & 1)
      r_ = easysimd__m128d_to_private(easysimd_x_mm_range_sd(a, b, imm8));
    else
      r_.f64[0] = EASYSIMD_FLOAT64_C(0.0);

    return easysimd__m128d_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_range_sd
  #define _mm_maskz_range_sd(k, a, b, imm8) easysimd_mm_mask_range_sd(k, a, b, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_RANGE_H) */
