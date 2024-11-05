#if !defined(EASYSIMD_X86_AVX512_RANGE_ROUND_H)
#define EASYSIMD_X86_AVX512_RANGE_ROUND_H

#include "types.h"
#include "range.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_range_round_ps(a, b, imm8, sae) _mm512_range_round_ps(a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_range_round_ps(a, b, imm8, sae) easysimd_mm512_range_ps(a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_range_round_ps(a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512 easysimd_mm512_range_round_ps_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_range_round_ps_envp; \
        int easysimd_mm512_range_round_ps_x = feholdexcept(&easysimd_mm512_range_round_ps_envp); \
        easysimd_mm512_range_round_ps_r = easysimd_mm512_range_ps(a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_range_round_ps_x == 0)) \
          fesetenv(&easysimd_mm512_range_round_ps_envp); \
      } \
      else { \
        easysimd_mm512_range_round_ps_r = easysimd_mm512_range_ps(a, b, imm8); \
      } \
      \
      easysimd_mm512_range_round_ps_r; \
    }))
  #else
    #define easysimd_mm512_range_round_ps(a, b, imm8, sae) easysimd_mm512_range_ps(a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_mm512_range_round_ps (easysimd__m512 a, easysimd__m512 b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_range_ps(a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_range_ps(a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_range_ps(a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_range_round_ps
  #define _mm512_range_round_ps(a, b, imm8, sae) easysimd_mm512_range_round_ps(a, b, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_mask_range_round_ps(src, k, a, b, imm8, sae) _mm512_mask_range_round_ps(src, k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_mask_range_round_ps(src, k, a, b, imm8, sae) easysimd_mm512_mask_range_ps(src, k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_mask_range_round_ps(src, k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512 easysimd_mm512_mask_range_round_ps_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_mask_range_round_ps_envp; \
        int easysimd_mm512_mask_range_round_ps_x = feholdexcept(&easysimd_mm512_mask_range_round_ps_envp); \
        easysimd_mm512_mask_range_round_ps_r = easysimd_mm512_mask_range_ps(src, k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_mask_range_round_ps_x == 0)) \
          fesetenv(&easysimd_mm512_mask_range_round_ps_envp); \
      } \
      else { \
        easysimd_mm512_mask_range_round_ps_r = easysimd_mm512_mask_range_ps(src, k, a, b, imm8); \
      } \
      \
      easysimd_mm512_mask_range_round_ps_r; \
    }))
  #else
    #define easysimd_mm512_mask_range_round_ps(src, k, a, b, imm8, sae) easysimd_mm512_mask_range_ps(src, k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_mm512_mask_range_round_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_mask_range_ps(src, k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_mask_range_ps(src, k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_mask_range_ps(src, k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_range_round_ps
  #define _mm512_mask_range_round_ps(src, k, a, b, imm8) easysimd_mm512_mask_range_round_ps(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_maskz_range_round_ps(k, a, b, imm8, sae) _mm512_maskz_range_round_ps(k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_maskz_range_round_ps(k, a, b, imm8, sae) easysimd_mm512_maskz_range_ps(k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_maskz_range_round_ps(k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512 easysimd_mm512_maskz_range_round_ps_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_maskz_range_round_ps_envp; \
        int easysimd_mm512_maskz_range_round_ps_x = feholdexcept(&easysimd_mm512_maskz_range_round_ps_envp); \
        easysimd_mm512_maskz_range_round_ps_r = easysimd_mm512_maskz_range_ps(k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_maskz_range_round_ps_x == 0)) \
          fesetenv(&easysimd_mm512_maskz_range_round_ps_envp); \
      } \
      else { \
        easysimd_mm512_maskz_range_round_ps_r = easysimd_mm512_maskz_range_ps(k, a, b, imm8); \
      } \
      \
      easysimd_mm512_maskz_range_round_ps_r; \
    }))
  #else
    #define easysimd_mm512_maskz_range_round_ps(k, a, b, imm8, sae) easysimd_mm512_maskz_range_ps(k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_mm512_maskz_range_round_ps (easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_maskz_range_ps(k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_maskz_range_ps(k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_maskz_range_ps(k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_range_round_ps
  #define _mm512_maskz_range_round_ps(k, a, b, imm8) easysimd_mm512_maskz_range_round_ps(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_range_round_pd(a, b, imm8, sae) _mm512_range_round_pd(a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_range_round_pd(a, b, imm8, sae) easysimd_mm512_range_pd(a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_range_round_pd(a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d easysimd_mm512_range_round_pd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_range_round_pd_envp; \
        int easysimd_mm512_range_round_pd_x = feholdexcept(&easysimd_mm512_range_round_pd_envp); \
        easysimd_mm512_range_round_pd_r = easysimd_mm512_range_pd(a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_range_round_pd_x == 0)) \
          fesetenv(&easysimd_mm512_range_round_pd_envp); \
      } \
      else { \
        easysimd_mm512_range_round_pd_r = easysimd_mm512_range_pd(a, b, imm8); \
      } \
      \
      easysimd_mm512_range_round_pd_r; \
    }))
  #else
    #define easysimd_mm512_range_round_pd(a, b, imm8, sae) easysimd_mm512_range_pd(a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_mm512_range_round_pd (easysimd__m512d a, easysimd__m512d b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_range_pd(a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_range_pd(a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_range_pd(a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_range_round_pd
  #define _mm512_range_round_pd(a, b, imm8, sae) easysimd_mm512_range_round_pd(a, b, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_mask_range_round_pd(src, k, a, b, imm8, sae) _mm512_mask_range_round_pd(src, k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_mask_range_round_pd(src, k, a, b, imm8, sae) easysimd_mm512_mask_range_pd(src, k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_mask_range_round_pd(src, k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d easysimd_mm512_mask_range_round_pd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_mask_range_round_pd_envp; \
        int easysimd_mm512_mask_range_round_pd_x = feholdexcept(&easysimd_mm512_mask_range_round_pd_envp); \
        easysimd_mm512_mask_range_round_pd_r = easysimd_mm512_mask_range_pd(src, k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_mask_range_round_pd_x == 0)) \
          fesetenv(&easysimd_mm512_mask_range_round_pd_envp); \
      } \
      else { \
        easysimd_mm512_mask_range_round_pd_r = easysimd_mm512_mask_range_pd(src, k, a, b, imm8); \
      } \
      \
      easysimd_mm512_mask_range_round_pd_r; \
    }))
  #else
    #define easysimd_mm512_mask_range_round_pd(src, k, a, b, imm8, sae) easysimd_mm512_mask_range_pd(src, k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_mm512_mask_range_round_pd (easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_mask_range_pd(src, k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_mask_range_pd(src, k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_mask_range_pd(src, k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_range_round_pd
  #define _mm512_mask_range_round_pd(src, k, a, b, imm8) easysimd_mm512_mask_range_round_pd(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_maskz_range_round_pd(k, a, b, imm8, sae) _mm512_maskz_range_round_pd(k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_maskz_range_round_pd(k, a, b, imm8, sae) easysimd_mm512_maskz_range_pd(k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_maskz_range_round_pd(k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d easysimd_mm512_maskz_range_round_pd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_maskz_range_round_pd_envp; \
        int easysimd_mm512_maskz_range_round_pd_x = feholdexcept(&easysimd_mm512_maskz_range_round_pd_envp); \
        easysimd_mm512_maskz_range_round_pd_r = easysimd_mm512_maskz_range_pd(k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_maskz_range_round_pd_x == 0)) \
          fesetenv(&easysimd_mm512_maskz_range_round_pd_envp); \
      } \
      else { \
        easysimd_mm512_maskz_range_round_pd_r = easysimd_mm512_maskz_range_pd(k, a, b, imm8); \
      } \
      \
      easysimd_mm512_maskz_range_round_pd_r; \
    }))
  #else
    #define easysimd_mm512_maskz_range_round_pd(k, a, b, imm8, sae) easysimd_mm512_maskz_range_pd(k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_mm512_maskz_range_round_pd (easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_maskz_range_pd(k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_maskz_range_pd(k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_maskz_range_pd(k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_range_round_pd
  #define _mm512_maskz_range_round_pd(k, a, b, imm8) easysimd_mm512_maskz_range_round_pd(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_range_round_ss(a, b, imm8, sae) _mm_range_round_ss(a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_range_round_ss(a, b, imm8, sae) easysimd_x_mm_range_ss(a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_range_round_ss(a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128 easysimd_mm_range_round_ss_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_range_round_ss_envp; \
        int easysimd_mm_range_round_ss_x = feholdexcept(&easysimd_mm_range_round_ss_envp); \
        easysimd_mm_range_round_ss_r = easysimd_x_mm_range_ss(a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_range_round_ss_x == 0)) \
          fesetenv(&easysimd_mm_range_round_ss_envp); \
      } \
      else { \
        easysimd_mm_range_round_ss_r = easysimd_x_mm_range_ss(a, b, imm8); \
      } \
      \
      easysimd_mm_range_round_ss_r; \
    }))
  #else
    #define easysimd_mm_range_round_ss(a, b, imm8, sae) easysimd_x_mm_range_ss(a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_range_round_ss (easysimd__m128 a, easysimd__m128 b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_x_mm_range_ss(a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_x_mm_range_ss(a, b, imm8);
      #endif
    }
    else {
      r = easysimd_x_mm_range_ss(a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_range_round_ss
  #define _mm_range_round_ss(a, b, imm8, sae) easysimd_mm_range_round_ss(a, b, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_mask_range_round_ss(src, k, a, b, imm8, sae) _mm_mask_range_round_ss(src, k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_mask_range_round_ss(src, k, a, b, imm8, sae) easysimd_mm_mask_range_ss(src, k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_mask_range_round_ss(src, k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128 easysimd_mm_mask_range_round_ss_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_mask_range_round_ss_envp; \
        int easysimd_mm_mask_range_round_ss_x = feholdexcept(&easysimd_mm_mask_range_round_ss_envp); \
        easysimd_mm_mask_range_round_ss_r = easysimd_mm_mask_range_ss(src, k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_mask_range_round_ss_x == 0)) \
          fesetenv(&easysimd_mm_mask_range_round_ss_envp); \
      } \
      else { \
        easysimd_mm_mask_range_round_ss_r = easysimd_mm_mask_range_ss(src, k, a, b, imm8); \
      } \
      \
      easysimd_mm_mask_range_round_ss_r; \
    }))
  #else
    #define easysimd_mm_mask_range_round_ss(src, k, a, b, imm8, sae) easysimd_mm_mask_range_ss(src, k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_mask_range_round_ss (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_mask_range_ss(src, k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_mask_range_ss(src, k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm_mask_range_ss(src, k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_range_round_ss
  #define _mm_mask_range_round_ss(src, k, a, b, imm8) easysimd_mm_mask_range_round_ss(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_maskz_range_round_ss(k, a, b, imm8, sae) _mm_maskz_range_round_ss(k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_maskz_range_round_ss(k, a, b, imm8, sae) easysimd_mm_maskz_range_ss(k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_maskz_range_round_ss(k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128 easysimd_mm_maskz_range_round_ss_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_maskz_range_round_ss_envp; \
        int easysimd_mm_maskz_range_round_ss_x = feholdexcept(&easysimd_mm_maskz_range_round_ss_envp); \
        easysimd_mm_maskz_range_round_ss_r = easysimd_mm_maskz_range_ss(k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_maskz_range_round_ss_x == 0)) \
          fesetenv(&easysimd_mm_maskz_range_round_ss_envp); \
      } \
      else { \
        easysimd_mm_maskz_range_round_ss_r = easysimd_mm_maskz_range_ss(k, a, b, imm8); \
      } \
      \
      easysimd_mm_maskz_range_round_ss_r; \
    }))
  #else
    #define easysimd_mm_maskz_range_round_ss(k, a, b, imm8, sae) easysimd_mm_maskz_range_ss(k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_maskz_range_round_ss (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_maskz_range_ss(k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_maskz_range_ss(k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm_maskz_range_ss(k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_range_round_ss
  #define _mm_maskz_range_round_ss(k, a, b, imm8) easysimd_mm_maskz_range_round_ss(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_range_round_sd(a, b, imm8, sae) _mm_range_round_sd(a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_range_round_sd(a, b, imm8, sae) easysimd_x_mm_range_sd(a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_range_round_sd(a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128d easysimd_mm_range_round_sd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_range_round_sd_envp; \
        int easysimd_mm_range_round_sd_x = feholdexcept(&easysimd_mm_range_round_sd_envp); \
        easysimd_mm_range_round_sd_r = easysimd_x_mm_range_sd(a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_range_round_sd_x == 0)) \
          fesetenv(&easysimd_mm_range_round_sd_envp); \
      } \
      else { \
        easysimd_mm_range_round_sd_r = easysimd_x_mm_range_sd(a, b, imm8); \
      } \
      \
      easysimd_mm_range_round_sd_r; \
    }))
  #else
    #define easysimd_mm_range_round_sd(a, b, imm8, sae) easysimd_x_mm_range_sd(a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_range_round_sd (easysimd__m128d a, easysimd__m128d b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_x_mm_range_sd(a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_x_mm_range_sd(a, b, imm8);
      #endif
    }
    else {
      r = easysimd_x_mm_range_sd(a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_range_round_sd
  #define _mm_range_round_sd(a, b, imm8, sae) easysimd_mm_range_round_sd(a, b, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_mask_range_round_sd(src, k, a, b, imm8, sae) _mm_mask_range_round_sd(src, k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_mask_range_round_sd(src, k, a, b, imm8, sae) easysimd_mm_mask_range_sd(src, k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_mask_range_round_sd(src, k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128d easysimd_mm_mask_range_round_sd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_mask_range_round_sd_envp; \
        int easysimd_mm_mask_range_round_sd_x = feholdexcept(&easysimd_mm_mask_range_round_sd_envp); \
        easysimd_mm_mask_range_round_sd_r = easysimd_mm_mask_range_sd(src, k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_mask_range_round_sd_x == 0)) \
          fesetenv(&easysimd_mm_mask_range_round_sd_envp); \
      } \
      else { \
        easysimd_mm_mask_range_round_sd_r = easysimd_mm_mask_range_sd(src, k, a, b, imm8); \
      } \
      \
      easysimd_mm_mask_range_round_sd_r; \
    }))
  #else
    #define easysimd_mm_mask_range_round_sd(src, k, a, b, imm8, sae) easysimd_mm_mask_range_sd(src, k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_mask_range_round_sd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_mask_range_sd(src, k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_mask_range_sd(src, k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm_mask_range_sd(src, k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_range_round_sd
  #define _mm_mask_range_round_sd(src, k, a, b, imm8) easysimd_mm_mask_range_round_sd(src, k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm_maskz_range_round_sd(k, a, b, imm8, sae) _mm_maskz_range_round_sd(k, a, b, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_maskz_range_round_sd(k, a, b, imm8, sae) easysimd_mm_maskz_range_sd(k, a, b, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_maskz_range_round_sd(k, a, b, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128d easysimd_mm_maskz_range_round_sd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_maskz_range_round_sd_envp; \
        int easysimd_mm_maskz_range_round_sd_x = feholdexcept(&easysimd_mm_maskz_range_round_sd_envp); \
        easysimd_mm_maskz_range_round_sd_r = easysimd_mm_maskz_range_sd(k, a, b, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_maskz_range_round_sd_x == 0)) \
          fesetenv(&easysimd_mm_maskz_range_round_sd_envp); \
      } \
      else { \
        easysimd_mm_maskz_range_round_sd_r = easysimd_mm_maskz_range_sd(k, a, b, imm8); \
      } \
      \
      easysimd_mm_maskz_range_round_sd_r; \
    }))
  #else
    #define easysimd_mm_maskz_range_round_sd(k, a, b, imm8, sae) easysimd_mm_maskz_range_sd(k, a, b, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_maskz_range_round_sd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_maskz_range_sd(k, a, b, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_maskz_range_sd(k, a, b, imm8);
      #endif
    }
    else {
      r = easysimd_mm_maskz_range_sd(k, a, b, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_range_round_sd
  #define _mm_maskz_range_round_sd(k, a, b, imm8) easysimd_mm_maskz_range_round_sd(k, a, b, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_RANGE_ROUND_H) */
