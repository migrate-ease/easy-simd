#if !defined(EASYSIMD_X86_AVX512_FIXUPIMM_ROUND_H)
#define EASYSIMD_X86_AVX512_FIXUPIMM_ROUND_H

#include "types.h"
#include "fixupimm.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_fixupimm_round_ps(a, b, c, imm8, sae) _mm512_fixupimm_round_ps(a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_fixupimm_round_ps(a, b, c, imm8, sae) easysimd_mm512_fixupimm_ps(a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_fixupimm_round_ps(a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512 easysimd_mm512_fixupimm_round_ps_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_fixupimm_round_ps_envp; \
        int easysimd_mm512_fixupimm_round_ps_x = feholdexcept(&easysimd_mm512_fixupimm_round_ps_envp); \
        easysimd_mm512_fixupimm_round_ps_r = easysimd_mm512_fixupimm_ps(a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_fixupimm_round_ps_x == 0)) \
          fesetenv(&easysimd_mm512_fixupimm_round_ps_envp); \
      } \
      else { \
        easysimd_mm512_fixupimm_round_ps_r = easysimd_mm512_fixupimm_ps(a, b, c, imm8); \
      } \
      \
      easysimd_mm512_fixupimm_round_ps_r; \
    }))
  #else
    #define easysimd_mm512_fixupimm_round_ps(a, b, c, imm8, sae) easysimd_mm512_fixupimm_ps(a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_mm512_fixupimm_round_ps (easysimd__m512 a, easysimd__m512 b, easysimd__m512i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_fixupimm_ps(a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_fixupimm_ps(a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_fixupimm_ps(a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_fixupimm_round_ps
  #define _mm512_fixupimm_round_ps(a, b, c, imm8, sae) easysimd_mm512_fixupimm_round_ps(a, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_fixupimm_round_ps(a, k, b, c, imm8, sae) _mm512_mask_fixupimm_round_ps(a, k, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_mask_fixupimm_round_ps(a, k, b, c, imm8, sae) easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_mask_fixupimm_round_ps(a, k, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512 easysimd_mm512_mask_fixupimm_round_ps_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_mask_fixupimm_round_ps_envp; \
        int easysimd_mm512_mask_fixupimm_round_ps_x = feholdexcept(&easysimd_mm512_mask_fixupimm_round_ps_envp); \
        easysimd_mm512_mask_fixupimm_round_ps_r = easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_mask_fixupimm_round_ps_x == 0)) \
          fesetenv(&easysimd_mm512_mask_fixupimm_round_ps_envp); \
      } \
      else { \
        easysimd_mm512_mask_fixupimm_round_ps_r = easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8); \
      } \
      \
      easysimd_mm512_mask_fixupimm_round_ps_r; \
    }))
  #else
    #define easysimd_mm512_mask_fixupimm_round_ps(a, k, b, c, imm8, sae) easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_mm512_mask_fixupimm_round_ps (easysimd__m512 a, easysimd__mmask16 k, easysimd__m512 b, easysimd__m512i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_mask_fixupimm_ps(a, k, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_fixupimm_round_ps
  #define _mm512_mask_fixupimm_round_ps(a, k, b, c, imm8, sae) easysimd_mm512_mask_fixupimm_round_ps(a, k, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_fixupimm_round_ps(k, a, b, c, imm8, sae) _mm512_maskz_fixupimm_round_ps(k, a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_maskz_fixupimm_round_ps(k, a, b, c, imm8, sae) easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_maskz_fixupimm_round_ps(k, a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512 easysimd_mm512_maskz_fixupimm_round_ps_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_maskz_fixupimm_round_ps_envp; \
        int easysimd_mm512_maskz_fixupimm_round_ps_x = feholdexcept(&easysimd_mm512_maskz_fixupimm_round_ps_envp); \
        easysimd_mm512_maskz_fixupimm_round_ps_r = easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_maskz_fixupimm_round_ps_x == 0)) \
          fesetenv(&easysimd_mm512_maskz_fixupimm_round_ps_envp); \
      } \
      else { \
        easysimd_mm512_maskz_fixupimm_round_ps_r = easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8); \
      } \
      \
      easysimd_mm512_maskz_fixupimm_round_ps_r; \
    }))
  #else
    #define easysimd_mm512_maskz_fixupimm_round_ps(k, a, b, c, imm8, sae) easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512
  easysimd_mm512_maskz_fixupimm_round_ps (easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b, easysimd__m512i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_maskz_fixupimm_ps(k, a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_fixupimm_round_ps
  #define _mm512_maskz_fixupimm_round_ps(k, a, b, c, imm8, sae) easysimd_mm512_maskz_fixupimm_round_ps(k, a, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_fixupimm_round_pd(a, b, c, imm8, sae) _mm512_fixupimm_round_pd(a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_fixupimm_round_pd(a, b, c, imm8, sae) easysimd_mm512_fixupimm_pd(a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_fixupimm_round_pd(a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d easysimd_mm512_fixupimm_round_pd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_fixupimm_round_pd_envp; \
        int easysimd_mm512_fixupimm_round_pd_x = feholdexcept(&easysimd_mm512_fixupimm_round_pd_envp); \
        easysimd_mm512_fixupimm_round_pd_r = easysimd_mm512_fixupimm_pd(a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_fixupimm_round_pd_x == 0)) \
          fesetenv(&easysimd_mm512_fixupimm_round_pd_envp); \
      } \
      else { \
        easysimd_mm512_fixupimm_round_pd_r = easysimd_mm512_fixupimm_pd(a, b, c, imm8); \
      } \
      \
      easysimd_mm512_fixupimm_round_pd_r; \
    }))
  #else
    #define easysimd_mm512_fixupimm_round_pd(a, b, c, imm8, sae) easysimd_mm512_fixupimm_pd(a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_mm512_fixupimm_round_pd (easysimd__m512d a, easysimd__m512d b, easysimd__m512i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_fixupimm_pd(a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_fixupimm_pd(a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_fixupimm_pd(a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_fixupimm_round_pd
  #define _mm512_fixupimm_round_pd(a, b, c, imm8, sae) easysimd_mm512_fixupimm_round_pd(a, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_fixupimm_round_pd(a, k, b, c, imm8, sae) _mm512_mask_fixupimm_round_pd(a, k, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_mask_fixupimm_round_pd(a, k, b, c, imm8, sae) easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_mask_fixupimm_round_pd(a, k, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d easysimd_mm512_mask_fixupimm_round_pd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_mask_fixupimm_round_pd_envp; \
        int easysimd_mm512_mask_fixupimm_round_pd_x = feholdexcept(&easysimd_mm512_mask_fixupimm_round_pd_envp); \
        easysimd_mm512_mask_fixupimm_round_pd_r = easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_mask_fixupimm_round_pd_x == 0)) \
          fesetenv(&easysimd_mm512_mask_fixupimm_round_pd_envp); \
      } \
      else { \
        easysimd_mm512_mask_fixupimm_round_pd_r = easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8); \
      } \
      \
      easysimd_mm512_mask_fixupimm_round_pd_r; \
    }))
  #else
    #define easysimd_mm512_mask_fixupimm_round_pd(a, k, b, c, imm8, sae) easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_mm512_mask_fixupimm_round_pd (easysimd__m512d a, easysimd__mmask8 k, easysimd__m512d b, easysimd__m512i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_mask_fixupimm_pd(a, k, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_fixupimm_round_pd
  #define _mm512_mask_fixupimm_round_pd(a, k, b, c, imm8, sae) easysimd_mm512_mask_fixupimm_round_pd(a, k, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_fixupimm_round_pd(k, a, b, c, imm8, sae) _mm512_maskz_fixupimm_round_pd(k, a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm512_maskz_fixupimm_round_pd(k, a, b, c, imm8, sae) easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm512_maskz_fixupimm_round_pd(k, a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m512d easysimd_mm512_maskz_fixupimm_round_pd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm512_maskz_fixupimm_round_pd_envp; \
        int easysimd_mm512_maskz_fixupimm_round_pd_x = feholdexcept(&easysimd_mm512_maskz_fixupimm_round_pd_envp); \
        easysimd_mm512_maskz_fixupimm_round_pd_r = easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm512_maskz_fixupimm_round_pd_x == 0)) \
          fesetenv(&easysimd_mm512_maskz_fixupimm_round_pd_envp); \
      } \
      else { \
        easysimd_mm512_maskz_fixupimm_round_pd_r = easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8); \
      } \
      \
      easysimd_mm512_maskz_fixupimm_round_pd_r; \
    }))
  #else
    #define easysimd_mm512_maskz_fixupimm_round_pd(k, a, b, c, imm8, sae) easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512d
  easysimd_mm512_maskz_fixupimm_round_pd (easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b, easysimd__m512i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m512d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm512_maskz_fixupimm_pd(k, a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_fixupimm_round_pd
  #define _mm512_maskz_fixupimm_round_pd(k, a, b, c, imm8, sae) easysimd_mm512_maskz_fixupimm_round_pd(k, a, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_fixupimm_round_ss(a, b, c, imm8, sae) _mm_fixupimm_round_ss(a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_fixupimm_round_ss(a, b, c, imm8, sae) easysimd_mm_fixupimm_ss(a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_fixupimm_round_ss(a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128 easysimd_mm_fixupimm_round_ss_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_fixupimm_round_ss_envp; \
        int easysimd_mm_fixupimm_round_ss_x = feholdexcept(&easysimd_mm_fixupimm_round_ss_envp); \
        easysimd_mm_fixupimm_round_ss_r = easysimd_mm_fixupimm_ss(a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_fixupimm_round_ss_x == 0)) \
          fesetenv(&easysimd_mm_fixupimm_round_ss_envp); \
      } \
      else { \
        easysimd_mm_fixupimm_round_ss_r = easysimd_mm_fixupimm_ss(a, b, c, imm8); \
      } \
      \
      easysimd_mm_fixupimm_round_ss_r; \
    }))
  #else
    #define easysimd_mm_fixupimm_round_ss(a, b, c, imm8, sae) easysimd_mm_fixupimm_ss(a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_fixupimm_round_ss (easysimd__m128 a, easysimd__m128 b, easysimd__m128i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_fixupimm_ss(a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_fixupimm_ss(a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm_fixupimm_ss(a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_fixupimm_round_ss
  #define _mm_fixupimm_round_ss(a, b, c, imm8, sae) easysimd_mm_fixupimm_round_ss(a, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae) _mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae) easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128 easysimd_mm_mask_fixupimm_round_ss_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_mask_fixupimm_round_ss_envp; \
        int easysimd_mm_mask_fixupimm_round_ss_x = feholdexcept(&easysimd_mm_mask_fixupimm_round_ss_envp); \
        easysimd_mm_mask_fixupimm_round_ss_r = easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_mask_fixupimm_round_ss_x == 0)) \
          fesetenv(&easysimd_mm_mask_fixupimm_round_ss_envp); \
      } \
      else { \
        easysimd_mm_mask_fixupimm_round_ss_r = easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8); \
      } \
      \
      easysimd_mm_mask_fixupimm_round_ss_r; \
    }))
  #else
    #define easysimd_mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae) easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_mask_fixupimm_round_ss (easysimd__m128 a, easysimd__mmask8 k, easysimd__m128 b, easysimd__m128i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm_mask_fixupimm_ss(a, k, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fixupimm_round_ss
  #define _mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae) easysimd_mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae) _mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae) easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128 easysimd_mm_maskz_fixupimm_round_ss_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_maskz_fixupimm_round_ss_envp; \
        int easysimd_mm_maskz_fixupimm_round_ss_x = feholdexcept(&easysimd_mm_maskz_fixupimm_round_ss_envp); \
        easysimd_mm_maskz_fixupimm_round_ss_r = easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_maskz_fixupimm_round_ss_x == 0)) \
          fesetenv(&easysimd_mm_maskz_fixupimm_round_ss_envp); \
      } \
      else { \
        easysimd_mm_maskz_fixupimm_round_ss_r = easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8); \
      } \
      \
      easysimd_mm_maskz_fixupimm_round_ss_r; \
    }))
  #else
    #define easysimd_mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae) easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128
  easysimd_mm_maskz_fixupimm_round_ss (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, easysimd__m128i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128 r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm_maskz_fixupimm_ss(k, a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fixupimm_round_ss
  #define _mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae) easysimd_mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_fixupimm_round_sd(a, b, c, imm8, sae) _mm_fixupimm_round_sd(a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_fixupimm_round_sd(a, b, c, imm8, sae) easysimd_mm_fixupimm_sd(a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_fixupimm_round_sd(a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128d easysimd_mm_fixupimm_round_sd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_fixupimm_round_sd_envp; \
        int easysimd_mm_fixupimm_round_sd_x = feholdexcept(&easysimd_mm_fixupimm_round_sd_envp); \
        easysimd_mm_fixupimm_round_sd_r = easysimd_mm_fixupimm_sd(a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_fixupimm_round_sd_x == 0)) \
          fesetenv(&easysimd_mm_fixupimm_round_sd_envp); \
      } \
      else { \
        easysimd_mm_fixupimm_round_sd_r = easysimd_mm_fixupimm_sd(a, b, c, imm8); \
      } \
      \
      easysimd_mm_fixupimm_round_sd_r; \
    }))
  #else
    #define easysimd_mm_fixupimm_round_sd(a, b, c, imm8, sae) easysimd_mm_fixupimm_sd(a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_fixupimm_round_sd (easysimd__m128d a, easysimd__m128d b, easysimd__m128i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_fixupimm_sd(a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_fixupimm_sd(a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm_fixupimm_sd(a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_fixupimm_round_sd
  #define _mm_fixupimm_round_sd(a, b, c, imm8, sae) easysimd_mm_fixupimm_round_sd(a, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae) _mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae) easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128d easysimd_mm_mask_fixupimm_round_sd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_mask_fixupimm_round_sd_envp; \
        int easysimd_mm_mask_fixupimm_round_sd_x = feholdexcept(&easysimd_mm_mask_fixupimm_round_sd_envp); \
        easysimd_mm_mask_fixupimm_round_sd_r = easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_mask_fixupimm_round_sd_x == 0)) \
          fesetenv(&easysimd_mm_mask_fixupimm_round_sd_envp); \
      } \
      else { \
        easysimd_mm_mask_fixupimm_round_sd_r = easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8); \
      } \
      \
      easysimd_mm_mask_fixupimm_round_sd_r; \
    }))
  #else
    #define easysimd_mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae) easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_mask_fixupimm_round_sd (easysimd__m128d a, easysimd__mmask8 k, easysimd__m128d b, easysimd__m128i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm_mask_fixupimm_sd(a, k, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_fixupimm_round_sd
  #define _mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae) easysimd_mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae) _mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae)
#elif defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae) easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8)
#elif defined(EASYSIMD_STATEMENT_EXPR_)
  #if defined(EASYSIMD_HAVE_FENV_H)
    #define easysimd_mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae) EASYSIMD_STATEMENT_EXPR_(({ \
      easysimd__m128d easysimd_mm_maskz_fixupimm_round_sd_r; \
      \
      if (sae & EASYSIMD_MM_FROUND_NO_EXC) { \
        fenv_t easysimd_mm_maskz_fixupimm_round_sd_envp; \
        int easysimd_mm_maskz_fixupimm_round_sd_x = feholdexcept(&easysimd_mm_maskz_fixupimm_round_sd_envp); \
        easysimd_mm_maskz_fixupimm_round_sd_r = easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8); \
        if (HEDLEY_LIKELY(easysimd_mm_maskz_fixupimm_round_sd_x == 0)) \
          fesetenv(&easysimd_mm_maskz_fixupimm_round_sd_envp); \
      } \
      else { \
        easysimd_mm_maskz_fixupimm_round_sd_r = easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8); \
      } \
      \
      easysimd_mm_maskz_fixupimm_round_sd_r; \
    }))
  #else
    #define easysimd_mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae) easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8)
  #endif
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128d
  easysimd_mm_maskz_fixupimm_round_sd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b, easysimd__m128i c, int imm8, int sae)
      EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)
      EASYSIMD_REQUIRE_CONSTANT(sae) {
    easysimd__m128d r;

    if (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(EASYSIMD_HAVE_FENV_H)
        fenv_t envp;
        int x = feholdexcept(&envp);
        r = easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8);
        if (HEDLEY_LIKELY(x == 0))
          fesetenv(&envp);
      #else
        r = easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8);
      #endif
    }
    else {
      r = easysimd_mm_maskz_fixupimm_sd(k, a, b, c, imm8);
    }

    return r;
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_fixupimm_round_sd
  #define _mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae) easysimd_mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_FIXUPIMM_ROUND_H) */
