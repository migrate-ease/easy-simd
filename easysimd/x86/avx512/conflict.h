#if !defined(EASYSIMD_X86_AVX512_CONFLICT_H)
#define EASYSIMD_X86_AVX512_CONFLICT_H

#include "types.h"
#include "mov_mask.h"
#include "mov.h"
#include "cmpeq.h"
#include "set1.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_conflict_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_conflict_epi32(a);
  #else
    easysimd__m128i_private
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128()),
      a_ = easysimd__m128i_to_private(a);

    for (size_t i = 1 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] =
          easysimd_mm_movemask_ps(
            easysimd_mm_castsi128_ps(
              easysimd_mm_cmpeq_epi32(easysimd_mm_set1_epi32(a_.i32[i]), a)
            )
          ) & ((1 << i) - 1);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_conflict_epi32
  #define _mm_conflict_epi32(a) easysimd_mm_conflict_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_conflict_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_mask_conflict_epi32(src, k, a);
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_conflict_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_conflict_epi32
  #define _mm_mask_conflict_epi32(src, k, a) easysimd_mm_mask_conflict_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_conflict_epi32 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_maskz_conflict_epi32(k, a);
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_conflict_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_conflict_epi32
  #define _mm_maskz_conflict_epi32(k, a) easysimd_mm_maskz_conflict_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_conflict_epi32 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_conflict_epi32(a);
  #else
    easysimd__m256i_private
      r_ = easysimd__m256i_to_private(easysimd_mm256_setzero_si256()),
      a_ = easysimd__m256i_to_private(a);

    for (size_t i = 1 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] =
          easysimd_mm256_movemask_ps(
            easysimd_mm256_castsi256_ps(
              easysimd_mm256_cmpeq_epi32(easysimd_mm256_set1_epi32(a_.i32[i]), a)
            )
          ) & ((1 << i) - 1);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_conflict_epi32
  #define _mm256_conflict_epi32(a) easysimd_mm256_conflict_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_conflict_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_mask_conflict_epi32(src, k, a);
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_conflict_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_conflict_epi32
  #define _mm256_mask_conflict_epi32(src, k, a) easysimd_mm256_mask_conflict_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_conflict_epi32 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_maskz_conflict_epi32(k, a);
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_conflict_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_conflict_epi32
  #define _mm256_maskz_conflict_epi32(k, a) easysimd_mm256_maskz_conflict_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_conflict_epi32 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_conflict_epi32(a);
  #else
    easysimd__m512i_private
      r_ = easysimd__m512i_to_private(easysimd_mm512_setzero_si512()),
      a_ = easysimd__m512i_to_private(a);

    for (size_t i = 1 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] =
        HEDLEY_STATIC_CAST(
          int32_t,
          easysimd_mm512_cmpeq_epi32_mask(easysimd_mm512_set1_epi32(a_.i32[i]), a)
        ) & ((1 << i) - 1);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_conflict_epi32
  #define _mm512_conflict_epi32(a) easysimd_mm512_conflict_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_conflict_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_mask_conflict_epi32(src, k, a);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_conflict_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_conflict_epi32
  #define _mm512_mask_conflict_epi32(src, k, a) easysimd_mm512_mask_conflict_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_conflict_epi32 (easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_maskz_conflict_epi32(k, a);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_conflict_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_conflict_epi32
  #define _mm512_maskz_conflict_epi32(k, a) easysimd_mm512_maskz_conflict_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_conflict_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_conflict_epi64(a);
  #else
    easysimd__m128i_private
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128()),
      a_ = easysimd__m128i_to_private(a);

    for (size_t i = 1 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] =
        HEDLEY_STATIC_CAST(
          int64_t,
          easysimd_mm_movemask_pd(
            easysimd_mm_castsi128_pd(
              easysimd_mm_cmpeq_epi64(easysimd_mm_set1_epi64x(a_.i64[i]), a)
            )
          )
        ) & ((1 << i) - 1);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_conflict_epi64
  #define _mm_conflict_epi64(a) easysimd_mm_conflict_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_conflict_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_mask_conflict_epi64(src, k, a);
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_conflict_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_conflict_epi64
  #define _mm_mask_conflict_epi64(src, k, a) easysimd_mm_mask_conflict_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_conflict_epi64 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_maskz_conflict_epi64(k, a);
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_conflict_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_conflict_epi64
  #define _mm_maskz_conflict_epi64(k, a) easysimd_mm_maskz_conflict_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_conflict_epi64 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_conflict_epi64(a);
  #else
    easysimd__m256i_private
      r_ = easysimd__m256i_to_private(easysimd_mm256_setzero_si256()),
      a_ = easysimd__m256i_to_private(a);

    for (size_t i = 1 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] =
        HEDLEY_STATIC_CAST(
          int64_t,
          easysimd_mm256_movemask_pd(
            easysimd_mm256_castsi256_pd(
              easysimd_mm256_cmpeq_epi64(easysimd_mm256_set1_epi64x(a_.i64[i]), a)
            )
          )
        ) & ((1 << i) - 1);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_conflict_epi64
  #define _mm256_conflict_epi64(a) easysimd_mm256_conflict_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_conflict_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_mask_conflict_epi64(src, k, a);
  #else
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_conflict_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_conflict_epi64
  #define _mm256_mask_conflict_epi64(src, k, a) easysimd_mm256_mask_conflict_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_conflict_epi64 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_maskz_conflict_epi64(k, a);
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_conflict_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_conflict_epi64
  #define _mm256_maskz_conflict_epi64(k, a) easysimd_mm256_maskz_conflict_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_conflict_epi64 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_conflict_epi64(a);
  #else
    easysimd__m512i_private
      r_ = easysimd__m512i_to_private(easysimd_mm512_setzero_si512()),
      a_ = easysimd__m512i_to_private(a);

    for (size_t i = 1 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] =
        HEDLEY_STATIC_CAST(
          int64_t,
          easysimd_mm512_cmpeq_epi64_mask(easysimd_mm512_set1_epi64(a_.i64[i]), a)
        ) & ((1 << i) - 1);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_conflict_epi64
  #define _mm512_conflict_epi64(a) easysimd_mm512_conflict_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_conflict_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_mask_conflict_epi64(src, k, a);
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_conflict_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_conflict_epi64
  #define _mm512_mask_conflict_epi64(src, k, a) easysimd_mm512_mask_conflict_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_conflict_epi64 (easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_maskz_conflict_epi64(k, a);
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_conflict_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_conflict_epi64
  #define _mm512_maskz_conflict_epi64(k, a) easysimd_mm512_maskz_conflict_epi64(k, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_CONFLICT_H) */
