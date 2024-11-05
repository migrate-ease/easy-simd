#if !defined(EASYSIMD_X86_AVX512_RORV_H)
#define EASYSIMD_X86_AVX512_RORV_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#include "srlv.h"
#include "sllv.h"
#include "or.h"
#include "and.h"
#include "sub.h"
#include "set1.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rorv_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_rorv_epi32(a, b);
  #else
    easysimd__m128i
      count1 = easysimd_mm_and_si128(b, easysimd_mm_set1_epi32(31)),
      count2 = easysimd_mm_sub_epi32(easysimd_mm_set1_epi32(32), count1);
    return easysimd_mm_or_si128(easysimd_mm_srlv_epi32(a, count1), easysimd_mm_sllv_epi32(a, count2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_rorv_epi32
  #define _mm_rorv_epi32(a, b) easysimd_mm_rorv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_rorv_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_rorv_epi32(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_rorv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_rorv_epi32
  #define _mm_mask_rorv_epi32(src, k, a, b) easysimd_mm_mask_rorv_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_rorv_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_rorv_epi32(k, a, b);
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_rorv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_rorv_epi32
  #define _mm_maskz_rorv_epi32(k, a, b) easysimd_mm_maskz_rorv_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rorv_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_rorv_epi32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    r_.m128i[0] = easysimd_mm_rorv_epi32(a_.m128i[0], b_.m128i[0]);
    r_.m128i[1] = easysimd_mm_rorv_epi32(a_.m128i[1], b_.m128i[1]);

    return easysimd__m256i_from_private(r_);
  #else
    easysimd__m256i
      count1 = easysimd_mm256_and_si256(b, easysimd_mm256_set1_epi32(31)),
      count2 = easysimd_mm256_sub_epi32(easysimd_mm256_set1_epi32(32), count1);
    return easysimd_mm256_or_si256(easysimd_mm256_srlv_epi32(a, count1), easysimd_mm256_sllv_epi32(a, count2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rorv_epi32
  #define _mm256_rorv_epi32(a, b) easysimd_mm256_rorv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_rorv_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_rorv_epi32(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_rorv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_rorv_epi32
  #define _mm256_mask_rorv_epi32(src, k, a, b) easysimd_mm256_mask_rorv_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_rorv_epi32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_rorv_epi32(k, a, b);
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_rorv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_rorv_epi32
  #define _mm256_maskz_rorv_epi32(k, a, b) easysimd_mm256_maskz_rorv_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rorv_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rorv_epi32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    r_.m256i[0] = easysimd_mm256_rorv_epi32(a_.m256i[0], b_.m256i[0]);
    r_.m256i[1] = easysimd_mm256_rorv_epi32(a_.m256i[1], b_.m256i[1]);

    return easysimd__m512i_from_private(r_);
  #else
    easysimd__m512i
      count1 = easysimd_mm512_and_si512(b, easysimd_mm512_set1_epi32(31)),
      count2 = easysimd_mm512_sub_epi32(easysimd_mm512_set1_epi32(32), count1);
    return easysimd_mm512_or_si512(easysimd_mm512_srlv_epi32(a, count1), easysimd_mm512_sllv_epi32(a, count2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rorv_epi32
  #define _mm512_rorv_epi32(a, b) easysimd_mm512_rorv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_rorv_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_rorv_epi32(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_rorv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rorv_epi32
  #define _mm512_mask_rorv_epi32(src, k, a, b) easysimd_mm512_mask_rorv_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_rorv_epi32 (easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_rorv_epi32(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_rorv_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_rorv_epi32
  #define _mm512_maskz_rorv_epi32(k, a, b) easysimd_mm512_maskz_rorv_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_rorv_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_rorv_epi64(a, b);
  #else
    easysimd__m128i
      count1 = easysimd_mm_and_si128(b, easysimd_mm_set1_epi64x(63)),
      count2 = easysimd_mm_sub_epi64(easysimd_mm_set1_epi64x(64), count1);
    return easysimd_mm_or_si128(easysimd_mm_srlv_epi64(a, count1), easysimd_mm_sllv_epi64(a, count2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_rorv_epi64
  #define _mm_rorv_epi64(a, b) easysimd_mm_rorv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_rorv_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_rorv_epi64(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_rorv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_rorv_epi64
  #define _mm_mask_rorv_epi64(src, k, a, b) easysimd_mm_mask_rorv_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_rorv_epi64 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_rorv_epi64(k, a, b);
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_rorv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_rorv_epi64
  #define _mm_maskz_rorv_epi64(k, a, b) easysimd_mm_maskz_rorv_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_rorv_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_rorv_epi64(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    r_.m128i[0] = easysimd_mm_rorv_epi64(a_.m128i[0], b_.m128i[0]);
    r_.m128i[1] = easysimd_mm_rorv_epi64(a_.m128i[1], b_.m128i[1]);

    return easysimd__m256i_from_private(r_);
  #else
    easysimd__m256i
      count1 = easysimd_mm256_and_si256(b, easysimd_mm256_set1_epi64x(63)),
      count2 = easysimd_mm256_sub_epi64(easysimd_mm256_set1_epi64x(64), count1);
    return easysimd_mm256_or_si256(easysimd_mm256_srlv_epi64(a, count1), easysimd_mm256_sllv_epi64(a, count2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rorv_epi64
  #define _mm256_rorv_epi64(a, b) easysimd_mm256_rorv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_rorv_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_rorv_epi64(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_rorv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_rorv_epi64
  #define _mm256_mask_rorv_epi64(src, k, a, b) easysimd_mm256_mask_rorv_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_rorv_epi64 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_rorv_epi64(k, a, b);
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_rorv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_rorv_epi64
  #define _mm256_maskz_rorv_epi64(k, a, b) easysimd_mm256_maskz_rorv_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_rorv_epi64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_rorv_epi64(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    r_.m256i[0] = easysimd_mm256_rorv_epi64(a_.m256i[0], b_.m256i[0]);
    r_.m256i[1] = easysimd_mm256_rorv_epi64(a_.m256i[1], b_.m256i[1]);

    return easysimd__m512i_from_private(r_);
  #else
    easysimd__m512i
      count1 = easysimd_mm512_and_si512(b, easysimd_mm512_set1_epi64(63)),
      count2 = easysimd_mm512_sub_epi64(easysimd_mm512_set1_epi64(64), count1);
    return easysimd_mm512_or_si512(easysimd_mm512_srlv_epi64(a, count1), easysimd_mm512_sllv_epi64(a, count2));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rorv_epi64
  #define _mm512_rorv_epi64(a, b) easysimd_mm512_rorv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_rorv_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_rorv_epi64(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_rorv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rorv_epi64
  #define _mm512_mask_rorv_epi64(src, k, a, b) easysimd_mm512_mask_rorv_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_rorv_epi64 (easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_rorv_epi64(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_rorv_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_rorv_epi64
  #define _mm512_maskz_rorv_epi64(k, a, b) easysimd_mm512_maskz_rorv_epi64(k, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_RORV_H) */
