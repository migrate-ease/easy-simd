#if !defined(EASYSIMD_X86_AVX512_DPBUSD_H)
#define EASYSIMD_X86_AVX512_DPBUSD_H

#include "types.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_dpbusd_epi32(easysimd__m128i src, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm_dpbusd_epi32(src, a, b);
  #else
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    #if defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_CONVERT_VECTOR_)
      uint32_t x1_ EASYSIMD_VECTOR(64);
      int32_t  x2_ EASYSIMD_VECTOR(64);
      easysimd__m128i_private
        r1_[4],
        r2_[4];

      a_.u8 =
        EASYSIMD_SHUFFLE_VECTOR_(
          8, 16,
          a_.u8, a_.u8,
           0,  4,  8, 12,
           1,  5,  9, 13,
           2,  6, 10, 14,
           3,  7, 11, 15
        );
      b_.i8 =
        EASYSIMD_SHUFFLE_VECTOR_(
          8, 16,
          b_.i8, b_.i8,
           0,  4,  8, 12,
           1,  5,  9, 13,
           2,  6, 10, 14,
           3,  7, 11, 15
        );

      EASYSIMD_CONVERT_VECTOR_(x1_, a_.u8);
      EASYSIMD_CONVERT_VECTOR_(x2_, b_.i8);

      easysimd_memcpy(&r1_, &x1_, sizeof(x1_));
      easysimd_memcpy(&r2_, &x2_, sizeof(x2_));

      src_.i32 +=
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[0].u32) * r2_[0].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[1].u32) * r2_[1].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[2].u32) * r2_[2].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[3].u32) * r2_[3].i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        src_.i32[i / 4] += HEDLEY_STATIC_CAST(uint16_t, a_.u8[i]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[i]);
      }
    #endif

    return easysimd__m128i_from_private(src_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_dpbusd_epi32
  #define _mm_dpbusd_epi32(src, a, b) easysimd_mm_dpbusd_epi32(src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_dpbusd_epi32(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm_mask_dpbusd_epi32(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_dpbusd_epi32(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_dpbusd_epi32
  #define _mm_mask_dpbusd_epi32(src, k, a, b) easysimd_mm_mask_dpbusd_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_dpbusd_epi32(easysimd__mmask8 k, easysimd__m128i src, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm_maskz_dpbusd_epi32(k, src, a, b);
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_dpbusd_epi32(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_dpbusd_epi32
  #define _mm_maskz_dpbusd_epi32(k, src, a, b) easysimd_mm_maskz_dpbusd_epi32(k, src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_dpbusd_epi32(easysimd__m256i src, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm256_dpbusd_epi32(src, a, b);
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      src_.m128i[0] = easysimd_mm_dpbusd_epi32(src_.m128i[0], a_.m128i[0], b_.m128i[0]);
      src_.m128i[1] = easysimd_mm_dpbusd_epi32(src_.m128i[1], a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_CONVERT_VECTOR_)
      uint32_t x1_ EASYSIMD_VECTOR(128);
      int32_t  x2_ EASYSIMD_VECTOR(128);
      easysimd__m256i_private
        r1_[4],
        r2_[4];

      a_.u8 =
        EASYSIMD_SHUFFLE_VECTOR_(
          8, 32,
          a_.u8, a_.u8,
           0,  4,  8, 12, 16, 20, 24, 28,
           1,  5,  9, 13, 17, 21, 25, 29,
           2,  6, 10, 14, 18, 22, 26, 30,
           3,  7, 11, 15, 19, 23, 27, 31
        );
      b_.i8 =
        EASYSIMD_SHUFFLE_VECTOR_(
          8, 32,
          b_.i8, b_.i8,
           0,  4,  8, 12, 16, 20, 24, 28,
           1,  5,  9, 13, 17, 21, 25, 29,
           2,  6, 10, 14, 18, 22, 26, 30,
           3,  7, 11, 15, 19, 23, 27, 31
        );

      EASYSIMD_CONVERT_VECTOR_(x1_, a_.u8);
      EASYSIMD_CONVERT_VECTOR_(x2_, b_.i8);

      easysimd_memcpy(&r1_, &x1_, sizeof(x1_));
      easysimd_memcpy(&r2_, &x2_, sizeof(x2_));

      src_.i32 +=
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[0].u32) * r2_[0].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[1].u32) * r2_[1].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[2].u32) * r2_[2].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[3].u32) * r2_[3].i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        src_.i32[i / 4] += HEDLEY_STATIC_CAST(uint16_t, a_.u8[i]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[i]);
      }
    #endif

    return easysimd__m256i_from_private(src_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_dpbusd_epi32
  #define _mm256_dpbusd_epi32(src, a, b) easysimd_mm256_dpbusd_epi32(src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_dpbusd_epi32(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm256_mask_dpbusd_epi32(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_dpbusd_epi32(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_dpbusd_epi32
  #define _mm256_mask_dpbusd_epi32(src, k, a, b) easysimd_mm256_mask_dpbusd_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_dpbusd_epi32(easysimd__mmask8 k, easysimd__m256i src, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm256_maskz_dpbusd_epi32(k, src, a, b);
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_dpbusd_epi32(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_dpbusd_epi32
  #define _mm256_maskz_dpbusd_epi32(k, src, a, b) easysimd_mm256_maskz_dpbusd_epi32(k, src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_dpbusd_epi32(easysimd__m512i src, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm512_dpbusd_epi32(src, a, b);
  #else
    easysimd__m512i_private
      src_ = easysimd__m512i_to_private(src),
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      src_.m256i[0] = easysimd_mm256_dpbusd_epi32(src_.m256i[0], a_.m256i[0], b_.m256i[0]);
      src_.m256i[1] = easysimd_mm256_dpbusd_epi32(src_.m256i[1], a_.m256i[1], b_.m256i[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_CONVERT_VECTOR_)
      uint32_t x1_ EASYSIMD_VECTOR(256);
      int32_t  x2_ EASYSIMD_VECTOR(256);
      easysimd__m512i_private
        r1_[4],
        r2_[4];

      a_.u8 =
        EASYSIMD_SHUFFLE_VECTOR_(
          8, 64,
          a_.u8, a_.u8,
            0,   4,   8,  12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,  56,  60,
            1,   5,   9,  13,  17,  21,  25,  29,  33,  37,  41,  45,  49,  53,  57,  61,
            2,   6,  10,  14,  18,  22,  26,  30,  34,  38,  42,  46,  50,  54,  58,  62,
            3,   7,  11,  15,  19,  23,  27,  31,  35,  39,  43,  47,  51,  55,  59,  63
        );
      b_.i8 =
        EASYSIMD_SHUFFLE_VECTOR_(
          8, 64,
          b_.i8, b_.i8,
            0,   4,   8,  12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,  56,  60,
            1,   5,   9,  13,  17,  21,  25,  29,  33,  37,  41,  45,  49,  53,  57,  61,
            2,   6,  10,  14,  18,  22,  26,  30,  34,  38,  42,  46,  50,  54,  58,  62,
            3,   7,  11,  15,  19,  23,  27,  31,  35,  39,  43,  47,  51,  55,  59,  63
        );

      EASYSIMD_CONVERT_VECTOR_(x1_, a_.u8);
      EASYSIMD_CONVERT_VECTOR_(x2_, b_.i8);

      easysimd_memcpy(&r1_, &x1_, sizeof(x1_));
      easysimd_memcpy(&r2_, &x2_, sizeof(x2_));

      src_.i32 +=
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[0].u32) * r2_[0].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[1].u32) * r2_[1].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[2].u32) * r2_[2].i32) +
        (HEDLEY_REINTERPRET_CAST(__typeof__(a_.i32), r1_[3].u32) * r2_[3].i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        src_.i32[i / 4] += HEDLEY_STATIC_CAST(uint16_t, a_.u8[i]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[i]);
      }
    #endif

    return easysimd__m512i_from_private(src_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_dpbusd_epi32
  #define _mm512_dpbusd_epi32(src, a, b) easysimd_mm512_dpbusd_epi32(src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_dpbusd_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm512_mask_dpbusd_epi32(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_dpbusd_epi32(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_dpbusd_epi32
  #define _mm512_mask_dpbusd_epi32(src, k, a, b) easysimd_mm512_mask_dpbusd_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_dpbusd_epi32(easysimd__mmask16 k, easysimd__m512i src, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    return _mm512_maskz_dpbusd_epi32(k, src, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_dpbusd_epi32(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_dpbusd_epi32
  #define _mm512_maskz_dpbusd_epi32(k, src, a, b) easysimd_mm512_maskz_dpbusd_epi32(k, src, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_DPBUSD_H) */
