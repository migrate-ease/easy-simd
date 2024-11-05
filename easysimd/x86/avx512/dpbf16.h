#if !defined(EASYSIMD_X86_AVX512_DPBF16_H)
#define EASYSIMD_X86_AVX512_DPBF16_H

#include "types.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_dpbf16_ps (easysimd__m128 src, easysimd__m128bh a, easysimd__m128bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_dpbf16_ps(src, a, b);
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src);
    easysimd__m128bh_private
      a_ = easysimd__m128bh_to_private(a),
      b_ = easysimd__m128bh_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      uint32_t x1 EASYSIMD_VECTOR(32);
      uint32_t x2 EASYSIMD_VECTOR(32);
      easysimd__m128_private
        r1_[2],
        r2_[2];

      a_.u16 =
        EASYSIMD_SHUFFLE_VECTOR_(
          16, 16,
          a_.u16, a_.u16,
          0, 2, 4, 6,
          1, 3, 5, 7
        );
      b_.u16 =
        EASYSIMD_SHUFFLE_VECTOR_(
          16, 16,
          b_.u16, b_.u16,
          0, 2, 4, 6,
          1, 3, 5, 7
        );

      EASYSIMD_CONVERT_VECTOR_(x1, a_.u16);
      EASYSIMD_CONVERT_VECTOR_(x2, b_.u16);

      x1 <<= 16;
      x2 <<= 16;

      easysimd_memcpy(&r1_, &x1, sizeof(x1));
      easysimd_memcpy(&r2_, &x2, sizeof(x2));

      src_.f32 +=
        HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r1_[0].u32) * HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r2_[0].u32) +
        HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r1_[1].u32) * HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r2_[1].u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
        src_.f32[i / 2] += (easysimd_uint32_as_float32(HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) * easysimd_uint32_as_float32(HEDLEY_STATIC_CAST(uint32_t, b_.u16[i]) << 16));
      }
    #endif

    return easysimd__m128_from_private(src_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_dpbf16_ps
  #define _mm_dpbf16_ps(src, a, b) easysimd_mm_dpbf16_ps(src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_dpbf16_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128bh a, easysimd__m128bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_dpbf16_ps(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_ps(src, k, easysimd_mm_dpbf16_ps(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_dpbf16_ps
  #define _mm_mask_dpbf16_ps(src, k, a, b) easysimd_mm_mask_dpbf16_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_dpbf16_ps (easysimd__mmask8 k, easysimd__m128 src, easysimd__m128bh a, easysimd__m128bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_dpbf16_ps(k, src, a, b);
  #else
    return easysimd_mm_maskz_mov_ps(k, easysimd_mm_dpbf16_ps(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_dpbf16_ps
  #define _mm_maskz_dpbf16_ps(k, src, a, b) easysimd_mm_maskz_dpbf16_ps(k, src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_dpbf16_ps (easysimd__m256 src, easysimd__m256bh a, easysimd__m256bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_dpbf16_ps(src, a, b);
  #else
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src);
    easysimd__m256bh_private
      a_ = easysimd__m256bh_to_private(a),
      b_ = easysimd__m256bh_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      uint32_t x1 EASYSIMD_VECTOR(64);
      uint32_t x2 EASYSIMD_VECTOR(64);
      easysimd__m256_private
        r1_[2],
        r2_[2];

      a_.u16 =
        EASYSIMD_SHUFFLE_VECTOR_(
          16, 32,
          a_.u16, a_.u16,
           0,  2,  4,  6,  8, 10, 12, 14,
           1,  3,  5,  7,  9, 11, 13, 15
        );
      b_.u16 =
        EASYSIMD_SHUFFLE_VECTOR_(
          16, 32,
          b_.u16, b_.u16,
           0,  2,  4,  6,  8, 10, 12, 14,
           1,  3,  5,  7,  9, 11, 13, 15
        );

      EASYSIMD_CONVERT_VECTOR_(x1, a_.u16);
      EASYSIMD_CONVERT_VECTOR_(x2, b_.u16);

      x1 <<= 16;
      x2 <<= 16;

      easysimd_memcpy(&r1_, &x1, sizeof(x1));
      easysimd_memcpy(&r2_, &x2, sizeof(x2));

      src_.f32 +=
        HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r1_[0].u32) * HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r2_[0].u32) +
        HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r1_[1].u32) * HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r2_[1].u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
        src_.f32[i / 2] += (easysimd_uint32_as_float32(HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) * easysimd_uint32_as_float32(HEDLEY_STATIC_CAST(uint32_t, b_.u16[i]) << 16));
      }
    #endif

    return easysimd__m256_from_private(src_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_dpbf16_ps
  #define _mm256_dpbf16_ps(src, a, b) easysimd_mm256_dpbf16_ps(src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_dpbf16_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256bh a, easysimd__m256bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_dpbf16_ps(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_dpbf16_ps(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_dpbf16_ps
  #define _mm256_mask_dpbf16_ps(src, k, a, b) easysimd_mm256_mask_dpbf16_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_dpbf16_ps (easysimd__mmask8 k, easysimd__m256 src, easysimd__m256bh a, easysimd__m256bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_dpbf16_ps(k, src, a, b);
  #else
    return easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_dpbf16_ps(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_dpbf16_ps
  #define _mm256_maskz_dpbf16_ps(k, src, a, b) easysimd_mm256_maskz_dpbf16_ps(k, src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_dpbf16_ps (easysimd__m512 src, easysimd__m512bh a, easysimd__m512bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE)
    return _mm512_dpbf16_ps(src, a, b);
  #else
    easysimd__m512_private
      src_ = easysimd__m512_to_private(src);
    easysimd__m512bh_private
      a_ = easysimd__m512bh_to_private(a),
      b_ = easysimd__m512bh_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      uint32_t x1 EASYSIMD_VECTOR(128);
      uint32_t x2 EASYSIMD_VECTOR(128);
      easysimd__m512_private
        r1_[2],
        r2_[2];

      a_.u16 =
        EASYSIMD_SHUFFLE_VECTOR_(
          16, 64,
          a_.u16, a_.u16,
           0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
           1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        );
      b_.u16 =
        EASYSIMD_SHUFFLE_VECTOR_(
          16, 64,
          b_.u16, b_.u16,
           0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
           1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        );

      EASYSIMD_CONVERT_VECTOR_(x1, a_.u16);
      EASYSIMD_CONVERT_VECTOR_(x2, b_.u16);

      x1 <<= 16;
      x2 <<= 16;

      easysimd_memcpy(&r1_, &x1, sizeof(x1));
      easysimd_memcpy(&r2_, &x2, sizeof(x2));

      src_.f32 +=
        HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r1_[0].u32) * HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r2_[0].u32) +
        HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r1_[1].u32) * HEDLEY_REINTERPRET_CAST(__typeof__(a_.f32), r2_[1].u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
        src_.f32[i / 2] += (easysimd_uint32_as_float32(HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) * easysimd_uint32_as_float32(HEDLEY_STATIC_CAST(uint32_t, b_.u16[i]) << 16));
      }
    #endif

    return easysimd__m512_from_private(src_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES)
  #undef _mm512_dpbf16_ps
  #define _mm512_dpbf16_ps(src, a, b) easysimd_mm512_dpbf16_ps(src, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_dpbf16_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512bh a, easysimd__m512bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE)
    return _mm512_mask_dpbf16_ps(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_dpbf16_ps(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_dpbf16_ps
  #define _mm512_mask_dpbf16_ps(src, k, a, b) easysimd_mm512_mask_dpbf16_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_dpbf16_ps (easysimd__mmask16 k, easysimd__m512 src, easysimd__m512bh a, easysimd__m512bh b) {
  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE)
    return _mm512_maskz_dpbf16_ps(k, src, a, b);
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_dpbf16_ps(src, a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_dpbf16_ps
  #define _mm512_maskz_dpbf16_ps(k, src, a, b) easysimd_mm512_maskz_dpbf16_ps(k, src, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_DPBF16_H) */
