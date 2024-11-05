#if !defined(EASYSIMD_X86_AVX512_COMPRESS_H)
#define EASYSIMD_X86_AVX512_COMPRESS_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_compress_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_compress_pd(src, k, a);
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      src_ = easysimd__m256d_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f64[ri++] = a_.f64[i];
      }
    }

    for ( ; ri < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; ri++) {
      a_.f64[ri] = src_.f64[ri];
    }

    return easysimd__m256d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compress_pd
  #define _mm256_mask_compress_pd(src, k, a) _mm256_mask_compress_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_compressstoreu_pd (void* base_addr, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm256_mask_compressstoreu_pd(base_addr, k, a);
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f64[ri++] = a_.f64[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.f64[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compressstoreu_pd
  #define _mm256_mask_compressstoreu_pd(base_addr, k, a) _mm256_mask_compressstoreu_pd(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_compress_pd (easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_compress_pd(k, a);
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f64[ri++] = a_.f64[i];
      }
    }

    for ( ; ri < (sizeof(a_.f64) / sizeof(a_.f64[0])); ri++) {
      a_.f64[ri] = EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_compress_pd
  #define _mm256_maskz_compress_pd(k, a) _mm256_maskz_compress_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_compress_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_compress_ps(src, k, a);
  #else
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      src_ = easysimd__m256_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f32[ri++] = a_.f32[i];
      }
    }

    for ( ; ri < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; ri++) {
      a_.f32[ri] = src_.f32[ri];
    }

    return easysimd__m256_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compress_ps
  #define _mm256_mask_compress_ps(src, k, a) _mm256_mask_compress_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_compressstoreu_ps (void* base_addr, easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm256_mask_compressstoreu_ps(base_addr, k, a);
  #else
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f32[ri++] = a_.f32[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.f32[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compressstoreu_pd
  #define _mm256_mask_compressstoreu_ps(base_addr, k, a) _mm256_mask_compressstoreu_ps(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_compress_ps (easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_compress_ps(k, a);
  #else
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f32[ri++] = a_.f32[i];
      }
    }

    for ( ; ri < (sizeof(a_.f32) / sizeof(a_.f32[0])); ri++) {
      a_.f32[ri] = EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_compress_ps
  #define _mm256_maskz_compress_ps(k, a) _mm256_maskz_compress_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_compress_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_compress_epi32(src, k, a);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      src_ = easysimd__m256i_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i32[ri++] = a_.i32[i];
      }
    }

    for ( ; ri < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; ri++) {
      a_.i32[ri] = src_.i32[ri];
    }

    return easysimd__m256i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compress_epi32
  #define _mm256_mask_compress_epi32(src, k, a) _mm256_mask_compress_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_compressstoreu_epi32 (void* base_addr, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm256_mask_compressstoreu_epi32(base_addr, k, a);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i32[ri++] = a_.i32[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.i32[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compressstoreu_epi32
  #define _mm256_mask_compressstoreu_epi32(base_addr, k, a) _mm256_mask_compressstoreu_epi32(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_compress_epi32 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_compress_epi32(k, a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    size_t idx = 0;
    for(size_t i = 0; i < (sizeof(a) / sizeof(a.i32[0])); i++) {
      if((k >> i) & 0x01) {
        a.i32[idx++] = a.i32[i];
      }
    }
    for(; idx < (sizeof(a) / sizeof(a.i32[0]));) {
      a.i32[idx++] = 0;
    }

    return a;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i32[ri++] = a_.i32[i];
      }
    }

    for ( ; ri < (sizeof(a_.i32) / sizeof(a_.i32[0])); ri++) {
      a_.f32[ri] = INT32_C(0);
    }

    return easysimd__m256i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_compress_epi32
  #define _mm256_maskz_compress_epi32(k, a) _mm256_maskz_compress_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_compress_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_compress_epi64(src, k, a);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      src_ = easysimd__m256i_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i64[ri++] = a_.i64[i];
      }
    }

    for ( ; ri < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; ri++) {
      a_.i64[ri] = src_.i64[ri];
    }

    return easysimd__m256i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compress_epi64
  #define _mm256_mask_compress_epi64(src, k, a) _mm256_mask_compress_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_compressstoreu_epi64 (void* base_addr, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm256_mask_compressstoreu_epi64(base_addr, k, a);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i64[ri++] = a_.i64[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.i64[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_compressstoreu_epi64
  #define _mm256_mask_compressstoreu_epi64(base_addr, k, a) _mm256_mask_compressstoreu_epi64(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_compress_epi64 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_compress_epi64(k, a);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i64[ri++] = a_.i64[i];
      }
    }

    for ( ; ri < (sizeof(a_.i64) / sizeof(a_.i64[0])); ri++) {
      a_.i64[ri] = INT64_C(0);
    }

    return easysimd__m256i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_compress_epi64
  #define _mm256_maskz_compress_epi64(k, a) _mm256_maskz_compress_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_compress_pd (easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_compress_pd(src, k, a);
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a),
      src_ = easysimd__m512d_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f64[ri++] = a_.f64[i];
      }
    }

    for ( ; ri < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; ri++) {
      a_.f64[ri] = src_.f64[ri];
    }

    return easysimd__m512d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compress_pd
  #define _mm512_mask_compress_pd(src, k, a) _mm512_mask_compress_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_compressstoreu_pd (void* base_addr, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_mask_compressstoreu_pd(base_addr, k, a);
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f64[ri++] = a_.f64[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.f64[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compressstoreu_pd
  #define _mm512_mask_compressstoreu_pd(base_addr, k, a) _mm512_mask_compressstoreu_pd(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_compress_pd (easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_compress_pd(k, a);
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f64[ri++] = a_.f64[i];
      }
    }

    for ( ; ri < (sizeof(a_.f64) / sizeof(a_.f64[0])); ri++) {
      a_.f64[ri] = EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m512d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_compress_pd
  #define _mm512_maskz_compress_pd(k, a) _mm512_maskz_compress_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_compress_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_compress_ps(src, k, a);
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a),
      src_ = easysimd__m512_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f32[ri++] = a_.f32[i];
      }
    }

    for ( ; ri < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; ri++) {
      a_.f32[ri] = src_.f32[ri];
    }

    return easysimd__m512_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compress_ps
  #define _mm512_mask_compress_ps(src, k, a) _mm512_mask_compress_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_compressstoreu_ps (void* base_addr, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_mask_compressstoreu_ps(base_addr, k, a);
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f32[ri++] = a_.f32[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.f32[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compressstoreu_pd
  #define _mm512_mask_compressstoreu_ps(base_addr, k, a) _mm512_mask_compressstoreu_ps(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_compress_ps (easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_compress_ps(k, a);
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.f32[ri++] = a_.f32[i];
      }
    }

    for ( ; ri < (sizeof(a_.f32) / sizeof(a_.f32[0])); ri++) {
      a_.f32[ri] = EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m512_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_compress_ps
  #define _mm512_maskz_compress_ps(k, a) _mm512_maskz_compress_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_compress_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_compress_epi32(src, k, a);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      src_ = easysimd__m512i_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i32[ri++] = a_.i32[i];
      }
    }

    for ( ; ri < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; ri++) {
      a_.i32[ri] = src_.i32[ri];
    }

    return easysimd__m512i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compress_epi32
  #define _mm512_mask_compress_epi32(src, k, a) _mm512_mask_compress_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_compressstoreu_epi32 (void* base_addr, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_mask_compressstoreu_epi32(base_addr, k, a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    size_t idx = 0;
    for(size_t i = 0; i < (sizeof(a) / sizeof(a.i32[0])); i++) {
      if((k >> i) & 0x01) {
        a.i32[idx++] = a.i32[i];
      }
    }

    for(; idx < (sizeof(a) / sizeof(a.i32[0]));) {
      a.i32[idx++] = 0;
    }

    vst1q_s32_x4((int32_t *)base_addr, a.neon_i32x4);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i32[ri++] = a_.i32[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.i32[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compressstoreu_epi32
  #define _mm512_mask_compressstoreu_epi32(base_addr, k, a) _mm512_mask_compressstoreu_epi32(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_compress_epi32 (easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_compress_epi32(k, a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    size_t idx = 0;
    for(size_t i = 0; i < (sizeof(a) / sizeof(a.i32[0])); i++) {
      if((k >> i) & 0x01) {
        a.i32[idx++] = a.i32[i];
      }
    }
    for(; idx < (sizeof(a) / sizeof(a.i32[0]));) {
      a.i32[idx++] = 0;
    }

    return a;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i32[ri++] = a_.i32[i];
      }
    }

    for ( ; ri < (sizeof(a_.i32) / sizeof(a_.i32[0])); ri++) {
      a_.f32[ri] = INT32_C(0);
    }

    return easysimd__m512i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_compress_epi32
  #define _mm512_maskz_compress_epi32(k, a) _mm512_maskz_compress_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_compress_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_compress_epi64(src, k, a);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      src_ = easysimd__m512i_to_private(src);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i64[ri++] = a_.i64[i];
      }
    }

    for ( ; ri < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; ri++) {
      a_.i64[ri] = src_.i64[ri];
    }

    return easysimd__m512i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compress_epi64
  #define _mm512_mask_compress_epi64(src, k, a) _mm512_mask_compress_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_compressstoreu_epi64 (void* base_addr, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    _mm512_mask_compressstoreu_epi64(base_addr, k, a);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i64[ri++] = a_.i64[i];
      }
    }

    easysimd_memcpy(base_addr, &a_, ri * sizeof(a_.i64[0]));

    return;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_compressstoreu_epi64
  #define _mm512_mask_compressstoreu_epi64(base_addr, k, a) _mm512_mask_compressstoreu_epi64(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_compress_epi64 (easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_compress_epi64(k, a);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a);
    size_t ri = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        a_.i64[ri++] = a_.i64[i];
      }
    }

    for ( ; ri < (sizeof(a_.i64) / sizeof(a_.i64[0])); ri++) {
      a_.i64[ri] = INT64_C(0);
    }

    return easysimd__m512i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_compress_epi64
  #define _mm512_maskz_compress_epi64(k, a) _mm512_maskz_compress_epi64(k, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_COMPRESS_H) */
