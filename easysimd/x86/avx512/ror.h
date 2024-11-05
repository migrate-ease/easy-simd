#if !defined(EASYSIMD_X86_AVX512_ROR_H)
#define EASYSIMD_X86_AVX512_ROR_H

#include "types.h"
#include "mov.h"
#include "or.h"
#include "srli.h"
#include "slli.h"
#include "../avx2.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_ror_epi32(a, imm8) _mm_ror_epi32(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128i
  easysimd_mm_ror_epi32 (easysimd__m128i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);
    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      switch (imm8 & 31) {
        case 0:
          r_ = a_;
          break;
        default:
          r_.u32 = (a_.u32 >> (imm8 & 31)) | (a_.u32 << (32 - (imm8 & 31)));
          break;
      }
    #else
      switch (imm8 & 31) {
        case 0:
          r_ = a_;
          break;
        default:
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
            r_.u32[i] = (a_.u32[i] >> (imm8 & 31)) | (a_.u32[i] << (32 - (imm8 & 31)));
          }
          break;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_ror_epi32
  #define _mm_ror_epi32(a, imm8) easysimd_mm_ror_epi32(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_ror_epi32(src, k, a, imm8) _mm_mask_ror_epi32(src, k, a, imm8)
#else
  #define easysimd_mm_mask_ror_epi32(src, k, a, imm8) easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_ror_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_ror_epi32
  #define _mm_mask_ror_epi32(src, k, a, imm8) easysimd_mm_mask_ror_epi32(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_ror_epi32(k, a, imm8) _mm_maskz_ror_epi32(k, a, imm8)
#else
  #define easysimd_mm_maskz_ror_epi32(k, a, imm8) easysimd_mm_maskz_mov_epi32(k, easysimd_mm_ror_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_ror_epi32
  #define _mm_maskz_ror_epi32(src, k, a, imm8) easysimd_mm_maskz_ror_epi32(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_ror_epi32(a, imm8) _mm256_ror_epi32(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m256i
  easysimd_mm256_ror_epi32 (easysimd__m256i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      switch (imm8 & 31) {
        case 0:
          r_ = a_;
          break;
        default:
          r_.u32 = (a_.u32 >> (imm8 & 31)) | (a_.u32 << (32 - (imm8 & 31)));
          break;
      }
    #else
      switch (imm8 & 31) {
        case 0:
          r_ = a_;
          break;
        default:
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
            r_.u32[i] = (a_.u32[i] >> (imm8 & 31)) | (a_.u32[i] << (32 - (imm8 & 31)));
          }
          break;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_ror_epi32
  #define _mm256_ror_epi32(a, imm8) easysimd_mm256_ror_epi32(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_ror_epi32(src, k, a, imm8) _mm256_mask_ror_epi32(src, k, a, imm8)
#else
  #define easysimd_mm256_mask_ror_epi32(src, k, a, imm8) easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_ror_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_ror_epi32
  #define _mm256_mask_ror_epi32(src, k, a, imm8) easysimd_mm256_mask_ror_epi32(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_ror_epi32(k, a, imm8) _mm256_maskz_ror_epi32(k, a, imm8)
#else
  #define easysimd_mm256_maskz_ror_epi32(k, a, imm8) easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_ror_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_ror_epi32
  #define _mm256_maskz_ror_epi32(k, a, imm8) easysimd_mm256_maskz_ror_epi32(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_ror_epi32(a, imm8) _mm512_ror_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE_NOTUSE)
  #define easysimd_mm512_ror_epi32(a, imm8)  \
    ({  \
      easysimd__m512i r; \
      svint32_t svzero = svdup_n_s32(0);  \
      r.sve_i32[EASYSIMD_SV_INDEX_0] = svxar_n_s32((a).sve_i32[EASYSIMD_SV_INDEX_0], svzero, ((imm8) > 32 ? 32 : (imm8))); \
      r.sve_i32[EASYSIMD_SV_INDEX_1] = svxar_n_s32((a).sve_i32[EASYSIMD_SV_INDEX_1], svzero, ((imm8) > 32 ? 32 : (imm8))); \
      r.sve_i32[EASYSIMD_SV_INDEX_2] = svxar_n_s32((a).sve_i32[EASYSIMD_SV_INDEX_2], svzero, ((imm8) > 32 ? 32 : (imm8))); \
      r.sve_i32[EASYSIMD_SV_INDEX_3] = svxar_n_s32((a).sve_i32[EASYSIMD_SV_INDEX_3], svzero, ((imm8) > 32 ? 32 : (imm8))); \
      r; \
    })
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512i
  easysimd_mm512_ror_epi32 (easysimd__m512i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      switch (imm8 & 31) {
        case 0:
          r_ = a_;
          break;
        default:
          r_.u32 = (a_.u32 >> (imm8 & 31)) | (a_.u32 << (32 - (imm8 & 31)));
          break;
      }
    #else
      switch (imm8 & 31) {
        case 0:
          r_ = a_;
          break;
        default:
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
            r_.u32[i] = (a_.u32[i] >> (imm8 & 31)) | (a_.u32[i] << (32 - (imm8 & 31)));
          }
          break;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_ror_epi32
  #define _mm512_ror_epi32(a, imm8) easysimd_mm512_ror_epi32(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_ror_epi32(src, k, a, imm8) _mm512_mask_ror_epi32(src, k, a, imm8)
#else
  #define easysimd_mm512_mask_ror_epi32(src, k, a, imm8) easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_ror_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_ror_epi32
  #define _mm512_mask_ror_epi32(src, k, a, imm8) easysimd_mm512_mask_ror_epi32(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_ror_epi32(k, a, imm8) _mm512_maskz_ror_epi32(k, a, imm8)
#else
  #define easysimd_mm512_maskz_ror_epi32(k, a, imm8) easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_ror_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_ror_epi32
  #define _mm512_maskz_ror_epi32(k, a, imm8) easysimd_mm512_maskz_ror_epi32(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_ror_epi64(a, imm8) _mm_ror_epi64(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128i
  easysimd_mm_ror_epi64 (easysimd__m128i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      switch (imm8 & 63) {
        case 0:
          r_ = a_;
          break;
        default:
          r_.u64 = (a_.u64 >> (imm8 & 63)) | (a_.u64 << (64 - (imm8 & 63)));
          break;
      }
    #else
      switch (imm8 & 63) {
        case 0:
          r_ = a_;
          break;
        default:
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
            r_.u64[i] = (a_.u64[i] >> (imm8 & 63)) | (a_.u64[i] << (64 - (imm8 & 63)));
          }
          break;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_ror_epi64
  #define _mm_ror_epi64(a, imm8) easysimd_mm_ror_epi64(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_ror_epi64(src, k, a, imm8) _mm_mask_ror_epi64(src, k, a, imm8)
#else
  #define easysimd_mm_mask_ror_epi64(src, k, a, imm8) easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_ror_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_ror_epi64
  #define _mm_mask_ror_epi64(src, k, a, imm8) easysimd_mm_mask_ror_epi64(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_ror_epi64(k, a, imm8) _mm_maskz_ror_epi64(k, a, imm8)
#else
  #define easysimd_mm_maskz_ror_epi64(k, a, imm8) easysimd_mm_maskz_mov_epi64(k, easysimd_mm_ror_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_ror_epi64
  #define _mm_maskz_ror_epi64(k, a, imm8) easysimd_mm_maskz_ror_epi64(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_ror_epi64(a, imm8) _mm256_ror_epi64(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m256i
  easysimd_mm256_ror_epi64 (easysimd__m256i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      switch (imm8 & 63) {
        case 0:
          r_ = a_;
          break;
        default:
          r_.u64 = (a_.u64 >> (imm8 & 63)) | (a_.u64 << (64 - (imm8 & 63)));
          break;
      }
    #else
      switch (imm8 & 63) {
        case 0:
          r_ = a_;
          break;
        default:
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
            r_.u64[i] = (a_.u64[i] >> (imm8 & 63)) | (a_.u64[i] << (64 - (imm8 & 63)));
          }
          break;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_ror_epi64
  #define _mm256_ror_epi64(a, imm8) easysimd_mm256_ror_epi64(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_ror_epi64(src, k, a, imm8) _mm256_mask_ror_epi64(src, k, a, imm8)
#else
  #define easysimd_mm256_mask_ror_epi64(src, k, a, imm8) easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_ror_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_ror_epi64
  #define _mm256_mask_ror_epi64(src, k, a, imm8) easysimd_mm256_mask_ror_epi64(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_ror_epi64(k, a, imm8) _mm256_maskz_ror_epi64(k, a, imm8)
#else
  #define easysimd_mm256_maskz_ror_epi64(k, a, imm8) easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_ror_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_ror_epi64
  #define _mm256_maskz_ror_epi64(k, a, imm8) easysimd_mm256_maskz_ror_epi64(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_ror_epi64(a, imm8) _mm512_ror_epi64(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512i
  easysimd_mm512_ror_epi64 (easysimd__m512i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      switch (imm8 & 63) {
        case 0:
          r_ = a_;
          break;
        default:
          r_.u64 = (a_.u64 >> (imm8 & 63)) | (a_.u64 << (64 - (imm8 & 63)));
          break;
      }
    #else
      switch (imm8 & 63) {
        case 0:
          r_ = a_;
          break;
        default:
          EASYSIMD_VECTORIZE
          for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
            r_.u64[i] = (a_.u64[i] >> (imm8 & 63)) | (a_.u64[i] << (64 - (imm8 & 63)));
          }
          break;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_ror_epi64
  #define _mm512_ror_epi64(a, imm8) easysimd_mm512_ror_epi64(a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_ror_epi64(src, k, a, imm8) _mm512_mask_ror_epi64(src, k, a, imm8)
#else
  #define easysimd_mm512_mask_ror_epi64(src, k, a, imm8) easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_ror_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_ror_epi64
  #define _mm512_mask_ror_epi64(src, k, a, imm8) easysimd_mm512_mask_ror_epi64(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_ror_epi64(k, a, imm8) _mm512_maskz_ror_epi64(k, a, imm8)
#else
  #define easysimd_mm512_maskz_ror_epi64(k, a, imm8) easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_ror_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_ror_epi64
  #define _mm512_maskz_ror_epi64(k, a, imm8) easysimd_mm512_maskz_ror_epi64(k, a, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_ROR_H) */
