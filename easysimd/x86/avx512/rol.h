#if !defined(EASYSIMD_X86_AVX512_ROL_H)
#define EASYSIMD_X86_AVX512_ROL_H

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
  #define easysimd_mm_rol_epi32(a, imm8) _mm_rol_epi32(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128i
  easysimd_mm_rol_epi32 (easysimd__m128i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m128i r;
      if ((imm8 & 31) == 0) {
        r = a;
      } else {
        easysimd_svbool_t pg = svptrue_b32();
        r.sve_u32 = svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32, (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32, (32 - (imm8 & 31))));
      }
      return r;
    #else
      easysimd__m128i_private
        r_,
        a_ = easysimd__m128i_to_private(a);

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        switch (imm8 & 31) {
          case 0:
            r_ = a_;
            break;
          default:
            r_.u32 = (a_.u32 << (imm8 & 31)) | (a_.u32 >> (32 - (imm8 & 31)));
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
              r_.u32[i] = (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31)));
            }
            break;
        }
      #endif

      return easysimd__m128i_from_private(r_);
    #endif
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_rol_epi32
  #define _mm_rol_epi32(a, imm8) easysimd_mm_rol_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_rol_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    if ((imm8 & 31) == 0) {
      r.sve_u32 = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32, src.sve_u32);
    } else {
      easysimd_svbool_t pg = svptrue_b32();
      r.sve_u32 = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32, (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32, (32 - (imm8 & 31)))),
                            src.sve_u32);
    }
    return r;
  #else
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? a_.u32[i] : src_.u32[i];
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31))) : src_.u32[i];
        }
        break;
      }
    return easysimd__m128i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_rol_epi32(src, k, a, imm8) _mm_mask_rol_epi32(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm_mask_rol_epi32(src, k, a, imm8) easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_rol_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_rol_epi32
  #define _mm_mask_rol_epi32(src, k, a, imm8) easysimd_mm_mask_rol_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_rol_epi32 (easysimd__mmask8 k, easysimd__m128i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    if ((imm8 & 31) == 0) {
      r.sve_u32 = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32, svdup_n_u32(0));
    } else {
      easysimd_svbool_t pg = svptrue_b32();
      r.sve_u32 = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32, (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32, (32 - (imm8 & 31)))),
                            svdup_n_u32(0));
    }
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? a_.u32[i] : UINT32_C(0);
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31))) : UINT32_C(0);
        }
        break;
      }
    return easysimd__m128i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_rol_epi32(k, a, imm8) _mm_maskz_rol_epi32(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm_maskz_rol_epi32(k, a, imm8) easysimd_mm_maskz_mov_epi32(k, easysimd_mm_rol_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_rol_epi32
  #define _mm_maskz_rol_epi32(src, k, a, imm8) easysimd_mm_maskz_rol_epi32(src, k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_rol_epi32(a, imm8) _mm256_rol_epi32(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m256i
  easysimd_mm256_rol_epi32 (easysimd__m256i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m256i r;
      if ((imm8 & 31) == 0) {
        r = a;
      } else {
        easysimd_svbool_t pg = svptrue_b32();
        r.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (32 - (imm8 & 31))));
        r.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (32 - (imm8 & 31))));
      }
      return r;
    #else
      easysimd__m256i_private
        r_,
        a_ = easysimd__m256i_to_private(a);

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        switch (imm8 & 31) {
          case 0:
            r_ = a_;
            break;
          default:
            r_.u32 = (a_.u32 << (imm8 & 31)) | (a_.u32 >> (32 - (imm8 & 31)));
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
              r_.u32[i] = (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31)));
            }
            break;
        }
      #endif

      return easysimd__m256i_from_private(r_);
    #endif
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rol_epi32
  #define _mm256_rol_epi32(a, imm8) easysimd_mm256_rol_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_rol_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    if ((imm8 & 31) == 0) {
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], src.sve_u32[EASYSIMD_SV_INDEX_0]);
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], src.sve_u32[EASYSIMD_SV_INDEX_1]);
    } else {
      easysimd_svbool_t pg = svptrue_b32();
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (32 - (imm8 & 31)))),
                            src.sve_u32[EASYSIMD_SV_INDEX_0]);
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (32 - (imm8 & 31)))),
                            src.sve_u32[EASYSIMD_SV_INDEX_1]);
    }
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? a_.u32[i] : src_.u32[i];
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31))) : src_.u32[i];
        }
        break;
      }
    return easysimd__m256i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_rol_epi32(src, k, a, imm8) _mm256_mask_rol_epi32(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm256_mask_rol_epi32(src, k, a, imm8) easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_rol_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_rol_epi32
  #define _mm256_mask_rol_epi32(src, k, a, imm8) easysimd_mm256_mask_rol_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_rol_epi32 (easysimd__mmask8 k, easysimd__m256i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    if ((imm8 & 31) == 0) {
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], svdup_n_u32(0));
    } else {
      easysimd_svbool_t pg = svptrue_b32();
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (32 - (imm8 & 31)))),
                            svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (32 - (imm8 & 31)))),
                            svdup_n_u32(0));
    }
    return r;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? a_.u32[i] : UINT32_C(0);
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31))) : UINT32_C(0);
        }
        break;
      }
    return easysimd__m256i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_rol_epi32(k, a, imm8) _mm256_maskz_rol_epi32(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm256_maskz_rol_epi32(k, a, imm8) easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_rol_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_rol_epi32
  #define _mm256_maskz_rol_epi32(k, a, imm8) easysimd_mm256_maskz_rol_epi32(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_rol_epi32(a, imm8) _mm512_rol_epi32(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512i
  easysimd_mm512_rol_epi32 (easysimd__m512i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m512i r;
      if ((imm8 & 31) == 0) {
        r = a;
      } else {
        easysimd_svbool_t pg = svptrue_b32();
        r.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (32 - (imm8 & 31))));
        r.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (32 - (imm8 & 31))));
        r.sve_u32[EASYSIMD_SV_INDEX_2] = svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], (32 - (imm8 & 31))));
        r.sve_u32[EASYSIMD_SV_INDEX_3] = svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], (32 - (imm8 & 31))));
      }
      return r;
    #else
      easysimd__m512i_private
        r_,
        a_ = easysimd__m512i_to_private(a);

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        switch (imm8 & 31) {
          case 0:
            r_ = a_;
            break;
          default:
            r_.u32 = (a_.u32 << (imm8 & 31)) | (a_.u32 >> (32 - (imm8 & 31)));
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
              r_.u32[i] = (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31)));
            }
            break;
        }
      #endif

      return easysimd__m512i_from_private(r_);
    #endif
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rol_epi32
  #define _mm512_rol_epi32(a, imm8) easysimd_mm512_rol_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_rol_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    if ((imm8 & 31) == 0) {
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], src.sve_u32[EASYSIMD_SV_INDEX_0]);
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], src.sve_u32[EASYSIMD_SV_INDEX_1]);
      r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_u32[EASYSIMD_SV_INDEX_2], src.sve_u32[EASYSIMD_SV_INDEX_2]);
      r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_u32[EASYSIMD_SV_INDEX_3], src.sve_u32[EASYSIMD_SV_INDEX_3]);
    } else {
      easysimd_svbool_t pg = svptrue_b32();
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (32 - (imm8 & 31)))),
                            src.sve_u32[EASYSIMD_SV_INDEX_0]);
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (32 - (imm8 & 31)))),
                            src.sve_u32[EASYSIMD_SV_INDEX_1]);
      r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], (32 - (imm8 & 31)))),
                            src.sve_u32[EASYSIMD_SV_INDEX_2]);
      r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], (32 - (imm8 & 31)))),
                            src.sve_u32[EASYSIMD_SV_INDEX_3]);
    }
    return r;
  #else
    easysimd__m512i_private
      src_ = easysimd__m512i_to_private(src),
      a_ = easysimd__m512i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? a_.u32[i] : src_.u32[i];
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31))) : src_.u32[i];
        }
        break;
      }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_rol_epi32(src, k, a, imm8) _mm512_mask_rol_epi32(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_mask_rol_epi32(src, k, a, imm8) easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_rol_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rol_epi32
  #define _mm512_mask_rol_epi32(src, k, a, imm8) easysimd_mm512_mask_rol_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_rol_epi32 (easysimd__mmask16 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    if ((imm8 & 31) == 0) {
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_u32[EASYSIMD_SV_INDEX_2], svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_u32[EASYSIMD_SV_INDEX_3], svdup_n_u32(0));
    } else {
      easysimd_svbool_t pg = svptrue_b32();
      r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], (32 - (imm8 & 31)))),
                            svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], (32 - (imm8 & 31)))),
                            svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], (32 - (imm8 & 31)))),
                            svdup_n_u32(0));
      r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                            svorr_u32_z(pg, svlsl_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], (imm8 & 31)), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], (32 - (imm8 & 31)))),
                            svdup_n_u32(0));
    }
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? a_.u32[i] : UINT32_C(0);
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
          r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] << (imm8 & 31)) | (a_.u32[i] >> (32 - (imm8 & 31))) : UINT32_C(0);
        }
        break;
      }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_rol_epi32(k, a, imm8) _mm512_maskz_rol_epi32(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_maskz_rol_epi32(k, a, imm8) easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_rol_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_rol_epi32
  #define _mm512_maskz_rol_epi32(k, a, imm8) easysimd_mm512_maskz_rol_epi32(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_rol_epi64(a, imm8) _mm_rol_epi64(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m128i
  easysimd_mm_rol_epi64 (easysimd__m128i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m128i r;
      if ((imm8 & 63) == 0) {
        r = a;
      } else {
        easysimd_svbool_t pg = svptrue_b64();
        r.sve_u64 = svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64, (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64, (64 - (imm8 & 63))));
      }
      return r;
    #else
      easysimd__m128i_private
        r_,
        a_ = easysimd__m128i_to_private(a);

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        switch (imm8 & 63) {
          case 0:
            r_ = a_;
            break;
          default:
            r_.u64 = (a_.u64 << (imm8 & 63)) | (a_.u64 >> (64 - (imm8 & 63)));
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
              r_.u64[i] = (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63)));
            }
            break;
        }
      #endif

      return easysimd__m128i_from_private(r_);
    #endif
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_rol_epi64
  #define _mm_rol_epi64(a, imm8) easysimd_mm_rol_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_rol_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    if ((imm8 & 63) == 0) {
      r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64, src.sve_u64);
    } else {
      easysimd_svbool_t pg = svptrue_b64();
      r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64, (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64, (64 - (imm8 & 63)))),
                            src.sve_u64);
    }
    return r;
  #else
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? a_.u64[i] : src_.u64[i];
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63))) : src_.u64[i];
        }
        break;
      }
    return easysimd__m128i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_rol_epi64(src, k, a, imm8) _mm_mask_rol_epi64(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm_mask_rol_epi64(src, k, a, imm8) easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_rol_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_rol_epi64
  #define _mm_mask_rol_epi64(src, k, a, imm8) easysimd_mm_mask_rol_epi64(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_rol_epi64 (easysimd__mmask8 k, easysimd__m128i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    if ((imm8 & 63) == 0) {
      r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64, svdup_n_u64(0));
    } else {
      easysimd_svbool_t pg = svptrue_b64();
      r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64, (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64, (64 - (imm8 & 63)))),
                            svdup_n_u64(0));
    }
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? a_.u64[i] : UINT64_C(0);
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63))) : UINT64_C(0);
        }
        break;
      }
    return easysimd__m128i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_rol_epi64(k, a, imm8) _mm_maskz_rol_epi64(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm_maskz_rol_epi64(k, a, imm8) easysimd_mm_maskz_mov_epi64(k, easysimd_mm_rol_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_rol_epi64
  #define _mm_maskz_rol_epi64(k, a, imm8) easysimd_mm_maskz_rol_epi64(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_rol_epi64(a, imm8) _mm256_rol_epi64(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m256i
  easysimd_mm256_rol_epi64 (easysimd__m256i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m256i r;
      if ((imm8 & 63) == 0) {
        r = a;
      } else {
        easysimd_svbool_t pg = svptrue_b64();
        r.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (64 - (imm8 & 63))));
        r.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (64 - (imm8 & 63))));
      }
      return r;
    #else
      easysimd__m256i_private
        r_,
        a_ = easysimd__m256i_to_private(a);

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        switch (imm8 & 63) {
          case 0:
            r_ = a_;
            break;
          default:
            r_.u64 = (a_.u64 << (imm8 & 63)) | (a_.u64 >> (64 - (imm8 & 63)));
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
              r_.u64[i] = (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63)));
            }
            break;
        }
      #endif

      return easysimd__m256i_from_private(r_);
    #endif
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rol_epi64
  #define _mm256_rol_epi64(a, imm8) easysimd_mm256_rol_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_rol_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    if ((imm8 & 63) == 0) {
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], src.sve_u64[EASYSIMD_SV_INDEX_0]);
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], src.sve_u64[EASYSIMD_SV_INDEX_1]);
    } else {
      easysimd_svbool_t pg = svptrue_b64();
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (64 - (imm8 & 63)))),
                            src.sve_u64[EASYSIMD_SV_INDEX_0]);
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (64 - (imm8 & 63)))),
                            src.sve_u64[EASYSIMD_SV_INDEX_1]);
    }
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? a_.u64[i] : src_.u64[i];
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63))) : src_.u64[i];
        }
        break;
      }
    return easysimd__m256i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_rol_epi64(src, k, a, imm8) _mm256_mask_rol_epi64(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm256_mask_rol_epi64(src, k, a, imm8) easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_rol_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_rol_epi64
  #define _mm256_mask_rol_epi64(src, k, a, imm8) easysimd_mm256_mask_rol_epi64(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_rol_epi64 (easysimd__mmask8 k, easysimd__m256i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    if ((imm8 & 63) == 0) {
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], svdup_n_u64(0));
    } else {
      easysimd_svbool_t pg = svptrue_b64();
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (64 - (imm8 & 63)))),
                            svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (64 - (imm8 & 63)))),
                            svdup_n_u64(0));
    }
    return r;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? a_.u64[i] : UINT64_C(0);
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63))) : UINT64_C(0);
        }
        break;
      }
    return easysimd__m256i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_rol_epi64(k, a, imm8) _mm256_maskz_rol_epi64(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm256_maskz_rol_epi64(k, a, imm8) easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_rol_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_rol_epi64
  #define _mm256_maskz_rol_epi64(k, a, imm8) easysimd_mm256_maskz_rol_epi64(k, a, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_rol_epi64(a, imm8) _mm512_rol_epi64(a, imm8)
#else
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m512i
  easysimd_mm512_rol_epi64 (easysimd__m512i a, int imm8)
      EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m512i r;
      if ((imm8 & 63) == 0) {
        r = a;
      } else {
        easysimd_svbool_t pg = svptrue_b64();
        r.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (64 - (imm8 & 63))));
        r.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (64 - (imm8 & 63))));
        r.sve_u64[EASYSIMD_SV_INDEX_2] = svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], (64 - (imm8 & 63))));
        r.sve_u64[EASYSIMD_SV_INDEX_3] = svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], (64 - (imm8 & 63))));
      }
      return r;
    #else
      easysimd__m512i_private
        r_,
        a_ = easysimd__m512i_to_private(a);

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        switch (imm8 & 63) {
          case 0:
            r_ = a_;
            break;
          default:
            r_.u64 = (a_.u64 << (imm8 & 63)) | (a_.u64 >> (64 - (imm8 & 63)));
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
              r_.u64[i] = (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63)));
            }
            break;
        }
      #endif

      return easysimd__m512i_from_private(r_);
    #endif
  }
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_rol_epi64
  #define _mm512_rol_epi64(a, imm8) easysimd_mm512_rol_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_rol_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    if ((imm8 & 63) == 0) {
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], src.sve_u64[EASYSIMD_SV_INDEX_0]);
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], src.sve_u64[EASYSIMD_SV_INDEX_1]);
      r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_u64[EASYSIMD_SV_INDEX_2], src.sve_u64[EASYSIMD_SV_INDEX_2]);
      r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_u64[EASYSIMD_SV_INDEX_3], src.sve_u64[EASYSIMD_SV_INDEX_3]);
    } else {
      easysimd_svbool_t pg = svptrue_b64();
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (64 - (imm8 & 63)))),
                            src.sve_u64[EASYSIMD_SV_INDEX_0]);
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (64 - (imm8 & 63)))),
                            src.sve_u64[EASYSIMD_SV_INDEX_1]);
      r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], (64 - (imm8 & 63)))),
                            src.sve_u64[EASYSIMD_SV_INDEX_2]);
      r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], (64 - (imm8 & 63)))),
                            src.sve_u64[EASYSIMD_SV_INDEX_3]);
    }
    return r;
  #else
    easysimd__m512i_private
      src_ = easysimd__m512i_to_private(src),
      a_ = easysimd__m512i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? a_.u64[i] : src_.u64[i];
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63))) : src_.u64[i];
        }
        break;
      }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_rol_epi64(src, k, a, imm8) _mm512_mask_rol_epi64(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_mask_rol_epi64(src, k, a, imm8) easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_rol_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_rol_epi64
  #define _mm512_mask_rol_epi64(src, k, a, imm8) easysimd_mm512_mask_rol_epi64(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_rol_epi64 (easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    if ((imm8 & 63) == 0) {
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_u64[EASYSIMD_SV_INDEX_2], svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_u64[EASYSIMD_SV_INDEX_3], svdup_n_u64(0));
    } else {
      easysimd_svbool_t pg = svptrue_b64();
      r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], (64 - (imm8 & 63)))),
                            svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], (64 - (imm8 & 63)))),
                            svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], (64 - (imm8 & 63)))),
                            svdup_n_u64(0));
      r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3),
                            svorr_u64_z(pg, svlsl_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], (imm8 & 63)), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], (64 - (imm8 & 63)))),
                            svdup_n_u64(0));
    }
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      r_;

    switch (imm8 & 31) {
      case 0:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? a_.u64[i] : UINT64_C(0);
        }
        break;
      default:
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
          r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] << (imm8 & 63)) | (a_.u64[i] >> (64 - (imm8 & 63))) : UINT64_C(0);
        }
        break;
      }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_rol_epi64(k, a, imm8) _mm512_maskz_rol_epi64(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_maskz_rol_epi64(k, a, imm8) easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_rol_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_rol_epi64
  #define _mm512_maskz_rol_epi64(k, a, imm8) easysimd_mm512_maskz_rol_epi64(k, a, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_ROL_H) */
