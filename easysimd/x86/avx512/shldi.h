#if !defined(EASYSIMD_X86_AVX512_SHLDI_H)
#define EASYSIMD_X86_AVX512_SHLDI_H

#include "types.h"
#include "../avx2.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shldi_epi16(easysimd__m128i a, easysimd__m128i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    uint16_t shift_l = imm8 & 0xF;
    a.sve_u16 = svorr_u16_x(pg, svlsl_n_u16_x(pg, a.sve_u16, shift_l), svlsr_n_u16_x(pg, b.sve_u16, 16 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 15;
    if(!shift_) return a;

    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] =(a_.u16[i] << shift_) | (b_.u16[i] >> (16 - shift_));
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm_shldi_epi16(a, b, imm8) _mm_shldi_epi16(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shldi_epi16
  #define _mm_shldi_epi16(a, b, c) easysimd_mm_shldi_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shldi_epi32(easysimd__m128i a, easysimd__m128i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    uint32_t shift_l = imm8 & 31;
    a.sve_u32 = svorr_u32_x(pg, svlsl_n_u32_x(pg, a.sve_u32, shift_l), svlsr_n_u32_x(pg, b.sve_u32, 32 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 31;
    if(!shift_) return a;
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] =(a_.u32[i] << shift_) | (b_.u32[i] >> (32 - shift_));
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm_shldi_epi32(a, b, imm8) _mm_shldi_epi32(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shldi_epi32
  #define _mm_shldi_epi32(a, b, c) easysimd_mm_shldi_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shldi_epi64(easysimd__m128i a, easysimd__m128i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    uint64_t shift_l = imm8 & 63;
    a.sve_u64 = svorr_u64_x(pg, svlsl_n_u64_x(pg, a.sve_u64, shift_l), svlsr_n_u64_x(pg, b.sve_u64, 64 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 63;
    if(!shift_) return a;
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] =(a_.u64[i] << shift_) | (b_.u64[i] >> (64 - shift_));
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm_shldi_epi64(a, b, imm8) _mm_shldi_epi64(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shldi_epi64
  #define _mm_shldi_epi64(a, b, c) easysimd_mm_shldi_epi64(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shldi_epi16(easysimd__m256i a, easysimd__m256i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    uint16_t shift_l = imm8 & 0xF;
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_x(pg, svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_x(pg, svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], 16 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 15;
    if(!shift_) return a;

    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (a_.u16[i] << shift_) | (b_.u16[i] >> (16 - shift_));
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm256_shldi_epi16(a, b, imm8) _mm256_shldi_epi16(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shldi_epi16
  #define _mm256_shldi_epi16(a, b, c) easysimd_mm256_shldi_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shldi_epi32(easysimd__m256i a, easysimd__m256i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    uint32_t shift_l = imm8 & 31;
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_x(pg, svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_x(pg, svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], 32 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 31;
    if(!shift_) return a;

    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] =(a_.u32[i] << shift_) | (b_.u32[i] >> (32 - shift_));
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm256_shldi_epi32(a, b, imm8) _mm256_shldi_epi32(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shldi_epi32
  #define _mm256_shldi_epi32(a, b, c) easysimd_mm256_shldi_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shldi_epi64(easysimd__m256i a, easysimd__m256i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    uint64_t shift_l = imm8 & 63;
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_x(pg, svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_x(pg, svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], 64 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 63;
    if(!shift_) return a;

    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] =(a_.u64[i] << shift_) | (b_.u64[i] >> (64 - shift_));
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm256_shldi_epi64(a, b, imm8) _mm256_shldi_epi64(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shldi_epi64
  #define _mm256_shldi_epi64(a, b, c) easysimd_mm256_shldi_epi64(a, b, c)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shldi_epi16(easysimd__m512i a, easysimd__m512i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    uint16_t shift_l = imm8 & 0xF;
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_x(pg, svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_x(pg, svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_2] = svorr_u16_x(pg, svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_2], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_3] = svorr_u16_x(pg, svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_3], 16 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 15;
    if(!shift_) return a;

    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (a_.u16[i] << shift_) | (b_.u16[i] >> (16 - shift_));
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm512_shldi_epi16(a, b, imm8) _mm512_shldi_epi16(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shldi_epi16
  #define _mm512_shldi_epi16(a, b, c) easysimd_mm512_shldi_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shldi_epi32(easysimd__m512i a, easysimd__m512i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    uint32_t shift_l = imm8 & 31;
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_x(pg, svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_x(pg, svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_2] = svorr_u32_x(pg, svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_2], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_3] = svorr_u32_x(pg, svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_3], 32 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 31;
    if(!shift_) return a;

    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] =(a_.u32[i] << shift_) | (b_.u32[i] >> (32 - shift_));
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm512_shldi_epi32(a, b, imm8) _mm512_shldi_epi32(a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shldi_epi32
  #define _mm512_shldi_epi32(a, b, c) easysimd_mm512_shldi_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shldi_epi64(easysimd__m512i a, easysimd__m512i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    uint64_t shift_l = imm8 & 63;
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_x(pg, svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_x(pg, svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_2] = svorr_u64_x(pg, svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_2], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_3] = svorr_u64_x(pg, svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_3], 64 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 63;
    if(!shift_) return a;

    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] =(a_.u64[i] << shift_) | (b_.u64[i] >> (64 - shift_));
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm512_shldi_epi64(a, b, imm8) _mm512_shldi_epi64(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shldi_epi16(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    uint16_t shift_l = imm8 & 0xF;
    a.sve_u16 = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_n_u16_x(pg, a.sve_u16, shift_l), svlsr_n_u16_x(pg, b.sve_u16, 16 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 15;
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u16[i] << shift_) | (b_.u16[i] >> (16 - shift_))) : a_.u16[i]) : 0;
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm_maskz_shldi_epi16(k, a, b, imm8) _mm_maskz_shldi_epi16(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shldi_epi16
  #define _mm_maskz_shldi_epi16(k, a, b, c) easysimd_mm_maskz_shldi_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shldi_epi32(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    uint32_t shift_l = imm8 & 31;
    a.sve_u32 = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_n_u32_x(pg, a.sve_u32, shift_l), svlsr_n_u32_x(pg, b.sve_u32, 32 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 31;

    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u32[i] << shift_) | (b_.u32[i] >> (32 - shift_))) : a_.u32[i]) : 0;
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm_maskz_shldi_epi32(k, a, b, imm8) _mm_maskz_shldi_epi32(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shldi_epi32
  #define _mm_maskz_shldi_epi32(k, a, b, c) easysimd_mm_maskz_shldi_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shldi_epi64(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    uint64_t shift_l = imm8 & 63;
    a.sve_u64 = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_n_u64_x(pg, a.sve_u64, shift_l), svlsr_n_u64_x(pg, b.sve_u64, 64 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 63;

    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u64[i] << shift_) | (b_.u64[i] >> (64 - shift_))) : a_.u64[i]) : 0;
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm_maskz_shldi_epi64(k, a, b, imm8) _mm_maskz_shldi_epi64(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shldi_epi64
  #define _mm_maskz_shldi_epi64(k, a, b, c) easysimd_mm_maskz_shldi_epi64(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shldi_epi16(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    uint16_t shift_l = imm8 & 0xF;
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], 16 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 15;
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u16[i] << shift_) | (b_.u16[i] >> (16 - shift_))) : a_.u16[i]) : 0;
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm256_maskz_shldi_epi16(k, a, b, imm8) _mm256_maskz_shldi_epi16(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shldi_epi16
  #define _mm256_maskz_shldi_epi16(k, a, b, c) easysimd_mm256_maskz_shldi_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shldi_epi32(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    uint32_t shift_l = imm8 & 31;
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], 32 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 31;

    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u32[i] << shift_) | (b_.u32[i] >> (32 - shift_))) : a_.u32[i]) : 0;
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm256_maskz_shldi_epi32(k, a, b, imm8) _mm256_maskz_shldi_epi32(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shldi_epi32
  #define _mm256_maskz_shldi_epi32(k, a, b, c) easysimd_mm256_maskz_shldi_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shldi_epi64(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    uint64_t shift_l = imm8 & 63;
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], 64 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 63;

    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u64[i] << shift_) | (b_.u64[i] >> (64 - shift_))) : a_.u64[i]) : 0;
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm256_maskz_shldi_epi64(k, a, b, imm8) _mm256_maskz_shldi_epi64(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shldi_epi64
  #define _mm256_maskz_shldi_epi64(k, a, b, c) easysimd_mm256_maskz_shldi_epi64(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shldi_epi16(easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    uint16_t shift_l = imm8 & 0xF;
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_2] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_2], 16 - shift_l));
    a.sve_u16[EASYSIMD_SV_INDEX_3] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svlsl_n_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], shift_l), svlsr_n_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_3], 16 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 15;
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u16[i] << shift_) | (b_.u16[i] >> (16 - shift_))) : a_.u16[i]) : 0;
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm512_maskz_shldi_epi16(k, a, b, imm8) _mm512_maskz_shldi_epi16(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shldi_epi16
  #define _mm512_maskz_shldi_epi16(k, a, b, c) easysimd_mm512_maskz_shldi_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shldi_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    uint32_t shift_l = imm8 & 31;
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_2] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_2], 32 - shift_l));
    a.sve_u32[EASYSIMD_SV_INDEX_3] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svlsl_n_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], shift_l), svlsr_n_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_3], 32 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 31;

    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u32[i] << shift_) | (b_.u32[i] >> (32 - shift_))) : a_.u32[i]) : 0;
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm512_maskz_shldi_epi32(k, a, b, imm8) _mm512_maskz_shldi_epi32(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shldi_epi32
  #define _mm512_maskz_shldi_epi32(k, a, b, c) easysimd_mm512_maskz_shldi_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shldi_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b, int imm8) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    uint64_t shift_l = imm8 & 63;
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_2] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_2], 64 - shift_l));
    a.sve_u64[EASYSIMD_SV_INDEX_3] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svlsl_n_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], shift_l), svlsr_n_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_3], 64 - shift_l));
    return a;
  #else
    int shift_ = imm8 & 63;

    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (k >> i) & 0x01 ? (shift_ ? ((a_.u64[i] << shift_) | (b_.u64[i] >> (64 - shift_))) : a_.u64[i]) : 0;
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define easysimd_mm512_maskz_shldi_epi64(k, a, b, imm8) _mm512_maskz_shldi_epi64(k, a, b, imm8)
#endif

#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shldi_epi64
  #define _mm512_maskz_shldi_epi64(k, a, b, c) easysimd_mm512_maskz_shldi_epi64(k, a, b, c)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SHLDI_H) */
