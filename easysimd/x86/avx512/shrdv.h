#if !defined(EASYSIMD_X86_AVX512_SHRDV_H)
#define EASYSIMD_X86_AVX512_SHRDV_H

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
easysimd_mm_shrdv_epi16(easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_shrdv_epi16(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b16();
  svuint16_t shift_r = svand_n_u16_x(pg, c.sve_u16, 15);
  svuint16_t shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16 = svorr_u16_x(pg, svlsr_u16_x(pg, a.sve_u16, shift_r), svlsl_u16_x(pg, b.sve_u16, shift_l));
  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    c_ = easysimd__m128i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i16[0]); i++) {
      int shift_ = c_.i16[i] & 0xF;
      r_.u16[i] = (uint16_t)(((uint32_t)b_.u16[i] << (16 - shift_)) | (a_.u16[i] >> shift_));
    }

    return easysimd__m128i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shrdv_epi16
  #define _mm_shrdv_epi16(a, b, c) easysimd_mm_shrdv_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shrdv_epi32(easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_shrdv_epi32(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svuint32_t shift_r = svand_n_u32_x(pg, c.sve_u32, 31);
  svuint32_t shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32 = svorr_u32_x(pg, svlsr_u32_x(pg, a.sve_u32, shift_r), svlsl_u32_x(pg, b.sve_u32, shift_l));
  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    c_ = easysimd__m128i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i32[0]); i++) {
      uint32_t shift_ = c_.i32[i] & 31;
      r_.u32[i] = (uint32_t)(((uint64_t)b_.u32[i] << (32 - shift_)) | (a_.u32[i] >> shift_));
    }

    return easysimd__m128i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shrdv_epi32
  #define _mm_shrdv_epi32(a, b, c) easysimd_mm_shrdv_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shrdv_epi64(easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_shrdv_epi64(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svuint64_t shift_r = svand_n_u64_x(pg, c.sve_u64, 63);
  svuint64_t shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64 = svorr_u64_x(pg, svlsr_u64_x(pg, a.sve_u64, shift_r), svlsl_u64_x(pg, b.sve_u64, shift_l));
  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    c_ = easysimd__m128i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i64[0]); i++) {
      int shift_ = c_.i64[i] & 0x3F;
      r_.u64[i] = (b_.u64[i] << (64 - shift_)) | (a_.u64[i] >> shift_);
    }

    return easysimd__m128i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shrdv_epi64
  #define _mm_shrdv_epi64(a, b, c) easysimd_mm_shrdv_epi64(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shrdv_epi16(easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_shrdv_epi16(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b16();
  a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_x(pg, 
    svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15)), 
    svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], svsub_u16_x(pg, svdup_n_u16(16), svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15))));
  a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_x(pg, 
    svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15)), 
    svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], svsub_u16_x(pg, svdup_n_u16(16), svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15))));
  return a;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    c_ = easysimd__m256i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i16[0]); i++) {
      int shift_ = c_.i16[i] & 0xF;
      r_.u16[i] = (uint16_t)(((uint32_t)b_.u16[i] << (16 - shift_)) | (a_.u16[i] >> shift_));
    }

    return easysimd__m256i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shrdv_epi16
  #define _mm256_shrdv_epi16(a, b, c) easysimd_mm256_shrdv_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shrdv_epi32(easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_shrdv_epi32(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_x(pg, 
    svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31)), 
    svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], svsub_u32_x(pg, svdup_n_u32(32), svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31))));
  a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_x(pg, 
    svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31)), 
    svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], svsub_u32_x(pg, svdup_n_u32(32), svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31))));
  return a;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    c_ = easysimd__m256i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i32[0]); i++) {
      int shift_ = c_.i32[i] & 0x1F;
      r_.u32[i] = (uint32_t)(((uint64_t)b_.u32[i] << (32 - shift_)) | (a_.u32[i] >> shift_));
    }

    return easysimd__m256i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shrdv_epi32
  #define _mm256_shrdv_epi32(a, b, c) easysimd_mm256_shrdv_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shrdv_epi64(easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_shrdv_epi64(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_x(pg, 
    svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_0], 63)), 
    svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], svsub_u64_x(pg, svdup_n_u64(64), svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_0], 63))));
  a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_x(pg, 
    svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_1], 63)), 
    svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], svsub_u64_x(pg, svdup_n_u64(64), svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_1], 63))));
  return a;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    c_ = easysimd__m256i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i64[0]); i++) {
      int shift_ = c_.i64[i] & 0x3F;
      r_.u64[i] = (b_.u64[i] << (64 - shift_)) | (a_.u64[i] >> shift_);
    }

    return easysimd__m256i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shrdv_epi64
  #define _mm256_shrdv_epi64(a, b, c) easysimd_mm256_shrdv_epi64(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shrdv_epi16(easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm512_shrdv_epi16(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b16();
  a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_x(pg, 
    svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15)), 
    svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], svsub_u16_x(pg, svdup_n_u16(16), svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15))));
  a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_x(pg, 
    svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15)), 
    svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], svsub_u16_x(pg, svdup_n_u16(16), svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15))));
  a.sve_u16[EASYSIMD_SV_INDEX_2] = svorr_u16_x(pg, 
    svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_2], 15)), 
    svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_2], svsub_u16_x(pg, svdup_n_u16(16), svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_2], 15))));
  a.sve_u16[EASYSIMD_SV_INDEX_3] = svorr_u16_x(pg, 
    svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_3], 15)), 
    svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_3], svsub_u16_x(pg, svdup_n_u16(16), svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_3], 15))));
  return a;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    c_ = easysimd__m512i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i16[0]); i++) {
      int shift_ = c_.i16[i] & 0xF;
      r_.u16[i] = (uint16_t)(((uint32_t)b_.u16[i] << (16 - shift_)) | (a_.u16[i] >> shift_));
    }

    return easysimd__m512i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shrdv_epi16
  #define _mm512_shrdv_epi16(a, b, c) easysimd_mm512_shrdv_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shrdv_epi32(easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm512_shrdv_epi32(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_x(pg, 
    svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31)), 
    svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], svsub_u32_x(pg, svdup_n_u32(32), svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31))));
  a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_x(pg, 
    svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31)), 
    svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], svsub_u32_x(pg, svdup_n_u32(32), svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31))));
  a.sve_u32[EASYSIMD_SV_INDEX_2] = svorr_u32_x(pg, 
    svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_2], 31)), 
    svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_2], svsub_u32_x(pg, svdup_n_u32(32), svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_2], 31))));
  a.sve_u32[EASYSIMD_SV_INDEX_3] = svorr_u32_x(pg, 
    svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_3], 31)), 
    svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_3], svsub_u32_x(pg, svdup_n_u32(32), svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_3], 31))));
  return a;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    c_ = easysimd__m512i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i32[0]); i++) {
      int shift_ = c_.i32[i] & 0x1F;
      r_.u32[i] = (uint32_t)(((uint64_t)b_.u32[i] << (32 - shift_)) | (a_.u32[i] >> shift_));
    }

    return easysimd__m512i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shrdv_epi32
  #define _mm512_shrdv_epi32(a, b, c) easysimd_mm512_shrdv_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shrdv_epi64(easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm512_shrdv_epi64(a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_x(pg, 
    svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_0], 63)), 
    svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], svsub_u64_x(pg, svdup_n_u64(64), svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_0], 63))));
  a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_x(pg, 
    svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_1], 63)), 
    svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], svsub_u64_x(pg, svdup_n_u64(64), svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_1], 63))));
  a.sve_u64[EASYSIMD_SV_INDEX_2] = svorr_u64_x(pg, 
    svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_2], 63)), 
    svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_2], svsub_u64_x(pg, svdup_n_u64(64), svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_2], 63))));
  a.sve_u64[EASYSIMD_SV_INDEX_3] = svorr_u64_x(pg, 
    svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_3], 63)), 
    svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_3], svsub_u64_x(pg, svdup_n_u64(64), svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_3], 63))));
  return a;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    c_ = easysimd__m512i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i64[0]); i++) {
      int shift_ = c_.i64[i] & 0x3F;
      r_.u64[i] = (b_.u64[i] << (64 - shift_)) | (a_.u64[i] >> shift_);
    }

    return easysimd__m512i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shrdv_epi64
  #define _mm512_shrdv_epi64(a, b, c) easysimd_mm512_shrdv_epi64(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shrdv_epi16(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_maskz_shrdv_epi16(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b16();
  svuint16_t shift_r = svand_n_u16_x(pg, c.sve_u16, 15);
  svuint16_t shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16 = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsr_u16_x(pg, a.sve_u16, shift_r), svlsl_u16_x(pg, b.sve_u16, shift_l));
  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    c_ = easysimd__m128i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i16[0]); i++) {
      int shift_ = c_.i16[i] & 0xF;
      r_.u16[i] = (k >> i) & 0x01 ? (uint16_t)(((uint32_t)b_.u16[i] << (16 - shift_)) | (a_.u16[i] >> shift_)) : 0;
    }

    return easysimd__m128i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shrdv_epi16
  #define _mm_maskz_shrdv_epi16(k, a, b, c) easysimd_mm_maskz_shrdv_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shrdv_epi32(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_maskz_shrdv_epi32(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svuint32_t shift_r = svand_n_u32_x(pg, c.sve_u32, 31);
  svuint32_t shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32 = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsr_u32_x(pg, a.sve_u32, shift_r), svlsl_u32_x(pg, b.sve_u32, shift_l));
  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    c_ = easysimd__m128i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i32[0]); i++) {
      int shift_ = c_.i32[i] & 31;
      r_.u32[i] = (k >> i) & 0x01 ? (uint32_t)(((uint64_t)b_.u32[i] << (32 - shift_)) | (a_.u32[i] >> shift_)) : 0;
    }

    return easysimd__m128i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shrdv_epi32
  #define _mm_maskz_shrdv_epi32(k, a, b, c) easysimd_mm_maskz_shrdv_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shrdv_epi64(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_maskz_shrdv_epi64(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svuint64_t shift_r = svand_n_u64_x(pg, c.sve_u64, 63);
  svuint64_t shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64 = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsr_u64_x(pg, a.sve_u64, shift_r), svlsl_u64_x(pg, b.sve_u64, shift_l));
  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    c_ = easysimd__m128i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i64[0]); i++) {
      int shift_ = c_.i64[i] & 31;
      r_.u64[i] = (k >> i) & 0x01 ? (uint64_t)(((uint64_t)b_.u64[i] << (64 - shift_)) | (a_.u64[i] >> shift_)) : 0;
    }

    return easysimd__m128i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shrdv_epi64
  #define _mm_maskz_shrdv_epi64(k, a, b, c) easysimd_mm_maskz_shrdv_epi64(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shrdv_epi16(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_shrdv_epi16(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b16();
  svuint16_t shift_r = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15);
  svuint16_t shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_r), svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], shift_l));

  shift_r = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15);
  shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_r), svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], shift_l));

  return a;

#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    c_ = easysimd__m256i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i16[0]); i++) {
      int shift_ = c_.i16[i] & 0xF;
      r_.u16[i] = (k >> i) & 0x01 ? (uint16_t)(((uint32_t)b_.u16[i] << (16 - shift_)) | (a_.u16[i] >> shift_)) : 0;
    }

    return easysimd__m256i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shrdv_epi16
  #define _mm256_maskz_shrdv_epi16(k, a, b, c) easysimd_mm256_maskz_shrdv_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shrdv_epi32(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_shrdv_epi32(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svuint32_t shift_r = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31);
  svuint32_t shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_r), svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], shift_l));

  shift_r = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31);
  shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_r), svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], shift_l));

  return a;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    c_ = easysimd__m256i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i32[0]); i++) {
      int shift_ = c_.i32[i] & 31;
      r_.u32[i] = (k >> i) & 0x01 ? (uint32_t)(((uint64_t)b_.u32[i] << (32 - shift_)) | (a_.u32[i] >> shift_)) : 0;
    }

    return easysimd__m256i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shrdv_epi32
  #define _mm256_maskz_shrdv_epi32(k, a, b, c) easysimd_mm256_maskz_shrdv_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shrdv_epi64(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_shrdv_epi64(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svuint64_t shift_r = svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_0], 63);
  svuint64_t shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_r), svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], shift_l));

  shift_r = svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_1], 63);
  shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_r), svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], shift_l));

  return a;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    c_ = easysimd__m256i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i64[0]); i++) {
      int shift_ = c_.i64[i] & 31;
      r_.u64[i] = (k >> i) & 0x01 ? (uint64_t)(((uint64_t)b_.u64[i] << (64 - shift_)) | (a_.u64[i] >> shift_)) : 0;
    }

    return easysimd__m256i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shrdv_epi64
  #define _mm256_maskz_shrdv_epi64(k, a, b, c) easysimd_mm256_maskz_shrdv_epi64(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shrdv_epi16(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm512_maskz_shrdv_epi16(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b16();
  svuint16_t shift_r = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15);
  svuint16_t shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_r), svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], shift_l));

  shift_r = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15);
  shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_r), svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], shift_l));

  shift_r = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_2], 15);
  shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16[EASYSIMD_SV_INDEX_2] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], shift_r), svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_2], shift_l));

  shift_r = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_3], 15);
  shift_l = svsub_u16_x(pg, svdup_n_u16(16), shift_r);
  a.sve_u16[EASYSIMD_SV_INDEX_3] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svlsr_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], shift_r), svlsl_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_3], shift_l));

  return a;

#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    c_ = easysimd__m512i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i16[0]); i++) {
      int shift_ = c_.i16[i] & 0xF;
      r_.u16[i] = (k >> i) & 0x01 ? (uint16_t)(((uint32_t)b_.u16[i] << (16 - shift_)) | (a_.u16[i] >> shift_)) : 0;
    }

    return easysimd__m512i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shrdv_epi16
  #define _mm512_maskz_shrdv_epi16(k, a, b, c) easysimd_mm512_maskz_shrdv_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shrdv_epi32(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm512_maskz_shrdv_epi32(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svuint32_t shift_r = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31);
  svuint32_t shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_r), svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], shift_l));

  shift_r = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31);
  shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_r), svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], shift_l));

  shift_r = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_2], 31);
  shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32[EASYSIMD_SV_INDEX_2] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], shift_r), svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_2], shift_l));

  shift_r = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_3], 31);
  shift_l = svsub_u32_x(pg, svdup_n_u32(32), shift_r);
  a.sve_u32[EASYSIMD_SV_INDEX_3] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svlsr_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], shift_r), svlsl_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_3], shift_l));

  return a;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    c_ = easysimd__m512i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i32[0]); i++) {
      int shift_ = c_.i32[i] & 31;
      r_.u32[i] = (k >> i) & 0x01 ? (uint32_t)(((uint64_t)b_.u32[i] << (32 - shift_)) | (a_.u32[i] >> shift_)) : 0;
    }

    return easysimd__m512i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shrdv_epi32
  #define _mm512_maskz_shrdv_epi32(k, a, b, c) easysimd_mm512_maskz_shrdv_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shrdv_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm512_maskz_shrdv_epi64(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svuint64_t shift_r = svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_0], 63);
  svuint64_t shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_r), svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], shift_l));

  shift_r = svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_1], 63);
  shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_r), svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], shift_l));

  shift_r = svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_2], 63);
  shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64[EASYSIMD_SV_INDEX_2] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], shift_r), svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_2], shift_l));

  shift_r = svand_n_u64_x(pg, c.sve_u64[EASYSIMD_SV_INDEX_3], 63);
  shift_l = svsub_u64_x(pg, svdup_n_u64(64), shift_r);
  a.sve_u64[EASYSIMD_SV_INDEX_3] = svorr_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svlsr_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], shift_r), svlsl_u64_x(pg, b.sve_u64[EASYSIMD_SV_INDEX_3], shift_l));

  return a;
#else
  easysimd__m512i_private
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b),
    c_ = easysimd__m512i_to_private(c),
    r_;

    for(size_t i = 0; i < sizeof(a_) / sizeof(a_.i64[0]); i++) {
      int shift_ = c_.i64[i] & 31;
      r_.u64[i] = (k >> i) & 0x01 ? (uint64_t)(((uint64_t)b_.u64[i] << (64 - shift_)) | (a_.u64[i] >> shift_)) : 0;
    }

    return easysimd__m512i_from_private(r_);

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shrdv_epi64
  #define _mm512_maskz_shrdv_epi64(k, a, b, c) easysimd_mm512_maskz_shrdv_epi64(k, a, b, c)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SHLDV_H) */
