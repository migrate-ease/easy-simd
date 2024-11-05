#if !defined(EASYSIMD_X86_AVX512_SHLDV_H)
#define EASYSIMD_X86_AVX512_SHLDV_H

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
easysimd_mm_shldv_epi16(easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_shldv_epi16(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    svuint16_t shift_l = svand_n_u16_x(pg, c.sve_u16, 15);
    svuint16_t shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16 = svorr_u16_x(pg, svlsl_u16_x(pg, a.sve_u16, shift_l), svlsr_u16_x(pg, b.sve_u16, shift_r));
    return a;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      c_ = easysimd__m128i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (((HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) | b_.u16[i]) << (c_.u16[i] & 15)) >> 16);
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shldv_epi16
  #define _mm_shldv_epi16(a, b, c) easysimd_mm_shldv_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shldv_epi32(easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
  #if defined(EASYSIMD_X86_AVX512VBMI2_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_shldv_epi32(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svuint32_t shift_l = svand_n_u32_x(pg, c.sve_u32, 31);
    svuint32_t shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32 = svorr_u32_x(pg, svlsl_u32_x(pg, a.sve_u32, shift_l), svlsr_u32_x(pg, b.sve_u32, shift_r));
    return a;
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      easysimd__m128i_private
        a_ = easysimd__m128i_to_private(a),
        b_ = easysimd__m128i_to_private(b),
        c_ = easysimd__m128i_to_private(c);

      uint64x2_t
        values_lo = vreinterpretq_u64_u32(vzip1q_u32(b_.neon_u32, a_.neon_u32)),
        values_hi = vreinterpretq_u64_u32(vzip2q_u32(b_.neon_u32, a_.neon_u32));

      int32x4_t count = vandq_s32(c_.neon_i32, vdupq_n_s32(31));

      values_lo = vshlq_u64(values_lo, vmovl_s32(vget_low_s32(count)));
      values_hi = vshlq_u64(values_hi, vmovl_high_s32(count));

      r_.neon_u32 =
        vuzp2q_u32(
          vreinterpretq_u32_u64(values_lo),
          vreinterpretq_u32_u64(values_hi)
        );
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      easysimd__m256i
        tmp1,
        lo =
          easysimd_mm256_castps_si256(
            easysimd_mm256_unpacklo_ps(
              easysimd_mm256_castsi256_ps(easysimd_mm256_castsi128_si256(b)),
              easysimd_mm256_castsi256_ps(easysimd_mm256_castsi128_si256(a))
            )
          ),
        hi =
          easysimd_mm256_castps_si256(
            easysimd_mm256_unpackhi_ps(
              easysimd_mm256_castsi256_ps(easysimd_mm256_castsi128_si256(b)),
              easysimd_mm256_castsi256_ps(easysimd_mm256_castsi128_si256(a))
            )
          ),
        tmp2 =
          easysimd_mm256_castpd_si256(
            easysimd_mm256_permute2f128_pd(
              easysimd_mm256_castsi256_pd(lo),
              easysimd_mm256_castsi256_pd(hi),
              32
            )
          );

      tmp2 =
        easysimd_mm256_sllv_epi64(
          tmp2,
          easysimd_mm256_cvtepi32_epi64(
            easysimd_mm_and_si128(
              c,
              easysimd_mm_set1_epi32(31)
            )
          )
        );

      tmp1 =
        easysimd_mm256_castpd_si256(
          easysimd_mm256_permute2f128_pd(
            easysimd_mm256_castsi256_pd(tmp2),
            easysimd_mm256_castsi256_pd(tmp2),
            1
          )
        );

      r_ =
        easysimd__m128i_to_private(
          easysimd_mm256_castsi256_si128(
            easysimd_mm256_castps_si256(
              easysimd_mm256_shuffle_ps(
                easysimd_mm256_castsi256_ps(tmp2),
                easysimd_mm256_castsi256_ps(tmp1),
                221
              )
            )
          )
        );
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      easysimd__m128i_private
        c_ = easysimd__m128i_to_private(c),
        lo = easysimd__m128i_to_private(easysimd_mm_unpacklo_epi32(b, a)),
        hi = easysimd__m128i_to_private(easysimd_mm_unpackhi_epi32(b, a));

      size_t halfway = (sizeof(r_.u32) / sizeof(r_.u32[0]) / 2);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < halfway ; i++) {
        lo.u64[i] <<= (c_.u32[i] & 31);
        hi.u64[i] <<= (c_.u32[halfway + i] & 31);
      }

      r_ =
        easysimd__m128i_to_private(
          easysimd_mm_castps_si128(
            easysimd_mm_shuffle_ps(
              easysimd_mm_castsi128_ps(easysimd__m128i_from_private(lo)),
              easysimd_mm_castsi128_ps(easysimd__m128i_from_private(hi)),
              221)
          )
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_CONVERT_VECTOR_) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      easysimd__m128i_private
        c_ = easysimd__m128i_to_private(c);
      easysimd__m256i_private
        a_ = easysimd__m256i_to_private(easysimd_mm256_castsi128_si256(a)),
        b_ = easysimd__m256i_to_private(easysimd_mm256_castsi128_si256(b)),
        tmp1,
        tmp2;

      tmp1.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(tmp1.u64), EASYSIMD_SHUFFLE_VECTOR_(32, 32, b_.i32, a_.i32, 0, 8, 1, 9, 2, 10, 3, 11));
      EASYSIMD_CONVERT_VECTOR_(tmp2.u64, c_.u32);

      tmp1.u64 <<= (tmp2.u64 & 31);

      r_.i32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, tmp1.m128i_private[0].i32, tmp1.m128i_private[1].i32, 1, 3, 5, 7);
    #else
      easysimd__m128i_private
        a_ = easysimd__m128i_to_private(a),
        b_ = easysimd__m128i_to_private(b),
        c_ = easysimd__m128i_to_private(c);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (((HEDLEY_STATIC_CAST(uint64_t, a_.u32[i]) << 32) | b_.u32[i]) << (c_.u32[i] & 31)) >> 32);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI2_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_shldv_epi32
  #define _mm_shldv_epi32(a, b, c) easysimd_mm_shldv_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shldv_epi16(easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_shldv_epi16(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    svuint16_t shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15);
    svuint16_t shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_x(pg, svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], shift_r));

    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_x(pg, svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], shift_r));

    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b),
      c_ = easysimd__m256i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (((HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) | b_.u16[i]) << (c_.u16[i] & 15)) >> 16);
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shldv_epi16
  #define _mm256_shldv_epi16(a, b, c) easysimd_mm256_shldv_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shldv_epi32(easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
  #if defined(EASYSIMD_X86_AVX512VBMI2_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_shldv_epi32(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svuint32_t shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31);
    svuint32_t shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_x(pg, svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], shift_r));

    shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31);
    shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_x(pg, svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], shift_r));

    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b),
      c_ = easysimd__m256i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (((HEDLEY_STATIC_CAST(uint64_t, a_.u32[i]) << 32) | b_.u32[i]) << (c_.u32[i] & 31)) >> 32);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI2_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shldv_epi32
  #define _mm256_shldv_epi32(a, b, c) easysimd_mm256_shldv_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shldv_epi16(easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm512_shldv_epi16(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    svuint16_t shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15);
    svuint16_t shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_x(pg, svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], shift_r));

    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_x(pg, svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], shift_r));

    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_2], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_2] = svorr_u16_x(pg, svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_2], shift_r));

    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_3], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_3] = svorr_u16_x(pg, svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_3], shift_r));

    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b),
      c_ = easysimd__m512i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (((HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) | b_.u16[i]) << (c_.u16[i] & 15)) >> 16);
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shldv_epi16
  #define _mm512_shldv_epi16(a, b, c) easysimd_mm512_shldv_epi16(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shldv_epi32(easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
  #if defined(EASYSIMD_X86_AVX512VBMI2_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm512_shldv_epi32(a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svuint32_t shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31);
    svuint32_t shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_x(pg, svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], shift_r));

    shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31);
    shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_x(pg, svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], shift_r));

    shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_2], 31);
    shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_2] = svorr_u32_x(pg, svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_2], shift_r));

    shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_3], 31);
    shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_3] = svorr_u32_x(pg, svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_3], shift_r));

    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b),
      c_ = easysimd__m512i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (((HEDLEY_STATIC_CAST(uint64_t, a_.u32[i]) << 32) | b_.u32[i]) << (c_.u32[i] & 31)) >> 32);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI2_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shldv_epi32
  #define _mm512_shldv_epi32(a, b, c) easysimd_mm512_shldv_epi32(a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shldv_epi16(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    // printf("fun %s on X86 was called!\n", __FUNCTION__);
    return _mm_maskz_shldv_epi16(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    svuint16_t shift_l = svand_n_u16_x(pg, c.sve_u16, 15);
    svuint16_t shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16 = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_u16_x(pg, a.sve_u16, shift_l), svlsr_u16_x(pg, b.sve_u16, shift_r));
    return a;
  #else
    // printf("fun %s on C was called!\n", __FUNCTION__);
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      c_ = easysimd__m128i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (k >> i ) & 0x01 ? HEDLEY_STATIC_CAST(uint16_t, (((HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) | b_.u16[i]) << (c_.u16[i] & 15)) >> 16) : 0;
    }
    
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shldv_epi16
  #define _mm_maskz_shldv_epi16(k, a, b, c) easysimd_mm_maskz_shldv_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_shldv_epi32(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b, easysimd__m128i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_shldv_epi32(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svuint32_t shift_l = svand_n_u32_x(pg, c.sve_u32, 31);
    svuint32_t shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32 = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_u32_x(pg, a.sve_u32, shift_l), svlsr_u32_x(pg, b.sve_u32, shift_r));
    return a;
  #else
    easysimd__m128i_private r_;

    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      c_ = easysimd__m128i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(uint32_t, (((HEDLEY_STATIC_CAST(uint64_t, a_.u32[i]) << 32) | b_.u32[i]) << (c_.u32[i] & 31)) >> 32) : 0;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shldv_epi32
  #define _mm_maskz_shldv_epi32(k, a, b, c) easysimd_mm_maskz_shldv_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shldv_epi16(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_shldv_epi16(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    svuint16_t shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15);
    svuint16_t shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], shift_r));
    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], shift_r));

    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b),
      c_ = easysimd__m256i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(uint16_t, (((HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) | b_.u16[i]) << (c_.u16[i] & 15)) >> 16) : 0;
    }
    
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shldv_epi16
  #define _mm256_maskz_shldv_epi16(k, a, b, c) easysimd_mm256_maskz_shldv_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_shldv_epi32(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b, easysimd__m256i c) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_shldv_epi32(k, a, b, c);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svuint32_t shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31);
  svuint32_t shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
  a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], shift_r));

  shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31);
  shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
  a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], shift_r));

  return a;
#else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    c_ = easysimd__m256i_to_private(c);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
    r_.u32[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(uint32_t, (((HEDLEY_STATIC_CAST(uint64_t, a_.u32[i]) << 32) | b_.u32[i]) << (c_.u32[i] & 31)) >> 32) : 0;
  }

  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_shldv_epi32
  #define _mm256_maskz_shldv_epi32(k, a, b, c) easysimd_mm256_maskz_shldv_epi32(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shldv_epi16(easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm512_maskz_shldv_epi16(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b16();
    svuint16_t shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_0], 15);
    svuint16_t shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_0] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_0], shift_r));

    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_1], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_1] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_1], shift_r));

    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_2], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_2] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_2], shift_r));

    shift_l = svand_n_u16_x(pg, c.sve_u16[EASYSIMD_SV_INDEX_3], 15);
    shift_r = svsub_u16_x(pg, svdup_n_u16(16), shift_l);
    a.sve_u16[EASYSIMD_SV_INDEX_3] = svorr_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svlsl_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], shift_l), svlsr_u16_x(pg, b.sve_u16[EASYSIMD_SV_INDEX_3], shift_r));

    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b),
      c_ = easysimd__m512i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(uint16_t, (((HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) << 16) | b_.u16[i]) << (c_.u16[i] & 15)) >> 16) : 0;
    }
    
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shldv_epi16
  #define _mm512_maskz_shldv_epi16(k, a, b, c) easysimd_mm512_maskz_shldv_epi16(k, a, b, c)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shldv_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b, easysimd__m512i c) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm512_maskz_shldv_epi32(k, a, b, c);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svuint32_t shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_0], 31);
    svuint32_t shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], shift_r));

    shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_1], 31);
    shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], shift_r));

    shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_2], 31);
    shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_2] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_2], shift_r));

    shift_l = svand_n_u32_x(pg, c.sve_u32[EASYSIMD_SV_INDEX_3], 31);
    shift_r = svsub_u32_x(pg, svdup_n_u32(32), shift_l);
    a.sve_u32[EASYSIMD_SV_INDEX_3] = svorr_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svlsl_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], shift_l), svlsr_u32_x(pg, b.sve_u32[EASYSIMD_SV_INDEX_3], shift_r));

    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b),
      c_ = easysimd__m512i_to_private(c);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(uint32_t, (((HEDLEY_STATIC_CAST(uint64_t, a_.u32[i]) << 32) | b_.u32[i]) << (c_.u32[i] & 31)) >> 32) : 0;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shldv_epi32
  #define _mm512_maskz_shldv_epi32(k, a, b, c) easysimd_mm512_maskz_shldv_epi32(k, a, b, c)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SHLDV_H) */
