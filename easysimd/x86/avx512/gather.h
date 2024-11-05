#if !defined(EASYSIMD_X86_AVX512_GATHER_H)
#define EASYSIMD_X86_AVX512_GATHER_H

#include "types.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_i64gather_ps(easysimd__m512i vindex, const void* base_addr, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index0 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_gather_s32offset_f32(pg, (const easysimd_float32*)base_addr, index0);

    sveint32_t index1 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[4])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[5])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[6])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[7])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_gather_s32offset_f32(pg, (const easysimd_float32*)base_addr, index1);
    return r;
  #else
    easysimd__m512i_private
      vindex_ = easysimd__m512i_to_private(vindex);
    easysimd__m256_private
      r_ = easysimd__m256_to_private(easysimd_mm256_setzero_ps());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float32 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f32[i] = dst;
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_i64gather_ps(vindex, base_addr, scale) _mm512_i64gather_ps(vindex, EASYSIMD_CHECKED_REINTERPRET_CAST(void const*, easysimd_float32 const*, base_addr), scale)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i64gather_ps
  #define _mm512_i64gather_ps(vindex, base_addr, scale) easysimd_mm512_i64gather_ps(vindex, EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, void const*, base_addr), scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_mask_i64gather_ps(easysimd__m256 src, easysimd__mmask8 k, easysimd__m512i vindex, const void* base_addr, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    sveint32_t index0 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_gather_s32offset_f32(pg, (const easysimd_float32*)base_addr, index0), src.sve_f32[EASYSIMD_SV_INDEX_0]);

    sveint32_t index1 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[4])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[5])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[6])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[7])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svld1_gather_s32offset_f32(pg, (const easysimd_float32*)base_addr, index1), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m512i_private
      vindex_ = easysimd__m512i_to_private(vindex);
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      r_ = easysimd__m256_to_private(easysimd_mm256_setzero_ps());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float32 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f32[i] = dst;
      }
      else {
        r_.f32[i] = src_.f32[i];
      }
    }

    return easysimd__m256_from_private(r_);
 #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_i64gather_ps(src, k, vindex, base_addr, scale) _mm512_mask_i64gather_ps(src, k, vindex, EASYSIMD_CHECKED_REINTERPRET_CAST(void const*, easysimd_float32 const*, base_addr), scale)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_i64gather_ps
  #define _mm512_mask_i64gather_ps(src, k, vindex, base_addr, scale) easysimd_mm512_mask_i64gather_ps(src, k, vindex, EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, void const*, base_addr), scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_i64gather_epi32(easysimd__m512i vindex, const void* base_addr, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    sveint32_t index0 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_gather_s32offset_s32(pg, (const int32_t*)base_addr, index0);

    sveint32_t index1 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[4])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[5])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[6])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[7])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_gather_s32offset_s32(pg, (const int32_t*)base_addr, index1);
    return r;
  #else
    easysimd__m512i_private
      vindex_ = easysimd__m512i_to_private(vindex);
    easysimd__m256i_private
      r_ = easysimd__m256i_to_private(easysimd_mm256_setzero_si256());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int32_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i32[i] = dst;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i64gather_epi32
  #define _mm512_i64gather_epi32(vindex, base_addr, scale) easysimd_mm512_i64gather_epi32(vindex, EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, void const*, base_addr), scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_mask_i64gather_epi32(easysimd__m256i src, easysimd__mmask8 k, easysimd__m512i vindex, const void* base_addr, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    sveint32_t index0 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_gather_s32offset_s32(pg, (const int32_t*)base_addr, index0), src.sve_i32[EASYSIMD_SV_INDEX_0]);

    sveint32_t index1 = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[4])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[5])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[6])* HEDLEY_STATIC_CAST(size_t, scale)),
                                     (HEDLEY_STATIC_CAST(size_t, vindex.i64[7])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svld1_gather_s32offset_s32(pg, (const int32_t*)base_addr, index1), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m512i_private
      vindex_ = easysimd__m512i_to_private(vindex);
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      r_ = easysimd__m256i_to_private(easysimd_mm256_setzero_si256());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((k >> i) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int32_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i32[i] = dst;
      }
      else {
        r_.i32[i] = src_.i32[i];
      }
    }

    return easysimd__m256i_from_private(r_);
 #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_i64gather_epi32
  #define _mm512_mask_i64gather_epi32(src, k, vindex, base_addr, scale) easysimd_mm512_mask_i64gather_epi32(src, k, vindex, EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, void const*, base_addr), scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_i32gather_epi32(easysimd__m512i vindex, void const* base_addr, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    sveint32_t index0 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_0], scale);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_gather_s32offset_s32(pg, (const int *)base_addr, index0);

    sveint32_t index1 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_1], scale);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_gather_s32offset_s32(pg, (const int *)base_addr, index1);

    sveint32_t index2 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_2], scale);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svld1_gather_s32offset_s32(pg, (const int *)base_addr, index2);

    sveint32_t index3 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_3], scale);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svld1_gather_s32offset_s32(pg, (const int *)base_addr, index3);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.i32[ 0] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 0] * scale));
    r.i32[ 1] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 1] * scale));
    r.i32[ 2] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 2] * scale));
    r.i32[ 3] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 3] * scale));
    r.i32[ 4] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 4] * scale));
    r.i32[ 5] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 5] * scale));
    r.i32[ 6] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 6] * scale));
    r.i32[ 7] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 7] * scale));
    r.i32[ 8] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 8] * scale));
    r.i32[ 9] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 9] * scale));
    r.i32[10] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[10] * scale));
    r.i32[11] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[11] * scale));
    r.i32[12] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[12] * scale));
    r.i32[13] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[13] * scale));
    r.i32[14] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[14] * scale));
    r.i32[15] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[15] * scale));
    return r;
  #else
    easysimd__m512i_private
      vindex_ = easysimd__m512i_to_private(vindex), r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int32_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i32[i] = dst;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm512_i32gather_epi32(vindex, base_addr, scale) _mm512_i32gather_epi32(vindex, base_addr, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i32gather_epi32
  #define _mm512_i32gather_epi32(vindex, base_addr, scale) easysimd_mm512_i32gather_epi32(vindex, base_addr, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_i32gather_ps(easysimd__m512i vindex, void const* base_addr, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index0 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_0], scale);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_gather_s32offset_f32(pg, (const float *)base_addr, index0);

    sveint32_t index1 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_1], scale);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_gather_s32offset_f32(pg, (const float *)base_addr, index1);

    sveint32_t index2 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_2], scale);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svld1_gather_s32offset_f32(pg, (const float *)base_addr, index2);

    sveint32_t index3 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_3], scale);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svld1_gather_s32offset_f32(pg, (const float *)base_addr, index3);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.f32[ 0] = *((float *)((uint8_t *)base_addr + vindex.i32[ 0] * scale));
    r.f32[ 1] = *((float *)((uint8_t *)base_addr + vindex.i32[ 1] * scale));
    r.f32[ 2] = *((float *)((uint8_t *)base_addr + vindex.i32[ 2] * scale));
    r.f32[ 3] = *((float *)((uint8_t *)base_addr + vindex.i32[ 3] * scale));
    r.f32[ 4] = *((float *)((uint8_t *)base_addr + vindex.i32[ 4] * scale));
    r.f32[ 5] = *((float *)((uint8_t *)base_addr + vindex.i32[ 5] * scale));
    r.f32[ 6] = *((float *)((uint8_t *)base_addr + vindex.i32[ 6] * scale));
    r.f32[ 7] = *((float *)((uint8_t *)base_addr + vindex.i32[ 7] * scale));
    r.f32[ 8] = *((float *)((uint8_t *)base_addr + vindex.i32[ 8] * scale));
    r.f32[ 9] = *((float *)((uint8_t *)base_addr + vindex.i32[ 9] * scale));
    r.f32[10] = *((float *)((uint8_t *)base_addr + vindex.i32[10] * scale));
    r.f32[11] = *((float *)((uint8_t *)base_addr + vindex.i32[11] * scale));
    r.f32[12] = *((float *)((uint8_t *)base_addr + vindex.i32[12] * scale));
    r.f32[13] = *((float *)((uint8_t *)base_addr + vindex.i32[13] * scale));
    r.f32[14] = *((float *)((uint8_t *)base_addr + vindex.i32[14] * scale));
    r.f32[15] = *((float *)((uint8_t *)base_addr + vindex.i32[15] * scale));
    return r;
  #else
    easysimd__m512i_private vindex_ = easysimd__m512i_to_private(vindex);
    easysimd__m512_private r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      float dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f32[i] = dst;
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm512_i32gather_ps(vindex, base_addr, scale) _mm512_i32gather_ps(vindex, base_addr, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i32gather_ps
  #define _mm512_i32gather_ps(vindex, base_addr, scale) easysimd_mm512_i32gather_ps(vindex, base_addr, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_i64gather_epi64(easysimd__m512i vindex, void const* base_addr, const int64_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    sveint64_t index0 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_0], scale);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_gather_s64offset_s64(pg, (const int64_t *)base_addr, index0);

    sveint64_t index1 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_1], scale);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_gather_s64offset_s64(pg, (const int64_t *)base_addr, index1);

    sveint64_t index2 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_2], scale);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svld1_gather_s64offset_s64(pg, (const int64_t *)base_addr, index2);

    sveint64_t index3 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_3], scale);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svld1_gather_s64offset_s64(pg, (const int64_t *)base_addr, index3);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.i64[ 0] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 0] * scale));
    r.i64[ 1] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 1] * scale));
    r.i64[ 2] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 2] * scale));
    r.i64[ 3] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 3] * scale));
    r.i64[ 4] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 4] * scale));
    r.i64[ 5] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 5] * scale));
    r.i64[ 6] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 6] * scale));
    r.i64[ 7] = *((int64_t *)((uint8_t *)base_addr + vindex.i64[ 7] * scale));
    return r;
  #else
    easysimd__m512i_private
      vindex_ = easysimd__m512i_to_private(vindex), r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int64_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i64[i] = dst;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_i64gather_epi64(vindex, base_addr, scale) _mm512_i64gather_epi64(vindex, base_addr, scale)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i64gather_epi64
  #define _mm512_i64gather_epi64(vindex, base_addr, scale) easysimd_mm512_i64gather_epi64(vindex, base_addr, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_i64gather_pd(easysimd__m512i vindex, void const* base_addr, const int64_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    svbool_t pg = svptrue_b64();
    sveint64_t index0 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_0], scale);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_gather_s64offset_f64(pg, (const double *)base_addr, index0);

    sveint64_t index1 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_1], scale);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_gather_s64offset_f64(pg, (const double *)base_addr, index1);

    sveint64_t index2 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_2], scale);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svld1_gather_s64offset_f64(pg, (const double *)base_addr, index2);

    sveint64_t index3 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_3], scale);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svld1_gather_s64offset_f64(pg, (const double *)base_addr, index3);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.f64[0] = *((double *)((uint8_t *)base_addr + vindex.i64[0] * scale));
    r.f64[1] = *((double *)((uint8_t *)base_addr + vindex.i64[1] * scale));
    r.f64[2] = *((double *)((uint8_t *)base_addr + vindex.i64[2] * scale));
    r.f64[3] = *((double *)((uint8_t *)base_addr + vindex.i64[3] * scale));
    r.f64[4] = *((double *)((uint8_t *)base_addr + vindex.i64[4] * scale));
    r.f64[5] = *((double *)((uint8_t *)base_addr + vindex.i64[5] * scale));
    r.f64[6] = *((double *)((uint8_t *)base_addr + vindex.i64[6] * scale));
    r.f64[7] = *((double *)((uint8_t *)base_addr + vindex.i64[7] * scale));
    return r;
  #else
    easysimd__m512i_private vindex_ = easysimd__m512i_to_private(vindex);
    easysimd__m512_private r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      double dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f64[i] = dst;
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm512_i64gather_pd(vindex, base_addr, scale) _mm512_i64gather_pd(vindex, base_addr, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i64gather_pd
  #define _mm512_i64gather_pd(vindex, base_addr, scale) easysimd_mm512_i64gather_pd(vindex, base_addr, scale)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_GATHER_H) */