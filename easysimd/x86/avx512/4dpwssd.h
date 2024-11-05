#if !defined(EASYSIMD_X86_AVX512_4DPWSSD_H)
#define EASYSIMD_X86_AVX512_4DPWSSD_H

#include "types.h"
#include "dpwssd.h"
#include "set1.h"
#include "mov.h"
#include "add.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_4dpwssd_epi32 (easysimd__m512i src, easysimd__m512i a0, easysimd__m512i a1, easysimd__m512i a2, easysimd__m512i a3, easysimd__m128i* b) {
  #if defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE)
    return _mm512_4dpwssd_epi32(src, a0, a1, a2, a3, b);
  #else
    easysimd__m128i_private bv = easysimd__m128i_to_private(easysimd_mm_loadu_epi32(b));
    easysimd__m512i r;

    r = easysimd_mm512_dpwssd_epi32(src, a0, easysimd_mm512_set1_epi32(bv.i32[0]));
    r = easysimd_mm512_add_epi32(easysimd_mm512_dpwssd_epi32(src, a1, easysimd_mm512_set1_epi32(bv.i32[1])), r);
    r = easysimd_mm512_add_epi32(easysimd_mm512_dpwssd_epi32(src, a2, easysimd_mm512_set1_epi32(bv.i32[2])), r);
    r = easysimd_mm512_add_epi32(easysimd_mm512_dpwssd_epi32(src, a3, easysimd_mm512_set1_epi32(bv.i32[3])), r);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX5124VNNIW_ENABLE_NATIVE_ALIASES)
  #undef easysimd_mm512_4dpwssd_epi32
  #define _mm512_4dpwssd_epi32(src, a0, a1, a2, a3, b) easysimd_mm512_4dpwssd_epi32(src, a0, a1, a2, a3, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_4dpwssd_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a0, easysimd__m512i a1, easysimd__m512i a2, easysimd__m512i a3, easysimd__m128i* b) {
  #if defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE)
    return _mm512_mask_4dpwssd_epi32(src, k, a0, a1, a2, a3, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_4dpwssd_epi32(src, a0, a1, a2, a3, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX5124VNNIW_ENABLE_NATIVE_ALIASES)
  #undef easysimd_mm512_mask_4dpwssd_epi32
  #define _mm512_mask_4dpwssd_epi32(src, k, a0, a1, a2, a3, b) easysimd_mm512_mask_4dpwssd_epi32(src, k, a0, a1, a2, a3, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_4dpwssd_epi32 (easysimd__mmask16 k, easysimd__m512i src, easysimd__m512i a0, easysimd__m512i a1, easysimd__m512i a2, easysimd__m512i a3, easysimd__m128i* b) {
  #if defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE)
    return _mm512_mask_4dpwssd_epi32(k, src, a0, a1, a2, a3, b);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_4dpwssd_epi32(src, a0, a1, a2, a3, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX5124VNNIW_ENABLE_NATIVE_ALIASES)
  #undef easysimd_mm512_maskz_4dpwssd_epi32
  #define _mm512_maskz_4dpwssd_epi32(k, src, a0, a1, a2, a3, b) easysimd_mm512_maskz_4dpwssd_epi32(k, src, a0, a1, a2, a3, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_dbsad_epu8(easysimd__m128i a, easysimd__m128i b, int imm8){
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i temp;
    svuint32_t svindex0 = svdupq_n_u32((imm8 >> 0) & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
    b.sve_u32 = svtbl_u32(b.sve_u32, svindex0);
    svuint8_t sva, svb;
    sveuint8_t svarr[4];
    svuint8_t svaindex = svdupq_n_u8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
    svuint8_t svbindex = svdupq_n_u8(0, 0, 1, 0, 2, 0, 3, 0, 8, 0, 9, 0, 10,  0, 11,  0);
    for(int i = 0; i < 4; i++){
      sva = svtbl_u8(a.sve_u8, svaindex);
      svb = svtbl_u8(b.sve_u8, svbindex);
      svarr[i] = svabd_u8_x(svptrue_b8(), sva, svb);
      svaindex = svadd_u8_x(svptrue_b8(), svaindex, svdup_n_u8(1));
      svbindex = svadd_u8_x(svptrue_b8(), svbindex, svdup_n_u8(1));
    }
    temp.sve_u16 = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[0], svarr[1]), svaddlb_u16(svarr[2], svarr[3]));
    return temp;
  #else
    easysimd__m128i_private
      r_,
      temp_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(b_.u32) / sizeof(b_.u32[0])) ; i++) {
      temp_.u32[i] = b_.u32[(imm8 >> (2 * i)) & 0x03];
    }
    #if !defined(easysimd_math_abs)
      #define easysimd_math_abs(v) __builtin_abs(v)
    #else
      int32_t index[4] = {0, 2, 8, 10};
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u16[  2 * i  ] = easysimd_math_abs((int16_t)(a_.u8[  4 * i  ] - temp_.u8[  index[i]  ])) + 
                            easysimd_math_abs((int16_t)(a_.u8[4 * i + 1] - temp_.u8[index[i] + 1])) + 
                            easysimd_math_abs((int16_t)(a_.u8[4 * i + 2] - temp_.u8[index[i] + 2])) + 
                            easysimd_math_abs((int16_t)(a_.u8[4 * i + 3] - temp_.u8[index[i] + 3]));
        r_.u16[2 * i + 1] = easysimd_math_abs((int16_t)(a_.u8[  4 * i  ] - temp_.u8[index[i] + 1])) + 
                            easysimd_math_abs((int16_t)(a_.u8[4 * i + 1] - temp_.u8[index[i] + 2])) + 
                            easysimd_math_abs((int16_t)(a_.u8[4 * i + 2] - temp_.u8[index[i] + 3])) + 
                            easysimd_math_abs((int16_t)(a_.u8[4 * i + 3] - temp_.u8[index[i] + 4]));
      }
    #endif
      return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_dbsad_epu8(a, b, imm8) _mm_dbsad_epu8(a, b, imm8); 
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_dbsad_epu8
  #define _mm_dbsad_epu8(a, b, imm8) easysimd_mm_dbsad_epu8(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_dbsad_epu8(easysimd__m256i a, easysimd__m256i b, int imm8){
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i temp;
    svuint32_t svindex0 = svdupq_n_u32((imm8 >> 0) & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
    b.sve_u32[EASYSIMD_SV_INDEX_0] = svtbl_u32(b.sve_u32[EASYSIMD_SV_INDEX_0], svindex0);
    b.sve_u32[EASYSIMD_SV_INDEX_1] = svtbl_u32(b.sve_u32[EASYSIMD_SV_INDEX_1], svindex0);
    svuint8_t sva, svb;
    sveuint8_t svarr[8];
    svuint8_t svaindex = svdupq_n_u8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
    svuint8_t svbindex = svdupq_n_u8(0, 0, 1, 0, 2, 0, 3, 0, 8, 0, 9, 0, 10,  0, 11,  0);
    for(int i = 0; i < 4; i++){
      sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svaindex);
      svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_0], svbindex);
      svarr[i] = svabd_u8_x(svptrue_b8(), sva, svb);
      sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svaindex);
      svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_1], svbindex);
      svarr[i + 4] = svabd_u8_x(svptrue_b8(), sva, svb);
      svaindex = svadd_u8_x(svptrue_b8(), svaindex, svdup_n_u8(1));
      svbindex = svadd_u8_x(svptrue_b8(), svbindex, svdup_n_u8(1));
    }
    temp.sve_u16[EASYSIMD_SV_INDEX_0] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[0], svarr[1]), svaddlb_u16(svarr[2], svarr[3]));
    temp.sve_u16[EASYSIMD_SV_INDEX_1] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[4], svarr[5]), svaddlb_u16(svarr[6], svarr[7]));
    return temp;

  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    r_.m128i[0] = easysimd_mm_dbsad_epu8(a_.m128i[0], b_.m128i[0], imm8);
    r_.m128i[1] = easysimd_mm_dbsad_epu8(a_.m128i[1], b_.m128i[1], imm8);
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_dbsad_epu8(a, b, imm8) _mm256_dbsad_epu8(a, b, imm8); 
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_dbsad_epu8
  #define _mm256_dbsad_epu8(a, b, imm8) easysimd_mm256_dbsad_epu8(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_dbsad_epu8(easysimd__m512i a, easysimd__m512i b, int imm8){
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i temp;
    svuint32_t svindex0 = svdupq_n_u32((imm8 >> 0) & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
    b.sve_u32[EASYSIMD_SV_INDEX_0] = svtbl_u32(b.sve_u32[EASYSIMD_SV_INDEX_0], svindex0);
    b.sve_u32[EASYSIMD_SV_INDEX_1] = svtbl_u32(b.sve_u32[EASYSIMD_SV_INDEX_1], svindex0);
    b.sve_u32[EASYSIMD_SV_INDEX_2] = svtbl_u32(b.sve_u32[EASYSIMD_SV_INDEX_2], svindex0);
    b.sve_u32[EASYSIMD_SV_INDEX_3] = svtbl_u32(b.sve_u32[EASYSIMD_SV_INDEX_3], svindex0);
    svuint8_t sva, svb;
    sveuint8_t svarr[16];
    svuint8_t svaindex = svdupq_n_u8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
    svuint8_t svbindex = svdupq_n_u8(0, 0, 1, 0, 2, 0, 3, 0, 8, 0, 9, 0, 10,  0, 11,  0);
    for(int i = 0; i < 4; i++){
      sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svaindex);
      svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_0], svbindex);
      svarr[i     ] = svabd_u8_x(svptrue_b8(), sva, svb);
      sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svaindex);
      svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_1], svbindex);
      svarr[i +  4] = svabd_u8_x(svptrue_b8(), sva, svb);
      sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_2], svaindex);
      svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_2], svbindex);
      svarr[i +  8] = svabd_u8_x(svptrue_b8(), sva, svb);
      sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_3], svaindex);
      svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_3], svbindex);
      svarr[i + 12] = svabd_u8_x(svptrue_b8(), sva, svb);

      svaindex = svadd_u8_x(svptrue_b8(), svaindex, svdup_n_u8(1));
      svbindex = svadd_u8_x(svptrue_b8(), svbindex, svdup_n_u8(1));
    }
    temp.sve_u16[EASYSIMD_SV_INDEX_0] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[ 0], svarr[ 1]), svaddlb_u16(svarr[ 2], svarr[ 3]));
    temp.sve_u16[EASYSIMD_SV_INDEX_1] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[ 4], svarr[ 5]), svaddlb_u16(svarr[ 6], svarr[ 7]));
    temp.sve_u16[EASYSIMD_SV_INDEX_2] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[ 8], svarr[ 9]), svaddlb_u16(svarr[10], svarr[11]));
    temp.sve_u16[EASYSIMD_SV_INDEX_3] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[12], svarr[13]), svaddlb_u16(svarr[14], svarr[15]));
    return temp;

  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    r_.m256i[0] = easysimd_mm256_dbsad_epu8(a_.m256i[0], b_.m256i[0], imm8);
    r_.m256i[1] = easysimd_mm256_dbsad_epu8(a_.m256i[1], b_.m256i[1], imm8);
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm512_dbsad_epu8(a, b, imm8) _mm512_dbsad_epu8(a, b, imm8); 
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_dbsad_epu8
  #define _mm512_dbsad_epu8(a, b, imm8) easysimd_mm512_dbsad_epu8(a, b, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_4DPWSSD_H) */
