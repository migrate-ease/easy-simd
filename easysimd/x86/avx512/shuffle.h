/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2020      Evan Nemerson <evan@nemerson.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_X86_AVX512_SHUFFLE_H)
#define EASYSIMD_X86_AVX512_SHUFFLE_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_shuffle_ps(easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;

  easysimd_svbool_t pgsel   = svdupq_n_b32(1, 1, 0, 0);
  svuint32_t svindexa = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, 0, 0);
  svuint32_t svindexb = svdupq_n_u32((imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03, 0, 0);

  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svsplice_f32(pgsel, svtbl_f32(a.sve_f32, svindexa), svtbl_f32(b.sve_f32, svindexb)), src.sve_f32);
  return r;
#else
    easysimd__m128_private
      r_,
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = a_.f32[(imm8 >> 0) & 3];
    r_.f32[1] = a_.f32[(imm8 >> 2) & 3];
    r_.f32[2] = b_.f32[(imm8 >> 4) & 3];
    r_.f32[3] = b_.f32[(imm8 >> 6) & 3];

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
#endif
  }
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_shuffle_ps
  #define _mm_mask_shuffle_ps(src, k, a, b, imm8) easysimd_mm_mask_shuffle_ps(src, k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_shuffle_ps(easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;

  easysimd_svbool_t pgsel   = svdupq_n_b32(1, 1, 0, 0);
  svuint32_t svindexa = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, 0, 0);
  svuint32_t svindexb = svdupq_n_u32((imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03, 0, 0);

  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svsplice_f32(pgsel, svtbl_f32(a.sve_f32, svindexa), svtbl_f32(b.sve_f32, svindexb)), svdup_n_f32(0.0));
  return r;
#else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = a_.f32[(imm8 >> 0) & 3];
    r_.f32[1] = a_.f32[(imm8 >> 2) & 3];
    r_.f32[2] = b_.f32[(imm8 >> 4) & 3];
    r_.f32[3] = b_.f32[(imm8 >> 6) & 3];

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
#endif
  }
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_shuffle_ps
  #define _mm_maskz_shuffle_ps(k, a, b, imm8) easysimd_mm_maskz_shuffle_ps(k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shuffle_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_shuffle_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_0], svdup_n_u8(0x8F)));
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_1], svdup_n_u8(0x8F)));
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_2], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_2], svdup_n_u8(0x8F)));
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_3], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_3], svdup_n_u8(0x8F)));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint8x16_t mask_and = vdupq_n_u8(0x8f);
    r.m128i[0].neon_u8 = vqtbl1q_u8(a.m128i[0].neon_u8, vandq_u8(b.m128i[0].neon_u8, mask_and));
    r.m128i[1].neon_u8 = vqtbl1q_u8(a.m128i[1].neon_u8, vandq_u8(b.m128i[1].neon_u8, mask_and));
    r.m128i[2].neon_u8 = vqtbl1q_u8(a.m128i[2].neon_u8, vandq_u8(b.m128i[2].neon_u8, mask_and));
    r.m128i[3].neon_u8 = vqtbl1q_u8(a.m128i[3].neon_u8, vandq_u8(b.m128i[3].neon_u8, mask_and));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
    for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
      r_.m256i[i] = easysimd_mm256_shuffle_epi8(a_.m256i[i], b_.m256i[i]);
    }
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = (b_.u8[i] & 0x80) ? 0 : a_.u8[(b_.u8[i] & 0x0f) + (i & 0x30)];
    }
  #endif

  return easysimd__m512i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shuffle_epi8
  #define _mm512_shuffle_epi8(a, b) easysimd_mm512_shuffle_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shuffle_epi32 (easysimd__m512i a, const int imm8) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_shuffle_epi32(a, imm8);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svuint32_t svindex = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svtbl_u32(a.sve_u32[EASYSIMD_SV_INDEX_0], svindex);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svtbl_u32(a.sve_u32[EASYSIMD_SV_INDEX_1], svindex);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svtbl_u32(a.sve_u32[EASYSIMD_SV_INDEX_2], svindex);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svtbl_u32(a.sve_u32[EASYSIMD_SV_INDEX_3], svindex);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i += 4) {
      r_.i32[i] = a_.i32[(imm8 & 0x03) + i];
      r_.i32[i + 1] = a_.i32[((imm8 >> 2) & 0x03) + i];
      r_.i32[i + 2] = a_.i32[((imm8 >> 4) & 0x03) + i];
      r_.i32[i + 3] = a_.i32[((imm8 >> 6) & 0x03) + i];
    }
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X326_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shuffle_epi32
  #define _mm512_shuffle_epi32(a, imm8) easysimd_mm512_shuffle_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_shuffle_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_shuffle_epi8(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_shuffle_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_shuffle_epi8
  #define _mm512_mask_shuffle_epi8(src, k, a, b) easysimd_mm512_mask_shuffle_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_shuffle_epi8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint8x8_t mk = vcreate_u8(k);
    uint8_t g_mask_epi8[16] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t mask_and = vld1q_u8(g_mask_epi8);

    easysimd__m512i tmp, r;
    tmp.m128i[0].neon_u8 = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 0), vdup_lane_u8(mk, 1)));
    tmp.m128i[1].neon_u8 = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 2), vdup_lane_u8(mk, 3)));
    tmp.m128i[2].neon_u8 = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 4), vdup_lane_u8(mk, 5)));
    tmp.m128i[3].neon_u8 = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 6), vdup_lane_u8(mk, 7)));
    mask_and = vdupq_n_u8(0x8f);
    r.m128i[0].neon_u8 = vqtbl1q_u8(a.m128i[0].neon_u8, vandq_u8(b.m128i[0].neon_u8, mask_and));
    r.m128i[1].neon_u8 = vqtbl1q_u8(a.m128i[1].neon_u8, vandq_u8(b.m128i[1].neon_u8, mask_and));
    r.m128i[2].neon_u8 = vqtbl1q_u8(a.m128i[2].neon_u8, vandq_u8(b.m128i[2].neon_u8, mask_and));
    r.m128i[3].neon_u8 = vqtbl1q_u8(a.m128i[3].neon_u8, vandq_u8(b.m128i[3].neon_u8, mask_and));
    r.m128i[0].neon_u8 = vminq_u8(r.m128i[0].neon_u8, tmp.m128i[0].neon_u8);
    r.m128i[1].neon_u8 = vminq_u8(r.m128i[1].neon_u8, tmp.m128i[1].neon_u8);
    r.m128i[2].neon_u8 = vminq_u8(r.m128i[2].neon_u8, tmp.m128i[2].neon_u8);
    r.m128i[3].neon_u8 = vminq_u8(r.m128i[3].neon_u8, tmp.m128i[3].neon_u8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0),
                                          svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_0], svdup_n_u8(0x8F))), svdup_n_u8(INT8_C(0)));
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1),
                                          svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_1], svdup_n_u8(0x8F))), svdup_n_u8(INT8_C(0)));
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2),
                                          svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_2], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_2], svdup_n_u8(0x8F))), svdup_n_u8(INT8_C(0)));
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3),
                                          svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_3], svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_3], svdup_n_u8(0x8F))), svdup_n_u8(INT8_C(0)));
    return r;
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_shuffle_epi8(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_shuffle_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shuffle_epi8
  #define _mm512_maskz_shuffle_epi8(k, a, b) easysimd_mm512_maskz_shuffle_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shuffle_i32x4 (easysimd__m256i a, easysimd__m256i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  r_.m128i[0] = a_.m128i[ imm8       & 1];
  r_.m128i[1] = b_.m128i[(imm8 >> 1) & 1];

  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_shuffle_i32x4(a, b, imm8) _mm256_shuffle_i32x4(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_i32x4
  #define _mm256_shuffle_i32x4(a, b, imm8) easysimd_mm256_shuffle_i32x4(a, b, imm8)
#endif

#define easysimd_mm256_maskz_shuffle_i32x4(k, a, b, imm8) easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_shuffle_i32x4(a, b, imm8))
#define easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, imm8) easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_shuffle_i32x4(a, b, imm8))

#define easysimd_mm256_shuffle_f32x4(a, b, imm8) easysimd_mm256_castsi256_ps(easysimd_mm256_shuffle_i32x4(easysimd_mm256_castps_si256(a), easysimd_mm256_castps_si256(b), imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_f32x4
  #define _mm256_shuffle_f32x4(a, b, imm8) easysimd_mm256_shuffle_f32x4(a, b, imm8)
#endif

#define easysimd_mm256_maskz_shuffle_f32x4(k, a, b, imm8) easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_shuffle_f32x4(a, b, imm8))
#define easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, imm8) easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_shuffle_f32x4(a, b, imm8))

#define easysimd_mm256_shuffle_i64x2(a, b, imm8) easysimd_mm256_shuffle_i32x4(a, b, imm8)
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_i64x2
  #define _mm256_shuffle_i64x2(a, b, imm8) easysimd_mm256_shuffle_i64x2(a, b, imm8)
#endif

#define easysimd_mm256_maskz_shuffle_i64x2(k, a, b, imm8) easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_shuffle_i64x2(a, b, imm8))
#define easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, imm8) easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_shuffle_i64x2(a, b, imm8))

#define easysimd_mm256_shuffle_f64x2(a, b, imm8) easysimd_mm256_castsi256_pd(easysimd_mm256_shuffle_i64x2(easysimd_mm256_castpd_si256(a), easysimd_mm256_castpd_si256(b), imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_f64x2
  #define _mm256_shuffle_f64x2(a, b, imm8) easysimd_mm256_shuffle_f64x2(a, b, imm8)
#endif

#define easysimd_mm256_maskz_shuffle_f64x2(k, a, b, imm8) easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_shuffle_f64x2(a, b, imm8))
#define easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, imm8) easysimd_mm256_mask_mov_pd(src, k, easysimd_mm256_shuffle_f64x2(a, b, imm8))

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_shuffle_i32x4 (easysimd__m512i a, easysimd__m512i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32[imm8 & 3];
    r.sve_i32[EASYSIMD_SV_INDEX_1] = a.sve_i32[(imm8 >> 2) & 3];
    r.sve_i32[EASYSIMD_SV_INDEX_2] = b.sve_i32[(imm8 >> 4) & 3];
    r.sve_i32[EASYSIMD_SV_INDEX_3] = b.sve_i32[(imm8 >> 6) & 3];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0] = a.m128i[ imm8       & 3];
    r.m128i[1] = a.m128i[(imm8 >> 2) & 3];
    r.m128i[2] = b.m128i[(imm8 >> 4) & 3];
    r.m128i[3] = b.m128i[(imm8 >> 6) & 3];
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    r_.m128i[0] = a_.m128i[ imm8       & 3];
    r_.m128i[1] = a_.m128i[(imm8 >> 2) & 3];
    r_.m128i[2] = b_.m128i[(imm8 >> 4) & 3];
    r_.m128i[3] = b_.m128i[(imm8 >> 6) & 3];
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_shuffle_i32x4(a, b, imm8) _mm512_shuffle_i32x4(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shuffle_i32x4
  #define _mm512_shuffle_i32x4(a, b, imm8) easysimd_mm512_shuffle_i32x4(a, b, imm8)
#endif

#define easysimd_mm512_maskz_shuffle_i32x4(k, a, b, imm8) easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_shuffle_i32x4(a, b, imm8))
#define easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, imm8) easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_shuffle_i32x4(a, b, imm8))

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_shuffle_f32x4 (easysimd__m512 a, easysimd__m512 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[imm8 & 3];
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[(imm8 >> 2) & 3];
    r.sve_f32[EASYSIMD_SV_INDEX_2] = b.sve_f32[(imm8 >> 4) & 3];
    r.sve_f32[EASYSIMD_SV_INDEX_3] = b.sve_f32[(imm8 >> 6) & 3];
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    r_.m128i[0] = a_.m128i[ imm8       & 3];
    r_.m128i[1] = a_.m128i[(imm8 >> 2) & 3];
    r_.m128i[2] = b_.m128i[(imm8 >> 4) & 3];
    r_.m128i[3] = b_.m128i[(imm8 >> 6) & 3];
    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_shuffle_f32x4(a, b, imm8) _mm512_shuffle_f32x4(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_shuffle_f32x4(a, b, imm8) easysimd_mm512_castsi512_ps(easysimd_mm512_shuffle_i32x4(easysimd_mm512_castps_si512(a), easysimd_mm512_castps_si512(b), imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shuffle_f32x4
  #define _mm512_shuffle_f32x4(a, b, imm8) easysimd_mm512_shuffle_f32x4(a, b, imm8)
#endif

#define easysimd_mm512_maskz_shuffle_f32x4(k, a, b, imm8) easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_shuffle_f32x4(a, b, imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shuffle_f32x4
  #define _mm512_maskz_shuffle_f32x4(k, a, b, imm8) easysimd_mm512_maskz_shuffle_f32x4(k, a, b, imm8)
#endif

#define easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, imm8) easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_shuffle_f32x4(a, b, imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_shuffle_f32x4
  #define _mm512_mask_shuffle_f32x4(src, k, a, b, imm8) easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, imm8)
#endif

#define easysimd_mm512_shuffle_i64x2(a, b, imm8) easysimd_mm512_shuffle_i32x4(a, b, imm8)
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shuffle_i64x2
  #define _mm512_shuffle_i64x2(a, b, imm8) easysimd_mm512_shuffle_i64x2(a, b, imm8)
#endif

#define easysimd_mm512_maskz_shuffle_i64x2(k, a, b, imm8) easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_shuffle_i64x2(a, b, imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shuffle_i64x2
  #define _mm512_maskz_shuffle_i64x2(k, a, b, imm8) easysimd_mm512_maskz_shuffle_i64x2(k, a, b, imm8)
#endif

#define easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, imm8) easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_shuffle_i64x2(a, b, imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_shuffle_i64x2
  #define _mm512_mask_shuffle_i64x2(src, k, a, b, imm8) easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, imm8)
#endif

#define easysimd_mm512_shuffle_f64x2(a, b, imm8) easysimd_mm512_castsi512_pd(easysimd_mm512_shuffle_i64x2(easysimd_mm512_castpd_si512(a), easysimd_mm512_castpd_si512(b), imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shuffle_f64x2
  #define _mm512_shuffle_f64x2(a, b, imm8) easysimd_mm512_shuffle_f64x2(a, b, imm8)
#endif

#define easysimd_mm512_maskz_shuffle_f64x2(k, a, b, imm8) easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_shuffle_f64x2(a, b, imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_shuffle_f64x2
  #define _mm512_maskz_shuffle_f64x2(k, a, b, imm8) easysimd_mm512_maskz_shuffle_f64x2(k, a, b, imm8)
#endif

#define easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, imm8) easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_shuffle_f64x2(a, b, imm8))
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_shuffle_f64x2
  #define _mm512_mask_shuffle_f64x2(src, k, a, b, imm8) easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_shuffle_ps(easysimd__m512 a, easysimd__m512 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE (imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t   pgsel   = svdupq_n_b32(1, 1, 0, 0);
    svuint32_t svindexa = svdupq_n_u32(imm8        & 0x03, (imm8 >> 2) & 0x03, 0, 0);
    svuint32_t svindexb = svdupq_n_u32((imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03, 0, 0);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsplice(pgsel, svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], svindexa), svtbl_f32(b.sve_f32[EASYSIMD_SV_INDEX_0], svindexb));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsplice(pgsel, svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], svindexa), svtbl_f32(b.sve_f32[EASYSIMD_SV_INDEX_1], svindexb));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsplice(pgsel, svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_2], svindexa), svtbl_f32(b.sve_f32[EASYSIMD_SV_INDEX_2], svindexb));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsplice(pgsel, svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_3], svindexa), svtbl_f32(b.sve_f32[EASYSIMD_SV_INDEX_3], svindexb));
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    for (size_t i = 0 ; i < (sizeof(r_.m128_private) / sizeof(r_.m128_private[0])) ; i++) {
      const size_t halfway = (sizeof(r_.m128_private[i].f32) / sizeof(r_.m128_private[i].f32[0]) / 2);
      EASYSIMD_VECTORIZE
      for (size_t j = 0 ; j < halfway ; j++) {
        r_.m128_private[i].f32[j] = a_.m128_private[i].f32[(imm8 >> (j * 2)) & 3];
        r_.m128_private[i].f32[halfway + j] = b_.m128_private[i].f32[(imm8 >> ((halfway + j) * 2)) & 3];
      }
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_shuffle_ps(a, b, imm8) _mm512_shuffle_ps(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_shuffle_ps(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512_private \
      easysimd_mm512_shuffle_ps_a_ = easysimd__m512_to_private(a), \
      easysimd_mm512_shuffle_ps_b_ = easysimd__m512_to_private(b); \
    \
    easysimd_mm512_shuffle_ps_a_.m256[0] = easysimd_mm256_shuffle_ps(easysimd_mm512_shuffle_ps_a_.m256[0], easysimd_mm512_shuffle_ps_b_.m256[0], imm8); \
    easysimd_mm512_shuffle_ps_a_.m256[1] = easysimd_mm256_shuffle_ps(easysimd_mm512_shuffle_ps_a_.m256[1], easysimd_mm512_shuffle_ps_b_.m256[1], imm8); \
    \
    easysimd__m512_from_private(easysimd_mm512_shuffle_ps_a_); \
  }))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm512_shuffle_ps(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m512_private \
      easysimd_mm512_shuffle_ps_a_ = easysimd__m512_to_private(a), \
      easysimd_mm512_shuffle_ps_b_ = easysimd__m512_to_private(b); \
    \
    easysimd_mm512_shuffle_ps_a_.f32 = \
      EASYSIMD_SHUFFLE_VECTOR_( \
        32, 64, \
        easysimd_mm512_shuffle_ps_a_.f32, \
        easysimd_mm512_shuffle_ps_b_.f32, \
        (((imm8)     ) & 3), \
        (((imm8) >> 2) & 3), \
        (((imm8) >> 4) & 3) + 16, \
        (((imm8) >> 6) & 3) + 16, \
        (((imm8)     ) & 3) + 4, \
        (((imm8) >> 2) & 3) + 4, \
        (((imm8) >> 4) & 3) + 20, \
        (((imm8) >> 6) & 3) + 20, \
        (((imm8)     ) & 3) + 8, \
        (((imm8) >> 2) & 3) + 8, \
        (((imm8) >> 4) & 3) + 24, \
        (((imm8) >> 6) & 3) + 24, \
        (((imm8)     ) & 3) + 12, \
        (((imm8) >> 2) & 3) + 12, \
        (((imm8) >> 4) & 3) + 28, \
        (((imm8) >> 6) & 3) + 28 \
      ); \
    \
    easysimd__m512_from_private(easysimd_mm512_shuffle_ps_a_); \
  }))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_shuffle_ps
  #define _mm512_shuffle_ps(a, b, imm8) easysimd_mm512_shuffle_ps(a, b, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_SHUFFLE_H) */
