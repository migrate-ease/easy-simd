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

#if !defined(EASYSIMD_X86_AVX512_PERMUTEXVAR_H)
#define EASYSIMD_X86_AVX512_PERMUTEXVAR_H

#include "types.h"
#include "and.h"
#include "andnot.h"
#include "blend.h"
#include "mov.h"
#include "or.h"
#include "set1.h"
#include "slli.h"
#include "srli.h"
#include "test.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_permute_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;

  svuint32_t svindex = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_f32(a.sve_f32, svindex), src.sve_f32);
  return r;
#else
  easysimd__m128_private
    r_,
    src_ = easysimd__m128_to_private(src),
    a_ = easysimd__m128_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? a_.f32[(imm8 >> ((i << 1) & 7)) & 3] : src_.f32[i];
  }

  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permute_ps
  #define _mm_mask_permute_ps(src, k, a, imm8) easysimd_mm_mask_permute_ps(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_permute_ps (easysimd__mmask8 k, easysimd__m128 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;

  svuint32_t svindex = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_f32(a.sve_f32, svindex), svdup_n_f32(0.0));
  return r;
#else
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? a_.f32[(imm8 >> ((i << 1) & 7)) & 3] : EASYSIMD_FLOAT32_C(0.0);
  }

  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permute_ps
  #define _mm_maskz_permute_ps(k, a, imm8) easysimd_mm_maskz_permute_ps(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_permute_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128d r;
  svuint64_t svindex = svdupq_n_u64((imm8 >> 0) & 1, (imm8 >> 1) & 1);
  r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svtbl_f64(a.sve_f64, svindex), src.sve_f64);
  return r;
#else
  easysimd__m128d_private
    r_,
    src_ = easysimd__m128d_to_private(src),
    a_ = easysimd__m128d_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = ((k >> i) & 1) ? a_.f64[(imm8 >> i) & 1] : src_.f64[i];
  }

  return easysimd__m128d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permute_pd
  #define _mm_mask_permute_pd(src, k, a, imm8) easysimd_mm_mask_permute_pd(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_permute_pd (easysimd__mmask8 k, easysimd__m128d a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128d r;
  svuint64_t svindex = svdupq_n_u64((imm8 >> 0) & 1, (imm8 >> 1) & 1);
  r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svtbl_f64(a.sve_f64, svindex), svdup_n_f64(0.0));
  return r;
#else
  easysimd__m128d_private
    r_,
    a_ = easysimd__m128d_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = ((k >> i) & 1) ? a_.f64[(imm8 >> i) & 1] : EASYSIMD_FLOAT64_C(0.0);
  }

  return easysimd__m128d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permute_pd
  #define _mm_maskz_permute_pd(k, a, imm8) easysimd_mm_maskz_permute_pd(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_permutevar_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128i b)
{
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  easysimd_svbool_t pg = svptrue_b32();

  svuint32_t svindex = svand_n_u32_z(pg, b.sve_u32, 3);
  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_f32(a.sve_f32, svindex), src.sve_f32);
  return r;
#else
  easysimd__m128_private
    r_,
    src_ = easysimd__m128_to_private(src),
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? a_.f32[b_.i32[i] & 3] : src_.f32[i];
  }

  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutevar_ps
  #define _mm_mask_permutevar_ps(src, k, a, b) easysimd_mm_mask_permutevar_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_permutevar_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128i b)
{
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  easysimd_svbool_t pg = svptrue_b32();

  svuint32_t svindex = svand_n_u32_z(pg, b.sve_u32, 3);
  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_f32(a.sve_f32, svindex), svdup_n_f32(0.0));
  return r;
#else
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? a_.f32[b_.i32[i] & 3] : EASYSIMD_FLOAT32_C(0.0);
  }

  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutevar_ps
  #define _mm_maskz_permutevar_ps(k, a, b) easysimd_mm_maskz_permutevar_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_permutexvar_epi16 (easysimd__m128i idx, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutexvar_epi16(idx, a);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
    easysimd__m128i mask16 = easysimd_mm_set1_epi16(0x0007);
    easysimd__m128i shift16 = easysimd_mm_set1_epi16(0x0202);
    easysimd__m128i byte_index16 = easysimd_mm_set1_epi16(0x0100);
    easysimd__m128i index16 = easysimd_mm_and_si128(idx, mask16);
    index16 = easysimd_mm_mullo_epi16(index16, shift16);
    index16 = easysimd_mm_add_epi16(index16, byte_index16);
    return easysimd_mm_shuffle_epi8(a, index16);
  #else
    easysimd__m128i_private
      idx_ = easysimd__m128i_to_private(idx),
      a_ = easysimd__m128i_to_private(a),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint16x8_t mask16 = vdupq_n_u16(0x0007);
      uint16x8_t byte_index16 = vdupq_n_u16(0x0100);
      uint16x8_t index16 = vandq_u16(idx_.neon_u16, mask16);
      index16 = vmulq_n_u16(index16, 0x0202);
      index16 = vaddq_u16(index16, byte_index16);
      r_.neon_u8 = vqtbl1q_u8(a_.neon_u8, vreinterpretq_u8_u16(index16));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[idx_.i16[i] & 0x07];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutexvar_epi16
  #define _mm_permutexvar_epi16(idx, a) easysimd_mm_permutexvar_epi16(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_permutexvar_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i idx, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutexvar_epi16(src, k, idx, a);
  #else
    return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_permutexvar_epi16(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutexvar_epi16
  #define _mm_mask_permutexvar_epi16(src, k, idx, a) easysimd_mm_mask_permutexvar_epi16(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_permutexvar_epi16 (easysimd__mmask8 k, easysimd__m128i idx, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutexvar_epi16(k, idx, a);
  #else
    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_permutexvar_epi16(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutexvar_epi16
  #define _mm_maskz_permutexvar_epi16(k, idx, a) easysimd_mm_maskz_permutexvar_epi16(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_permutexvar_epi8 (easysimd__m128i idx, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutexvar_epi8(idx, a);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
    easysimd__m128i mask = easysimd_mm_set1_epi8(0x0F);
    easysimd__m128i index = easysimd_mm_and_si128(idx, mask);
    return easysimd_mm_shuffle_epi8(a, index);
  #else
    easysimd__m128i_private
      idx_ = easysimd__m128i_to_private(idx),
      a_ = easysimd__m128i_to_private(a),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint8x16_t mask = vdupq_n_u8(0x0F);
      uint8x16_t index = vandq_u8(idx_.neon_u8, mask);
      r_.neon_u8 = vqtbl1q_u8(a_.neon_u8, index);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[idx_.i8[i] & 0x0F];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutexvar_epi8
  #define _mm_permutexvar_epi8(idx, a) easysimd_mm_permutexvar_epi8(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_permutexvar_epi8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i idx, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutexvar_epi8(src, k, idx, a);
  #else
    return easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_permutexvar_epi8(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutexvar_epi8
  #define _mm_mask_permutexvar_epi8(src, k, idx, a) easysimd_mm_mask_permutexvar_epi8(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_permutexvar_epi8 (easysimd__mmask16 k, easysimd__m128i idx, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutexvar_epi8(k, idx, a);
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_permutexvar_epi8(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutexvar_epi8
  #define _mm_maskz_permutexvar_epi8(k, idx, a) easysimd_mm_maskz_permutexvar_epi8(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutexvar_epi16 (easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutexvar_epi16(idx, a);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    easysimd__m256i mask16 = easysimd_mm256_set1_epi16(0x001F);
    easysimd__m256i shift16 = easysimd_mm256_set1_epi16(0x0202);
    easysimd__m256i byte_index16 = easysimd_mm256_set1_epi16(0x0100);
    easysimd__m256i index16 = easysimd_mm256_and_si256(idx, mask16);
    index16 = easysimd_mm256_mullo_epi16(index16, shift16);
    easysimd__m256i lo = easysimd_mm256_permute4x64_epi64(a, (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
    easysimd__m256i hi = easysimd_mm256_permute4x64_epi64(a, (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));
    easysimd__m256i select = easysimd_mm256_slli_epi64(index16, 3);
    index16 = easysimd_mm256_add_epi16(index16, byte_index16);
    lo = easysimd_mm256_shuffle_epi8(lo, index16);
    hi = easysimd_mm256_shuffle_epi8(hi, index16);
    return easysimd_mm256_blendv_epi8(lo, hi, select);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint8x16x2_t table = { { a.m128i[0].neon_u8, a.m128i[1].neon_u8 } };
    uint16x8_t mask16 = vdupq_n_u16(0x000F);
    uint16x8_t byte_index16 = vdupq_n_u16(0x0100);

    EASYSIMD_VECTORIZE
    for (size_t i = 0; i < (sizeof(a.m128i) / sizeof(a.m128i[0])); i++) {
      uint16x8_t index16 = vandq_u16(idx.m128i[i].neon_u16, mask16);
      index16 = vmulq_n_u16(index16, 0x0202);
      index16 = vaddq_u16(index16, byte_index16);
      a.m128i[i].neon_u8 = vqtbl2q_u8(table, vreinterpretq_u8_u16(index16));
    }
    return a;
  #else
    easysimd__m256i_private
      idx_ = easysimd__m256i_to_private(idx),
      a_ = easysimd__m256i_to_private(a),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = a_.i16[idx_.i16[i] & 0x0F];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutexvar_epi16
  #define _mm256_permutexvar_epi16(idx, a) easysimd_mm256_permutexvar_epi16(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutexvar_epi16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutexvar_epi16(src, k, idx, a);
  #else
    return easysimd_mm256_mask_mov_epi16(src, k, easysimd_mm256_permutexvar_epi16(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutexvar_epi16
  #define _mm256_mask_permutexvar_epi16(src, k, idx, a) easysimd_mm256_mask_permutexvar_epi16(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutexvar_epi16 (easysimd__mmask16 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutexvar_epi16(k, idx, a);
  #else
    return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_permutexvar_epi16(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutexvar_epi16
  #define _mm256_maskz_permutexvar_epi16(k, idx, a) easysimd_mm256_maskz_permutexvar_epi16(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutexvar_epi32 (easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutexvar_epi32(idx, a);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    return easysimd_mm256_permutevar8x32_epi32(a, idx);
  #else
    easysimd__m256i_private
      idx_ = easysimd__m256i_to_private(idx),
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint8x16x2_t table = { { a_.m128i_private[0].neon_u8,
                               a_.m128i_private[1].neon_u8 } };
      uint32x4_t mask32 = vdupq_n_u32(0x00000007);
      uint32x4_t byte_index32 = vdupq_n_u32(0x03020100);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m128i_private) / sizeof(r_.m128i_private[0])) ; i++) {
        uint32x4_t index32 = vandq_u32(idx_.m128i_private[i].neon_u32, mask32);
        index32 = vmulq_n_u32(index32, 0x04040404);
        index32 = vaddq_u32(index32, byte_index32);
        r_.m128i_private[i].neon_u8 = vqtbl2q_u8(table, vreinterpretq_u8_u32(index32));
      }
    #else
      #if !defined(__INTEL_COMPILER)
        EASYSIMD_VECTORIZE
      #endif
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[idx_.i32[i] & 0x07];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutexvar_epi32
  #define _mm256_permutexvar_epi32(idx, a) easysimd_mm256_permutexvar_epi32(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutexvar_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutexvar_epi32(src, k, idx, a);
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_permutexvar_epi32(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutexvar_epi32
  #define _mm256_mask_permutexvar_epi32(src, k, idx, a) easysimd_mm256_mask_permutexvar_epi32(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutexvar_epi32 (easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutexvar_epi32(k, idx, a);
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_permutexvar_epi32(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutexvar_epi32
  #define _mm256_maskz_permutexvar_epi32(k, idx, a) easysimd_mm256_maskz_permutexvar_epi32(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutexvar_epi64 (easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutexvar_epi64(idx, a);
  #else
    easysimd__m256i_private
      idx_ = easysimd__m256i_to_private(idx),
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[idx_.i64[i] & 3];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutexvar_epi64
  #define _mm256_permutexvar_epi64(idx, a) easysimd_mm256_permutexvar_epi64(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutexvar_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutexvar_epi64(src, k, idx, a);
  #else
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_permutexvar_epi64(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutexvar_epi64
  #define _mm256_mask_permutexvar_epi64(src, k, idx, a) easysimd_mm256_mask_permutexvar_epi64(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutexvar_epi64 (easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutexvar_epi64(k, idx, a);
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_permutexvar_epi64(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutexvar_epi64
  #define _mm256_maskz_permutexvar_epi64(k, idx, a) easysimd_mm256_maskz_permutexvar_epi64(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutexvar_epi8 (easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutexvar_epi8(idx, a);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    easysimd__m256i mask = easysimd_mm256_set1_epi8(0x0F);
    easysimd__m256i lo = easysimd_mm256_permute4x64_epi64(a, (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
    easysimd__m256i hi = easysimd_mm256_permute4x64_epi64(a, (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));
    easysimd__m256i index = easysimd_mm256_and_si256(idx, mask);
    easysimd__m256i select = easysimd_mm256_slli_epi64(idx, 3);
    lo = easysimd_mm256_shuffle_epi8(lo, index);
    hi = easysimd_mm256_shuffle_epi8(hi, index);
    return easysimd_mm256_blendv_epi8(lo, hi, select);
  #else
    easysimd__m256i_private
      idx_ = easysimd__m256i_to_private(idx),
      a_ = easysimd__m256i_to_private(a),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint8x16x2_t table = { { a_.m128i_private[0].neon_u8,
                               a_.m128i_private[1].neon_u8 } };
      uint8x16_t mask = vdupq_n_u8(0x1F);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m128i_private) / sizeof(r_.m128i_private[0])) ; i++) {
        r_.m128i_private[i].neon_u8 = vqtbl2q_u8(table, vandq_u8(idx_.m128i_private[i].neon_u8, mask));
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[idx_.i8[i] & 0x1F];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutexvar_epi8
  #define _mm256_permutexvar_epi8(idx, a) easysimd_mm256_permutexvar_epi8(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutexvar_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutexvar_epi8(src, k, idx, a);
  #else
    return easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_permutexvar_epi8(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutexvar_epi8
  #define _mm256_mask_permutexvar_epi8(src, k, idx, a) easysimd_mm256_mask_permutexvar_epi8(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutexvar_epi8 (easysimd__mmask32 k, easysimd__m256i idx, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutexvar_epi8(k, idx, a);
  #else
    return easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_permutexvar_epi8(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutexvar_epi8
  #define _mm256_maskz_permutexvar_epi8(k, idx, a) easysimd_mm256_maskz_permutexvar_epi8(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_permutexvar_pd (easysimd__m256i idx, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutexvar_pd(idx, a);
  #else
    return easysimd_mm256_castsi256_pd(easysimd_mm256_permutexvar_epi64(idx, easysimd_mm256_castpd_si256(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutexvar_pd
  #define _mm256_permutexvar_pd(idx, a) easysimd_mm256_permutexvar_pd(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_permutexvar_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutexvar_pd(src, k, idx, a);
  #else
    return easysimd_mm256_mask_mov_pd(src, k, easysimd_mm256_permutexvar_pd(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutexvar_pd
  #define _mm256_mask_permutexvar_pd(src, k, idx, a) easysimd_mm256_mask_permutexvar_pd(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_permutexvar_pd (easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutexvar_pd(k, idx, a);
  #else
    return easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_permutexvar_pd(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutexvar_pd
  #define _mm256_maskz_permutexvar_pd(k, idx, a) easysimd_mm256_maskz_permutexvar_pd(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_permutexvar_ps (easysimd__m256i idx, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutexvar_ps(idx, a);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    return easysimd_mm256_permutevar8x32_ps(a, idx);
  #else
    return easysimd_mm256_castsi256_ps(easysimd_mm256_permutexvar_epi32(idx, easysimd_mm256_castps_si256(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutexvar_ps
  #define _mm256_permutexvar_ps(idx, a) easysimd_mm256_permutexvar_ps(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_permutexvar_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutexvar_ps(src, k, idx, a);
  #else
    return easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_permutexvar_ps(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutexvar_ps
  #define _mm256_mask_permutexvar_ps(src, k, idx, a) easysimd_mm256_mask_permutexvar_ps(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_permutexvar_ps (easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutexvar_ps(k, idx, a);
  #else
    return easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_permutexvar_ps(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutexvar_ps
  #define _mm256_maskz_permutexvar_ps(k, idx, a) easysimd_mm256_maskz_permutexvar_ps(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutexvar_epi16 (easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_permutexvar_epi16(idx, a);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    easysimd__m512i_private
      idx_ = easysimd__m512i_to_private(idx),
      a_ = easysimd__m512i_to_private(a),
      r_;
    easysimd__m256i t0, t1, index, select, a01, a23;
    easysimd__m256i mask = easysimd_mm256_set1_epi16(0x001F);
    easysimd__m256i shift = easysimd_mm256_set1_epi16(0x0202);
    easysimd__m256i byte_index = easysimd_mm256_set1_epi16(0x0100);
    easysimd__m256i a0 = easysimd_mm256_broadcastsi128_si256(a_.m128i[0]);
    easysimd__m256i a1 = easysimd_mm256_broadcastsi128_si256(a_.m128i[1]);
    easysimd__m256i a2 = easysimd_mm256_broadcastsi128_si256(a_.m128i[2]);
    easysimd__m256i a3 = easysimd_mm256_broadcastsi128_si256(a_.m128i[3]);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.m256i_private) / sizeof(r_.m256i_private[0])) ; i++) {
      index = idx_.m256i[i];
      index = easysimd_mm256_and_si256(index, mask);
      index = easysimd_mm256_mullo_epi16(index, shift);
      index = easysimd_mm256_add_epi16(index, byte_index);
      t0 = easysimd_mm256_shuffle_epi8(a0, index);
      t1 = easysimd_mm256_shuffle_epi8(a1, index);
      select = easysimd_mm256_slli_epi64(index, 3);
      a01 = easysimd_mm256_blendv_epi8(t0, t1, select);
      t0 = easysimd_mm256_shuffle_epi8(a2, index);
      t1 = easysimd_mm256_shuffle_epi8(a3, index);
      a23 = easysimd_mm256_blendv_epi8(t0, t1, select);
      select = easysimd_mm256_slli_epi64(index, 2);
      r_.m256i[i] = easysimd_mm256_blendv_epi8(a01, a23, select);
    }
    return easysimd__m512i_from_private(r_);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    sveint16_t index;
    index = svand_n_s16_z(pg, idx.sve_i16[EASYSIMD_SV_INDEX_0], 0x1F);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svdupq_n_s16(a.i16[index[0]], a.i16[index[1]], a.i16[index[2]], a.i16[index[3]],
                                                a.i16[index[4]], a.i16[index[5]], a.i16[index[6]], a.i16[index[7]]);
    index = svand_n_s16_z(pg, idx.sve_i16[EASYSIMD_SV_INDEX_1], 0x1F);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svdupq_n_s16(a.i16[index[0]], a.i16[index[1]], a.i16[index[2]], a.i16[index[3]],
                                                a.i16[index[4]], a.i16[index[5]], a.i16[index[6]], a.i16[index[7]]);
    index = svand_n_s16_z(pg, idx.sve_i16[EASYSIMD_SV_INDEX_2], 0x1F);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svdupq_n_s16(a.i16[index[0]], a.i16[index[1]], a.i16[index[2]], a.i16[index[3]],
                                                a.i16[index[4]], a.i16[index[5]], a.i16[index[6]], a.i16[index[7]]);
    index = svand_n_s16_z(pg, idx.sve_i16[EASYSIMD_SV_INDEX_3], 0x1F);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svdupq_n_s16(a.i16[index[0]], a.i16[index[1]], a.i16[index[2]], a.i16[index[3]],
                                                a.i16[index[4]], a.i16[index[5]], a.i16[index[6]], a.i16[index[7]]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint16x8_t mask16 = vdupq_n_u16(0x001F);
    uint16x8_t byte_index16 = vdupq_n_u16(0x0100);
    uint16x8_t index16[4];

    index16[0] = vaddq_u16(vmulq_n_u16(vandq_u16(idx.m128i[0].neon_u16, mask16), 0x0202), byte_index16);
    index16[1] = vaddq_u16(vmulq_n_u16(vandq_u16(idx.m128i[1].neon_u16, mask16), 0x0202), byte_index16);
    index16[2] = vaddq_u16(vmulq_n_u16(vandq_u16(idx.m128i[2].neon_u16, mask16), 0x0202), byte_index16);
    index16[3] = vaddq_u16(vmulq_n_u16(vandq_u16(idx.m128i[3].neon_u16, mask16), 0x0202), byte_index16);

    r.m128i[0].neon_u8 = vqtbl4q_u8(a.neon_u8x4, vreinterpretq_u8_u16(index16[0]));
    r.m128i[1].neon_u8 = vqtbl4q_u8(a.neon_u8x4, vreinterpretq_u8_u16(index16[1]));
    r.m128i[2].neon_u8 = vqtbl4q_u8(a.neon_u8x4, vreinterpretq_u8_u16(index16[2]));
    r.m128i[3].neon_u8 = vqtbl4q_u8(a.neon_u8x4, vreinterpretq_u8_u16(index16[3]));

    return r;
  #else
    easysimd__m512i_private
      idx_ = easysimd__m512i_to_private(idx),
      a_ = easysimd__m512i_to_private(a),
      r_;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = a_.i16[idx_.i16[i] & 0x1F];
    }
    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutexvar_epi16
  #define _mm512_permutexvar_epi16(idx, a) easysimd_mm512_permutexvar_epi16(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutexvar_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_permutexvar_epi16(src, k, idx, a);
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_permutexvar_epi16(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutexvar_epi16
  #define _mm512_mask_permutexvar_epi16(src, k, idx, a) easysimd_mm512_mask_permutexvar_epi16(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutexvar_epi16 (easysimd__mmask32 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_permutexvar_epi16(k, idx, a);
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_permutexvar_epi16(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutexvar_epi16
  #define _mm512_maskz_permutexvar_epi16(k, idx, a) easysimd_mm512_maskz_permutexvar_epi16(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutexvar_epi32 (easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_permutexvar_epi32(idx, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    sveint32_t index;
    index = svand_n_s32_z(pg, idx.sve_i32[EASYSIMD_SV_INDEX_0], 0x0F);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_gather_s32index_s32(pg, &(a.i32[0]), index);
    index = svand_n_s32_z(pg, idx.sve_i32[EASYSIMD_SV_INDEX_1], 0x0F);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_gather_s32index_s32(pg, &(a.i32[0]), index);
    index = svand_n_s32_z(pg, idx.sve_i32[EASYSIMD_SV_INDEX_2], 0x0F);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svld1_gather_s32index_s32(pg, &(a.i32[0]), index);
    index = svand_n_s32_z(pg, idx.sve_i32[EASYSIMD_SV_INDEX_3], 0x0F);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svld1_gather_s32index_s32(pg, &(a.i32[0]), index);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE_NOT)
    easysimd__m512i r = easysimd_mm512_setzero_si512();
    int32x4_t low4bit = vdupq_n_s32(0x0f);
    idx.m128i[0].neon_i32 = vandq_s32(idx.m128i[0].neon_i32, low4bit);
    idx.m128i[1].neon_i32 = vandq_s32(idx.m128i[1].neon_i32, low4bit);
    idx.m128i[2].neon_i32 = vandq_s32(idx.m128i[2].neon_i32, low4bit);
    idx.m128i[3].neon_i32 = vandq_s32(idx.m128i[3].neon_i32, low4bit);
    int32_t p_a[16], p_i[16];
    vst1q_s32(p_a, a.m128i[0].neon_i32);
    vst1q_s32(p_a + 4, a.m128i[1].neon_i32);
    vst1q_s32(p_a + 8, a.m128i[2].neon_i32);
    vst1q_s32(p_a + 12, a.m128i[3].neon_i32);
    vst1q_s32(p_i, idx.m128i[0].neon_i32);
    vst1q_s32(p_i + 4, idx.m128i[1].neon_i32);
    vst1q_s32(p_i + 8, idx.m128i[2].neon_i32);
    vst1q_s32(p_i + 12, idx.m128i[3].neon_i32);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[p_i[0]], r.m128i[0].neon_i32, 0);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[p_i[1]], r.m128i[0].neon_i32, 1);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[p_i[2]], r.m128i[0].neon_i32, 2);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[p_i[3]], r.m128i[0].neon_i32, 3);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[p_i[4]], r.m128i[1].neon_i32, 0);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[p_i[5]], r.m128i[1].neon_i32, 1);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[p_i[6]], r.m128i[1].neon_i32, 2);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[p_i[7]], r.m128i[1].neon_i32, 3);
    r.m128i[2].neon_i32 = vsetq_lane_s32(p_a[p_i[8]], r.m128i[2].neon_i32, 0);
    r.m128i[2].neon_i32 = vsetq_lane_s32(p_a[p_i[9]], r.m128i[2].neon_i32, 1);
    r.m128i[2].neon_i32 = vsetq_lane_s32(p_a[p_i[10]], r.m128i[2].neon_i32, 2);
    r.m128i[2].neon_i32 = vsetq_lane_s32(p_a[p_i[11]], r.m128i[2].neon_i32, 3);
    r.m128i[3].neon_i32 = vsetq_lane_s32(p_a[p_i[12]], r.m128i[3].neon_i32, 0);
    r.m128i[3].neon_i32 = vsetq_lane_s32(p_a[p_i[13]], r.m128i[3].neon_i32, 1);
    r.m128i[3].neon_i32 = vsetq_lane_s32(p_a[p_i[14]], r.m128i[3].neon_i32, 2);
    r.m128i[3].neon_i32 = vsetq_lane_s32(p_a[p_i[15]], r.m128i[3].neon_i32, 3);
    return r;
  #else
    easysimd__m512i_private
      idx_ = easysimd__m512i_to_private(idx),
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      easysimd__m256i index, r0, r1, select;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i_private) / sizeof(r_.m256i_private[0])) ; i++) {
        index = idx_.m256i[i];
        r0 = easysimd_mm256_permutevar8x32_epi32(a_.m256i[0], index);
        r1 = easysimd_mm256_permutevar8x32_epi32(a_.m256i[1], index);
        select = easysimd_mm256_slli_epi32(index, 28);
        r_.m256i[i] = easysimd_mm256_castps_si256(easysimd_mm256_blendv_ps(easysimd_mm256_castsi256_ps(r0),
                                                                     easysimd_mm256_castsi256_ps(r1),
                                                                     easysimd_mm256_castsi256_ps(select)));
      }
    #else
      #if !defined(__INTEL_COMPILER)
        EASYSIMD_VECTORIZE
      #endif
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[idx_.i32[i] & 0x0F];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutexvar_epi32
  #define _mm512_permutexvar_epi32(idx, a) easysimd_mm512_permutexvar_epi32(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutexvar_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutexvar_epi32(src, k, idx, a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r = easysimd_mm512_setzero_si512();
    int32x4_t low4bit = vdupq_n_s32(0x0F);
    idx.m128i[0].neon_i32 = vandq_s32(idx.m128i[0].neon_i32, low4bit);
    idx.m128i[1].neon_i32 = vandq_s32(idx.m128i[1].neon_i32, low4bit);
    idx.m128i[2].neon_i32 = vandq_s32(idx.m128i[2].neon_i32, low4bit);
    idx.m128i[3].neon_i32 = vandq_s32(idx.m128i[3].neon_i32, low4bit);
    int32_t p_a[16], p_i[16];
    vst1q_s32(p_a     , a.m128i[0].neon_i32);
    vst1q_s32(p_a +  4, a.m128i[1].neon_i32);
    vst1q_s32(p_a +  8, a.m128i[2].neon_i32);
    vst1q_s32(p_a + 12, a.m128i[3].neon_i32);
    vst1q_s32(p_i     , idx.m128i[0].neon_i32);
    vst1q_s32(p_i +  4, idx.m128i[1].neon_i32);
    vst1q_s32(p_i +  8, idx.m128i[2].neon_i32);
    vst1q_s32(p_i + 12, idx.m128i[3].neon_i32);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  0) & 0x01 ? p_a[p_i[ 0]] : src.m128i[0].neon_i32[0], r.m128i[0].neon_i32, 0);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  1) & 0x01 ? p_a[p_i[ 1]] : src.m128i[0].neon_i32[1], r.m128i[0].neon_i32, 1);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  2) & 0x01 ? p_a[p_i[ 2]] : src.m128i[0].neon_i32[2], r.m128i[0].neon_i32, 2);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  3) & 0x01 ? p_a[p_i[ 3]] : src.m128i[0].neon_i32[3], r.m128i[0].neon_i32, 3);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  4) & 0x01 ? p_a[p_i[ 4]] : src.m128i[1].neon_i32[0], r.m128i[1].neon_i32, 0);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  5) & 0x01 ? p_a[p_i[ 5]] : src.m128i[1].neon_i32[1], r.m128i[1].neon_i32, 1);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  6) & 0x01 ? p_a[p_i[ 6]] : src.m128i[1].neon_i32[2], r.m128i[1].neon_i32, 2);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  7) & 0x01 ? p_a[p_i[ 7]] : src.m128i[1].neon_i32[3], r.m128i[1].neon_i32, 3);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >>  8) & 0x01 ? p_a[p_i[ 8]] : src.m128i[2].neon_i32[0], r.m128i[2].neon_i32, 0);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >>  9) & 0x01 ? p_a[p_i[ 9]] : src.m128i[2].neon_i32[1], r.m128i[2].neon_i32, 1);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >> 10) & 0x01 ? p_a[p_i[10]] : src.m128i[2].neon_i32[2], r.m128i[2].neon_i32, 2);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >> 11) & 0x01 ? p_a[p_i[11]] : src.m128i[2].neon_i32[3], r.m128i[2].neon_i32, 3);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 12) & 0x01 ? p_a[p_i[12]] : src.m128i[3].neon_i32[0], r.m128i[3].neon_i32, 0);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 13) & 0x01 ? p_a[p_i[13]] : src.m128i[3].neon_i32[1], r.m128i[3].neon_i32, 1);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 14) & 0x01 ? p_a[p_i[14]] : src.m128i[3].neon_i32[2], r.m128i[3].neon_i32, 2);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 15) & 0x01 ? p_a[p_i[15]] : src.m128i[3].neon_i32[3], r.m128i[3].neon_i32, 3);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_permutexvar_epi32(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutexvar_epi32
  #define _mm512_mask_permutexvar_epi32(src, k, idx, a) easysimd_mm512_mask_permutexvar_epi32(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutexvar_epi32 (easysimd__mmask16 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutexvar_epi32(k, idx, a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r = easysimd_mm512_setzero_si512();
    int32x4_t low4bit = vdupq_n_s32(0x0F);
    idx.m128i[0].neon_i32 = vandq_s32(idx.m128i[0].neon_i32, low4bit);
    idx.m128i[1].neon_i32 = vandq_s32(idx.m128i[1].neon_i32, low4bit);
    idx.m128i[2].neon_i32 = vandq_s32(idx.m128i[2].neon_i32, low4bit);
    idx.m128i[3].neon_i32 = vandq_s32(idx.m128i[3].neon_i32, low4bit);
    int32_t p_a[16], p_i[16];
    vst1q_s32(p_a     , a.m128i[0].neon_i32);
    vst1q_s32(p_a +  4, a.m128i[1].neon_i32);
    vst1q_s32(p_a +  8, a.m128i[2].neon_i32);
    vst1q_s32(p_a + 12, a.m128i[3].neon_i32);
    vst1q_s32(p_i     , idx.m128i[0].neon_i32);
    vst1q_s32(p_i +  4, idx.m128i[1].neon_i32);
    vst1q_s32(p_i +  8, idx.m128i[2].neon_i32);
    vst1q_s32(p_i + 12, idx.m128i[3].neon_i32);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  0) & 0x01 ? p_a[p_i[ 0]] : 0, r.m128i[0].neon_i32, 0);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  1) & 0x01 ? p_a[p_i[ 1]] : 0, r.m128i[0].neon_i32, 1);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  2) & 0x01 ? p_a[p_i[ 2]] : 0, r.m128i[0].neon_i32, 2);
    r.m128i[0].neon_i32 = vsetq_lane_s32((k >>  3) & 0x01 ? p_a[p_i[ 3]] : 0, r.m128i[0].neon_i32, 3);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  4) & 0x01 ? p_a[p_i[ 4]] : 0, r.m128i[1].neon_i32, 0);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  5) & 0x01 ? p_a[p_i[ 5]] : 0, r.m128i[1].neon_i32, 1);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  6) & 0x01 ? p_a[p_i[ 6]] : 0, r.m128i[1].neon_i32, 2);
    r.m128i[1].neon_i32 = vsetq_lane_s32((k >>  7) & 0x01 ? p_a[p_i[ 7]] : 0, r.m128i[1].neon_i32, 3);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >>  8) & 0x01 ? p_a[p_i[ 8]] : 0, r.m128i[2].neon_i32, 0);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >>  9) & 0x01 ? p_a[p_i[ 9]] : 0, r.m128i[2].neon_i32, 1);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >> 10) & 0x01 ? p_a[p_i[10]] : 0, r.m128i[2].neon_i32, 2);
    r.m128i[2].neon_i32 = vsetq_lane_s32((k >> 11) & 0x01 ? p_a[p_i[11]] : 0, r.m128i[2].neon_i32, 3);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 12) & 0x01 ? p_a[p_i[12]] : 0, r.m128i[3].neon_i32, 0);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 13) & 0x01 ? p_a[p_i[13]] : 0, r.m128i[3].neon_i32, 1);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 14) & 0x01 ? p_a[p_i[14]] : 0, r.m128i[3].neon_i32, 2);
    r.m128i[3].neon_i32 = vsetq_lane_s32((k >> 15) & 0x01 ? p_a[p_i[15]] : 0, r.m128i[3].neon_i32, 3);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_permutexvar_epi32(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutexvar_epi32
  #define _mm512_maskz_permutexvar_epi32(k, idx, a) easysimd_mm512_maskz_permutexvar_epi32(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutexvar_epi64 (easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_permutexvar_epi64(idx, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    sveint64_t index;
    index = svand_n_s64_z(pg, idx.sve_i64[EASYSIMD_SV_INDEX_0], 7);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(a.i64[index[0]], a.i64[index[1]]);
    index = svand_n_s64_z(pg, idx.sve_i64[EASYSIMD_SV_INDEX_1], 7);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdupq_n_s64(a.i64[index[0]], a.i64[index[1]]);
    index = svand_n_s64_z(pg, idx.sve_i64[EASYSIMD_SV_INDEX_2], 7);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svdupq_n_s64(a.i64[index[0]], a.i64[index[1]]);
    index = svand_n_s64_z(pg, idx.sve_i64[EASYSIMD_SV_INDEX_3], 7);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svdupq_n_s64(a.i64[index[0]], a.i64[index[1]]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r = easysimd_mm512_setzero_si512();
    int64x2_t low3bit = vdupq_n_s64(0x07);
    idx.m128i[0].neon_i64 = vandq_s64(idx.m128i[0].neon_i64, low3bit);
    idx.m128i[1].neon_i64 = vandq_s64(idx.m128i[1].neon_i64, low3bit);
    idx.m128i[2].neon_i64 = vandq_s64(idx.m128i[2].neon_i64, low3bit);
    idx.m128i[3].neon_i64 = vandq_s64(idx.m128i[3].neon_i64, low3bit);
    int64_t p_a[8], p_i[8];
    vst1q_s64(p_a, a.m128i[0].neon_i64);
    vst1q_s64(p_a + 2, a.m128i[1].neon_i64);
    vst1q_s64(p_a + 4, a.m128i[2].neon_i64);
    vst1q_s64(p_a + 6, a.m128i[3].neon_i64);
    vst1q_s64(p_i, idx.m128i[0].neon_i64);
    vst1q_s64(p_i + 2, idx.m128i[1].neon_i64);
    vst1q_s64(p_i + 4, idx.m128i[2].neon_i64);
    vst1q_s64(p_i + 6, idx.m128i[3].neon_i64);
    r.m128i[0].neon_i64 = vsetq_lane_s64(p_a[p_i[0]], r.m128i[0].neon_i64, 0);
    r.m128i[0].neon_i64 = vsetq_lane_s64(p_a[p_i[1]], r.m128i[0].neon_i64, 1);
    r.m128i[1].neon_i64 = vsetq_lane_s64(p_a[p_i[2]], r.m128i[1].neon_i64, 0);
    r.m128i[1].neon_i64 = vsetq_lane_s64(p_a[p_i[3]], r.m128i[1].neon_i64, 1);
    r.m128i[2].neon_i64 = vsetq_lane_s64(p_a[p_i[4]], r.m128i[2].neon_i64, 0);
    r.m128i[2].neon_i64 = vsetq_lane_s64(p_a[p_i[5]], r.m128i[2].neon_i64, 1);
    r.m128i[3].neon_i64 = vsetq_lane_s64(p_a[p_i[6]], r.m128i[3].neon_i64, 0);
    r.m128i[3].neon_i64 = vsetq_lane_s64(p_a[p_i[7]], r.m128i[3].neon_i64, 1);
    return r;
  #else
    easysimd__m512i_private
      idx_ = easysimd__m512i_to_private(idx),
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[idx_.i64[i] & 7];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutexvar_epi64
  #define _mm512_permutexvar_epi64(idx, a) easysimd_mm512_permutexvar_epi64(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutexvar_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutexvar_epi64(src, k, idx, a);
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_permutexvar_epi64(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutexvar_epi64
  #define _mm512_mask_permutexvar_epi64(src, k, idx, a) easysimd_mm512_mask_permutexvar_epi64(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutexvar_epi64 (easysimd__mmask8 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutexvar_epi64(k, idx, a);
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_permutexvar_epi64(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutexvar_epi64
  #define _mm512_maskz_permutexvar_epi64(k, idx, a) easysimd_mm512_maskz_permutexvar_epi64(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutexvar_epi8 (easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_permutexvar_epi8(idx, a);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    easysimd__m512i hilo, hi, lo, hi2, lo2, idx2;
    easysimd__m512i ones = easysimd_mm512_set1_epi8(1);
    easysimd__m512i low_bytes = easysimd_mm512_set1_epi16(0x00FF);

    idx2 = easysimd_mm512_srli_epi16(idx, 1);
    hilo = easysimd_mm512_permutexvar_epi16(idx2, a);
    easysimd__mmask64 mask = easysimd_mm512_test_epi8_mask(idx, ones);
    lo = easysimd_mm512_and_si512(hilo, low_bytes);
    hi = easysimd_mm512_srli_epi16(hilo, 8);

    idx2 = easysimd_mm512_srli_epi16(idx, 9);
    hilo = easysimd_mm512_permutexvar_epi16(idx2, a);
    lo2 = easysimd_mm512_slli_epi16(hilo, 8);
    hi2 = easysimd_mm512_andnot_si512(low_bytes, hilo);

    lo = easysimd_mm512_or_si512(lo, lo2);
    hi = easysimd_mm512_or_si512(hi, hi2);

    return easysimd_mm512_mask_blend_epi8(mask, lo, hi);
  #else
    easysimd__m512i_private
      idx_ = easysimd__m512i_to_private(idx),
      a_ = easysimd__m512i_to_private(a),
      r_;

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      easysimd__m256i t0, t1, index, select, a01, a23;
      easysimd__m256i mask = easysimd_mm256_set1_epi8(0x3F);
      easysimd__m256i a0 = easysimd_mm256_broadcastsi128_si256(a_.m128i[0]);
      easysimd__m256i a1 = easysimd_mm256_broadcastsi128_si256(a_.m128i[1]);
      easysimd__m256i a2 = easysimd_mm256_broadcastsi128_si256(a_.m128i[2]);
      easysimd__m256i a3 = easysimd_mm256_broadcastsi128_si256(a_.m128i[3]);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i_private) / sizeof(r_.m256i_private[0])) ; i++) {
        index = idx_.m256i[i];
        index = easysimd_mm256_and_si256(index, mask);
        select = easysimd_mm256_slli_epi64(index, 3);
        t0 = easysimd_mm256_shuffle_epi8(a0, index);
        t1 = easysimd_mm256_shuffle_epi8(a1, index);
        a01 = easysimd_mm256_blendv_epi8(t0, t1, select);
        t0 = easysimd_mm256_shuffle_epi8(a2, index);
        t1 = easysimd_mm256_shuffle_epi8(a3, index);
        a23 = easysimd_mm256_blendv_epi8(t0, t1, select);
        select = easysimd_mm256_slli_epi64(index, 2);
        r_.m256i[i] = easysimd_mm256_blendv_epi8(a01, a23, select);
      }
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint8x16x4_t table = { { a_.m128i_private[0].neon_u8,
                               a_.m128i_private[1].neon_u8,
                               a_.m128i_private[2].neon_u8,
                               a_.m128i_private[3].neon_u8 } };
      uint8x16_t mask = vdupq_n_u8(0x3F);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m128i_private) / sizeof(r_.m128i_private[0])) ; i++) {
        r_.m128i_private[i].neon_u8 = vqtbl4q_u8(table, vandq_u8(idx_.m128i_private[i].neon_u8, mask));
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[idx_.i8[i] & 0x3F];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutexvar_epi8
  #define _mm512_permutexvar_epi8(idx, a) easysimd_mm512_permutexvar_epi8(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutexvar_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_mask_permutexvar_epi8(src, k, idx, a);
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_permutexvar_epi8(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutexvar_epi8
  #define _mm512_mask_permutexvar_epi8(src, k, idx, a) easysimd_mm512_mask_permutexvar_epi8(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutexvar_epi8 (easysimd__mmask64 k, easysimd__m512i idx, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_maskz_permutexvar_epi8(k, idx, a);
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_permutexvar_epi8(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutexvar_epi8
  #define _mm512_maskz_permutexvar_epi8(k, idx, a) easysimd_mm512_maskz_permutexvar_epi8(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_permutexvar_pd (easysimd__m512i idx, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_permutexvar_pd(idx, a);
  #else
    return easysimd_mm512_castsi512_pd(easysimd_mm512_permutexvar_epi64(idx, easysimd_mm512_castpd_si512(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutexvar_pd
  #define _mm512_permutexvar_pd(idx, a) easysimd_mm512_permutexvar_pd(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_permutexvar_pd (easysimd__m512d src, easysimd__mmask8 k, easysimd__m512i idx, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutexvar_pd(src, k, idx, a);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_permutexvar_pd(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutexvar_pd
  #define _mm512_mask_permutexvar_pd(src, k, idx, a) easysimd_mm512_mask_permutexvar_pd(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_permutexvar_pd (easysimd__mmask8 k, easysimd__m512i idx, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutexvar_pd(k, idx, a);
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_permutexvar_pd(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutexvar_pd
  #define _mm512_maskz_permutexvar_pd(k, idx, a) easysimd_mm512_maskz_permutexvar_pd(k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_permutexvar_ps (easysimd__m512i idx, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_permutexvar_ps(idx, a);
  #else
    return easysimd_mm512_castsi512_ps(easysimd_mm512_permutexvar_epi32(idx, easysimd_mm512_castps_si512(a)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutexvar_ps
  #define _mm512_permutexvar_ps(idx, a) easysimd_mm512_permutexvar_ps(idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_permutexvar_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512i idx, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutexvar_ps(src, k, idx, a);
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_permutexvar_ps(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutexvar_ps
  #define _mm512_mask_permutexvar_ps(src, k, idx, a) easysimd_mm512_mask_permutexvar_ps(src, k, idx, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_permutexvar_ps (easysimd__mmask16 k, easysimd__m512i idx, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutexvar_ps(k, idx, a);
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_permutexvar_ps(idx, a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutexvar_ps
  #define _mm512_maskz_permutexvar_ps(k, idx, a) easysimd_mm512_maskz_permutexvar_ps(k, idx, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_PERMUTEXVAR_H) */
