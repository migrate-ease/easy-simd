/* Permission is hereby granted, free of charge, to any person
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
 *   2020-2021 Christopher Moore <moore@free.fr>
 *   2020      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_X86_GFNI_H)
#define EASYSIMD_X86_GFNI_H

#include "avx512/add.h"
#include "avx512/and.h"
#include "avx512/broadcast.h"
#include "avx512/cmpeq.h"
#include "avx512/cmpge.h"
#include "avx512/cmpgt.h"
#include "avx512/cmplt.h"
#include "avx512/extract.h"
#include "avx512/insert.h"
#include "avx512/kshift.h"
#include "avx512/mov.h"
#include "avx512/mov_mask.h"
#include "avx512/permutex2var.h"
#include "avx512/set.h"
#include "avx512/set1.h"
#include "avx512/setzero.h"
#include "avx512/shuffle.h"
#include "avx512/srli.h"
#include "avx512/test.h"
#include "avx512/xor.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

/* In all the *gf2p8affine* intrinsics the argument b must be a compile-time constant so we must use macros and easysimd_x_mm* helper functions */

/* N.B. The _mm*gf2p8affineinv_epi64_epi8 and _mm*gf2p8mul_epi8 intrinsics are for a Field Generator Polynomial (FGP) (aka reduction polynomial) of 0x11B */
/* Only the _mm*gf2p8affine_epi64_epi8 intrinsics do not assume this specific FGP */

/* The field generator polynomial is 0x11B but we make the 0x100 bit implicit to fit inside 8 bits */
#define EASYSIMD_X86_GFNI_FGP 0x1B

/* Computing the inverse of a GF element is expensive so use this LUT for an FGP of 0x11B */

static const union {
  uint8_t      u8[256];
  easysimd__m128i m128i[16];
} easysimd_x_gf2p8inverse_lut = {
  {
   0x00, 0x01, 0x8d, 0xf6, 0xcb, 0x52, 0x7b, 0xd1, 0xe8, 0x4f, 0x29, 0xc0, 0xb0, 0xe1, 0xe5, 0xc7,
   0x74, 0xb4, 0xaa, 0x4b, 0x99, 0x2b, 0x60, 0x5f, 0x58, 0x3f, 0xfd, 0xcc, 0xff, 0x40, 0xee, 0xb2,
   0x3a, 0x6e, 0x5a, 0xf1, 0x55, 0x4d, 0xa8, 0xc9, 0xc1, 0x0a, 0x98, 0x15, 0x30, 0x44, 0xa2, 0xc2,
   0x2c, 0x45, 0x92, 0x6c, 0xf3, 0x39, 0x66, 0x42, 0xf2, 0x35, 0x20, 0x6f, 0x77, 0xbb, 0x59, 0x19,
   0x1d, 0xfe, 0x37, 0x67, 0x2d, 0x31, 0xf5, 0x69, 0xa7, 0x64, 0xab, 0x13, 0x54, 0x25, 0xe9, 0x09,
   0xed, 0x5c, 0x05, 0xca, 0x4c, 0x24, 0x87, 0xbf, 0x18, 0x3e, 0x22, 0xf0, 0x51, 0xec, 0x61, 0x17,
   0x16, 0x5e, 0xaf, 0xd3, 0x49, 0xa6, 0x36, 0x43, 0xf4, 0x47, 0x91, 0xdf, 0x33, 0x93, 0x21, 0x3b,
   0x79, 0xb7, 0x97, 0x85, 0x10, 0xb5, 0xba, 0x3c, 0xb6, 0x70, 0xd0, 0x06, 0xa1, 0xfa, 0x81, 0x82,
   0x83, 0x7e, 0x7f, 0x80, 0x96, 0x73, 0xbe, 0x56, 0x9b, 0x9e, 0x95, 0xd9, 0xf7, 0x02, 0xb9, 0xa4,
   0xde, 0x6a, 0x32, 0x6d, 0xd8, 0x8a, 0x84, 0x72, 0x2a, 0x14, 0x9f, 0x88, 0xf9, 0xdc, 0x89, 0x9a,
   0xfb, 0x7c, 0x2e, 0xc3, 0x8f, 0xb8, 0x65, 0x48, 0x26, 0xc8, 0x12, 0x4a, 0xce, 0xe7, 0xd2, 0x62,
   0x0c, 0xe0, 0x1f, 0xef, 0x11, 0x75, 0x78, 0x71, 0xa5, 0x8e, 0x76, 0x3d, 0xbd, 0xbc, 0x86, 0x57,
   0x0b, 0x28, 0x2f, 0xa3, 0xda, 0xd4, 0xe4, 0x0f, 0xa9, 0x27, 0x53, 0x04, 0x1b, 0xfc, 0xac, 0xe6,
   0x7a, 0x07, 0xae, 0x63, 0xc5, 0xdb, 0xe2, 0xea, 0x94, 0x8b, 0xc4, 0xd5, 0x9d, 0xf8, 0x90, 0x6b,
   0xb1, 0x0d, 0xd6, 0xeb, 0xc6, 0x0e, 0xcf, 0xad, 0x08, 0x4e, 0xd7, 0xe3, 0x5d, 0x50, 0x1e, 0xb3,
   0x5b, 0x23, 0x38, 0x34, 0x68, 0x46, 0x03, 0x8c, 0xdd, 0x9c, 0x7d, 0xa0, 0xcd, 0x1a, 0x41, 0x1c
  }
};

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_gf2p8matrix_multiply_epi64_epi8 (easysimd__m128i x, easysimd__m128i A) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    const __m128i byte_select = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m128i zero = _mm_setzero_si128();
    __m128i r, a, p, X;

    a = _mm_shuffle_epi8(A, _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8));
    X = x;
    r = zero;

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 8 ; i++) {
      p = _mm_insert_epi16(zero, _mm_movemask_epi8(a), 0);
      p = _mm_shuffle_epi8(p, byte_select);
      p = _mm_and_si128(p, _mm_cmpgt_epi8(zero, X));
      r = _mm_xor_si128(r, p);
      a = _mm_add_epi8(a, a);
      X = _mm_add_epi8(X, X);
    }

    return r;
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    const __m128i zero = _mm_setzero_si128();
    __m128i r, a, p, X;

    a = _mm_shufflehi_epi16(A, (0 << 6) + (1 << 4) + (2 << 2) + (3 << 0));
    a = _mm_shufflelo_epi16(a, (0 << 6) + (1 << 4) + (2 << 2) + (3 << 0));
    a = _mm_or_si128(_mm_slli_epi16(a, 8), _mm_srli_epi16(a, 8));
    X = _mm_unpacklo_epi8(x, _mm_unpackhi_epi64(x, x));
    r = zero;

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 8 ; i++) {
      p = _mm_set1_epi16(HEDLEY_STATIC_CAST(short, _mm_movemask_epi8(a)));
      p = _mm_and_si128(p, _mm_cmpgt_epi8(zero, X));
      r = _mm_xor_si128(r, p);
      a = _mm_add_epi8(a, a);
      X = _mm_add_epi8(X, X);
    }

    return _mm_packus_epi16(_mm_srli_epi16(_mm_slli_epi16(r, 8), 8), _mm_srli_epi16(r, 8));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    static const uint8_t byte_interleave[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    static const uint8_t byte_deinterleave[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    static const uint8_t mask_d[16] = {128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1};
    const int8x16_t mask = vreinterpretq_s8_u8(vld1q_u8(mask_d));
    int8x16_t r, a, t, X;

    t = easysimd__m128i_to_neon_i8(A);
    a = vqtbl1q_s8(t, vld1q_u8(byte_interleave));
    t = easysimd__m128i_to_neon_i8(x);
    X = vqtbl1q_s8(t, vld1q_u8(byte_interleave));
    r = vdupq_n_s8(0);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 8 ; i++) {
      t = vshrq_n_s8(a, 7);
      t = vandq_s8(t, mask);
      t = vreinterpretq_s8_u16(vdupq_n_u16(vaddvq_u16(vreinterpretq_u16_s8(t))));
      t = vandq_s8(t, vshrq_n_s8(X, 7));
      r = veorq_s8(r, t);
      a = vshlq_n_s8(a, 1);
      X = vshlq_n_s8(X, 1);
    }

    r = vqtbl1q_s8(r, vld1q_u8(byte_deinterleave));
    return easysimd__m128i_from_neon_i8(r);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    static const uint8_t mask_d[16] = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
    const int8x16_t mask = vreinterpretq_s8_u8(vld1q_u8(mask_d));
    int8x16_t r, a, t, X;
    int16x8_t t16;
    int32x4_t t32;

    a = easysimd__m128i_to_neon_i8(A);
    X = easysimd__m128i_to_neon_i8(x);
    r = vdupq_n_s8(0);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 8 ; i++) {
      t = vshrq_n_s8(a, 7);
      t = vandq_s8(t, mask);
      t16 = vreinterpretq_s16_s8 (vorrq_s8 (t  , vrev64q_s8 (t  )));
      t32 = vreinterpretq_s32_s16(vorrq_s16(t16, vrev64q_s16(t16)));
      t   = vreinterpretq_s8_s32 (vorrq_s32(t32, vrev64q_s32(t32)));
      t = vandq_s8(t, vshrq_n_s8(X, 7));
      r = veorq_s8(r, t);
      a = vshlq_n_s8(a, 1);
      X = vshlq_n_s8(X, 1);
    }

    return easysimd__m128i_from_neon_i8(r);
  #else
    easysimd__m128i_private
      r_,
      x_ = easysimd__m128i_to_private(x),
      A_ = easysimd__m128i_to_private(A);

    const uint64_t ones = UINT64_C(0x0101010101010101);
    const uint64_t mask = UINT64_C(0x0102040810204080);
    uint64_t q;

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      q = easysimd_endian_bswap64_le(A_.u64[i / 8]);
      q &= HEDLEY_STATIC_CAST(uint64_t, x_.u8[i]) * ones;
      q ^= q >> 4;
      q ^= q >> 2;
      q ^= q >> 1;
      q &= ones;
      q *= 255;
      q &= mask;
      q |= q >> 32;
      q |= q >> 16;
      q |= q >> 8;
      r_.u8[i] = HEDLEY_STATIC_CAST(uint8_t, q);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_gf2p8matrix_multiply_epi64_epi8 (easysimd__m256i x, easysimd__m256i A) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    easysimd__m256i r, a, p;
    const easysimd__m256i byte_select = easysimd_x_mm256_set_epu64x(UINT64_C(0x0303030303030303), UINT64_C(0x0202020202020202),
                                                              UINT64_C(0x0101010101010101), UINT64_C(0x0000000000000000));
    a = easysimd_mm256_shuffle_epi8(A, easysimd_mm256_broadcastsi128_si256(easysimd_x_mm_set_epu64x(UINT64_C(0x08090A0B0C0D0E0F), UINT64_C(0x0001020304050607))));
    r = easysimd_mm256_setzero_si256();

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 8 ; i++) {
      p = easysimd_mm256_set1_epi32(easysimd_mm256_movemask_epi8(a));
      p = easysimd_mm256_shuffle_epi8(p, byte_select);
      p = easysimd_mm256_xor_si256(r, p);
      r = easysimd_mm256_blendv_epi8(r, p, x);
      a = easysimd_mm256_add_epi8(a, a);
      x = easysimd_mm256_add_epi8(x, x);
    }

    return r;
  #else
    easysimd__m256i_private
      r_,
      x_ = easysimd__m256i_to_private(x),
      A_ = easysimd__m256i_to_private(A);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
      r_.m128i[i] = easysimd_x_mm_gf2p8matrix_multiply_epi64_epi8(x_.m128i[i], A_.m128i[i]);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_gf2p8matrix_multiply_epi64_epi8 (easysimd__m512i x, easysimd__m512i A) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    easysimd__m512i r, a, p;
    const easysimd__m512i byte_select = easysimd_x_mm512_set_epu64(UINT64_C(0x0707070707070707), UINT64_C(0x0606060606060606), UINT64_C(0x0505050505050505), UINT64_C(0x0404040404040404),
                                                             UINT64_C(0x0303030303030303), UINT64_C(0x0202020202020202), UINT64_C(0x0101010101010101), UINT64_C(0X0000000000000000));
    a = easysimd_mm512_shuffle_epi8(A, easysimd_mm512_broadcast_i32x4(easysimd_x_mm_set_epu64x(UINT64_C(0x08090A0B0C0D0E0F), UINT64_C(0x0001020304050607))));
    r = easysimd_mm512_setzero_si512();

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 8 ; i++) {
      p = easysimd_mm512_set1_epi64(HEDLEY_STATIC_CAST(int64_t, easysimd_mm512_movepi8_mask(a)));
      p = easysimd_mm512_maskz_shuffle_epi8(easysimd_mm512_movepi8_mask(x), p, byte_select);
      r = easysimd_mm512_xor_si512(r, p);
      a = easysimd_mm512_add_epi8(a, a);
      x = easysimd_mm512_add_epi8(x, x);
    }

    return r;
  #else
    easysimd__m512i_private
      r_,
      x_ = easysimd__m512i_to_private(x),
      A_ = easysimd__m512i_to_private(A);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
      r_.m256i[i] = easysimd_x_mm256_gf2p8matrix_multiply_epi64_epi8(x_.m256i[i], A_.m256i[i]);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_gf2p8inverse_epi8 (easysimd__m128i x) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    /* N.B. CM: this fallback may not be faster */
    easysimd__m128i r, u, t, test;
    const easysimd__m128i sixteens = easysimd_mm_set1_epi8(16);
    const easysimd__m128i masked_x = easysimd_mm_and_si128(x, easysimd_mm_set1_epi8(0x0F));

    test = easysimd_mm_set1_epi8(INT8_MIN /* 0x80 */);
    x = easysimd_mm_xor_si128(x, test);
    r = easysimd_mm_shuffle_epi8(easysimd_x_gf2p8inverse_lut.m128i[0], masked_x);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 1 ; i < 16 ; i++) {
      t = easysimd_mm_shuffle_epi8(easysimd_x_gf2p8inverse_lut.m128i[i], masked_x);
      test = easysimd_mm_add_epi8(test, sixteens);
      u = easysimd_mm_cmplt_epi8(x, test);
      r = easysimd_mm_blendv_epi8(t, r, u);
    }

    return r;
  #else
    easysimd__m128i_private
      r_,
      x_ = easysimd__m128i_to_private(x);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = easysimd_x_gf2p8inverse_lut.u8[x_.u8[i]];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_gf2p8inverse_epi8 (easysimd__m256i x) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    /* N.B. CM: this fallback may not be faster */
    easysimd__m256i r, u, t, test;
    const easysimd__m256i sixteens = easysimd_mm256_set1_epi8(16);
    const easysimd__m256i masked_x = easysimd_mm256_and_si256(x, easysimd_mm256_set1_epi8(0x0F));

    test = easysimd_mm256_set1_epi8(INT8_MIN /* 0x80 */);
    x = easysimd_mm256_xor_si256(x, test);
    r = easysimd_mm256_shuffle_epi8(easysimd_mm256_broadcastsi128_si256(easysimd_x_gf2p8inverse_lut.m128i[0]), masked_x);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 1 ; i < 16 ; i++) {
      t = easysimd_mm256_shuffle_epi8(easysimd_mm256_broadcastsi128_si256(easysimd_x_gf2p8inverse_lut.m128i[i]), masked_x);
      test = easysimd_mm256_add_epi8(test, sixteens);
      u = easysimd_mm256_cmpgt_epi8(test, x);
      r = easysimd_mm256_blendv_epi8(t, r, u);
    }

    return r;
  #else
    easysimd__m256i_private
      r_,
      x_ = easysimd__m256i_to_private(x);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
      r_.m128i[i] = easysimd_x_mm_gf2p8inverse_epi8(x_.m128i[i]);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_gf2p8inverse_epi8 (easysimd__m512i x) {
  /* N.B. CM: TODO: later add VBMI version using just two _mm512_permutex2var_epi8 and friends */
  /* But except for Cannon Lake all processors with VBMI also have GFNI */
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    /* N.B. CM: this fallback may not be faster */
    easysimd__m512i r, test;
    const easysimd__m512i sixteens = easysimd_mm512_set1_epi8(16);
    const easysimd__m512i masked_x = easysimd_mm512_and_si512(x, easysimd_mm512_set1_epi8(0x0F));

    r = easysimd_mm512_shuffle_epi8(easysimd_mm512_broadcast_i32x4(easysimd_x_gf2p8inverse_lut.m128i[0]), masked_x);
    test = sixteens;

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 1 ; i < 16 ; i++) {
      r = easysimd_mm512_mask_shuffle_epi8(r, easysimd_mm512_cmpge_epu8_mask(x, test), easysimd_mm512_broadcast_i32x4(easysimd_x_gf2p8inverse_lut.m128i[i]), masked_x);
      test = easysimd_mm512_add_epi8(test, sixteens);
    }

    return r;
  #else
    easysimd__m512i_private
      r_,
      x_ = easysimd__m512i_to_private(x);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
      r_.m256i[i] = easysimd_x_mm256_gf2p8inverse_epi8(x_.m256i[i]);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}

#define easysimd_x_mm_gf2p8matrix_multiply_inverse_epi64_epi8(x, A) easysimd_x_mm_gf2p8matrix_multiply_epi64_epi8(easysimd_x_mm_gf2p8inverse_epi8(x), A)
#define easysimd_x_mm256_gf2p8matrix_multiply_inverse_epi64_epi8(x, A) easysimd_x_mm256_gf2p8matrix_multiply_epi64_epi8(easysimd_x_mm256_gf2p8inverse_epi8(x), A)
#define easysimd_x_mm512_gf2p8matrix_multiply_inverse_epi64_epi8(x, A) easysimd_x_mm512_gf2p8matrix_multiply_epi64_epi8(easysimd_x_mm512_gf2p8inverse_epi8(x), A)

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_gf2p8affine_epi64_epi8 (easysimd__m128i x, easysimd__m128i A, int b)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(b, 0, 255) {
  return easysimd_mm_xor_si128(easysimd_x_mm_gf2p8matrix_multiply_epi64_epi8(x, A), easysimd_mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, b)));
}
#if defined(EASYSIMD_X86_GFNI_NATIVE)
  #define easysimd_mm_gf2p8affine_epi64_epi8(x, A, b) _mm_gf2p8affine_epi64_epi8(x, A, b)
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_gf2p8affine_epi64_epi8
  #define _mm_gf2p8affine_epi64_epi8(x, A, b) easysimd_mm_gf2p8affine_epi64_epi8(x, A, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_gf2p8affine_epi64_epi8 (easysimd__m256i x, easysimd__m256i A, int b)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(b, 0, 255) {
  return easysimd_mm256_xor_si256(easysimd_x_mm256_gf2p8matrix_multiply_epi64_epi8(x, A), easysimd_mm256_set1_epi8(HEDLEY_STATIC_CAST(int8_t, b)));
}
#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_gf2p8affine_epi64_epi8(x, A, b) _mm256_gf2p8affine_epi64_epi8(x, A, b)
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_gf2p8affine_epi64_epi8
  #define _mm256_gf2p8affine_epi64_epi8(x, A, b) easysimd_mm256_gf2p8affine_epi64_epi8(x, A, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_gf2p8affine_epi64_epi8 (easysimd__m512i x, easysimd__m512i A, int b)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(b, 0, 255) {
  return easysimd_mm512_xor_si512(easysimd_x_mm512_gf2p8matrix_multiply_epi64_epi8(x, A), easysimd_mm512_set1_epi8(HEDLEY_STATIC_CAST(int8_t, b)));
}
#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_gf2p8affine_epi64_epi8(x, A, b) _mm512_gf2p8affine_epi64_epi8(x, A, b)
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_gf2p8affine_epi64_epi8
  #define _mm512_gf2p8affine_epi64_epi8(x, A, b) easysimd_mm512_gf2p8affine_epi64_epi8(x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) _mm_mask_gf2p8affine_epi64_epi8(src, k, x, A, b)
#else
  #define easysimd_mm_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_gf2p8affine_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_gf2p8affine_epi64_epi8
  #define _mm_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) easysimd_mm_mask_gf2p8affine_epi64_epi8(src, k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) _mm256_mask_gf2p8affine_epi64_epi8(src, k, x, A, b)
#else
  #define easysimd_mm256_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_gf2p8affine_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_gf2p8affine_epi64_epi8
  #define _mm256_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) easysimd_mm256_mask_gf2p8affine_epi64_epi8(src, k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) _mm512_mask_gf2p8affine_epi64_epi8(src, k, x, A, b)
#else
  #define easysimd_mm512_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_gf2p8affine_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_gf2p8affine_epi64_epi8
  #define _mm512_mask_gf2p8affine_epi64_epi8(src, k, x, A, b) easysimd_mm512_mask_gf2p8affine_epi64_epi8(src, k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_gf2p8affine_epi64_epi8(k, x, A, b) _mm_maskz_gf2p8affine_epi64_epi8(k, x, A, b)
#else
  #define easysimd_mm_maskz_gf2p8affine_epi64_epi8(k, x, A, b) easysimd_mm_maskz_mov_epi8(k, easysimd_mm_gf2p8affine_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_gf2p8affine_epi64_epi8
  #define _mm_maskz_gf2p8affine_epi64_epi8(k, x, A, b) easysimd_mm_maskz_gf2p8affine_epi64_epi8(k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_gf2p8affine_epi64_epi8(k, x, A, b) _mm256_maskz_gf2p8affine_epi64_epi8(k, x, A, b)
#else
  #define easysimd_mm256_maskz_gf2p8affine_epi64_epi8(k, x, A, b) easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_gf2p8affine_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_gf2p8affine_epi64_epi8
  #define _mm256_maskz_gf2p8affine_epi64_epi8(k, x, A, b) easysimd_mm256_maskz_gf2p8affine_epi64_epi8(k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_gf2p8affine_epi64_epi8(k, x, A, b) _mm512_maskz_gf2p8affine_epi64_epi8(k, x, A, b)
#else
  #define easysimd_mm512_maskz_gf2p8affine_epi64_epi8(k, x, A, b) easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_gf2p8affine_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_gf2p8affine_epi64_epi8
  #define _mm512_maskz_gf2p8affine_epi64_epi8(k, x, A, b) easysimd_mm512_maskz_gf2p8affine_epi64_epi8(k, x, A, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_gf2p8affineinv_epi64_epi8 (easysimd__m128i x, easysimd__m128i A, int b)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(b, 0, 255) {
  return easysimd_mm_xor_si128(easysimd_x_mm_gf2p8matrix_multiply_inverse_epi64_epi8(x, A), easysimd_mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, b)));
}
#if defined(EASYSIMD_X86_GFNI_NATIVE)
  #define easysimd_mm_gf2p8affineinv_epi64_epi8(x, A, b) _mm_gf2p8affineinv_epi64_epi8(x, A, b)
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_gf2p8affineinv_epi64_epi8
  #define _mm_gf2p8affineinv_epi64_epi8(x, A, b) easysimd_mm_gf2p8affineinv_epi64_epi8(x, A, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_gf2p8affineinv_epi64_epi8 (easysimd__m256i x, easysimd__m256i A, int b)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(b, 0, 255) {
  return easysimd_mm256_xor_si256(easysimd_x_mm256_gf2p8matrix_multiply_inverse_epi64_epi8(x, A), easysimd_mm256_set1_epi8(HEDLEY_STATIC_CAST(int8_t, b)));
}
#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_gf2p8affineinv_epi64_epi8(x, A, b) _mm256_gf2p8affineinv_epi64_epi8(x, A, b)
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_gf2p8affineinv_epi64_epi8
  #define _mm256_gf2p8affineinv_epi64_epi8(x, A, b) easysimd_mm256_gf2p8affineinv_epi64_epi8(x, A, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_gf2p8affineinv_epi64_epi8 (easysimd__m512i x, easysimd__m512i A, int b)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(b, 0, 255) {
  return easysimd_mm512_xor_si512(easysimd_x_mm512_gf2p8matrix_multiply_inverse_epi64_epi8(x, A), easysimd_mm512_set1_epi8(HEDLEY_STATIC_CAST(int8_t, b)));
}
#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_gf2p8affineinv_epi64_epi8(x, A, b) _mm512_gf2p8affineinv_epi64_epi8(x, A, b)
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_gf2p8affineinv_epi64_epi8
  #define _mm512_gf2p8affineinv_epi64_epi8(x, A, b) easysimd_mm512_gf2p8affineinv_epi64_epi8(x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) _mm_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b)
#else
  #define easysimd_mm_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_gf2p8affineinv_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_gf2p8affineinv_epi64_epi8
  #define _mm_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) easysimd_mm_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) _mm256_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b)
#else
  #define easysimd_mm256_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_gf2p8affineinv_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_gf2p8affineinv_epi64_epi8
  #define _mm256_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) easysimd_mm256_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) _mm512_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b)
#else
  #define easysimd_mm512_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_gf2p8affineinv_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_gf2p8affineinv_epi64_epi8
  #define _mm512_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b) easysimd_mm512_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) _mm_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b)
#else
  #define easysimd_mm_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) easysimd_mm_maskz_mov_epi8(k, easysimd_mm_gf2p8affineinv_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_gf2p8affineinv_epi64_epi8
  #define _mm_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) easysimd_mm_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) _mm256_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b)
#else
  #define easysimd_mm256_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_gf2p8affineinv_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_gf2p8affineinv_epi64_epi8
  #define _mm256_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) easysimd_mm256_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b)
#endif

#if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define easysimd_mm512_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) _mm512_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b)
#else
  #define easysimd_mm512_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_gf2p8affineinv_epi64_epi8(x, A, b))
#endif
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_gf2p8affineinv_epi64_epi8
  #define _mm512_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b) easysimd_mm512_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i easysimd_mm_gf2p8mul_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && (defined(EASYSIMD_X86_AVX512VL_NATIVE) || !defined(EASYSIMD_X86_AVX512F_NATIVE))
    return _mm_gf2p8mul_epi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    const poly8x16_t pa = vreinterpretq_p8_u8(easysimd__m128i_to_neon_u8(a));
    const poly8x16_t pb = vreinterpretq_p8_u8(easysimd__m128i_to_neon_u8(b));
    const uint8x16_t lo = vreinterpretq_u8_p16(vmull_p8(vget_low_p8(pa), vget_low_p8(pb)));
    #if defined (EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint8x16_t hi = vreinterpretq_u8_p16(vmull_high_p8(pa, pb));
    #else
      uint8x16_t hi = vreinterpretq_u8_p16(vmull_p8(vget_high_p8(pa), vget_high_p8(pb)));
    #endif
    uint8x16x2_t hilo = vuzpq_u8(lo, hi);
    uint8x16_t r = hilo.val[0];
    hi = hilo.val[1];
    const uint8x16_t idxHi = vshrq_n_u8(hi, 4);
    const uint8x16_t idxLo = vandq_u8(hi, vdupq_n_u8(0xF));

    #if defined (EASYSIMD_ARM_NEON_A64V8_NATIVE)
      static const uint8_t reduceLutHiData[] = {
        0x00, 0xab, 0x4d, 0xe6, 0x9a, 0x31, 0xd7, 0x7c,
        0x2f, 0x84, 0x62, 0xc9, 0xb5, 0x1e, 0xf8, 0x53
      };
      static const uint8_t reduceLutLoData[] = {
        0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41,
        0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99
      };
      const uint8x16_t reduceLutHi = vld1q_u8(reduceLutHiData);
      const uint8x16_t reduceLutLo = vld1q_u8(reduceLutLoData);
      r = veorq_u8(r, vqtbl1q_u8(reduceLutHi, idxHi));
      r = veorq_u8(r, vqtbl1q_u8(reduceLutLo, idxLo));
    #else
      static const uint8_t reduceLutHiData[] = {
        0x00, 0x2f,
        0xab, 0x84,
        0x4d, 0x62,
        0xe6, 0xc9,
        0x9a, 0xb5,
        0x31, 0x1e,
        0xd7, 0xf8,
        0x7c, 0x53
      };
      static const uint8_t reduceLutLoData[] = {
        0x00, 0xd8,
        0x1b, 0xc3,
        0x36, 0xee,
        0x2d, 0xf5,
        0x6c, 0xb4,
        0x77, 0xaf,
        0x5a, 0x82,
        0x41, 0x99
      };
      const uint8x8x2_t reduceLutHi = vld2_u8(reduceLutHiData);
      const uint8x8x2_t reduceLutLo = vld2_u8(reduceLutLoData);
      r = veorq_u8(r, vcombine_u8(vtbl2_u8(reduceLutHi, vget_low_u8(idxHi)), vtbl2_u8(reduceLutHi, vget_high_u8(idxHi))));
      r = veorq_u8(r, vcombine_u8(vtbl2_u8(reduceLutLo, vget_low_u8(idxLo)), vtbl2_u8(reduceLutLo, vget_high_u8(idxLo))));
    #endif
    return easysimd__m128i_from_neon_u8(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE)
    easysimd__m512i r4, t4, u4;
    easysimd__mmask64 ma, mb;

    easysimd__m512i a4 = easysimd_mm512_broadcast_i32x4(a);
    const easysimd__m512i zero = easysimd_mm512_setzero_si512();
    easysimd__mmask16 m8 = easysimd_mm512_cmpeq_epi32_mask(zero, zero);

    const easysimd__m512i b4 = easysimd_mm512_broadcast_i32x4(b);

    easysimd__m512i bits = easysimd_mm512_set_epi64(0x4040404040404040,
                                              0x4040404040404040,
                                              0x1010101010101010,
                                              0x1010101010101010,
                                              0x0404040404040404,
                                              0x0404040404040404,
                                              0x0101010101010101,
                                              0x0101010101010101);

    const easysimd__m512i fgp = easysimd_mm512_set1_epi8(EASYSIMD_X86_GFNI_FGP);

    for (int i = 0 ; i < 3 ; i++) {
      m8 = easysimd_kshiftli_mask16(m8, 4);

      ma = easysimd_mm512_cmplt_epi8_mask(a4, zero);
      u4 = easysimd_mm512_add_epi8(a4, a4);
      t4 = easysimd_mm512_maskz_mov_epi8(ma, fgp);
      u4 = easysimd_mm512_xor_epi32(u4, t4);

      ma = easysimd_mm512_cmplt_epi8_mask(u4, zero);
      u4 = easysimd_mm512_add_epi8(u4, u4);
      t4 = easysimd_mm512_maskz_mov_epi8(ma, fgp);
      a4 = easysimd_mm512_mask_xor_epi32(a4, m8, u4, t4);
    }

    mb = easysimd_mm512_test_epi8_mask(b4, bits);
    bits = easysimd_mm512_add_epi8(bits, bits);
    ma = easysimd_mm512_cmplt_epi8_mask(a4, zero);
    r4 = easysimd_mm512_maskz_mov_epi8(mb, a4);
    mb = easysimd_mm512_test_epi8_mask(b4, bits);
    a4 = easysimd_mm512_add_epi8(a4, a4);
    t4 = easysimd_mm512_maskz_mov_epi8(ma, fgp);
    a4 = easysimd_mm512_xor_si512(a4, t4);
    t4 = easysimd_mm512_maskz_mov_epi8(mb, a4);
    r4 = easysimd_mm512_xor_si512(r4, t4);

    r4 = easysimd_mm512_xor_si512(r4, easysimd_mm512_shuffle_i32x4(r4, r4, (1 << 6) + (0 << 4) + (3 << 2) + 2));
    r4 = easysimd_mm512_xor_si512(r4, easysimd_mm512_shuffle_i32x4(r4, r4, (0 << 6) + (3 << 4) + (2 << 2) + 1));

    return easysimd_mm512_extracti32x4_epi32(r4, 0);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    easysimd__m256i r2, t2;
    easysimd__m256i a2 = easysimd_mm256_broadcastsi128_si256(a);
    const easysimd__m256i zero = easysimd_mm256_setzero_si256();
    const easysimd__m256i fgp = easysimd_mm256_set1_epi8(EASYSIMD_X86_GFNI_FGP);
    const easysimd__m256i ones = easysimd_mm256_set1_epi8(0x01);
    easysimd__m256i b2 = easysimd_mm256_set_m128i(easysimd_mm_srli_epi64(b, 4), b);

    for (int i = 0 ; i < 4 ; i++) {
      t2 = easysimd_mm256_cmpgt_epi8(zero, a2);
      t2 = easysimd_mm256_and_si256(fgp, t2);
      a2 = easysimd_mm256_add_epi8(a2, a2);
      a2 = easysimd_mm256_xor_si256(a2, t2);
    }

    a2 = easysimd_mm256_inserti128_si256(a2, a, 0);

    t2 = easysimd_mm256_and_si256(b2, ones);
    t2 = easysimd_mm256_cmpeq_epi8(t2, ones);
    r2 = easysimd_mm256_and_si256(a2, t2);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 3 ; i++) {
      t2 = easysimd_mm256_cmpgt_epi8(zero, a2);
      t2 = easysimd_mm256_and_si256(fgp, t2);
      a2 = easysimd_mm256_add_epi8(a2, a2);
      a2 = easysimd_mm256_xor_si256(a2, t2);
      b2 = easysimd_mm256_srli_epi64(b2, 1);
      t2 = easysimd_mm256_and_si256(b2, ones);
      t2 = easysimd_mm256_cmpeq_epi8(t2, ones);
      t2 = easysimd_mm256_and_si256(a2, t2);
      r2 = easysimd_mm256_xor_si256(r2, t2);
    }

    return easysimd_mm_xor_si128(easysimd_mm256_extracti128_si256(r2, 1),
                              easysimd_mm256_extracti128_si256(r2, 0));
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd__m128i r, t;
    const easysimd__m128i zero = easysimd_mm_setzero_si128();
    const easysimd__m128i ones = easysimd_mm_set1_epi8(0x01);

    const easysimd__m128i fgp = easysimd_mm_set1_epi8(EASYSIMD_X86_GFNI_FGP);

    t = easysimd_mm_and_si128(b, ones);
    t = easysimd_mm_cmpeq_epi8(t, ones);
    r = easysimd_mm_and_si128(a, t);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 7 ; i++) {
      t = easysimd_mm_cmpgt_epi8(zero, a);
      t = easysimd_mm_and_si128(fgp, t);
      a = easysimd_mm_add_epi8(a, a);
      a = easysimd_mm_xor_si128(a, t);
      b = easysimd_mm_srli_epi64(b, 1);
      t = easysimd_mm_and_si128(b, ones);
      t = easysimd_mm_cmpeq_epi8(t, ones);
      t = easysimd_mm_and_si128(a, t);
      r = easysimd_mm_xor_si128(r, t);
    }

    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    const uint8_t fgp = EASYSIMD_X86_GFNI_FGP;

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = 0;
      while ((a_.u8[i] != 0) && (b_.u8[i] != 0)) {
        if (b_.u8[i] & 1)
          r_.u8[i] ^= a_.u8[i];

        if (a_.u8[i] & 0x80)
          a_.u8[i] = HEDLEY_STATIC_CAST(uint8_t, (a_.u8[i] << 1) ^ fgp);
        else
          a_.u8[i] <<= 1;

        b_.u8[i] >>= 1;
      }
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_gf2p8mul_epi8
  #define _mm_gf2p8mul_epi8(a, b) easysimd_mm_gf2p8mul_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_gf2p8mul_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && (defined(EASYSIMD_X86_AVX512VL_NATIVE) || (defined(EASYSIMD_X86_AVX_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)))
    return _mm256_gf2p8mul_epi8(a, b);
  #elif !defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    easysimd__mmask64 ma, mb;
    easysimd__m512i r, t, s;
    easysimd__m512i a2 = easysimd_mm512_broadcast_i64x4(a);
    const easysimd__m512i zero = easysimd_mm512_setzero_si512();

    const easysimd__m512i fgp = easysimd_mm512_set1_epi8(EASYSIMD_X86_GFNI_FGP);

    s = easysimd_mm512_set1_epi8(0x01);

    for (int i = 0 ; i < 4 ; i++) {
      ma = easysimd_mm512_cmplt_epi8_mask(a2, zero);
      a2 = easysimd_mm512_add_epi8(a2, a2);
      t = easysimd_mm512_xor_si512(a2, fgp);
      a2 = easysimd_mm512_mask_mov_epi8(a2, ma, t);
    }

    easysimd__m512i b2 = easysimd_mm512_inserti64x4(zero, easysimd_mm256_srli_epi64(b, 4), 1);
    b2 = easysimd_mm512_inserti64x4(b2, b, 0);
    a2 = easysimd_mm512_inserti64x4(a2, a, 0);

    mb = easysimd_mm512_test_epi8_mask(b2, s);
    r = easysimd_mm512_maskz_mov_epi8(mb, a2);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 3 ; i++) {
      ma = easysimd_mm512_cmplt_epi8_mask(a2, zero);
      s = easysimd_mm512_add_epi8(s, s);
      mb = easysimd_mm512_test_epi8_mask(b2, s);
      a2 = easysimd_mm512_add_epi8(a2, a2);
      t = easysimd_mm512_maskz_mov_epi8(ma, fgp);
      a2 = easysimd_mm512_xor_si512(a2, t);
      t = easysimd_mm512_maskz_mov_epi8(mb, a2);
      r = easysimd_mm512_xor_si512(r, t);
    }

    return easysimd_mm256_xor_si256(easysimd_mm512_extracti64x4_epi64(r, 1),
                                 easysimd_mm512_extracti64x4_epi64(r, 0));
  #elif !defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX2_NATIVE)
    easysimd__m256i r, t;
    const easysimd__m256i zero = easysimd_mm256_setzero_si256();
    const easysimd__m256i ones = easysimd_mm256_set1_epi8(0x01);

    const easysimd__m256i fgp = easysimd_mm256_set1_epi8(EASYSIMD_X86_GFNI_FGP);

    t = easysimd_mm256_and_si256(b, ones);
    t = easysimd_mm256_cmpeq_epi8(t, ones);
    r = easysimd_mm256_and_si256(a, t);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 7 ; i++) {
      t = easysimd_mm256_cmpgt_epi8(zero, a);
      t = easysimd_mm256_and_si256(fgp, t);
      a = easysimd_mm256_add_epi8(a, a);
      a = easysimd_mm256_xor_si256(a, t);
      b = easysimd_mm256_srli_epi64(b, 1);
      t = easysimd_mm256_and_si256(b, ones);
      t = easysimd_mm256_cmpeq_epi8(t, ones);
      t = easysimd_mm256_and_si256(a, t);
      r = easysimd_mm256_xor_si256(r, t);
    }

    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
      r_.m128i[i] = easysimd_mm_gf2p8mul_epi8(a_.m128i[i], b_.m128i[i]);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_gf2p8mul_epi8
  #define _mm256_gf2p8mul_epi8(a, b) easysimd_mm256_gf2p8mul_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_gf2p8mul_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_gf2p8mul_epi8(a, b);
  #elif !defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    easysimd__m512i r, s, t;
    easysimd__mmask64 ma, mb;
    const easysimd__m512i zero = easysimd_mm512_setzero_si512();

    const easysimd__m512i fgp = easysimd_mm512_set1_epi8(EASYSIMD_X86_GFNI_FGP);

    s = easysimd_mm512_set1_epi8(0x01);

    mb = easysimd_mm512_test_epi8_mask(b, s);
    r = easysimd_mm512_maskz_mov_epi8(mb, a);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (int i = 0 ; i < 7 ; i++) {
      ma = easysimd_mm512_cmplt_epi8_mask(a, zero);
      s = easysimd_mm512_add_epi8(s, s);
      mb = easysimd_mm512_test_epi8_mask(b, s);
      a = easysimd_mm512_add_epi8(a, a);
      t = easysimd_mm512_maskz_mov_epi8(ma, fgp);
      a = easysimd_mm512_xor_si512(a, t);
      t = easysimd_mm512_maskz_mov_epi8(mb, a);
      r = easysimd_mm512_xor_si512(r, t);
    }

    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if !defined(__INTEL_COMPILER)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
      r_.m128i[i] = easysimd_mm_gf2p8mul_epi8(a_.m128i[i], b_.m128i[i]);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_gf2p8mul_epi8
  #define _mm512_gf2p8mul_epi8(a, b) easysimd_mm512_gf2p8mul_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_gf2p8mul_epi8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_gf2p8mul_epi8(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_gf2p8mul_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_gf2p8mul_epi8
  #define _mm_mask_gf2p8mul_epi8(src, k, a, b) easysimd_mm_mask_gf2p8mul_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_gf2p8mul_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_gf2p8mul_epi8(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_gf2p8mul_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_gf2p8mul_epi8
  #define _mm256_mask_gf2p8mul_epi8(src, k, a, b) easysimd_mm256_mask_gf2p8mul_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_gf2p8mul_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_gf2p8mul_epi8(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_gf2p8mul_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_gf2p8mul_epi8
  #define _mm512_mask_gf2p8mul_epi8(src, k, a, b) easysimd_mm512_mask_gf2p8mul_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_gf2p8mul_epi8 (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_gf2p8mul_epi8(k, a, b);
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_gf2p8mul_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_gf2p8mul_epi8
  #define _mm_maskz_gf2p8mul_epi8(k, a, b) easysimd_mm_maskz_gf2p8mul_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_gf2p8mul_epi8 (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_gf2p8mul_epi8(k, a, b);
  #else
    return  easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_gf2p8mul_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_gf2p8mul_epi8
  #define _mm256_maskz_gf2p8mul_epi8(k, a, b) easysimd_mm256_maskz_gf2p8mul_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_gf2p8mul_epi8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_GFNI_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_gf2p8mul_epi8(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_gf2p8mul_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_gf2p8mul_epi8
  #define _mm512_maskz_gf2p8mul_epi8(k, a, b) easysimd_mm512_maskz_gf2p8mul_epi8(k, a, b)
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_GFNI_H) */
