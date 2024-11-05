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
 *   2020      Hidayat Khan <huk2209@gmail.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_X86_AVX512_BROADCAST_H)
#define EASYSIMD_X86_AVX512_BROADCAST_H

#include "types.h"
#include "../avx2.h"

#include "mov.h"
#include "cast.h"
#include "set1.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_broadcastmw_epi32 (easysimd__mmask16 k) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_broadcastmw_epi32(k);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u32 = svdup_n_u32((uint32_t)k);
    return r;
  #else
    easysimd__m128i_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.u32[i] = (uint32_t)k;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastmw_epi32
  #define _mm_broadcastmw_epi32(k) easysimd_mm_broadcastmw_epi32(k)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_broadcastmb_epi64 (easysimd__mmask8 k) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_broadcastmb_epi64(k);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svdup_n_u64((uint64_t)k);
    return r;
  #else
    easysimd__m128i_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.u64[i] = (uint64_t)k;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastmb_epi64
  #define _mm_broadcastmb_epi64(k) easysimd_mm_broadcastmb_epi64(k)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_broadcastb_epi8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_broadcastb_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svdup_n_s8(a.i8[0]), src.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_= easysimd__m128i_to_private(src),
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? a_.i8[0] : src_.i8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_broadcastb_epi8
  #define _mm_mask_broadcastb_epi8(src, k, a) easysimd_mm_mask_broadcastb_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_broadcastb_epi8 (easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_broadcastb_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svdup_n_s8(a.i8[0]), svdup_n_s8(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? a_.i8[0] : INT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_broadcastb_epi8
  #define _mm_maskz_broadcastb_epi8(k, a) easysimd_mm_maskz_broadcastb_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_broadcastw_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_broadcastw_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svdup_n_s16(a.i16[0]), src.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_= easysimd__m128i_to_private(src),
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? a_.i16[0] : src_.i16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_broadcastw_epi16
  #define _mm_mask_broadcastw_epi16(src, k, a) easysimd_mm_mask_broadcastw_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_broadcastw_epi16 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_broadcastw_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svdup_n_s16(a.i16[0]), svdup_n_s16(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? a_.i16[0] : INT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_broadcastw_epi16
  #define _mm_maskz_broadcastw_epi16(k, a) easysimd_mm_maskz_broadcastw_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_broadcastd_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_broadcastd_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svdup_n_s32(a.i32[0]), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_= easysimd__m128i_to_private(src),
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[0] : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_broadcastd_epi32
  #define _mm_mask_broadcastd_epi32(src, k, a) easysimd_mm_mask_broadcastd_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_broadcastd_epi32 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_broadcastd_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svdup_n_s32(a.i32[0]), svdup_n_s32(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[0] : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_broadcastd_epi32
  #define _mm_maskz_broadcastd_epi32(k, a) easysimd_mm_maskz_broadcastd_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_broadcastq_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_broadcastq_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_s64(a.i64[0]), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_= easysimd__m128i_to_private(src),
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[0] : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_broadcastq_epi64
  #define _mm_mask_broadcastq_epi64(src, k, a) easysimd_mm_mask_broadcastq_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_broadcastq_epi64 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_broadcastq_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdup_n_s64(a.i64[0]), svdup_n_s64(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[0] : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_broadcastq_epi64
  #define _mm_maskz_broadcastq_epi64(k, a) easysimd_mm_maskz_broadcastq_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_broadcastss_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_broadcastss_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svdup_n_f32(a.f32[0]), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      src_= easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[0] : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_broadcastss_ps
  #define _mm_mask_broadcastss_ps(src, k, a) easysimd_mm_mask_broadcastss_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_broadcast_i32x2 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_broadcast_i32x2(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1)), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a);

    r_.i32[0] = a_.i32[0];
    r_.i32[1] = a_.i32[1];
    r_.i32[2] = a_.i32[0];
    r_.i32[3] = a_.i32[1];

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_broadcast_i32x2
  #define _mm_mask_broadcast_i32x2(src, k, a) easysimd_mm_mask_broadcast_i32x2(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_broadcast_i32x2 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_broadcast_i32x2(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1)), svdup_n_s32(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    r_.i32[0] = a_.i32[0];
    r_.i32[1] = a_.i32[1];
    r_.i32[2] = a_.i32[0];
    r_.i32[3] = a_.i32[1];

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_broadcast_i32x2
  #define _mm_maskz_broadcast_i32x2(k, a) easysimd_mm_maskz_broadcast_i32x2(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcast_i32x2 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_broadcast_i32x2(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i += 2) {
        r_.i32[  i  ] = a_.i32[0];
        r_.i32[i + 1] = a_.i32[1];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_i32x2
  #define _mm256_broadcast_i32x2(a) easysimd_mm256_broadcast_i32x2(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_broadcast_i32x2 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_mask_broadcast_i32x2(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1)), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1)), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src);
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i += 2) {
        r_.i32[  i  ] = ((k >> i) & 1) ? a_.i32[0] : src_.i32[i];
        r_.i32[i + 1] = ((k >> (i + 1)) & 1) ? a_.i32[1] : src_.i32[i + 1];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_broadcast_i32x2
  #define _mm256_mask_broadcast_i32x2(src, k, a) easysimd_mm256_mask_broadcast_i32x2(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_broadcast_i32x2 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_maskz_broadcast_i32x2(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1)), svdup_n_s32(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1)), svdup_n_s32(0));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i += 2) {
        r_.i32[  i  ] = ((k >> i) & 1) ? a_.i32[0] : INT32_C(0);
        r_.i32[i + 1] = ((k >> (i + 1)) & 1) ? a_.i32[1] : INT32_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_broadcast_i32x2
  #define _mm256_maskz_broadcast_i32x2(k, a) easysimd_mm256_maskz_broadcast_i32x2(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_broadcast_f32x2 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_broadcast_f32x2(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1));
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.f32 = __builtin_shufflevector(a_.f32, a_.f32, 0, 1, 0, 1, 0, 1, 0, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
         r_.f32[  i  ] = a_.f32[0];
         r_.f32[i + 1] = a_.f32[1];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_f32x2
  #define _mm256_broadcast_f32x2(a) easysimd_mm256_broadcast_f32x2(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_broadcast_f32x2(easysimd__m256 src, easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_mask_broadcast_f32x2(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1)), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1)), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_broadcast_f32x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_broadcast_f32x2
  #define _mm256_mask_broadcast_f32x2(src, k, a) easysimd_mm256_mask_broadcast_f32x2(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_broadcast_f32x2(easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_maskz_broadcast_f32x2(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1)), svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1)), svdup_n_f32(0.0));
    return r;
  #else
    return easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_broadcast_f32x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_broadcast_f32x2
  #define _mm256_maskz_broadcast_f32x2(k, a) easysimd_mm256_maskz_broadcast_f32x2(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_broadcast_f32x2 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_broadcast_f32x2(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    a.sve_f32 = svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1));
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_2] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_3] = a.sve_f32;
    return r;
  #else
    easysimd__m512_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.f32 = __builtin_shufflevector(a_.f32, a_.f32, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i+=2) {
         r_.f32[  i  ] = a_.f32[0];
         r_.f32[i + 1] = a_.f32[1];
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcast_f32x2
  #define _mm512_broadcast_f32x2(a) easysimd_mm512_broadcast_f32x2(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_broadcast_f32x2(easysimd__m512 src, easysimd__mmask16 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_mask_broadcast_f32x2(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    a.sve_f32 = svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1));
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_broadcast_f32x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcast_f32x2
  #define _mm512_mask_broadcast_f32x2(src, k, a) easysimd_mm512_mask_broadcast_f32x2(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_broadcast_f32x2(easysimd__mmask16 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_maskz_broadcast_f32x2(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    a.sve_f32 = svtbl_f32(a.sve_f32, svdupq_n_u32(0, 1, 0, 1));
    svfloat32_t svzero = svdup_n_f32(0.0);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32, svzero);
    return r;
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_broadcast_f32x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcast_f32x2
  #define _mm512_maskz_broadcast_f32x2(k, a) easysimd_mm512_maskz_broadcast_f32x2(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_broadcast_f32x8 (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_broadcast_f32x8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[EASYSIMD_SV_INDEX_0];
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[EASYSIMD_SV_INDEX_1];
    r.sve_f32[EASYSIMD_SV_INDEX_2] = a.sve_f32[EASYSIMD_SV_INDEX_0];
    r.sve_f32[EASYSIMD_SV_INDEX_3] = a.sve_f32[EASYSIMD_SV_INDEX_1];
    return r;
  #else
    easysimd__m512_private r_;
    easysimd__m256_private a_ = easysimd__m256_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.f32 = __builtin_shufflevector(a_.f32, a_.f32, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i+=8) {
         r_.f32[  i  ] = a_.f32[0];
         r_.f32[i + 1] = a_.f32[1];
         r_.f32[i + 2] = a_.f32[2];
         r_.f32[i + 3] = a_.f32[3];
         r_.f32[i + 4] = a_.f32[4];
         r_.f32[i + 5] = a_.f32[5];
         r_.f32[i + 6] = a_.f32[6];
         r_.f32[i + 7] = a_.f32[7];
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcast_f32x8
  #define _mm512_broadcast_f32x8(a) easysimd_mm512_broadcast_f32x8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_broadcast_f32x8(easysimd__m512 src, easysimd__mmask16 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_mask_broadcast_f32x8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_0], src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_1], src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_broadcast_f32x8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcast_f32x8
  #define _mm512_mask_broadcast_f32x8(src, k, a) easysimd_mm512_mask_broadcast_f32x8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_broadcast_f32x8(easysimd__mmask16 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_maskz_broadcast_f32x8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svfloat32_t svzero = svdup_n_f32(0.0);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_1], svzero);
    return r;
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_broadcast_f32x8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcast_f32x8
  #define _mm512_maskz_broadcast_f32x8(k, a) easysimd_mm512_maskz_broadcast_f32x8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_broadcast_f64x2 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_broadcast_f64x2(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64;
    r.sve_f64[EASYSIMD_SV_INDEX_1] = a.sve_f64;
    r.sve_f64[EASYSIMD_SV_INDEX_2] = a.sve_f64;
    r.sve_f64[EASYSIMD_SV_INDEX_3] = a.sve_f64;
    return r;
  #else
    easysimd__m512d_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector) && !defined(EASYSIMD_BUG_CLANG_BAD_VI64_OPS)
      r_.f64 = __builtin_shufflevector(a_.f64, a_.f64, 0, 1, 0, 1, 0, 1, 0, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
         r_.f64[  i  ] = a_.f64[0];
         r_.f64[i + 1] = a_.f64[1];
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcast_f64x2
  #define _mm512_broadcast_f64x2(a) easysimd_mm512_broadcast_f64x2(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_broadcast_f64x2(easysimd__m512d src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_mask_broadcast_f64x2(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64, src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64, src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64, src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64, src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_broadcast_f64x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcast_f64x2
  #define _mm512_mask_broadcast_f64x2(src, k, a) easysimd_mm512_mask_broadcast_f64x2(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_broadcast_f64x2(easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_maskz_broadcast_f64x2(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    svfloat64_t svzero = svdup_n_f64(0.0);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64, svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64, svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64, svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64, svzero);
    return r;
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_broadcast_f64x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcast_f64x2
  #define _mm512_maskz_broadcast_f64x2(k, a) easysimd_mm512_maskz_broadcast_f64x2(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcast_i32x4 (easysimd__m128i a) {
  #if  defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_broadcast_i32x4(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32;
    r.sve_i32[EASYSIMD_SV_INDEX_1] = a.sve_i32;
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i += 4) {
        r_.i32[  i  ] = a_.i32[0];
        r_.i32[i + 1] = a_.i32[1];
        r_.i32[i + 2] = a_.i32[2];
        r_.i32[i + 3] = a_.i32[3];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_i32x4
  #define _mm256_broadcast_i32x4(a) easysimd_mm256_broadcast_i32x4(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_broadcast_i32x4 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if  defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_broadcast_i32x4(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32, src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src);
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i += 4) {
        r_.i32[  i  ] = ((k >> i) & 1) ?  a_.i32[0] : src_.i32[i];
        r_.i32[i + 1] = ((k >> (i + 1)) & 1) ?  a_.i32[1] : src_.i32[i + 1];
        r_.i32[i + 2] = ((k >> (i + 2)) & 1) ?  a_.i32[2] : src_.i32[i + 2];
        r_.i32[i + 3] = ((k >> (i + 3)) & 1) ?  a_.i32[3] : src_.i32[i + 3];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_broadcast_i32x4
  #define _mm256_mask_broadcast_i32x4(src, k, a) easysimd_mm256_mask_broadcast_i32x4(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_broadcast_i32x4 (easysimd__mmask8 k, easysimd__m128i a) {
  #if  defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_broadcast_i32x4(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, svdup_n_s32(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32, svdup_n_s32(0));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i += 4) {
        r_.i32[  i  ] = ((k >> i) & 1) ?  a_.i32[0] : INT32_C(0);
        r_.i32[i + 1] = ((k >> (i + 1)) & 1) ?  a_.i32[1] : INT32_C(0);
        r_.i32[i + 2] = ((k >> (i + 2)) & 1) ?  a_.i32[2] : INT32_C(0);
        r_.i32[i + 3] = ((k >> (i + 3)) & 1) ?  a_.i32[3] : INT32_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_broadcast_i32x4
  #define _mm256_maskz_broadcast_i32x4(k, a) easysimd_mm256_maskz_broadcast_i32x4(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_broadcast_f32x4 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_broadcast_f32x4(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32;
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
        r_.m128_private[0] = a_;
        r_.m128_private[1] = a_;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.f32 = __builtin_shufflevector(a_.f32, a_.f32, 0, 1, 2, 3, 0, 1, 2, 3);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 4) {
         r_.f32[  i  ] = a_.f32[0];
         r_.f32[i + 1] = a_.f32[1];
         r_.f32[i + 2] = a_.f32[2];
         r_.f32[i + 3] = a_.f32[3];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_f32x4
  #define _mm256_broadcast_f32x4(a) easysimd_mm256_broadcast_f32x4(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_broadcast_f32x4(easysimd__m256 src, easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_broadcast_f32x4(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_ps(src, k, easysimd_mm256_broadcast_f32x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_broadcast_f32x4
  #define _mm256_mask_broadcast_f32x4(src, k, a) easysimd_mm256_mask_broadcast_f32x4(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_broadcast_f32x4(easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_broadcast_f32x4(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32, svdup_n_f32(0.0));
    return r;
  #else
    return easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_broadcast_f32x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_broadcast_f32x4
  #define _mm256_maskz_broadcast_f32x4(k, a) easysimd_mm256_maskz_broadcast_f32x4(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcast_i64x2 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_broadcast_i64x2(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = a.sve_i64;
    r.sve_i64[EASYSIMD_SV_INDEX_1] = a.sve_i64;
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i += 2) {
        r_.i64[  i  ] = a_.i64[0];
        r_.i64[i + 1] = a_.i64[1];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_i64x2
  #define _mm256_broadcast_i64x2(a) easysimd_mm256_broadcast_i64x2(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_broadcast_i64x2 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_broadcast_i64x2(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64, src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64, src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src);
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i += 2) {
        r_.i64[  i  ] = ((k >> i) & 1) ? a_.i64[0] : src_.i64[i];
        r_.i64[i + 1] = ((k >> (i + 1)) & 1) ? a_.i64[1] : src_.i64[i + 1];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_broadcast_i64x2
  #define _mm256_mask_broadcast_i64x2(src, k, a) easysimd_mm256_mask_broadcast_i64x2(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_broadcast_i64x2 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_broadcast_i64x2(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64, svdup_n_s64(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64, svdup_n_s64(0));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i += 2) {
        r_.i64[  i  ] = ((k >> i) & 1) ? a_.i64[0] : INT64_C(0);
        r_.i64[i + 1] = ((k >> (i + 1)) & 1) ? a_.i64[1] : INT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_broadcast_i64x2
  #define _mm256_maskz_broadcast_i64x2(k, a) easysimd_mm256_maskz_broadcast_i64x2(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_broadcast_f64x2 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_broadcast_f64x2(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64;
    r.sve_f64[EASYSIMD_SV_INDEX_1] = a.sve_f64;
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    /* I don't have a bug # for this, but when compiled with clang-10 without optimization on aarch64
     * the __builtin_shufflevector version doesn't work correctly.  clang 9 and 11 aren't a problem */
    #if defined(EASYSIMD_VECTOR_SUBSCRIPT) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector) && \
        (!defined(__clang__) || (EASYSIMD_DETECT_CLANG_VERSION < 100000 || EASYSIMD_DETECT_CLANG_VERSION > 100000))
      r_.f64 = __builtin_shufflevector(a_.f64, a_.f64, 0, 1, 0, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
         r_.f64[  i  ] = a_.f64[0];
         r_.f64[i + 1] = a_.f64[1];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_f64x2
  #define _mm256_broadcast_f64x2(a) easysimd_mm256_broadcast_f64x2(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_broadcast_f64x2(easysimd__m256d src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_mask_broadcast_f64x2(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64, src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64, src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_pd(src, k, easysimd_mm256_broadcast_f64x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_broadcast_f64x2
  #define _mm256_mask_broadcast_f64x2(src, k, a) easysimd_mm256_mask_broadcast_f64x2(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_broadcast_f64x2(easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_maskz_broadcast_f64x2(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64, svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64, svdup_n_f64(0.0));
    return r;
  #else
    return easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_broadcast_f64x2(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_broadcast_f64x2
  #define _mm256_maskz_broadcast_f64x2(k, a) easysimd_mm256_maskz_broadcast_f64x2(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_broadcast_f32x4 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcast_f32x4(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_2] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_3] = a.sve_f32;
    return r;
  #else
    easysimd__m512_private r_;

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      r_.m256[1] = r_.m256[0] = easysimd_mm256_castsi256_ps(easysimd_mm256_broadcastsi128_si256(easysimd_mm_castps_si128(a)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        r_.m128[i] = a;
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcast_f32x4
  #define _mm512_broadcast_f32x4(a) easysimd_mm512_broadcast_f32x4(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_broadcast_f32x4(easysimd__m512 src, easysimd__mmask16 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcast_f32x4(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32, src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_broadcast_f32x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcast_f32x4
  #define _mm512_mask_broadcast_f32x4(src, k, a) easysimd_mm512_mask_broadcast_f32x4(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_broadcast_f32x4(easysimd__mmask16 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcast_f32x4(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svfloat32_t svzero = svdup_n_f32(0.0);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32, svzero);
    return r;
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_broadcast_f32x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcast_f32x4
  #define _mm512_maskz_broadcast_f32x4(k, a) easysimd_mm512_maskz_broadcast_f32x4(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_broadcast_f64x4 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcast_f64x4(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64[EASYSIMD_SV_INDEX_0];
    r.sve_f64[EASYSIMD_SV_INDEX_1] = a.sve_f64[EASYSIMD_SV_INDEX_1];
    r.sve_f64[EASYSIMD_SV_INDEX_2] = a.sve_f64[EASYSIMD_SV_INDEX_0];
    r.sve_f64[EASYSIMD_SV_INDEX_3] = a.sve_f64[EASYSIMD_SV_INDEX_1];
    return r;
  #else
    easysimd__m512d_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
      r_.m256d[i] = a;
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcast_f64x4
  #define _mm512_broadcast_f64x4(a) easysimd_mm512_broadcast_f64x4(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_broadcast_f64x4(easysimd__m512d src, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcast_f64x4(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_0], src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_1], src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_broadcast_f64x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcast_f64x4
  #define _mm512_mask_broadcast_f64x4(src, k, a) easysimd_mm512_mask_broadcast_f64x4(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_broadcast_f64x4(easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcast_f64x4(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    svfloat64_t svzero = svdup_n_f64(0.0);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_1], svzero);
    return r;
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_broadcast_f64x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcast_f64x4
  #define _mm512_maskz_broadcast_f64x4(k, a) easysimd_mm512_maskz_broadcast_f64x4(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_broadcast_i32x4 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcast_i32x4(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32;
    r.sve_i32[EASYSIMD_SV_INDEX_1] = a.sve_i32;
    r.sve_i32[EASYSIMD_SV_INDEX_2] = a.sve_i32;
    r.sve_i32[EASYSIMD_SV_INDEX_3] = a.sve_i32;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = a.neon_i32;
    r.m128i[1].neon_i32 = a.neon_i32;
    r.m128i[2].neon_i32 = a.neon_i32;
    r.m128i[3].neon_i32 = a.neon_i32;
    return r;
  #else
    easysimd__m512i_private r_;

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      r_.m256i[1] = r_.m256i[0] = easysimd_mm256_broadcastsi128_si256(a);
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i[3] = r_.m128i[2] = r_.m128i[1] = r_.m128i[0] = a;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = a;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcast_i32x4
  #define _mm512_broadcast_i32x4(a) easysimd_mm512_broadcast_i32x4(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_broadcast_i32x4(easysimd__m512i src, easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcast_i32x4(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32, src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32, src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32, src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_broadcast_i32x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcast_i32x4
  #define _mm512_mask_broadcast_i32x4(src, k, a) easysimd_mm512_mask_broadcast_i32x4(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_broadcast_i32x4(easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcast_i32x4(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svint32_t svzero = svdup_n_s32(0);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, svzero);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32, svzero);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32, svzero);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32, svzero);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_broadcast_i32x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcast_i32x4
  #define _mm512_maskz_broadcast_i32x4(k, a) easysimd_mm512_maskz_broadcast_i32x4(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_broadcast_i64x4 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcast_i64x4(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = a.sve_i64[EASYSIMD_SV_INDEX_0];
    r.sve_i64[EASYSIMD_SV_INDEX_1] = a.sve_i64[EASYSIMD_SV_INDEX_1];
    r.sve_i64[EASYSIMD_SV_INDEX_2] = a.sve_i64[EASYSIMD_SV_INDEX_0];
    r.sve_i64[EASYSIMD_SV_INDEX_3] = a.sve_i64[EASYSIMD_SV_INDEX_1];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i64 = a.m128i[0].neon_i64;
    r.m128i[1].neon_i64 = a.m128i[1].neon_i64;
    r.m128i[2].neon_i64 = a.m128i[0].neon_i64;
    r.m128i[3].neon_i64 = a.m128i[1].neon_i64;
    return r;
  #else
    easysimd__m512i_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
      r_.m256i[i] = a;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcast_i64x4
  #define _mm512_broadcast_i64x4(a) easysimd_mm512_broadcast_i64x4(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_broadcast_i64x4(easysimd__m512i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint64_t g_mask_epi64[2] __attribute__((aligned(16))) = {0x01, 0x02};
    uint64x2_t vect_mask = vld1q_u64(g_mask_epi64);
    uint64x2_t tmp[4];
    tmp[0] = vtstq_u64(vdupq_n_u64(k & 0x03), vect_mask);
    tmp[1] = vtstq_u64(vdupq_n_u64((k & 0x0c) >> 2), vect_mask);
    tmp[2] = vtstq_u64(vdupq_n_u64((k & 0x30) >> 4), vect_mask);
    tmp[3] = vtstq_u64(vdupq_n_u64((k & 0xc0) >> 6), vect_mask);
    r.m128i[0].neon_i64 = vbslq_s64(tmp[0], a.m128i[0].neon_i64, src.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vbslq_s64(tmp[1], a.m128i[1].neon_i64, src.m128i[1].neon_i64);
    r.m128i[2].neon_i64 = vbslq_s64(tmp[2], a.m128i[0].neon_i64, src.m128i[2].neon_i64);
    r.m128i[3].neon_i64 = vbslq_s64(tmp[3], a.m128i[1].neon_i64, src.m128i[3].neon_i64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], src.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_0], src.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_1], src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcast_i64x4(src, k, a);
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_broadcast_i64x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcast_i64x4
  #define _mm512_mask_broadcast_i64x4(src, k, a) easysimd_mm512_mask_broadcast_i64x4(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_broadcast_i64x4(easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcast_i64x4(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svint64_t svzero = svdup_n_s64(0);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], svzero);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_0], svzero);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_1], svzero);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_broadcast_i64x4(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcast_i64x4
  #define _mm512_maskz_broadcast_i64x4(k, a) easysimd_mm512_maskz_broadcast_i64x4(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_broadcastd_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcastd_epi32(a);
  #else
    easysimd__m512i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[0];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcastd_epi32
  #define _mm512_broadcastd_epi32(a) easysimd_mm512_broadcastd_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_broadcastd_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcastd_epi32(src, k, a);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_broadcastd_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcastd_epi32
  #define _mm512_mask_broadcastd_epi32(src, k, a) easysimd_mm512_mask_broadcastd_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_broadcastd_epi32(easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcastd_epi32(k, a);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_broadcastd_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcastd_epi32
  #define _mm512_maskz_broadcastd_epi32(k, a) easysimd_mm512_maskz_broadcastd_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_broadcastq_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcastq_epi64(a);
  #else
    easysimd__m512i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[0];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcastq_epi64
  #define _mm512_broadcastq_epi64(a) easysimd_mm512_broadcastq_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_broadcastq_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcastq_epi64(src, k, a);
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_broadcastq_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcastq_epi64
  #define _mm512_mask_broadcastq_epi64(src, k, a) easysimd_mm512_mask_broadcastq_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_broadcastq_epi64(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcastq_epi64(k, a);
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_broadcastq_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcastq_epi64
  #define _mm512_maskz_broadcastq_epi64(k, a) easysimd_mm512_maskz_broadcastq_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_broadcastss_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcastss_ps(a);
  #else
    easysimd__m512_private r_;
    easysimd__m128_private a_= easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[0];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcastss_ps
  #define _mm512_broadcastss_ps(a) easysimd_mm512_broadcastss_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_broadcastss_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcastss_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svfloat32_t svtemp = svdup_n_f32(a.f32[0]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtemp, src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svtemp, src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svtemp, src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svtemp, src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      src_ = easysimd__m512_to_private(src),
      r_;
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a);


    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[0] : src_.f32[i];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcastss_ps
  #define _mm512_mask_broadcastss_ps(src, k, a) easysimd_mm512_mask_broadcastss_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_broadcastss_ps(easysimd__mmask16 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcastss_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svfloat32_t svtemp = svdup_n_f32(a.f32[0]),
                svzero = svdup_n_f32(0);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svtemp, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svtemp, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svtemp, svzero);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svtemp, svzero);
    return r;
  #else
    easysimd__m512_private
      r_;
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[0] : INT32_C(0);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcastss_ps
  #define _mm512_maskz_broadcastss_ps(k, a) easysimd_mm512_maskz_broadcastss_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_broadcastsd_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_broadcastsd_pd(a);
  #else
    easysimd__m512d_private r_;
    easysimd__m128d_private a_= easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = a_.f64[0];
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcastsd_pd
  #define _mm512_broadcastsd_pd(a) easysimd_mm512_broadcastsd_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_broadcastsd_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_broadcastsd_pd(src, k, a);
  #else
    easysimd__m512d_private
      src_ = easysimd__m512d_to_private(src),
      r_;
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[0] : src_.f64[i];
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcastsd_pd
  #define _mm512_mask_broadcastsd_pd(src, k, a) easysimd_mm512_mask_broadcastsd_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_broadcastsd_pd(easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_broadcastsd_pd(k, a);
  #else
    easysimd__m512d_private
      r_;
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[0] : INT64_C(0);
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcastsd_pd
  #define _mm512_maskz_broadcastsd_pd(k, a) easysimd_mm512_maskz_broadcastsd_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_broadcastb_epi8 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_broadcastb_epi8(a);
  #else
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);
    return easysimd_mm512_set1_epi8(a_.i8[0]);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcastb_epi8
  #define _mm512_broadcastb_epi8(a) easysimd_mm512_broadcastb_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_broadcastb_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_broadcastb_epi8(src, k, a);
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_broadcastb_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_broadcastb_epi8
  #define _mm512_mask_broadcastb_epi8(src, k, a) easysimd_mm512_mask_broadcastb_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_broadcastb_epi8 (easysimd__mmask64 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_broadcastb_epi8(k, a);
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_broadcastb_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_broadcastb_epi8
  #define _mm512_maskz_broadcastb_epi8(k, a) easysimd_mm512_maskz_broadcastb_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_broadcastw_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_broadcastw_epi16(a);
  #else
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);
    return easysimd_mm512_set1_epi16(a_.i16[0]);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_broadcastw_epi16
  #define _mm512_broadcastw_epi16(a) easysimd_mm512_broadcastw_epi16(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_BROADCAST_H) */
