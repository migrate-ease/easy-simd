#if !defined(EASYSIMD_X86_AVX512_POPCNT_H)
#define EASYSIMD_X86_AVX512_POPCNT_H

#include "types.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_popcnt_epi8 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_popcnt_epi8(a);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i8 = vcntq_s8(a_.neon_i8);
    #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
      const __m128i low_nibble_set = _mm_set1_epi8(0x0f);
      const __m128i high_nibble_of_input = _mm_andnot_si128(low_nibble_set, a_.ni);
      const __m128i low_nibble_of_input = _mm_and_si128(low_nibble_set, a_.ni);
      const __m128i lut = _mm_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

      r_.ni =
        _mm_add_epi8(
          _mm_shuffle_epi8(
            lut,
            low_nibble_of_input
          ),
          _mm_shuffle_epi8(
            lut,
            _mm_srli_epi16(
              high_nibble_of_input,
              4
            )
          )
        );
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      /* v -= ((v >> 1) & UINT8_C(0x55)); */
      r_.ni =
        _mm_sub_epi8(
          a_.ni,
          _mm_and_si128(
            _mm_srli_epi16(a_.ni, 1),
            _mm_set1_epi8(0x55)
          )
        );

      /* v  = (v & 0x33) + ((v >> 2) & 0x33); */
      r_.ni =
        _mm_add_epi8(
          _mm_and_si128(
            r_.ni,
            _mm_set1_epi8(0x33)
          ),
          _mm_and_si128(
            _mm_srli_epi16(r_.ni, 2),
            _mm_set1_epi8(0x33)
          )
        );

      /* v = (v + (v >> 4)) & 0xf */
      r_.ni =
        _mm_and_si128(
          _mm_add_epi8(
            r_.ni,
            _mm_srli_epi16(r_.ni, 4)
          ),
          _mm_set1_epi8(0x0f)
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u8 -= ((a_.u8 >> 1) & 0x55);
      a_.u8  = ((a_.u8 & 0x33) + ((a_.u8 >> 2) & 0x33));
      a_.u8  = (a_.u8 + (a_.u8 >> 4)) & 15;
      r_.u8  = a_.u8 >> ((sizeof(uint8_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        uint8_t v = HEDLEY_STATIC_CAST(uint8_t, a_.u8[i]);
        v -= ((v >> 1) & 0x55);
        v  = (v & 0x33) + ((v >> 2) & 0x33);
        v  = (v + (v >> 4)) & 0xf;
        r_.u8[i] = v >> (sizeof(uint8_t) - 1) * CHAR_BIT;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_popcnt_epi8
  #define _mm_popcnt_epi8(a) easysimd_mm_popcnt_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_popcnt_epi8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_popcnt_epi8(src, k, a);
  #else
    return easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_popcnt_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_popcnt_epi8
  #define _mm_mask_popcnt_epi8(src, k, a) easysimd_mm_mask_popcnt_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_popcnt_epi8 (easysimd__mmask16 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_popcnt_epi8(k, a);
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_popcnt_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_popcnt_epi8
  #define _mm_maskz_popcnt_epi8(k, a) easysimd_mm_maskz_popcnt_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_popcnt_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_popcnt_epi16(a);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i16 = vpaddlq_s8(vcntq_s8(a_.neon_i8));
    #elif defined(EASYSIMD_X86_XOP_NATIVE)
      const __m128i low_nibble_set = _mm_set1_epi8(0x0f);
      const __m128i high_nibble_of_input = _mm_andnot_si128(low_nibble_set, a_.n);
      const __m128i low_nibble_of_input = _mm_and_si128(low_nibble_set, a_.n);
      const __m128i lut = _mm_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

      r_.n =
        _mm_haddw_epi8(
          _mm_add_epi8(
            _mm_shuffle_epi8(
              lut,
              low_nibble_of_input
            ),
            _mm_shuffle_epi8(
              lut,
              _mm_srli_epi16(high_nibble_of_input, 4)
            )
          )
        );
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.ni =
        _mm_sub_epi16(
          a_.ni,
          _mm_and_si128(
            _mm_srli_epi16(a_.ni, 1),
            _mm_set1_epi16(0x5555)
          )
        );

      r_.ni =
        _mm_add_epi16(
          _mm_and_si128(
            r_.ni,
            _mm_set1_epi16(0x3333)
          ),
          _mm_and_si128(
            _mm_srli_epi16(r_.ni, 2),
            _mm_set1_epi16(0x3333)
          )
        );

      r_.ni =
        _mm_and_si128(
          _mm_add_epi16(
            r_.ni,
            _mm_srli_epi16(r_.ni, 4)
          ),
          _mm_set1_epi16(0x0f0f)
        );

      r_.ni =
        _mm_srli_epi16(
          _mm_mullo_epi16(
            r_.ni,
            _mm_set1_epi16(0x0101)
          ),
          (sizeof(uint16_t) - 1) * CHAR_BIT
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u16 -= ((a_.u16 >> 1) & UINT16_C(0x5555));
      a_.u16  = ((a_.u16 & UINT16_C(0x3333)) + ((a_.u16 >> 2) & UINT16_C(0x3333)));
      a_.u16  = (a_.u16 + (a_.u16 >> 4)) & UINT16_C(0x0f0f);
      r_.u16  = (a_.u16 * UINT16_C(0x0101)) >> ((sizeof(uint16_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        uint16_t v = HEDLEY_STATIC_CAST(uint16_t, a_.u16[i]);
        v -= ((v >> 1) & UINT16_C(0x5555));
        v  = ((v & UINT16_C(0x3333)) + ((v >> 2) & UINT16_C(0x3333)));
        v  = (v + (v >> 4)) & UINT16_C(0x0f0f);
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (v * UINT16_C(0x0101))) >> ((sizeof(uint16_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_popcnt_epi16
  #define _mm_popcnt_epi16(a) easysimd_mm_popcnt_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_popcnt_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_popcnt_epi16(src, k, a);
  #else
    return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_popcnt_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_popcnt_epi16
  #define _mm_mask_popcnt_epi16(src, k, a) easysimd_mm_mask_popcnt_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_popcnt_epi16 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_popcnt_epi16(k, a);
  #else
    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_popcnt_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_popcnt_epi16
  #define _mm_maskz_popcnt_epi16(k, a) easysimd_mm_maskz_popcnt_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_popcnt_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_popcnt_epi32(a);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vpaddlq_s16(vpaddlq_s8(vcntq_s8(a_.neon_i8)));
    #elif defined(EASYSIMD_X86_XOP_NATIVE)
      const __m128i low_nibble_set = _mm_set1_epi8(0x0f);
      const __m128i high_nibble_of_input = _mm_andnot_si128(low_nibble_set, a_.n);
      const __m128i low_nibble_of_input = _mm_and_si128(low_nibble_set, a_.n);
      const __m128i lut = _mm_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

      r_.n =
        _mm_haddd_epi8(
          _mm_add_epi8(
            _mm_shuffle_epi8(
              lut,
              low_nibble_of_input
            ),
            _mm_shuffle_epi8(
              lut,
              _mm_srli_epi16(high_nibble_of_input, 4)
            )
          )
        );
    #elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.ni =
        _mm_sub_epi32(
          a_.ni,
          _mm_and_si128(
            _mm_srli_epi32(a_.ni, 1),
            _mm_set1_epi32(0x55555555)
          )
        );

      r_.ni =
        _mm_add_epi32(
          _mm_and_si128(
            r_.ni,
            _mm_set1_epi32(0x33333333)
          ),
          _mm_and_si128(
            _mm_srli_epi32(r_.ni, 2),
            _mm_set1_epi32(0x33333333)
          )
        );

      r_.ni =
        _mm_and_si128(
          _mm_add_epi32(
            r_.ni,
            _mm_srli_epi32(r_.ni, 4)
          ),
          _mm_set1_epi32(0x0f0f0f0f)
        );

      r_.ni =
        _mm_srli_epi32(
          _mm_mullo_epi32(
            r_.ni,
            _mm_set1_epi32(0x01010101)
          ),
          (sizeof(uint32_t) - 1) * CHAR_BIT
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u32 -= ((a_.u32 >> 1) & UINT32_C(0x55555555));
      a_.u32  = ((a_.u32 & UINT32_C(0x33333333)) + ((a_.u32 >> 2) & UINT32_C(0x33333333)));
      a_.u32  = (a_.u32 + (a_.u32 >> 4)) & UINT32_C(0x0f0f0f0f);
      r_.u32  = (a_.u32 * UINT32_C(0x01010101)) >> ((sizeof(uint32_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        uint32_t v = HEDLEY_STATIC_CAST(uint32_t, a_.u32[i]);
        v -= ((v >> 1) & UINT32_C(0x55555555));
        v  = ((v & UINT32_C(0x33333333)) + ((v >> 2) & UINT32_C(0x33333333)));
        v  = (v + (v >> 4)) & UINT32_C(0x0f0f0f0f);
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (v * UINT32_C(0x01010101))) >> ((sizeof(uint32_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_popcnt_epi32
  #define _mm_popcnt_epi32(a) easysimd_mm_popcnt_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_popcnt_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_popcnt_epi32(src, k, a);
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_popcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_popcnt_epi32
  #define _mm_mask_popcnt_epi32(src, k, a) easysimd_mm_mask_popcnt_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_popcnt_epi32 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_popcnt_epi32(k, a);
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_popcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_popcnt_epi32
  #define _mm_maskz_popcnt_epi32(k, a) easysimd_mm_maskz_popcnt_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_mm_popcnt_u32(unsigned int a)
{
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svdupq_n_b32(1, 0, 0, 0);
    return (int32_t)svaddv_u32(pg, svcnt_u32_z(pg, svdup_n_u32(a)));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return (int32_t)vaddlv_u8(vcnt_u8(vcreate_u8((uint64_t)a)));
  #elif defined (EASYSIMD_X86_POPCNT_NATIVE)
    return _mm_popcnt_u32(a);
  #else
    int count = 0;
    for(int i = 0; i < 32; i++){
        count += ((a >> i) & 1);
    }
    return count;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_popcnt_u32
  #define _mm_popcnt_u32(a) easysimd_mm_popcnt_u32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_mm_popcnt_u64(uint64_t a)
{
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t    pg = svdupq_n_b64(1, 0);
    return (int32_t)svaddv_u64(pg, svcnt_u64_z(pg, svdup_n_u64(a)));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return (int32_t)vaddlv_u8(vcnt_u8(vcreate_u8(a)));
  #elif defined (EASYSIMD_X86_POPCNT_NATIVE)
    return _mm_popcnt_u64(a);
  #else
    int count = 0;
    for(int i = 0; i < 64; i++){
        count += ((a >> i) & 1);
    }
    return count;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_popcnt_u64
  #define _mm_popcnt_u64(a) easysimd_mm_popcnt_u64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_popcnt_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_popcnt_epi64(a);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vpaddlq_s32(vpaddlq_s16(vpaddlq_s8(vcntq_s8(a_.neon_i8))));
    #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
      const __m128i low_nibble_set = _mm_set1_epi8(0x0f);
      const __m128i high_nibble_of_input = _mm_andnot_si128(low_nibble_set, a_.ni);
      const __m128i low_nibble_of_input = _mm_and_si128(low_nibble_set, a_.ni);
      const __m128i lut = _mm_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

      r_.ni =
        _mm_sad_epu8(
          _mm_add_epi8(
            _mm_shuffle_epi8(
              lut,
              low_nibble_of_input
            ),
            _mm_shuffle_epi8(
              lut,
              _mm_srli_epi16(high_nibble_of_input, 4)
            )
          ),
          _mm_setzero_si128()
        );
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.ni =
        _mm_sub_epi8(
          a_.ni,
          _mm_and_si128(
            _mm_srli_epi16(a_.ni, 1),
            _mm_set1_epi8(0x55)
          )
        );

      r_.ni =
        _mm_add_epi8(
          _mm_and_si128(
            r_.ni,
            _mm_set1_epi8(0x33)
          ),
          _mm_and_si128(
            _mm_srli_epi16(r_.ni, 2),
            _mm_set1_epi8(0x33)
          )
        );

      r_.ni =
        _mm_and_si128(
          _mm_add_epi8(
            r_.ni,
            _mm_srli_epi16(r_.ni, 4)
          ),
          _mm_set1_epi8(0x0f)
        );

      r_.ni =
        _mm_sad_epu8(
          r_.ni,
          _mm_setzero_si128()
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u64 -= ((a_.u64 >> 1) & UINT64_C(0x5555555555555555));
      a_.u64  = ((a_.u64 & UINT64_C(0x3333333333333333)) + ((a_.u64 >> 2) & UINT64_C(0x3333333333333333)));
      a_.u64  = (a_.u64 + (a_.u64 >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
      r_.u64  = (a_.u64 * UINT64_C(0x0101010101010101)) >> ((sizeof(uint64_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        uint64_t v = HEDLEY_STATIC_CAST(uint64_t, a_.u64[i]);
        v -= ((v >> 1) & UINT64_C(0x5555555555555555));
        v  = ((v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333)));
        v  = (v + (v >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
        r_.u64[i] = HEDLEY_STATIC_CAST(uint64_t, (v * UINT64_C(0x0101010101010101))) >> ((sizeof(uint64_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_popcnt_epi64
  #define _mm_popcnt_epi64(a) easysimd_mm_popcnt_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_popcnt_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_popcnt_epi64(src, k, a);
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_popcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_popcnt_epi64
  #define _mm_mask_popcnt_epi64(src, k, a) easysimd_mm_mask_popcnt_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_popcnt_epi64 (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_popcnt_epi64(k, a);
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_popcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_popcnt_epi64
  #define _mm_maskz_popcnt_epi64(k, a) easysimd_mm_maskz_popcnt_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_popcnt_epi8 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_popcnt_epi8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svcnt_u8_x(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svcnt_u8_x(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi8(a_.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      const __m256i low_nibble_set = _mm256_set1_epi8(0x0f);
      const __m256i high_nibble_of_input = _mm256_andnot_si256(low_nibble_set, a_.ni);
      const __m256i low_nibble_of_input = _mm256_and_si256(low_nibble_set, a_.ni);
      const __m256i lut =
        _mm256_set_epi8(
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
        );

      r_.ni =
        _mm256_add_epi8(
          _mm256_shuffle_epi8(
            lut,
            low_nibble_of_input
          ),
          _mm256_shuffle_epi8(
            lut,
            _mm256_srli_epi16(
              high_nibble_of_input,
              4
            )
          )
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u8 -= ((a_.u8 >> 1) & 0x55);
      a_.u8  = ((a_.u8 & 0x33) + ((a_.u8 >> 2) & 0x33));
      a_.u8  = (a_.u8 + (a_.u8 >> 4)) & 15;
      r_.u8  = a_.u8 >> ((sizeof(uint8_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        uint8_t v = HEDLEY_STATIC_CAST(uint8_t, a_.u8[i]);
        v -= ((v >> 1) & 0x55);
        v  = (v & 0x33) + ((v >> 2) & 0x33);
        v  = (v + (v >> 4)) & 0xf;
        r_.u8[i] = v >> (sizeof(uint8_t) - 1) * CHAR_BIT;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_popcnt_epi8
  #define _mm256_popcnt_epi8(a) easysimd_mm256_popcnt_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_popcnt_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_popcnt_epi8(src, k, a);
  #else
    return easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_popcnt_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_popcnt_epi8
  #define _mm256_mask_popcnt_epi8(src, k, a) easysimd_mm256_mask_popcnt_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_popcnt_epi8 (easysimd__mmask32 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_popcnt_epi8(k, a);
  #else
    return easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_popcnt_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_popcnt_epi8
  #define _mm256_maskz_popcnt_epi8(k, a) easysimd_mm256_maskz_popcnt_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_popcnt_epi16 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_popcnt_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svcnt_u16_x(svptrue_b8(), a.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svcnt_u16_x(svptrue_b8(), a.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi16(a_.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      r_.ni =
        _mm256_sub_epi16(
          a_.ni,
          _mm256_and_si256(
            _mm256_srli_epi16(a_.ni, 1),
            _mm256_set1_epi16(0x5555)
          )
        );

      r_.ni =
        _mm256_add_epi16(
          _mm256_and_si256(
            r_.ni,
            _mm256_set1_epi16(0x3333)
          ),
          _mm256_and_si256(
            _mm256_srli_epi16(r_.ni, 2),
            _mm256_set1_epi16(0x3333)
          )
        );

      r_.ni =
        _mm256_and_si256(
          _mm256_add_epi16(
            r_.ni,
            _mm256_srli_epi16(r_.ni, 4)
          ),
          _mm256_set1_epi16(0x0f0f)
        );

      r_.ni =
        _mm256_srli_epi16(
          _mm256_mullo_epi16(
            r_.ni,
            _mm256_set1_epi16(0x0101)
          ),
          (sizeof(uint16_t) - 1) * CHAR_BIT
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u16 -= ((a_.u16 >> 1) & UINT16_C(0x5555));
      a_.u16  = ((a_.u16 & UINT16_C(0x3333)) + ((a_.u16 >> 2) & UINT16_C(0x3333)));
      a_.u16  = (a_.u16 + (a_.u16 >> 4)) & UINT16_C(0x0f0f);
      r_.u16  = (a_.u16 * UINT16_C(0x0101)) >> ((sizeof(uint16_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        uint16_t v = HEDLEY_STATIC_CAST(uint16_t, a_.u16[i]);
        v -= ((v >> 1) & UINT16_C(0x5555));
        v  = ((v & UINT16_C(0x3333)) + ((v >> 2) & UINT16_C(0x3333)));
        v  = (v + (v >> 4)) & UINT16_C(0x0f0f);
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (v * UINT16_C(0x0101))) >> ((sizeof(uint16_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_popcnt_epi16
  #define _mm256_popcnt_epi16(a) easysimd_mm256_popcnt_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_popcnt_epi16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_popcnt_epi16(src, k, a);
  #else
    return easysimd_mm256_mask_mov_epi16(src, k, easysimd_mm256_popcnt_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_popcnt_epi16
  #define _mm256_mask_popcnt_epi16(src, k, a) easysimd_mm256_mask_popcnt_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_popcnt_epi16 (easysimd__mmask16 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_popcnt_epi16(k, a);
  #else
    return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_popcnt_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_popcnt_epi16
  #define _mm256_maskz_popcnt_epi16(k, a) easysimd_mm256_maskz_popcnt_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_popcnt_epi32 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_popcnt_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svcnt_u32_x(svptrue_b8(), a.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svcnt_u32_x(svptrue_b8(), a.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi32(a_.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      r_.ni =
        _mm256_sub_epi32(
          a_.ni,
          _mm256_and_si256(
            _mm256_srli_epi32(a_.ni, 1),
            _mm256_set1_epi32(0x55555555)
          )
        );

      r_.ni =
        _mm256_add_epi32(
          _mm256_and_si256(
            r_.ni,
            _mm256_set1_epi32(0x33333333)
          ),
          _mm256_and_si256(
            _mm256_srli_epi32(r_.ni, 2),
            _mm256_set1_epi32(0x33333333)
          )
        );

      r_.ni =
        _mm256_and_si256(
          _mm256_add_epi32(
            r_.ni,
            _mm256_srli_epi32(r_.ni, 4)
          ),
          _mm256_set1_epi32(0x0f0f0f0f)
        );

      r_.ni =
        _mm256_srli_epi32(
          _mm256_mullo_epi32(
            r_.ni,
            _mm256_set1_epi32(0x01010101)
          ),
          (sizeof(uint32_t) - 1) * CHAR_BIT
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u32 -= ((a_.u32 >> 1) & UINT32_C(0x55555555));
      a_.u32  = ((a_.u32 & UINT32_C(0x33333333)) + ((a_.u32 >> 2) & UINT32_C(0x33333333)));
      a_.u32  = (a_.u32 + (a_.u32 >> 4)) & UINT32_C(0x0f0f0f0f);
      r_.u32  = (a_.u32 * UINT32_C(0x01010101)) >> ((sizeof(uint32_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        uint32_t v = HEDLEY_STATIC_CAST(uint32_t, a_.u32[i]);
        v -= ((v >> 1) & UINT32_C(0x55555555));
        v  = ((v & UINT32_C(0x33333333)) + ((v >> 2) & UINT32_C(0x33333333)));
        v  = (v + (v >> 4)) & UINT32_C(0x0f0f0f0f);
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (v * UINT32_C(0x01010101))) >> ((sizeof(uint32_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_popcnt_epi32
  #define _mm256_popcnt_epi32(a) easysimd_mm256_popcnt_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_popcnt_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_popcnt_epi32(src, k, a);
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_popcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_popcnt_epi32
  #define _mm256_mask_popcnt_epi32(src, k, a) easysimd_mm256_mask_popcnt_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_popcnt_epi32 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_popcnt_epi32(k, a);
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_popcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_popcnt_epi32
  #define _mm256_maskz_popcnt_epi32(k, a) easysimd_mm256_maskz_popcnt_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_popcnt_epi64 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_popcnt_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svcnt_u64_x(svptrue_b8(), a.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svcnt_u64_x(svptrue_b8(), a.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < sizeof(r_.m128i) / sizeof(r_.m128i[0]) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi64(a_.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      const __m256i low_nibble_set = _mm256_set1_epi8(0x0f);
      const __m256i high_nibble_of_input = _mm256_andnot_si256(low_nibble_set, a_.ni);
      const __m256i low_nibble_of_input = _mm256_and_si256(low_nibble_set, a_.ni);
      const __m256i lut =
        _mm256_set_epi8(
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
        );

      r_.ni =
        _mm256_sad_epu8(
          _mm256_add_epi8(
            _mm256_shuffle_epi8(
              lut,
              low_nibble_of_input
            ),
            _mm256_shuffle_epi8(
              lut,
              _mm256_srli_epi16(high_nibble_of_input, 4)
            )
          ),
          _mm256_setzero_si256()
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u64 -= ((a_.u64 >> 1) & UINT64_C(0x5555555555555555));
      a_.u64  = ((a_.u64 & UINT64_C(0x3333333333333333)) + ((a_.u64 >> 2) & UINT64_C(0x3333333333333333)));
      a_.u64  = (a_.u64 + (a_.u64 >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
      r_.u64  = (a_.u64 * UINT64_C(0x0101010101010101)) >> ((sizeof(uint64_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        uint64_t v = HEDLEY_STATIC_CAST(uint64_t, a_.u64[i]);
        v -= ((v >> 1) & UINT64_C(0x5555555555555555));
        v  = ((v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333)));
        v  = (v + (v >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
        r_.u64[i] = HEDLEY_STATIC_CAST(uint64_t, (v * UINT64_C(0x0101010101010101))) >> ((sizeof(uint64_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_popcnt_epi64
  #define _mm256_popcnt_epi64(a) easysimd_mm256_popcnt_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_popcnt_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_popcnt_epi64(src, k, a);
  #else
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_popcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_popcnt_epi64
  #define _mm256_mask_popcnt_epi64(src, k, a) easysimd_mm256_mask_popcnt_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_popcnt_epi64 (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_popcnt_epi64(k, a);
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_popcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_popcnt_epi64
  #define _mm256_maskz_popcnt_epi64(k, a) easysimd_mm256_maskz_popcnt_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_popcnt_epi8 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_popcnt_epi8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svcnt_u8_x(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svcnt_u8_x(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svcnt_u8_x(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svcnt_u8_x(svptrue_b8(), a.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi8(a_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_popcnt_epi8(a_.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
      const __m512i low_nibble_set = _mm512_set1_epi8(0x0f);
      const __m512i high_nibble_of_input = _mm512_andnot_si512(low_nibble_set, a_.n);
      const __m512i low_nibble_of_input = _mm512_and_si512(low_nibble_set, a_.n);
      const __m512i lut =
        easysimd_mm512_set_epi8(
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
        );

      r_.n =
        _mm512_add_epi8(
          _mm512_shuffle_epi8(
            lut,
            low_nibble_of_input
          ),
          _mm512_shuffle_epi8(
            lut,
            _mm512_srli_epi16(
              high_nibble_of_input,
              4
            )
          )
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u8 -= ((a_.u8 >> 1) & 0x55);
      a_.u8  = ((a_.u8 & 0x33) + ((a_.u8 >> 2) & 0x33));
      a_.u8  = (a_.u8 + (a_.u8 >> 4)) & 15;
      r_.u8  = a_.u8 >> ((sizeof(uint8_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        uint8_t v = HEDLEY_STATIC_CAST(uint8_t, a_.u8[i]);
        v -= ((v >> 1) & 0x55);
        v  = (v & 0x33) + ((v >> 2) & 0x33);
        v  = (v + (v >> 4)) & 0xf;
        r_.u8[i] = v >> (sizeof(uint8_t) - 1) * CHAR_BIT;
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_popcnt_epi8
  #define _mm512_popcnt_epi8(a) easysimd_mm512_popcnt_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_popcnt_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_mask_popcnt_epi8(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svcnt_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svcnt_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), svcnt_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_2]), src.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), svcnt_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_3]), src.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_popcnt_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_popcnt_epi8
  #define _mm512_mask_popcnt_epi8(src, k, a) easysimd_mm512_mask_popcnt_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_popcnt_epi8 (easysimd__mmask64 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_maskz_popcnt_epi8(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svcnt_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svcnt_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svcnt_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svcnt_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_popcnt_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_popcnt_epi8
  #define _mm512_maskz_popcnt_epi8(k, a) easysimd_mm512_maskz_popcnt_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_popcnt_epi16 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_popcnt_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svcnt_u16_x(svptrue_b8(), a.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svcnt_u16_x(svptrue_b8(), a.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svcnt_u16_x(svptrue_b8(), a.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svcnt_u16_x(svptrue_b8(), a.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi16(a_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_popcnt_epi16(a_.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
      r_.n =
        _mm512_sub_epi16(
          a_.n,
          _mm512_and_si512(
            _mm512_srli_epi16(a_.n, 1),
            _mm512_set1_epi16(0x5555)
          )
        );

      r_.n =
        _mm512_add_epi16(
          _mm512_and_si512(
            r_.n,
            _mm512_set1_epi16(0x3333)
          ),
          _mm512_and_si512(
            _mm512_srli_epi16(r_.n, 2),
            _mm512_set1_epi16(0x3333)
          )
        );

      r_.n =
        _mm512_and_si512(
          _mm512_add_epi16(
            r_.n,
            _mm512_srli_epi16(r_.n, 4)
          ),
          _mm512_set1_epi16(0x0f0f)
        );

      r_.n =
        _mm512_srli_epi16(
          _mm512_mullo_epi16(
            r_.n,
            _mm512_set1_epi16(0x0101)
          ),
          (sizeof(uint16_t) - 1) * CHAR_BIT
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u16 -= ((a_.u16 >> 1) & UINT16_C(0x5555));
      a_.u16  = ((a_.u16 & UINT16_C(0x3333)) + ((a_.u16 >> 2) & UINT16_C(0x3333)));
      a_.u16  = (a_.u16 + (a_.u16 >> 4)) & UINT16_C(0x0f0f);
      r_.u16  = (a_.u16 * UINT16_C(0x0101)) >> ((sizeof(uint16_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        uint16_t v = HEDLEY_STATIC_CAST(uint16_t, a_.u16[i]);
        v -= ((v >> 1) & UINT16_C(0x5555));
        v  = ((v & UINT16_C(0x3333)) + ((v >> 2) & UINT16_C(0x3333)));
        v  = (v + (v >> 4)) & UINT16_C(0x0f0f);
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (v * UINT16_C(0x0101))) >> ((sizeof(uint16_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_popcnt_epi16
  #define _mm512_popcnt_epi16(a) easysimd_mm512_popcnt_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_popcnt_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_mask_popcnt_epi16(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svcnt_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svcnt_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svcnt_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_2]), src.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svcnt_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_3]), src.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_popcnt_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_popcnt_epi16
  #define _mm512_mask_popcnt_epi16(src, k, a) easysimd_mm512_mask_popcnt_epi16(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_popcnt_epi16 (easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    return _mm512_maskz_popcnt_epi16(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svcnt_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svcnt_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svcnt_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svcnt_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_popcnt_epi16(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_popcnt_epi16
  #define _mm512_maskz_popcnt_epi16(k, a) easysimd_mm512_maskz_popcnt_epi16(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_popcnt_epi32 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE)
    return _mm512_popcnt_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svcnt_u32_x(svptrue_b8(), a.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svcnt_u32_x(svptrue_b8(), a.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svcnt_u32_x(svptrue_b8(), a.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svcnt_u32_x(svptrue_b8(), a.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi32(a_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_popcnt_epi32(a_.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
      r_.ni =
        _mm512_sub_epi32(
          r_.ni,
          _mm512_and_si512(
            _mm512_srli_epi32(r_.ni, 1),
            _mm512_set1_epi32(0x55555555)
          )
        );

      r_.ni =
        _mm512_add_epi32(
          _mm512_and_si512(
            r_.ni,
            _mm512_set1_epi32(0x33333333)
          ),
          _mm512_and_si512(
            _mm512_srli_epi32(r_.ni, 2),
            _mm512_set1_epi32(0x33333333)
          )
        );

      r_.ni =
        _mm512_and_si512(
          _mm512_add_epi32(
            r_.ni,
            _mm512_srli_epi32(r_.ni, 4)
          ),
          _mm512_set1_epi32(0x0f0f0f0f)
        );

      r_.ni =
        _mm512_srli_epi32(
          _mm512_mullo_epi32(
            r_.ni,
            _mm512_set1_epi32(0x01010101)
          ),
          (sizeof(uint32_t) - 1) * CHAR_BIT
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u32 -= ((a_.u32 >> 1) & UINT32_C(0x55555555));
      a_.u32  = ((a_.u32 & UINT32_C(0x33333333)) + ((a_.u32 >> 2) & UINT32_C(0x33333333)));
      a_.u32  = (a_.u32 + (a_.u32 >> 4)) & UINT32_C(0x0f0f0f0f);
      r_.u32  = (a_.u32 * UINT32_C(0x01010101)) >> ((sizeof(uint32_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        uint32_t v = HEDLEY_STATIC_CAST(uint32_t, a_.u32[i]);
        v -= ((v >> 1) & UINT32_C(0x55555555));
        v  = ((v & UINT32_C(0x33333333)) + ((v >> 2) & UINT32_C(0x33333333)));
        v  = (v + (v >> 4)) & UINT32_C(0x0f0f0f0f);
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (v * UINT32_C(0x01010101))) >> ((sizeof(uint32_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_popcnt_epi32
  #define _mm512_popcnt_epi32(a) easysimd_mm512_popcnt_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_popcnt_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE)
    return _mm512_mask_popcnt_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcnt_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svcnt_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svcnt_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2]), src.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svcnt_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3]), src.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_popcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_popcnt_epi32
  #define _mm512_mask_popcnt_epi32(src, k, a) easysimd_mm512_mask_popcnt_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_popcnt_epi32 (easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE)
    return _mm512_maskz_popcnt_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svcnt_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svcnt_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svcnt_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svcnt_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_popcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_popcnt_epi32
  #define _mm512_maskz_popcnt_epi32(k, a) easysimd_mm512_maskz_popcnt_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_popcnt_epi64 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE)
    return _mm512_popcnt_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svcnt_u64_x(svptrue_b8(), a.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svcnt_u64_x(svptrue_b8(), a.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svcnt_u64_x(svptrue_b8(), a.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svcnt_u64_x(svptrue_b8(), a.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_popcnt_epi64(a_.m128i[i]);
      }
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < sizeof(r_.m256i) / sizeof(r_.m256i[0]) ; i++) {
        r_.m256i[i] = easysimd_mm256_popcnt_epi64(a_.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
      const __m512i low_nibble_set = _mm512_set1_epi8(0x0f);
      const __m512i high_nibble_of_input = _mm512_andnot_si512(low_nibble_set, a_.n);
      const __m512i low_nibble_of_input = _mm512_and_si512(low_nibble_set, a_.n);
      const __m512i lut =
        easysimd_mm512_set_epi8(
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
          4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
        );

      r_.n =
        _mm512_sad_epu8(
          _mm512_add_epi8(
            _mm512_shuffle_epi8(
              lut,
              low_nibble_of_input
            ),
            _mm512_shuffle_epi8(
              lut,
              _mm512_srli_epi16(high_nibble_of_input, 4)
            )
          ),
          _mm512_setzero_si512()
        );
    #elif defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
      r_.ni =
        _mm512_sub_epi64(
          a_.ni,
          _mm512_and_si512(
            _mm512_srli_epi64(a_.ni, 1),
            _mm512_set1_epi64(0x5555555555555555)
          )
        );

      r_.ni =
        _mm512_add_epi64(
          _mm512_and_si512(
            r_.ni,
            _mm512_set1_epi64(0x3333333333333333)
          ),
          _mm512_and_si512(
            _mm512_srli_epi64(r_.ni, 2),
            _mm512_set1_epi64(0x3333333333333333)
          )
        );

      r_.ni =
        _mm512_and_si512(
          _mm512_add_epi64(
            r_.ni,
            _mm512_srli_epi64(r_.ni, 4)
          ),
          _mm512_set1_epi64(0x0f0f0f0f0f0f0f0f)
        );

      r_.ni =
        _mm512_srli_epi64(
          _mm512_mullo_epi64(
            r_.ni,
            _mm512_set1_epi64(0x0101010101010101)
          ),
          (sizeof(uint64_t) - 1) * CHAR_BIT
        );
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      a_.u64 -= ((a_.u64 >> 1) & UINT64_C(0x5555555555555555));
      a_.u64  = ((a_.u64 & UINT64_C(0x3333333333333333)) + ((a_.u64 >> 2) & UINT64_C(0x3333333333333333)));
      a_.u64  = (a_.u64 + (a_.u64 >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
      r_.u64  = (a_.u64 * UINT64_C(0x0101010101010101)) >> ((sizeof(uint64_t) - 1) * CHAR_BIT);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        uint64_t v = HEDLEY_STATIC_CAST(uint64_t, a_.u64[i]);
        v -= ((v >> 1) & UINT64_C(0x5555555555555555));
        v  = ((v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333)));
        v  = (v + (v >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
        r_.u64[i] = HEDLEY_STATIC_CAST(uint64_t, (v * UINT64_C(0x0101010101010101))) >> ((sizeof(uint64_t) - 1) * CHAR_BIT);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_popcnt_epi64
  #define _mm512_popcnt_epi64(a) easysimd_mm512_popcnt_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_popcnt_epi64 (easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE)
    return _mm512_mask_popcnt_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcnt_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svcnt_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svcnt_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]), src.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svcnt_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]), src.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_popcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_popcnt_epi64
  #define _mm512_mask_popcnt_epi64(src, k, a) easysimd_mm512_mask_popcnt_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_popcnt_epi64 (easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE)
    return _mm512_maskz_popcnt_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svcnt_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svcnt_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svcnt_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svcnt_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_popcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_popcnt_epi64
  #define _mm512_maskz_popcnt_epi64(k, a) easysimd_mm512_maskz_popcnt_epi64(k, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_POPCNT_H) */
