#if !defined(EASYSIMD_X86_AVX512_4DPWSSDS_H)
#define EASYSIMD_X86_AVX512_4DPWSSDS_H

#include "types.h"
#include "dpwssds.h"
#include "set1.h"
#include "mov.h"
#include "adds.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_4dpwssds_epi32 (easysimd__m512i src, easysimd__m512i a0, easysimd__m512i a1, easysimd__m512i a2, easysimd__m512i a3, easysimd__m128i* b) {
  #if defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE)
    return _mm512_4dpwssds_epi32(src, a0, a1, a2, a3, b);
  #else
    easysimd__m128i_private bv = easysimd__m128i_to_private(easysimd_mm_loadu_epi32(b));
    easysimd__m512i r;

    r = easysimd_mm512_dpwssds_epi32(src, a0, easysimd_mm512_set1_epi32(bv.i32[0]));
    r = easysimd_x_mm512_adds_epi32(easysimd_mm512_dpwssds_epi32(src, a1, easysimd_mm512_set1_epi32(bv.i32[1])), r);
    r = easysimd_x_mm512_adds_epi32(easysimd_mm512_dpwssds_epi32(src, a2, easysimd_mm512_set1_epi32(bv.i32[2])), r);
    r = easysimd_x_mm512_adds_epi32(easysimd_mm512_dpwssds_epi32(src, a3, easysimd_mm512_set1_epi32(bv.i32[3])), r);

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX5124VNNIW_ENABLE_NATIVE_ALIASES)
  #undef easysimd_mm512_4dpwssds_epi32
  #define _mm512_4dpwssds_epi32(src, a0, a1, a2, a3, b) easysimd_mm512_4dpwssds_epi32(src, a0, a1, a2, a3, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_4dpwssds_epi32 (easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a0, easysimd__m512i a1, easysimd__m512i a2, easysimd__m512i a3, easysimd__m128i* b) {
  #if defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE)
    return _mm512_mask_4dpwssds_epi32(src, k, a0, a1, a2, a3, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_4dpwssds_epi32(src, a0, a1, a2, a3, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX5124VNNIW_ENABLE_NATIVE_ALIASES)
  #undef easysimd_mm512_mask_4dpwssds_epi32
  #define _mm512_mask_4dpwssds_epi32(src, k, a0, a1, a2, a3, b) easysimd_mm512_mask_4dpwssds_epi32(src, k, a0, a1, a2, a3, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_4dpwssds_epi32 (easysimd__mmask16 k, easysimd__m512i src, easysimd__m512i a0, easysimd__m512i a1, easysimd__m512i a2, easysimd__m512i a3, easysimd__m128i* b) {
  #if defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE)
    return _mm512_mask_4dpwssds_epi32(k, src, a0, a1, a2, a3, b);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_4dpwssds_epi32(src, a0, a1, a2, a3, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX5124VNNIW_ENABLE_NATIVE_ALIASES)
  #undef easysimd_mm512_maskz_4dpwssds_epi32
  #define _mm512_maskz_4dpwssds_epi32(k, src, a0, a1, a2, a3, b) easysimd_mm512_maskz_4dpwssds_epi32(k, src, a0, a1, a2, a3, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_4DPWSSDS_H) */
