#if !defined(EASYSIMD_X86_AVX512_MULTISHIFT_H)
#define EASYSIMD_X86_AVX512_MULTISHIFT_H

#include "types.h"
#include "mov.h"
#if defined(EASYSIMD_ARM_SVE_NATIVE)
#include "../../arm/sve.h"
#endif
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_multishift_epi64_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_multishift_epi64_epi8(a, b);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < sizeof(r_.u8) / sizeof(r_.u8[0]) ; i++) {
      r_.u8[i] = HEDLEY_STATIC_CAST(uint8_t, (b_.u64[i / 8] >> (a_.u8[i] & 63)) | (b_.u64[i / 8] << (64 - (a_.u8[i] & 63))));
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_multishift_epi64_epi8
  #define _mm_multishift_epi64_epi8(a, b) easysimd_mm_multishift_epi64_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_multishift_epi64_epi8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_multishift_epi64_epi8(src, k, a, b);
  #else
    return easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_multishift_epi64_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_multishift_epi64_epi8
  #define _mm_mask_multishift_epi64_epi8(src, k, a, b) easysimd_mm_mask_multishift_epi64_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_multishift_epi64_epi8 (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_multishift_epi64_epi8(k, a, b);
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_multishift_epi64_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_multishift_epi64_epi8
  #define _mm_maskz_multishift_epi64_epi8(src, k, a, b) easysimd_mm_maskz_multishift_epi64_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_multishift_epi64_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_multishift_epi64_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    svuint64_t sva, svb, svbr, svbl;
    svb  = svdup_n_u64(b.u64[0]);
    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[0])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[0]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[2])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[2]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[4])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[4]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[6])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[6]), svorr_u64_x(pg, svbr, svbl));

    svb  = svdup_n_u64(b.u64[1]);
    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[8])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[8]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[10])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[10]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[12])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[12]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[14])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[14]), svorr_u64_x(pg, svbr, svbl));

    svb  = svdup_n_u64(b.u64[2]);
    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[16])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[16]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[18])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[18]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[20])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[20]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[22])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[22]), svorr_u64_x(pg, svbr, svbl));

    svb  = svdup_n_u64(b.u64[3]);
    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[24])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[24]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[26])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[26]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[28])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[28]), svorr_u64_x(pg, svbr, svbl));

    sva  = svand_n_u64_x(pg, svld1sb_u64(pg, &(a.i8[30])), 0x3F);
    svbr = svlsr_u64_x(pg, svb, sva);
    svbl = svlsl_u64_x(pg, svb, svsub_u64_x(pg, svdup_n_u64(0x40), sva));
    svst1b_u64(pg, &(r.u8[30]), svorr_u64_x(pg, svbr, svbl));

    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < sizeof(r_.u8) / sizeof(r_.u8[0]) ; i++) {
      r_.u8[i] = HEDLEY_STATIC_CAST(uint8_t, (b_.u64[i / 8] >> (a_.u8[i] & 63)) | (b_.u64[i / 8] << (64 - (a_.u8[i] & 63))));
    }

  return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_multishift_epi64_epi8
  #define _mm256_multishift_epi64_epi8(a, b) easysimd_mm256_multishift_epi64_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_multishift_epi64_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_multishift_epi64_epi8(src, k, a, b);
  #else
    return easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_multishift_epi64_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_multishift_epi64_epi8
  #define _mm256_mask_multishift_epi64_epi8(src, k, a, b) easysimd_mm256_mask_multishift_epi64_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_multishift_epi64_epi8 (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_multishift_epi64_epi8(k, a, b);
  #else
    return easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_multishift_epi64_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_multishift_epi64_epi8
  #define _mm256_maskz_multishift_epi64_epi8(src, k, a, b) easysimd_mm256_maskz_multishift_epi64_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_multishift_epi64_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_multishift_epi64_epi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m256i[0] = easysimd_mm256_multishift_epi64_epi8(a.m256i[0], b.m256i[0]);
    r.m256i[1] = easysimd_mm256_multishift_epi64_epi8(a.m256i[1], b.m256i[1]);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    size_t n64 = sizeof(a.u64) / sizeof(a.u64[0]),
            i64 = 1, i8 = 0;
    svbool_t pg  = svptrue_b64();
    do {
      svuint64_t svb64 = svdup_n_u64(b.u64[i64 - 1]);
      do {
        svuint64_t 
          sva64  = svld1sb_u64(pg, &(a.i8[i8]));
          sva64  = svand_n_u64_z(pg, sva64, 0x3F);
        svuint64_t 
          svb64r = svlsr_u64_z(pg, svb64, sva64),
          svb64l = svlsl_u64_z(pg, svb64, svsub_u64_z(pg, svdup_n_u64(0x40), sva64));
          sva64  = svorr_u64_z(pg, svb64r, svb64l);
          svst1b_u64(pg, &(r.u8[i8]), sva64);
          i8 += svcntd();
      } while(i8 < (i64 << 3));
      i64++;
    }while(i64 <= n64);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < sizeof(r_.u8) / sizeof(r_.u8[0]) ; i++) {
      r_.u8[i] = HEDLEY_STATIC_CAST(uint8_t, (b_.u64[i / 8] >> (a_.u8[i] & 63)) | (b_.u64[i / 8] << (64 - (a_.u8[i] & 63))));
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_multishift_epi64_epi8
  #define _mm512_multishift_epi64_epi8(a, b) easysimd_mm512_multishift_epi64_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_multishift_epi64_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_mask_multishift_epi64_epi8(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_multishift_epi64_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_multishift_epi64_epi8
  #define _mm512_mask_multishift_epi64_epi8(src, k, a, b) easysimd_mm512_mask_multishift_epi64_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_multishift_epi64_epi8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_maskz_multishift_epi64_epi8(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_multishift_epi64_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_multishift_epi64_epi8
  #define _mm512_maskz_multishift_epi64_epi8(src, k, a, b) easysimd_mm512_maskz_multishift_epi64_epi8(src, k, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_MULTISHIFT_H) */
