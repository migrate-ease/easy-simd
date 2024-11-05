#if !defined(EASYSIMD_X86_AVX512_REDUCE_H)
#define EASYSIMD_X86_AVX512_REDUCE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_reduce_add_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_add_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svadd_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svadd_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_3]);
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svadd_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return a.i64[0] + a.i64[1];
  #else
    int64_t res = 0;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        res += a_.i64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_add_epi64
  #define _mm512_reduce_add_epi64(a) easysimd_mm512_reduce_add_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_mask_reduce_add_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_add_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t r0 = svaddv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    int64_t r1 = svaddv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    int64_t r2 = svaddv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    int64_t r3 = svaddv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return (r0 + r1 + r2 + r3);
  #else
    int64_t r = 0;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        r += ((k >> i) & 0x01) ? a_.i64[i] : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_add_epi64
  #define _mm512_mask_reduce_add_epi64(k, a) easysimd_mm512_mask_reduce_add_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_reduce_and_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_and_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svand_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svand_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_3]);
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svand_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return a.i64[0] & a.i64[1];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t res = a_.i64[0];
    for(size_t i = 1; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        res &= a_.i64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_and_epi64
  #define _mm512_reduce_and_epi64(a) easysimd_mm512_reduce_and_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_mask_reduce_and_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_and_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t r0 = svandv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    int64_t r1 = svandv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    int64_t r2 = svandv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    int64_t r3 = svandv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return (r0 & r1 & r2 & r3);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t r = ~UINT64_C(0);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        r &= ((k >> i) & 0x01) ? a_.i64[i] : ~UINT64_C(0);
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_and_epi64
  #define _mm512_mask_reduce_and_epi64(k, a) easysimd_mm512_mask_reduce_and_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_reduce_mul_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_mul_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svmul_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svmul_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_3]);
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svmul_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return a.i64[0] * a.i64[1];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t res = a_.i64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i64[0])); i++){
        res *= a_.i64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_mul_epi64
  #define _mm512_reduce_mul_epi64(a) easysimd_mm512_reduce_mul_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_mask_reduce_mul_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_mul_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b64();
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], svdup_n_s64(1));
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(1));
    a.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], svdup_n_s64(1));
    a.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], svdup_n_s64(1));

    a.sve_i64[EASYSIMD_SV_INDEX_0] = svmul_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svmul_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_3]);
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svmul_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return (a.i64[0] * a.i64[1]);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t r = UINT64_C(1);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        r *= ((k >> i) & 0x01) ? a_.i64[i] : 1;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_mul_epi64
  #define _mm512_mask_reduce_mul_epi64(k, a) easysimd_mm512_mask_reduce_mul_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_reduce_or_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_or_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svorr_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svorr_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_3]);
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svorr_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return a.i64[0] | a.i64[1] | a.i64[2] | a.i64[3];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t res = a_.i64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i64[0])); i++){
        res |= a_.i64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_or_epi64
  #define _mm512_reduce_or_epi64(a) easysimd_mm512_reduce_or_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_mask_reduce_or_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_or_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t r0 = svorv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    int64_t r1 = svorv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    int64_t r2 = svorv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    int64_t r3 = svorv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return (r0 | r1 | r2 | r3);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t r = UINT64_C(0);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        r |= ((k >> i) & 0x01) ? a_.i64[i] : UINT64_C(0);
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_or_epi64
  #define _mm512_mask_reduce_or_epi64(k, a) easysimd_mm512_mask_reduce_or_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_reduce_add_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_add_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return (int32_t)svaddv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  #else
    int32_t res = 0;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i32[0])); i++){
        res += a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_add_epi32
  #define _mm512_reduce_add_epi32(a) easysimd_mm512_reduce_add_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_reduce_and_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_and_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svand_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svand_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svand_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return a.i32[0] & a.i32[1] & a.i32[2] & a.i32[3];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int32_t res = a_.i32[0];
    for(size_t i = 1; i < (sizeof(a_) / sizeof(a_.i32[0])); i++){
        res &= a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_and_epi32
  #define _mm512_reduce_and_epi32(a) easysimd_mm512_reduce_and_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_reduce_mul_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_mul_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svmul_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svmul_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svmul_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return a.i32[0] * a.i32[1] * a.i32[2] * a.i32[3];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int32_t res = a_.i32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i32[0])); i++){
        res *= a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_mul_epi32
  #define _mm512_reduce_mul_epi32(a) easysimd_mm512_reduce_mul_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_reduce_or_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_or_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svorr_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_i32[EASYSIMD_SV_INDEX_1] = svorr_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svorr_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return a.i32[0] | a.i32[1] | a.i32[2] | a.i32[3];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int32_t res = a_.i32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i32[0])); i++){
        res |= a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_or_epi32
  #define _mm512_reduce_or_epi32(a) easysimd_mm512_reduce_or_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
double
easysimd_mm512_reduce_add_pd(easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_add_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_2]);
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], a.sve_f64[EASYSIMD_SV_INDEX_3]);
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return svaddv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
  #else
    double res = 0;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f64[0])); i++){
        res += a_.f64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_add_pd
  #define _mm512_reduce_add_pd(a) easysimd_mm512_reduce_add_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
double
easysimd_mm512_reduce_mul_pd(easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE_)
    return _mm512_reduce_mul_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svmul_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_2]);
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svmul_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], a.sve_f64[EASYSIMD_SV_INDEX_3]);
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svmul_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return a.f64[0] * a.f64[1];
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    double res = a_.f64[0];
    for(size_t i = 1; i < (sizeof(a_) / sizeof(a_.f64[0])); i++){
        res *= a_.f64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_mul_pd
  #define _mm512_reduce_mul_pd(a) easysimd_mm512_reduce_mul_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
float
easysimd_mm512_reduce_add_ps(easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_add_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_2]);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_3]);
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return svaddv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]);
  #else
    float res = 0;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f32[0])); i++){
        res += a_.f32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_add_ps
  #define _mm512_reduce_add_ps(a) easysimd_mm512_reduce_add_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
float
easysimd_mm512_reduce_mul_ps(easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_mul_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svmul_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_2]);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svmul_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_3]);
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svmul_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return a.f32[0] * a.f32[1] * a.f32[2] * a.f32[3];
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    float res = 1;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f32[0])); i++){
        res *= a_.f32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_mul_ps
  #define _mm512_reduce_mul_ps(a) easysimd_mm512_reduce_mul_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_reduce_max_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_max_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(svmaxv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]), svmaxv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]));
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svdupq_n_s64(svmaxv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]), svmaxv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]));
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(svmaxv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]), svmaxv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]));
    return svmaxv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t res = a_.i64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i64[0])); i++){
        if(res < a_.i64[i])
            res = a_.i64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_max_epi64
  #define _mm512_reduce_max_epi64(a) easysimd_mm512_reduce_max_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_mask_reduce_max_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_max_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t r0 = svmaxv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    int64_t r1 = svmaxv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    int64_t r2 = svmaxv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    int64_t r3 = svmaxv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    int64_t r = svmaxv_s64(svptrue_b64(), svmax_s64_z(svptrue_b64(), svdupq_n_s64(r0, r1), svdupq_n_s64(r2, r3)));
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t r = INT64_MIN;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        r = ((k >> i) & 0x01) ? ((r > a_.i64[i]) ? r : a_.i64[i]) : ((r > (int64_t)(-0x8000000000000000)) ? r : (int64_t)(-0x8000000000000000));
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_max_epi64
  #define _mm512_mask_reduce_max_epi64(k, a) easysimd_mm512_mask_reduce_max_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_reduce_min_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_min_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(svminv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]), svminv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]));
    a.sve_i64[EASYSIMD_SV_INDEX_1] = svdupq_n_s64(svminv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]), svminv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]));
    a.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(svminv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]), svminv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]));
    return svminv_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t res = a_.i64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i64[0])); i++){
        if(res > a_.i64[i])
            res = a_.i64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_min_epi64
  #define _mm512_reduce_min_epi64(a) easysimd_mm512_reduce_min_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm512_mask_reduce_min_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_min_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t r0 = svminv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    int64_t r1 = svminv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    int64_t r2 = svminv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    int64_t r3 = svminv_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    int64_t r = svminv_s64(svptrue_b64(), svmin_s64_z(svptrue_b64(), svdupq_n_s64(r0, r1), svdupq_n_s64(r2, r3)));
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int64_t r = INT64_MAX;
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i64[0])); i++){
        r = ((k >> i) & 0x01) ? ((r < a_.i64[i]) ? r : a_.i64[i]) : ((r < (int64_t)(0x7FFFFFFFFFFFFFFF)) ? r : (int64_t)(0x7FFFFFFFFFFFFFFF));
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_min_epi64
  #define _mm512_mask_reduce_min_epi64(k, a) easysimd_mm512_mask_reduce_min_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
double
easysimd_mm512_reduce_max_pd(easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_max_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svdupq_n_f64(svmaxv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]), svmaxv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_2]));
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svdupq_n_f64(svmaxv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]), svmaxv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_3]));
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svdupq_n_f64(svmaxv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]), svmaxv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]));
    return svmaxv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    double res = a_.f64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.f64[0])); i++){
        if(res < a_.f64[i])
            res = a_.f64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_max_pd
  #define _mm512_reduce_max_pd(a) easysimd_mm512_reduce_max_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
double
easysimd_mm512_reduce_min_pd(easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_min_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svdupq_n_f64(svminv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]), svminv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_2]));
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svdupq_n_f64(svminv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]), svminv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_3]));
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svdupq_n_f64(svminv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]), svminv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]));
    return svminv_f64(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    double res = a_.f64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.f64[0])); i++){
        if(res > a_.f64[i])
            res = a_.f64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_min_pd
  #define _mm512_reduce_min_pd(a) easysimd_mm512_reduce_min_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
float
easysimd_mm512_reduce_max_ps(easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_max_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svdupq_n_f32(svmaxv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]),
                                               svmaxv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]),
                                               svmaxv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_2]),
                                               svmaxv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_3]));
    return svmaxv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    float res = a_.f32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.f32[0])); i++){
        if(res < a_.f32[i])
            res = a_.f32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_max_ps
  #define _mm512_reduce_max_ps(a) easysimd_mm512_reduce_max_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
float
easysimd_mm512_reduce_min_ps(easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_min_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svdupq_n_f32(svminv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]),
                                               svminv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]),
                                               svminv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_2]),
                                               svminv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_3]));
    return svminv_f32(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    float res = a_.f32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.f32[0])); i++){
        if(res > a_.f32[i])
            res = a_.f32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_min_ps
  #define _mm512_reduce_min_ps(a) easysimd_mm512_reduce_min_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_reduce_max_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_max_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svdupq_n_s32(svmaxv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]),
                                               svmaxv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]),
                                               svmaxv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_2]),
                                               svmaxv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_3]));
    return svmaxv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    a.m128i[0].neon_i32 = vmaxq_s32(a.m128i[0].neon_i32, a.m128i[2].neon_i32);
    a.m128i[1].neon_i32 = vmaxq_s32(a.m128i[1].neon_i32, a.m128i[3].neon_i32);
    a.m128i[0].neon_i32 = vmaxq_s32(a.m128i[0].neon_i32, a.m128i[1].neon_i32);
    return vmaxvq_s32(a.m128i[0].neon_i32);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int32_t res = a_.i32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i32[0])); i++){
        if(res < a_.i32[i])
            res = a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_max_epi32
  #define _mm512_reduce_max_epi32(a) easysimd_mm512_reduce_max_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_reduce_min_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_min_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svdupq_n_s32(svminv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]),
                                               svminv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]),
                                               svminv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_2]),
                                               svminv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_3]));
    return svminv_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    a.m128i[0].neon_i32 = vminq_s32(a.m128i[0].neon_i32, a.m128i[2].neon_i32);
    a.m128i[1].neon_i32 = vminq_s32(a.m128i[1].neon_i32, a.m128i[3].neon_i32);
    a.m128i[0].neon_i32 = vminq_s32(a.m128i[0].neon_i32, a.m128i[1].neon_i32);
    return vminvq_s32(a.m128i[0].neon_i32);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int32_t res = a_.i32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i32[0])); i++){
        if(res > a_.i32[i])
            res = a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_min_epi32
  #define _mm512_reduce_min_epi32(a) easysimd_mm512_reduce_min_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_mm512_reduce_max_epu64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_max_epu64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svdupq_n_u64(svmaxv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]), svmaxv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_2]));
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svdupq_n_u64(svmaxv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]), svmaxv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_3]));
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svdupq_n_u64(svmaxv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]), svmaxv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]));
    return svmaxv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    uint64_t res = a_.u64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.u64[0])); i++){
        if(res < a_.u64[i])
            res = a_.u64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_max_epu64
  #define _mm512_reduce_max_epu64(a) easysimd_mm512_reduce_max_epu64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_mm512_reduce_min_epu64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_min_epu64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svdupq_n_u64(svminv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]), svminv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_2]));
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svdupq_n_u64(svminv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]), svminv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_3]));
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svdupq_n_u64(svminv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]), svminv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]));
    return svminv_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    uint64_t res = a_.u64[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.u64[0])); i++){
        if(res > a_.u64[i])
            res = a_.u64[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_min_epu64
  #define _mm512_reduce_min_epu64(a) easysimd_mm512_reduce_min_epu64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_mm512_reduce_max_epu32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_max_epu32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svdupq_n_u32(svmaxv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0]),
                                               svmaxv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1]),
                                               svmaxv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_2]),
                                               svmaxv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_3]));
    return svmaxv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    uint32_t res = a_.u32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.u32[0])); i++){
        if(res < a_.u32[i])
            res = a_.u32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_max_epu32
  #define _mm512_reduce_max_epu32(a) easysimd_mm512_reduce_max_epu32(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_mm512_reduce_min_epu32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_reduce_min_epu32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svdupq_n_u32(svminv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0]),
                                               svminv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1]),
                                               svminv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_2]),
                                               svminv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_3]));
    return svminv_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    uint32_t res = a_.u32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.u32[0])); i++){
        if(res > a_.u32[i])
            res = a_.u32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_reduce_min_epu32
  #define _mm512_reduce_min_epu32(a) easysimd_mm512_reduce_min_epu32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_mask_reduce_add_epi32(easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_add_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return svaddv_s32(svptrue_b32(),
                      svdupq_n_s32(svaddv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]),
                                   svaddv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]),
                                   svaddv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2]),
                                   svaddv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3])));

  #else
    int32_t res = 0;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.i32[0])); i++){
        res += ((k >> i) & 0x01 ? a_.i32[i] : 0);
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_add_epi32
  #define _mm512_mask_reduce_add_epi32(k, a) easysimd_mm512_mask_reduce_add_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
float
easysimd_mm512_mask_reduce_add_ps(easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_add_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return svaddv_f32(svptrue_b32(),
                      svdupq_n_f32(svaddv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0]),
                                   svaddv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1]),
                                   svaddv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2]),
                                   svaddv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3])));

  #else
    float res = 0;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    for(size_t i = 0; i < (sizeof(a_) / sizeof(a_.f32[0])); i++){
        res += ((k >> i) & 0x01 ? a_.f32[i] : 0);
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_add_ps
  #define _mm512_mask_reduce_add_ps(k, a) easysimd_mm512_mask_reduce_add_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_mask_reduce_max_epi32(easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_max_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svdupq_n_s32(svmaxv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]),
                                               svmaxv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]),
                                               svmaxv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2]),
                                               svmaxv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3]));
    return svmaxv_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    for(size_t i = 1; i < (sizeof(a) / sizeof(a.i32[0])); i++){
        if(!((k >> i) & 0x01))
            a.i32[i] = -0x80000000;
    }
    a.m128i[0].neon_i32 = vmaxq_s32(a.m128i[0].neon_i32, a.m128i[2].neon_i32);
    a.m128i[1].neon_i32 = vmaxq_s32(a.m128i[1].neon_i32, a.m128i[3].neon_i32);
    a.m128i[0].neon_i32 = vmaxq_s32(a.m128i[0].neon_i32, a.m128i[1].neon_i32);
    return vmaxvq_s32(a.m128i[0].neon_i32);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int32_t res = a_.i32[0];
    for(size_t i = 1; i < (sizeof(a_) / sizeof(a_.i32[0])); i++){
        if(res < a_.i32[i] && ((k >> i) & 0x01))
            res = a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_max_epi32
  #define _mm512_mask_reduce_max_epi32(k, a) easysimd_mm512_mask_reduce_max_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
float
easysimd_mm512_mask_reduce_max_ps(easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_max_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svdupq_n_f32(svmaxv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0]),
                                               svmaxv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1]),
                                               svmaxv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2]),
                                               svmaxv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3]));
    return svmaxv_f32(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    float res = a_.f32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.f32[0])); i++){
        if(res < a_.f32[i]  && ((k >> i) & 0x01))
            res = a_.f32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_max_ps
  #define _mm512_mask_reduce_max_ps(k, a) easysimd_mm512_mask_reduce_max_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm512_mask_reduce_min_epi32(easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_min_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i32[EASYSIMD_SV_INDEX_0] = svdupq_n_s32(svminv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]),
                                               svminv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]),
                                               svminv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2]),
                                               svminv_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3]));
    return svminv_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    for(size_t i = 1; i < (sizeof(a) / sizeof(a.i32[0])); i++){
        if(!((k >> i) & 0x01))
            a.i32[i] = 0x7FFFFFFF;
    }
    a.m128i[0].neon_i32 = vminq_s32(a.m128i[0].neon_i32, a.m128i[2].neon_i32);
    a.m128i[1].neon_i32 = vminq_s32(a.m128i[1].neon_i32, a.m128i[3].neon_i32);
    a.m128i[0].neon_i32 = vminq_s32(a.m128i[0].neon_i32, a.m128i[1].neon_i32);
    return vminvq_s32(a.m128i[0].neon_i32);
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    int32_t res = a_.i32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.i32[0])); i++){
        if(res > a_.i32[i] && ((k >> i) & 0x01))
            res = a_.i32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_min_epi32
  #define _mm512_mask_reduce_min_epi32(k, a) easysimd_mm512_mask_reduce_min_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
float
easysimd_mm512_mask_reduce_min_ps(easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_reduce_min_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svdupq_n_f32(svminv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0]),
                                               svminv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1]),
                                               svminv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2]),
                                               svminv_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3]));
    return svminv_f32(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_0]);
  #else
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    float res = a_.f32[0];
    for(size_t i = 1; i < (sizeof(a) / sizeof(a_.f32[0])); i++){
        if(res > a_.f32[i]  && ((k >> i) & 0x01))
            res = a_.f32[i];
    }
    return res;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_reduce_min_ps
  #define _mm512_mask_reduce_min_ps(k, a) easysimd_mm512_mask_reduce_min_ps(k, a)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP


#endif /* !defined(EASYSIMD_X86_AVX512_REDUCE_H) */

