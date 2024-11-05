#define EASYSIMD_TEST_X86_AVX512_INSN flushsubnormal

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/flushsubnormal.h>
#include <easysimd/x86/avx512/set1.h>

static int
test_easysimd_x_mm_flushsubnormal_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128 a, r;
    easysimd__m128i e;
    if (easysimd_test_codegen_random_i32() & 1) {
      a = easysimd_mm_set1_ps(EASYSIMD_MATH_FLT_MIN / 2.0f);
      e = easysimd_mm_set1_epi32(0);
    } else {
      a = easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0));
      e = easysimd_mm_castps_si128(easysimd_mm_set1_ps(EASYSIMD_FLOAT32_C(1.0)));
    }
    r = easysimd_x_mm_flushsubnormal_ps(a);

    easysimd_test_x86_assert_equal_i32x4(easysimd_mm_castps_si128(r), e);
  }

  return 0;
}

static int
test_easysimd_x_mm256_flushsubnormal_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m256 a, r;
    easysimd__m256i e;
    if (easysimd_test_codegen_random_i32() & 1) {
      a = easysimd_mm256_set1_ps(EASYSIMD_MATH_FLT_MIN / 2.0f);
      e = easysimd_mm256_set1_epi32(0);
    } else {
      a = easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0));
      e = easysimd_mm256_castps_si256(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(1.0)));
    }
    r = easysimd_x_mm256_flushsubnormal_ps(a);

    easysimd_test_x86_assert_equal_i32x8(easysimd_mm256_castps_si256(r), e);
  }

  return 0;
}

static int
test_easysimd_x_mm512_flushsubnormal_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m512 a, r;
    easysimd__m512i e;
    if (easysimd_test_codegen_random_i32() & 1) {
      a = easysimd_mm512_set1_ps(EASYSIMD_MATH_FLT_MIN / 2.0f);
      e = easysimd_mm512_set1_epi32(0);
    } else {
      a = easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0));
      e = easysimd_mm512_castps_si512(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(1.0)));
    }
    r = easysimd_x_mm512_flushsubnormal_ps(a);

    easysimd_test_x86_assert_equal_i32x16(easysimd_mm512_castps_si512(r), e);
  }

  return 0;
}

static int
test_easysimd_x_mm_flushsubnormal_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128d a, r;
    easysimd__m128i e;
    if (easysimd_test_codegen_random_i32() & 1) {
      a = easysimd_mm_set1_pd(EASYSIMD_MATH_DBL_MIN / 2.0);
      e = easysimd_mm_set1_epi64x(0);
    } else {
      a = easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0));
      e = easysimd_mm_castpd_si128(easysimd_mm_set1_pd(EASYSIMD_FLOAT64_C(1.0)));
    }
    r = easysimd_x_mm_flushsubnormal_pd(a);

    easysimd_test_x86_assert_equal_i64x2(easysimd_mm_castpd_si128(r), e);
  }

  return 0;
}

static int
test_easysimd_x_mm256_flushsubnormal_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m256d a, r;
    easysimd__m256i e;
    if (easysimd_test_codegen_random_i32() & 1) {
      a = easysimd_mm256_set1_pd(EASYSIMD_MATH_DBL_MIN / 2.0);
      e = easysimd_mm256_set1_epi64x(0);
    } else {
      a = easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0));
      e = easysimd_mm256_castpd_si256(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(1.0)));
    }
    r = easysimd_x_mm256_flushsubnormal_pd(a);

    easysimd_test_x86_assert_equal_i64x4(easysimd_mm256_castpd_si256(r), e);
  }

  return 0;
}

static int
test_easysimd_x_mm512_flushsubnormal_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m512d a, r;
    easysimd__m512i e;
    if (easysimd_test_codegen_random_i32() & 1) {
      a = easysimd_mm512_set1_pd(EASYSIMD_MATH_DBL_MIN / 2.0);
      e = easysimd_mm512_set1_epi64(0);
    } else {
      a = easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0));
      e = easysimd_mm512_castpd_si512(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(1.0)));
    }
    r = easysimd_x_mm512_flushsubnormal_pd(a);

    easysimd_test_x86_assert_equal_i64x8(easysimd_mm512_castpd_si512(r), e);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm_flushsubnormal_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm256_flushsubnormal_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm512_flushsubnormal_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm_flushsubnormal_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm256_flushsubnormal_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm512_flushsubnormal_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
