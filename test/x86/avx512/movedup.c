#define EASYSIMD_TEST_X86_AVX512_INSN move

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/movedup.h>

static int
test_easysimd_mm_mask_movedup_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -774.99), EASYSIMD_FLOAT64_C(   131.43) },
      UINT8_C(227),
      { EASYSIMD_FLOAT64_C(   485.09), EASYSIMD_FLOAT64_C(   125.06) },
      { EASYSIMD_FLOAT64_C(   485.09), EASYSIMD_FLOAT64_C(   485.09) } },
    { { EASYSIMD_FLOAT64_C(  -687.44), EASYSIMD_FLOAT64_C(  -882.74) },
      UINT8_C(  9),
      { EASYSIMD_FLOAT64_C(   736.43), EASYSIMD_FLOAT64_C(  -280.03) },
      { EASYSIMD_FLOAT64_C(   736.43), EASYSIMD_FLOAT64_C(  -882.74) } },
    { { EASYSIMD_FLOAT64_C(  -238.23), EASYSIMD_FLOAT64_C(   599.36) },
      UINT8_C(116),
      { EASYSIMD_FLOAT64_C(   449.34), EASYSIMD_FLOAT64_C(  -692.99) },
      { EASYSIMD_FLOAT64_C(  -238.23), EASYSIMD_FLOAT64_C(   599.36) } },
    { { EASYSIMD_FLOAT64_C(  -193.95), EASYSIMD_FLOAT64_C(   527.48) },
      UINT8_C( 53),
      { EASYSIMD_FLOAT64_C(   652.67), EASYSIMD_FLOAT64_C(   -19.72) },
      { EASYSIMD_FLOAT64_C(   652.67), EASYSIMD_FLOAT64_C(   527.48) } },
    { { EASYSIMD_FLOAT64_C(  -992.34), EASYSIMD_FLOAT64_C(  -766.27) },
      UINT8_C(203),
      { EASYSIMD_FLOAT64_C(   -84.60), EASYSIMD_FLOAT64_C(  -147.42) },
      { EASYSIMD_FLOAT64_C(   -84.60), EASYSIMD_FLOAT64_C(   -84.60) } },
    { { EASYSIMD_FLOAT64_C(   897.15), EASYSIMD_FLOAT64_C(  -180.14) },
      UINT8_C(208),
      { EASYSIMD_FLOAT64_C(  -996.30), EASYSIMD_FLOAT64_C(    87.76) },
      { EASYSIMD_FLOAT64_C(   897.15), EASYSIMD_FLOAT64_C(  -180.14) } },
    { { EASYSIMD_FLOAT64_C(  -259.91), EASYSIMD_FLOAT64_C(  -771.28) },
      UINT8_C( 39),
      { EASYSIMD_FLOAT64_C(  -644.91), EASYSIMD_FLOAT64_C(   713.80) },
      { EASYSIMD_FLOAT64_C(  -644.91), EASYSIMD_FLOAT64_C(  -644.91) } },
    { { EASYSIMD_FLOAT64_C(   344.25), EASYSIMD_FLOAT64_C(  -332.35) },
      UINT8_C(121),
      { EASYSIMD_FLOAT64_C(   120.99), EASYSIMD_FLOAT64_C(  -595.93) },
      { EASYSIMD_FLOAT64_C(   120.99), EASYSIMD_FLOAT64_C(  -332.35) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_movedup_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_movedup_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_movedup_pd(src, k, a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_movedup_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(  8),
      { EASYSIMD_FLOAT64_C(   882.76), EASYSIMD_FLOAT64_C(  -996.56) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(124),
      { EASYSIMD_FLOAT64_C(   332.10), EASYSIMD_FLOAT64_C(  -689.55) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(206),
      { EASYSIMD_FLOAT64_C(  -140.43), EASYSIMD_FLOAT64_C(   876.52) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -140.43) } },
    { UINT8_C(143),
      { EASYSIMD_FLOAT64_C(   839.85), EASYSIMD_FLOAT64_C(   884.18) },
      { EASYSIMD_FLOAT64_C(   839.85), EASYSIMD_FLOAT64_C(   839.85) } },
    { UINT8_C( 49),
      { EASYSIMD_FLOAT64_C(  -896.99), EASYSIMD_FLOAT64_C(  -200.43) },
      { EASYSIMD_FLOAT64_C(  -896.99), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(249),
      { EASYSIMD_FLOAT64_C(  -999.83), EASYSIMD_FLOAT64_C(   619.43) },
      { EASYSIMD_FLOAT64_C(  -999.83), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(201),
      { EASYSIMD_FLOAT64_C(  -996.13), EASYSIMD_FLOAT64_C(  -292.81) },
      { EASYSIMD_FLOAT64_C(  -996.13), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 16),
      { EASYSIMD_FLOAT64_C(  -767.41), EASYSIMD_FLOAT64_C(   -73.61) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_movedup_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_movedup_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_movedup_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_movedup_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -320.71), EASYSIMD_FLOAT64_C(   643.03), EASYSIMD_FLOAT64_C(  -424.40), EASYSIMD_FLOAT64_C(  -201.96) },
      UINT8_C(243),
      { EASYSIMD_FLOAT64_C(   725.11), EASYSIMD_FLOAT64_C(   605.75), EASYSIMD_FLOAT64_C(  -266.50), EASYSIMD_FLOAT64_C(  -655.85) },
      { EASYSIMD_FLOAT64_C(   725.11), EASYSIMD_FLOAT64_C(   725.11), EASYSIMD_FLOAT64_C(  -424.40), EASYSIMD_FLOAT64_C(  -201.96) } },
    { { EASYSIMD_FLOAT64_C(   618.75), EASYSIMD_FLOAT64_C(   484.87), EASYSIMD_FLOAT64_C(  -637.80), EASYSIMD_FLOAT64_C(  -409.16) },
      UINT8_C(200),
      { EASYSIMD_FLOAT64_C(   172.09), EASYSIMD_FLOAT64_C(  -853.56), EASYSIMD_FLOAT64_C(   146.92), EASYSIMD_FLOAT64_C(   675.08) },
      { EASYSIMD_FLOAT64_C(   618.75), EASYSIMD_FLOAT64_C(   484.87), EASYSIMD_FLOAT64_C(  -637.80), EASYSIMD_FLOAT64_C(   146.92) } },
    { { EASYSIMD_FLOAT64_C(     8.59), EASYSIMD_FLOAT64_C(   378.87), EASYSIMD_FLOAT64_C(   499.71), EASYSIMD_FLOAT64_C(   945.82) },
      UINT8_C(186),
      { EASYSIMD_FLOAT64_C(    31.45), EASYSIMD_FLOAT64_C(  -231.74), EASYSIMD_FLOAT64_C(   -44.64), EASYSIMD_FLOAT64_C(   840.37) },
      { EASYSIMD_FLOAT64_C(     8.59), EASYSIMD_FLOAT64_C(    31.45), EASYSIMD_FLOAT64_C(   499.71), EASYSIMD_FLOAT64_C(   -44.64) } },
    { { EASYSIMD_FLOAT64_C(  -333.83), EASYSIMD_FLOAT64_C(  -632.87), EASYSIMD_FLOAT64_C(  -677.95), EASYSIMD_FLOAT64_C(   686.61) },
      UINT8_C( 57),
      { EASYSIMD_FLOAT64_C(   965.09), EASYSIMD_FLOAT64_C(  -737.79), EASYSIMD_FLOAT64_C(   844.46), EASYSIMD_FLOAT64_C(  -835.15) },
      { EASYSIMD_FLOAT64_C(   965.09), EASYSIMD_FLOAT64_C(  -632.87), EASYSIMD_FLOAT64_C(  -677.95), EASYSIMD_FLOAT64_C(   844.46) } },
    { { EASYSIMD_FLOAT64_C(   987.32), EASYSIMD_FLOAT64_C(   450.21), EASYSIMD_FLOAT64_C(  -101.65), EASYSIMD_FLOAT64_C(  -668.53) },
      UINT8_C( 74),
      { EASYSIMD_FLOAT64_C(  -616.79), EASYSIMD_FLOAT64_C(  -306.33), EASYSIMD_FLOAT64_C(   659.79), EASYSIMD_FLOAT64_C(   543.17) },
      { EASYSIMD_FLOAT64_C(   987.32), EASYSIMD_FLOAT64_C(  -616.79), EASYSIMD_FLOAT64_C(  -101.65), EASYSIMD_FLOAT64_C(   659.79) } },
    { { EASYSIMD_FLOAT64_C(   865.76), EASYSIMD_FLOAT64_C(   806.23), EASYSIMD_FLOAT64_C(  -309.91), EASYSIMD_FLOAT64_C(   540.83) },
      UINT8_C( 55),
      { EASYSIMD_FLOAT64_C(  -931.04), EASYSIMD_FLOAT64_C(    40.55), EASYSIMD_FLOAT64_C(  -239.35), EASYSIMD_FLOAT64_C(   266.44) },
      { EASYSIMD_FLOAT64_C(  -931.04), EASYSIMD_FLOAT64_C(  -931.04), EASYSIMD_FLOAT64_C(  -239.35), EASYSIMD_FLOAT64_C(   540.83) } },
    { { EASYSIMD_FLOAT64_C(  -928.01), EASYSIMD_FLOAT64_C(   528.91), EASYSIMD_FLOAT64_C(  -778.19), EASYSIMD_FLOAT64_C(   912.37) },
      UINT8_C(113),
      { EASYSIMD_FLOAT64_C(  -411.07), EASYSIMD_FLOAT64_C(  -765.58), EASYSIMD_FLOAT64_C(   881.70), EASYSIMD_FLOAT64_C(   635.35) },
      { EASYSIMD_FLOAT64_C(  -411.07), EASYSIMD_FLOAT64_C(   528.91), EASYSIMD_FLOAT64_C(  -778.19), EASYSIMD_FLOAT64_C(   912.37) } },
    { { EASYSIMD_FLOAT64_C(  -800.50), EASYSIMD_FLOAT64_C(  -856.09), EASYSIMD_FLOAT64_C(   479.81), EASYSIMD_FLOAT64_C(  -635.65) },
      UINT8_C( 86),
      { EASYSIMD_FLOAT64_C(   -69.98), EASYSIMD_FLOAT64_C(   262.70), EASYSIMD_FLOAT64_C(  -537.29), EASYSIMD_FLOAT64_C(   998.97) },
      { EASYSIMD_FLOAT64_C(  -800.50), EASYSIMD_FLOAT64_C(   -69.98), EASYSIMD_FLOAT64_C(  -537.29), EASYSIMD_FLOAT64_C(  -635.65) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_movedup_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_movedup_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_movedup_pd(src, k, a);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_movedup_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(147),
      { EASYSIMD_FLOAT64_C(   156.38), EASYSIMD_FLOAT64_C(   658.76), EASYSIMD_FLOAT64_C(   189.08), EASYSIMD_FLOAT64_C(    22.13) },
      { EASYSIMD_FLOAT64_C(   156.38), EASYSIMD_FLOAT64_C(   156.38), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(165),
      { EASYSIMD_FLOAT64_C(   879.16), EASYSIMD_FLOAT64_C(  -437.03), EASYSIMD_FLOAT64_C(  -720.18), EASYSIMD_FLOAT64_C(   948.12) },
      { EASYSIMD_FLOAT64_C(   879.16), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -720.18), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(249),
      { EASYSIMD_FLOAT64_C(    40.47), EASYSIMD_FLOAT64_C(   214.57), EASYSIMD_FLOAT64_C(   675.51), EASYSIMD_FLOAT64_C(  -430.61) },
      { EASYSIMD_FLOAT64_C(    40.47), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   675.51) } },
    { UINT8_C( 53),
      { EASYSIMD_FLOAT64_C(   587.87), EASYSIMD_FLOAT64_C(  -235.53), EASYSIMD_FLOAT64_C(  -974.69), EASYSIMD_FLOAT64_C(   822.29) },
      { EASYSIMD_FLOAT64_C(   587.87), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -974.69), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(230),
      { EASYSIMD_FLOAT64_C(   660.66), EASYSIMD_FLOAT64_C(  -978.21), EASYSIMD_FLOAT64_C(  -209.92), EASYSIMD_FLOAT64_C(   140.47) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   660.66), EASYSIMD_FLOAT64_C(  -209.92), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(206),
      { EASYSIMD_FLOAT64_C(   -78.69), EASYSIMD_FLOAT64_C(  -929.51), EASYSIMD_FLOAT64_C(   648.84), EASYSIMD_FLOAT64_C(   384.02) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   -78.69), EASYSIMD_FLOAT64_C(   648.84), EASYSIMD_FLOAT64_C(   648.84) } },
    { UINT8_C(155),
      { EASYSIMD_FLOAT64_C(   294.76), EASYSIMD_FLOAT64_C(  -459.60), EASYSIMD_FLOAT64_C(   728.23), EASYSIMD_FLOAT64_C(  -516.17) },
      { EASYSIMD_FLOAT64_C(   294.76), EASYSIMD_FLOAT64_C(   294.76), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   728.23) } },
    { UINT8_C(120),
      { EASYSIMD_FLOAT64_C(   193.22), EASYSIMD_FLOAT64_C(  -637.00), EASYSIMD_FLOAT64_C(  -874.51), EASYSIMD_FLOAT64_C(   473.05) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -874.51) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_movedup_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_movedup_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_movedup_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_movedup_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -688.88), EASYSIMD_FLOAT64_C(   729.01), EASYSIMD_FLOAT64_C(  -486.48), EASYSIMD_FLOAT64_C(   525.69),
        EASYSIMD_FLOAT64_C(   404.51), EASYSIMD_FLOAT64_C(    82.91), EASYSIMD_FLOAT64_C(   -37.93), EASYSIMD_FLOAT64_C(    -7.61) },
      { EASYSIMD_FLOAT64_C(  -688.88), EASYSIMD_FLOAT64_C(  -688.88), EASYSIMD_FLOAT64_C(  -486.48), EASYSIMD_FLOAT64_C(  -486.48),
        EASYSIMD_FLOAT64_C(   404.51), EASYSIMD_FLOAT64_C(   404.51), EASYSIMD_FLOAT64_C(   -37.93), EASYSIMD_FLOAT64_C(   -37.93) } },
    { { EASYSIMD_FLOAT64_C(   847.38), EASYSIMD_FLOAT64_C(   -12.62), EASYSIMD_FLOAT64_C(  -185.32), EASYSIMD_FLOAT64_C(  -506.45),
        EASYSIMD_FLOAT64_C(  -351.96), EASYSIMD_FLOAT64_C(  -163.53), EASYSIMD_FLOAT64_C(   283.63), EASYSIMD_FLOAT64_C(   788.52) },
      { EASYSIMD_FLOAT64_C(   847.38), EASYSIMD_FLOAT64_C(   847.38), EASYSIMD_FLOAT64_C(  -185.32), EASYSIMD_FLOAT64_C(  -185.32),
        EASYSIMD_FLOAT64_C(  -351.96), EASYSIMD_FLOAT64_C(  -351.96), EASYSIMD_FLOAT64_C(   283.63), EASYSIMD_FLOAT64_C(   283.63) } },
    { { EASYSIMD_FLOAT64_C(   222.62), EASYSIMD_FLOAT64_C(  -795.06), EASYSIMD_FLOAT64_C(   859.01), EASYSIMD_FLOAT64_C(  -128.54),
        EASYSIMD_FLOAT64_C(   588.96), EASYSIMD_FLOAT64_C(   928.47), EASYSIMD_FLOAT64_C(  -833.79), EASYSIMD_FLOAT64_C(  -870.64) },
      { EASYSIMD_FLOAT64_C(   222.62), EASYSIMD_FLOAT64_C(   222.62), EASYSIMD_FLOAT64_C(   859.01), EASYSIMD_FLOAT64_C(   859.01),
        EASYSIMD_FLOAT64_C(   588.96), EASYSIMD_FLOAT64_C(   588.96), EASYSIMD_FLOAT64_C(  -833.79), EASYSIMD_FLOAT64_C(  -833.79) } },
    { { EASYSIMD_FLOAT64_C(   656.70), EASYSIMD_FLOAT64_C(  -349.95), EASYSIMD_FLOAT64_C(   691.88), EASYSIMD_FLOAT64_C(  -150.08),
        EASYSIMD_FLOAT64_C(    13.05), EASYSIMD_FLOAT64_C(   817.38), EASYSIMD_FLOAT64_C(  -677.03), EASYSIMD_FLOAT64_C(   324.17) },
      { EASYSIMD_FLOAT64_C(   656.70), EASYSIMD_FLOAT64_C(   656.70), EASYSIMD_FLOAT64_C(   691.88), EASYSIMD_FLOAT64_C(   691.88),
        EASYSIMD_FLOAT64_C(    13.05), EASYSIMD_FLOAT64_C(    13.05), EASYSIMD_FLOAT64_C(  -677.03), EASYSIMD_FLOAT64_C(  -677.03) } },
    { { EASYSIMD_FLOAT64_C(   546.39), EASYSIMD_FLOAT64_C(  -163.51), EASYSIMD_FLOAT64_C(  -150.14), EASYSIMD_FLOAT64_C(   -49.10),
        EASYSIMD_FLOAT64_C(   919.40), EASYSIMD_FLOAT64_C(   811.93), EASYSIMD_FLOAT64_C(   943.28), EASYSIMD_FLOAT64_C(   766.78) },
      { EASYSIMD_FLOAT64_C(   546.39), EASYSIMD_FLOAT64_C(   546.39), EASYSIMD_FLOAT64_C(  -150.14), EASYSIMD_FLOAT64_C(  -150.14),
        EASYSIMD_FLOAT64_C(   919.40), EASYSIMD_FLOAT64_C(   919.40), EASYSIMD_FLOAT64_C(   943.28), EASYSIMD_FLOAT64_C(   943.28) } },
    { { EASYSIMD_FLOAT64_C(  -200.70), EASYSIMD_FLOAT64_C(  -242.04), EASYSIMD_FLOAT64_C(  -739.67), EASYSIMD_FLOAT64_C(   447.35),
        EASYSIMD_FLOAT64_C(   594.43), EASYSIMD_FLOAT64_C(   543.95), EASYSIMD_FLOAT64_C(   235.87), EASYSIMD_FLOAT64_C(  -182.95) },
      { EASYSIMD_FLOAT64_C(  -200.70), EASYSIMD_FLOAT64_C(  -200.70), EASYSIMD_FLOAT64_C(  -739.67), EASYSIMD_FLOAT64_C(  -739.67),
        EASYSIMD_FLOAT64_C(   594.43), EASYSIMD_FLOAT64_C(   594.43), EASYSIMD_FLOAT64_C(   235.87), EASYSIMD_FLOAT64_C(   235.87) } },
    { { EASYSIMD_FLOAT64_C(   748.89), EASYSIMD_FLOAT64_C(    94.87), EASYSIMD_FLOAT64_C(   688.50), EASYSIMD_FLOAT64_C(   337.85),
        EASYSIMD_FLOAT64_C(    23.35), EASYSIMD_FLOAT64_C(   854.72), EASYSIMD_FLOAT64_C(   467.21), EASYSIMD_FLOAT64_C(  -319.96) },
      { EASYSIMD_FLOAT64_C(   748.89), EASYSIMD_FLOAT64_C(   748.89), EASYSIMD_FLOAT64_C(   688.50), EASYSIMD_FLOAT64_C(   688.50),
        EASYSIMD_FLOAT64_C(    23.35), EASYSIMD_FLOAT64_C(    23.35), EASYSIMD_FLOAT64_C(   467.21), EASYSIMD_FLOAT64_C(   467.21) } },
    { { EASYSIMD_FLOAT64_C(  -495.23), EASYSIMD_FLOAT64_C(   159.10), EASYSIMD_FLOAT64_C(   529.96), EASYSIMD_FLOAT64_C(   517.81),
        EASYSIMD_FLOAT64_C(   -23.53), EASYSIMD_FLOAT64_C(   852.93), EASYSIMD_FLOAT64_C(  -158.02), EASYSIMD_FLOAT64_C(  -477.14) },
      { EASYSIMD_FLOAT64_C(  -495.23), EASYSIMD_FLOAT64_C(  -495.23), EASYSIMD_FLOAT64_C(   529.96), EASYSIMD_FLOAT64_C(   529.96),
        EASYSIMD_FLOAT64_C(   -23.53), EASYSIMD_FLOAT64_C(   -23.53), EASYSIMD_FLOAT64_C(  -158.02), EASYSIMD_FLOAT64_C(  -158.02) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_movedup_pd(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_movedup_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r = easysimd_mm512_movedup_pd(a);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_movedup_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[8];
    const uint8_t k;
    const easysimd_float64 a[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -310.58), EASYSIMD_FLOAT64_C(   691.84), EASYSIMD_FLOAT64_C(   473.76), EASYSIMD_FLOAT64_C(  -391.18),
        EASYSIMD_FLOAT64_C(   503.76), EASYSIMD_FLOAT64_C(   417.04), EASYSIMD_FLOAT64_C(  -624.40), EASYSIMD_FLOAT64_C(  -696.93) },
      UINT8_C(172),
      { EASYSIMD_FLOAT64_C(  -364.08), EASYSIMD_FLOAT64_C(   750.42), EASYSIMD_FLOAT64_C(   769.43), EASYSIMD_FLOAT64_C(  -820.12),
        EASYSIMD_FLOAT64_C(   -13.72), EASYSIMD_FLOAT64_C(  -413.53), EASYSIMD_FLOAT64_C(   928.77), EASYSIMD_FLOAT64_C(  -918.84) },
      { EASYSIMD_FLOAT64_C(  -310.58), EASYSIMD_FLOAT64_C(   691.84), EASYSIMD_FLOAT64_C(   769.43), EASYSIMD_FLOAT64_C(   769.43),
        EASYSIMD_FLOAT64_C(   503.76), EASYSIMD_FLOAT64_C(   -13.72), EASYSIMD_FLOAT64_C(  -624.40), EASYSIMD_FLOAT64_C(   928.77) } },
    { { EASYSIMD_FLOAT64_C(  -725.02), EASYSIMD_FLOAT64_C(   266.63), EASYSIMD_FLOAT64_C(   104.50), EASYSIMD_FLOAT64_C(  -870.30),
        EASYSIMD_FLOAT64_C(  -266.16), EASYSIMD_FLOAT64_C(   784.54), EASYSIMD_FLOAT64_C(  -365.53), EASYSIMD_FLOAT64_C(   892.93) },
      UINT8_C(  2),
      { EASYSIMD_FLOAT64_C(  -847.72), EASYSIMD_FLOAT64_C(  -130.59), EASYSIMD_FLOAT64_C(   167.43), EASYSIMD_FLOAT64_C(    -5.74),
        EASYSIMD_FLOAT64_C(   392.27), EASYSIMD_FLOAT64_C(   856.85), EASYSIMD_FLOAT64_C(  -313.90), EASYSIMD_FLOAT64_C(  -133.98) },
      { EASYSIMD_FLOAT64_C(  -725.02), EASYSIMD_FLOAT64_C(  -847.72), EASYSIMD_FLOAT64_C(   104.50), EASYSIMD_FLOAT64_C(  -870.30),
        EASYSIMD_FLOAT64_C(  -266.16), EASYSIMD_FLOAT64_C(   784.54), EASYSIMD_FLOAT64_C(  -365.53), EASYSIMD_FLOAT64_C(   892.93) } },
    { { EASYSIMD_FLOAT64_C(  -534.33), EASYSIMD_FLOAT64_C(  -810.14), EASYSIMD_FLOAT64_C(  -716.94), EASYSIMD_FLOAT64_C(  -158.73),
        EASYSIMD_FLOAT64_C(  -507.07), EASYSIMD_FLOAT64_C(  -541.94), EASYSIMD_FLOAT64_C(   477.19), EASYSIMD_FLOAT64_C(  -756.65) },
      UINT8_C(177),
      { EASYSIMD_FLOAT64_C(   657.07), EASYSIMD_FLOAT64_C(   229.63), EASYSIMD_FLOAT64_C(  -186.03), EASYSIMD_FLOAT64_C(   585.84),
        EASYSIMD_FLOAT64_C(   310.79), EASYSIMD_FLOAT64_C(    88.94), EASYSIMD_FLOAT64_C(  -147.53), EASYSIMD_FLOAT64_C(  -584.71) },
      { EASYSIMD_FLOAT64_C(   657.07), EASYSIMD_FLOAT64_C(  -810.14), EASYSIMD_FLOAT64_C(  -716.94), EASYSIMD_FLOAT64_C(  -158.73),
        EASYSIMD_FLOAT64_C(   310.79), EASYSIMD_FLOAT64_C(   310.79), EASYSIMD_FLOAT64_C(   477.19), EASYSIMD_FLOAT64_C(  -147.53) } },
    { { EASYSIMD_FLOAT64_C(   218.64), EASYSIMD_FLOAT64_C(   586.31), EASYSIMD_FLOAT64_C(  -800.16), EASYSIMD_FLOAT64_C(   853.11),
        EASYSIMD_FLOAT64_C(   479.24), EASYSIMD_FLOAT64_C(   514.34), EASYSIMD_FLOAT64_C(  -994.61), EASYSIMD_FLOAT64_C(  -651.35) },
      UINT8_C(235),
      { EASYSIMD_FLOAT64_C(    -0.35), EASYSIMD_FLOAT64_C(   740.92), EASYSIMD_FLOAT64_C(  -461.37), EASYSIMD_FLOAT64_C(   685.74),
        EASYSIMD_FLOAT64_C(  -393.06), EASYSIMD_FLOAT64_C(     4.30), EASYSIMD_FLOAT64_C(   875.61), EASYSIMD_FLOAT64_C(  -109.99) },
      { EASYSIMD_FLOAT64_C(    -0.35), EASYSIMD_FLOAT64_C(    -0.35), EASYSIMD_FLOAT64_C(  -800.16), EASYSIMD_FLOAT64_C(  -461.37),
        EASYSIMD_FLOAT64_C(   479.24), EASYSIMD_FLOAT64_C(  -393.06), EASYSIMD_FLOAT64_C(   875.61), EASYSIMD_FLOAT64_C(   875.61) } },
    { { EASYSIMD_FLOAT64_C(   845.57), EASYSIMD_FLOAT64_C(  -631.46), EASYSIMD_FLOAT64_C(   348.07), EASYSIMD_FLOAT64_C(   322.76),
        EASYSIMD_FLOAT64_C(  -388.11), EASYSIMD_FLOAT64_C(   575.56), EASYSIMD_FLOAT64_C(   -20.17), EASYSIMD_FLOAT64_C(   841.52) },
      UINT8_C(227),
      { EASYSIMD_FLOAT64_C(  -434.32), EASYSIMD_FLOAT64_C(   152.31), EASYSIMD_FLOAT64_C(   478.48), EASYSIMD_FLOAT64_C(   418.15),
        EASYSIMD_FLOAT64_C(   567.60), EASYSIMD_FLOAT64_C(  -302.88), EASYSIMD_FLOAT64_C(     4.46), EASYSIMD_FLOAT64_C(   767.44) },
      { EASYSIMD_FLOAT64_C(  -434.32), EASYSIMD_FLOAT64_C(  -434.32), EASYSIMD_FLOAT64_C(   348.07), EASYSIMD_FLOAT64_C(   322.76),
        EASYSIMD_FLOAT64_C(  -388.11), EASYSIMD_FLOAT64_C(   567.60), EASYSIMD_FLOAT64_C(     4.46), EASYSIMD_FLOAT64_C(     4.46) } },
    { { EASYSIMD_FLOAT64_C(  -449.77), EASYSIMD_FLOAT64_C(  -516.30), EASYSIMD_FLOAT64_C(   281.78), EASYSIMD_FLOAT64_C(  -444.39),
        EASYSIMD_FLOAT64_C(  -167.65), EASYSIMD_FLOAT64_C(   963.56), EASYSIMD_FLOAT64_C(   555.26), EASYSIMD_FLOAT64_C(  -426.73) },
      UINT8_C(111),
      { EASYSIMD_FLOAT64_C(   241.00), EASYSIMD_FLOAT64_C(   180.22), EASYSIMD_FLOAT64_C(   506.49), EASYSIMD_FLOAT64_C(   116.61),
        EASYSIMD_FLOAT64_C(  -929.77), EASYSIMD_FLOAT64_C(   352.05), EASYSIMD_FLOAT64_C(   485.14), EASYSIMD_FLOAT64_C(   418.30) },
      { EASYSIMD_FLOAT64_C(   241.00), EASYSIMD_FLOAT64_C(   241.00), EASYSIMD_FLOAT64_C(   506.49), EASYSIMD_FLOAT64_C(   506.49),
        EASYSIMD_FLOAT64_C(  -167.65), EASYSIMD_FLOAT64_C(  -929.77), EASYSIMD_FLOAT64_C(   485.14), EASYSIMD_FLOAT64_C(  -426.73) } },
    { { EASYSIMD_FLOAT64_C(  -325.19), EASYSIMD_FLOAT64_C(  -902.97), EASYSIMD_FLOAT64_C(    -6.14), EASYSIMD_FLOAT64_C(   654.64),
        EASYSIMD_FLOAT64_C(   938.55), EASYSIMD_FLOAT64_C(   383.39), EASYSIMD_FLOAT64_C(  -779.68), EASYSIMD_FLOAT64_C(    90.85) },
      UINT8_C(  0),
      { EASYSIMD_FLOAT64_C(   638.47), EASYSIMD_FLOAT64_C(  -341.55), EASYSIMD_FLOAT64_C(   558.99), EASYSIMD_FLOAT64_C(  -357.07),
        EASYSIMD_FLOAT64_C(  -574.11), EASYSIMD_FLOAT64_C(  -890.79), EASYSIMD_FLOAT64_C(   126.63), EASYSIMD_FLOAT64_C(   707.67) },
      { EASYSIMD_FLOAT64_C(  -325.19), EASYSIMD_FLOAT64_C(  -902.97), EASYSIMD_FLOAT64_C(    -6.14), EASYSIMD_FLOAT64_C(   654.64),
        EASYSIMD_FLOAT64_C(   938.55), EASYSIMD_FLOAT64_C(   383.39), EASYSIMD_FLOAT64_C(  -779.68), EASYSIMD_FLOAT64_C(    90.85) } },
    { { EASYSIMD_FLOAT64_C(  -335.17), EASYSIMD_FLOAT64_C(   958.98), EASYSIMD_FLOAT64_C(   671.22), EASYSIMD_FLOAT64_C(  -779.92),
        EASYSIMD_FLOAT64_C(  -467.74), EASYSIMD_FLOAT64_C(  -826.59), EASYSIMD_FLOAT64_C(   461.09), EASYSIMD_FLOAT64_C(   712.48) },
      UINT8_C(206),
      { EASYSIMD_FLOAT64_C(  -422.31), EASYSIMD_FLOAT64_C(   782.70), EASYSIMD_FLOAT64_C(    31.95), EASYSIMD_FLOAT64_C(  -937.16),
        EASYSIMD_FLOAT64_C(   201.00), EASYSIMD_FLOAT64_C(   706.76), EASYSIMD_FLOAT64_C(  -840.13), EASYSIMD_FLOAT64_C(  -805.13) },
      { EASYSIMD_FLOAT64_C(  -335.17), EASYSIMD_FLOAT64_C(  -422.31), EASYSIMD_FLOAT64_C(    31.95), EASYSIMD_FLOAT64_C(    31.95),
        EASYSIMD_FLOAT64_C(  -467.74), EASYSIMD_FLOAT64_C(  -826.59), EASYSIMD_FLOAT64_C(  -840.13), EASYSIMD_FLOAT64_C(  -840.13) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_movedup_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_movedup_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d src = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r = easysimd_mm512_mask_movedup_pd(src, k, a);

    easysimd_test_x86_write_f64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_movedup_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { UINT8_C(115),
      { EASYSIMD_FLOAT64_C(  -901.59), EASYSIMD_FLOAT64_C(   578.26), EASYSIMD_FLOAT64_C(   581.73), EASYSIMD_FLOAT64_C(   189.27),
        EASYSIMD_FLOAT64_C(  -559.87), EASYSIMD_FLOAT64_C(   220.19), EASYSIMD_FLOAT64_C(   847.72), EASYSIMD_FLOAT64_C(   999.12) },
      { EASYSIMD_FLOAT64_C(  -901.59), EASYSIMD_FLOAT64_C(  -901.59), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(  -559.87), EASYSIMD_FLOAT64_C(  -559.87), EASYSIMD_FLOAT64_C(   847.72), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 61),
      { EASYSIMD_FLOAT64_C(  -726.39), EASYSIMD_FLOAT64_C(  -891.67), EASYSIMD_FLOAT64_C(   -10.25), EASYSIMD_FLOAT64_C(   981.28),
        EASYSIMD_FLOAT64_C(  -226.84), EASYSIMD_FLOAT64_C(   -51.27), EASYSIMD_FLOAT64_C(   652.51), EASYSIMD_FLOAT64_C(    -6.76) },
      { EASYSIMD_FLOAT64_C(  -726.39), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   -10.25), EASYSIMD_FLOAT64_C(   -10.25),
        EASYSIMD_FLOAT64_C(  -226.84), EASYSIMD_FLOAT64_C(  -226.84), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(150),
      { EASYSIMD_FLOAT64_C(   825.91), EASYSIMD_FLOAT64_C(  -545.67), EASYSIMD_FLOAT64_C(   193.47), EASYSIMD_FLOAT64_C(   505.81),
        EASYSIMD_FLOAT64_C(    32.02), EASYSIMD_FLOAT64_C(   -23.83), EASYSIMD_FLOAT64_C(  -462.25), EASYSIMD_FLOAT64_C(    94.86) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   825.91), EASYSIMD_FLOAT64_C(   193.47), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(    32.02), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -462.25) } },
    { UINT8_C(240),
      { EASYSIMD_FLOAT64_C(  -755.49), EASYSIMD_FLOAT64_C(   254.72), EASYSIMD_FLOAT64_C(  -627.96), EASYSIMD_FLOAT64_C(   605.92),
        EASYSIMD_FLOAT64_C(   353.14), EASYSIMD_FLOAT64_C(   950.30), EASYSIMD_FLOAT64_C(   187.64), EASYSIMD_FLOAT64_C(  -457.59) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   353.14), EASYSIMD_FLOAT64_C(   353.14), EASYSIMD_FLOAT64_C(   187.64), EASYSIMD_FLOAT64_C(   187.64) } },
    { UINT8_C(118),
      { EASYSIMD_FLOAT64_C(  -592.16), EASYSIMD_FLOAT64_C(  -609.87), EASYSIMD_FLOAT64_C(  -610.45), EASYSIMD_FLOAT64_C(  -729.04),
        EASYSIMD_FLOAT64_C(  -336.26), EASYSIMD_FLOAT64_C(  -502.12), EASYSIMD_FLOAT64_C(   260.71), EASYSIMD_FLOAT64_C(  -354.98) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -592.16), EASYSIMD_FLOAT64_C(  -610.45), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(  -336.26), EASYSIMD_FLOAT64_C(  -336.26), EASYSIMD_FLOAT64_C(   260.71), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(146),
      { EASYSIMD_FLOAT64_C(  -790.56), EASYSIMD_FLOAT64_C(  -702.47), EASYSIMD_FLOAT64_C(  -735.73), EASYSIMD_FLOAT64_C(   690.43),
        EASYSIMD_FLOAT64_C(  -876.56), EASYSIMD_FLOAT64_C(  -281.40), EASYSIMD_FLOAT64_C(  -116.10), EASYSIMD_FLOAT64_C(   629.25) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -790.56), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(  -876.56), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -116.10) } },
    { UINT8_C( 40),
      { EASYSIMD_FLOAT64_C(   860.07), EASYSIMD_FLOAT64_C(  -832.99), EASYSIMD_FLOAT64_C(  -154.53), EASYSIMD_FLOAT64_C(  -962.76),
        EASYSIMD_FLOAT64_C(  -588.48), EASYSIMD_FLOAT64_C(  -899.80), EASYSIMD_FLOAT64_C(  -590.72), EASYSIMD_FLOAT64_C(  -982.56) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -154.53),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -588.48), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(149),
      { EASYSIMD_FLOAT64_C(  -640.42), EASYSIMD_FLOAT64_C(   205.09), EASYSIMD_FLOAT64_C(   995.74), EASYSIMD_FLOAT64_C(  -249.99),
        EASYSIMD_FLOAT64_C(   612.92), EASYSIMD_FLOAT64_C(  -614.13), EASYSIMD_FLOAT64_C(   139.56), EASYSIMD_FLOAT64_C(   883.88) },
      { EASYSIMD_FLOAT64_C(  -640.42), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   995.74), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   612.92), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   139.56) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_movedup_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_movedup_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r = easysimd_mm512_maskz_movedup_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_movedup_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_movedup_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_movedup_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_movedup_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_movedup_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_movedup_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_movedup_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>