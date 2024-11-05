#define EASYSIMD_TEST_X86_AVX512_INSN range

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/rcp.h>

static int
test_easysimd_mm_rcp14_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   510.86), EASYSIMD_FLOAT32_C(    69.87), EASYSIMD_FLOAT32_C(    90.85), EASYSIMD_FLOAT32_C(  -555.27) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.01), EASYSIMD_FLOAT32_C(     0.01), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(   655.60), EASYSIMD_FLOAT32_C(  -963.20), EASYSIMD_FLOAT32_C(   801.68), EASYSIMD_FLOAT32_C(  -212.46) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(  -217.71), EASYSIMD_FLOAT32_C(   214.30), EASYSIMD_FLOAT32_C(   -66.91), EASYSIMD_FLOAT32_C(  -639.31) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(  -304.05), EASYSIMD_FLOAT32_C(  -153.42), EASYSIMD_FLOAT32_C(   588.37), EASYSIMD_FLOAT32_C(   418.94) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   801.39), EASYSIMD_FLOAT32_C(  -131.20), EASYSIMD_FLOAT32_C(  -451.03), EASYSIMD_FLOAT32_C(   205.16) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   842.54), EASYSIMD_FLOAT32_C(   965.34), EASYSIMD_FLOAT32_C(   352.85), EASYSIMD_FLOAT32_C(   897.83) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -754.15), EASYSIMD_FLOAT32_C(   957.58), EASYSIMD_FLOAT32_C(   950.90), EASYSIMD_FLOAT32_C(  -657.78) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(  -959.94), EASYSIMD_FLOAT32_C(  -123.19), EASYSIMD_FLOAT32_C(  -185.42), EASYSIMD_FLOAT32_C(   550.93) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_rcp14_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_rcp14_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_rcp14_ps(a);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_rcp14_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   498.80), EASYSIMD_FLOAT32_C(  -670.29), EASYSIMD_FLOAT32_C(   835.38), EASYSIMD_FLOAT32_C(  -279.63) },
      UINT8_C(166),
      { EASYSIMD_FLOAT32_C(   948.30), EASYSIMD_FLOAT32_C(  -803.82), EASYSIMD_FLOAT32_C(  -774.85), EASYSIMD_FLOAT32_C(   308.12) },
      { EASYSIMD_FLOAT32_C(   498.80), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -279.63) } },
    { { EASYSIMD_FLOAT32_C(  -331.68), EASYSIMD_FLOAT32_C(   889.29), EASYSIMD_FLOAT32_C(  -874.67), EASYSIMD_FLOAT32_C(   433.03) },
      UINT8_C(178),
      { EASYSIMD_FLOAT32_C(  -510.38), EASYSIMD_FLOAT32_C(   882.73), EASYSIMD_FLOAT32_C(   298.12), EASYSIMD_FLOAT32_C(   471.15) },
      { EASYSIMD_FLOAT32_C(  -331.68), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -874.67), EASYSIMD_FLOAT32_C(   433.03) } },
    { { EASYSIMD_FLOAT32_C(   349.77), EASYSIMD_FLOAT32_C(  -398.86), EASYSIMD_FLOAT32_C(  -171.04), EASYSIMD_FLOAT32_C(   540.78) },
      UINT8_C( 55),
      { EASYSIMD_FLOAT32_C(  -102.21), EASYSIMD_FLOAT32_C(  -275.85), EASYSIMD_FLOAT32_C(  -197.39), EASYSIMD_FLOAT32_C(   393.15) },
      { EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(   540.78) } },
    { { EASYSIMD_FLOAT32_C(   714.41), EASYSIMD_FLOAT32_C(  -925.19), EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(  -276.30) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT32_C(   775.71), EASYSIMD_FLOAT32_C(  -440.92), EASYSIMD_FLOAT32_C(  -706.02), EASYSIMD_FLOAT32_C(   230.47) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -925.19), EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(  -276.30) } },
    { { EASYSIMD_FLOAT32_C(  -492.62), EASYSIMD_FLOAT32_C(  -509.84), EASYSIMD_FLOAT32_C(   455.62), EASYSIMD_FLOAT32_C(   815.49) },
      UINT8_C( 18),
      { EASYSIMD_FLOAT32_C(   344.91), EASYSIMD_FLOAT32_C(   940.82), EASYSIMD_FLOAT32_C(  -408.49), EASYSIMD_FLOAT32_C(  -469.37) },
      { EASYSIMD_FLOAT32_C(  -492.62), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   455.62), EASYSIMD_FLOAT32_C(   815.49) } },
    { { EASYSIMD_FLOAT32_C(  -569.56), EASYSIMD_FLOAT32_C(  -525.76), EASYSIMD_FLOAT32_C(   828.75), EASYSIMD_FLOAT32_C(   901.59) },
      UINT8_C(206),
      { EASYSIMD_FLOAT32_C(  -570.11), EASYSIMD_FLOAT32_C(  -269.45), EASYSIMD_FLOAT32_C(   364.78), EASYSIMD_FLOAT32_C(   961.85) },
      { EASYSIMD_FLOAT32_C(  -569.56), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   628.34), EASYSIMD_FLOAT32_C(  -911.06), EASYSIMD_FLOAT32_C(  -235.53), EASYSIMD_FLOAT32_C(    21.49) },
      UINT8_C(170),
      { EASYSIMD_FLOAT32_C(  -160.72), EASYSIMD_FLOAT32_C(  -532.51), EASYSIMD_FLOAT32_C(  -472.96), EASYSIMD_FLOAT32_C(  -587.11) },
      { EASYSIMD_FLOAT32_C(   628.34), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -235.53), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(  -756.80), EASYSIMD_FLOAT32_C(    86.12), EASYSIMD_FLOAT32_C(  -293.13), EASYSIMD_FLOAT32_C(   473.67) },
      UINT8_C(161),
      { EASYSIMD_FLOAT32_C(   197.04), EASYSIMD_FLOAT32_C(   -70.71), EASYSIMD_FLOAT32_C(   408.99), EASYSIMD_FLOAT32_C(  -644.48) },
      { EASYSIMD_FLOAT32_C(     0.01), EASYSIMD_FLOAT32_C(    86.12), EASYSIMD_FLOAT32_C(  -293.13), EASYSIMD_FLOAT32_C(   473.67) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_rcp14_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_rcp14_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_rcp14_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_rcp14_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(102),
      { EASYSIMD_FLOAT32_C(   349.81), EASYSIMD_FLOAT32_C(   -52.97), EASYSIMD_FLOAT32_C(  -195.17), EASYSIMD_FLOAT32_C(   780.25) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.02), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 25),
      { EASYSIMD_FLOAT32_C(  -366.42), EASYSIMD_FLOAT32_C(   681.85), EASYSIMD_FLOAT32_C(   245.27), EASYSIMD_FLOAT32_C(    63.48) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.02) } },
    { UINT8_C(253),
      { EASYSIMD_FLOAT32_C(  -389.94), EASYSIMD_FLOAT32_C(    25.33), EASYSIMD_FLOAT32_C(  -959.26), EASYSIMD_FLOAT32_C(  -301.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00) } },
    { UINT8_C( 22),
      { EASYSIMD_FLOAT32_C(    62.23), EASYSIMD_FLOAT32_C(  -497.66), EASYSIMD_FLOAT32_C(  -370.92), EASYSIMD_FLOAT32_C(   529.72) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(116),
      { EASYSIMD_FLOAT32_C(    41.97), EASYSIMD_FLOAT32_C(   772.91), EASYSIMD_FLOAT32_C(  -884.50), EASYSIMD_FLOAT32_C(   748.84) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 48),
      { EASYSIMD_FLOAT32_C(   709.00), EASYSIMD_FLOAT32_C(   -54.12), EASYSIMD_FLOAT32_C(  -824.13), EASYSIMD_FLOAT32_C(   117.99) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 83),
      { EASYSIMD_FLOAT32_C(  -549.93), EASYSIMD_FLOAT32_C(  -532.19), EASYSIMD_FLOAT32_C(  -751.57), EASYSIMD_FLOAT32_C(   254.90) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(147),
      { EASYSIMD_FLOAT32_C(   669.70), EASYSIMD_FLOAT32_C(   888.48), EASYSIMD_FLOAT32_C(   929.91), EASYSIMD_FLOAT32_C(   -85.03) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_rcp14_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_rcp14_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_rcp14_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_rcp14_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -48.05), EASYSIMD_FLOAT64_C(  -657.68) },
      { EASYSIMD_FLOAT64_C(    -0.02), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(   525.03), EASYSIMD_FLOAT64_C(   977.28) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -616.94), EASYSIMD_FLOAT64_C(  -775.98) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(   767.08), EASYSIMD_FLOAT64_C(   445.29) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -273.64), EASYSIMD_FLOAT64_C(  -603.85) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(   -25.00), EASYSIMD_FLOAT64_C(   755.75) },
      { EASYSIMD_FLOAT64_C(    -0.04), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   438.12), EASYSIMD_FLOAT64_C(  -252.08) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(   871.25), EASYSIMD_FLOAT64_C(   186.96) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.01) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_rcp14_pd(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_rcp14_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_rcp14_pd(a);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_rcp14_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   994.50), EASYSIMD_FLOAT64_C(   580.25) },
      UINT8_C(152),
      { EASYSIMD_FLOAT64_C(  -829.63), EASYSIMD_FLOAT64_C(  -301.75) },
      { EASYSIMD_FLOAT64_C(   994.50), EASYSIMD_FLOAT64_C(   580.25) } },
    { { EASYSIMD_FLOAT64_C(   434.24), EASYSIMD_FLOAT64_C(  -379.57) },
      UINT8_C(236),
      { EASYSIMD_FLOAT64_C(   682.67), EASYSIMD_FLOAT64_C(   875.33) },
      { EASYSIMD_FLOAT64_C(   434.24), EASYSIMD_FLOAT64_C(  -379.57) } },
    { { EASYSIMD_FLOAT64_C(   414.12), EASYSIMD_FLOAT64_C(   352.37) },
      UINT8_C( 34),
      { EASYSIMD_FLOAT64_C(   344.03), EASYSIMD_FLOAT64_C(  -732.66) },
      { EASYSIMD_FLOAT64_C(   414.12), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(  -284.24), EASYSIMD_FLOAT64_C(   686.35) },
      UINT8_C(167),
      { EASYSIMD_FLOAT64_C(  -306.96), EASYSIMD_FLOAT64_C(  -930.60) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(  -983.61), EASYSIMD_FLOAT64_C(  -539.88) },
      UINT8_C(230),
      { EASYSIMD_FLOAT64_C(  -257.25), EASYSIMD_FLOAT64_C(  -143.72) },
      { EASYSIMD_FLOAT64_C(  -983.61), EASYSIMD_FLOAT64_C(    -0.01) } },
    { { EASYSIMD_FLOAT64_C(  -510.31), EASYSIMD_FLOAT64_C(  -501.50) },
      UINT8_C(  3),
      { EASYSIMD_FLOAT64_C(   237.61), EASYSIMD_FLOAT64_C(  -630.25) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(   481.36), EASYSIMD_FLOAT64_C(   232.10) },
      UINT8_C(116),
      { EASYSIMD_FLOAT64_C(   614.21), EASYSIMD_FLOAT64_C(   402.47) },
      { EASYSIMD_FLOAT64_C(   481.36), EASYSIMD_FLOAT64_C(   232.10) } },
    { { EASYSIMD_FLOAT64_C(  -351.75), EASYSIMD_FLOAT64_C(    48.45) },
      UINT8_C(160),
      { EASYSIMD_FLOAT64_C(   814.30), EASYSIMD_FLOAT64_C(  -268.88) },
      { EASYSIMD_FLOAT64_C(  -351.75), EASYSIMD_FLOAT64_C(    48.45) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_rcp14_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_rcp14_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_rcp14_pd(src, k, a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_rcp14_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C( 72),
      { EASYSIMD_FLOAT64_C(   228.42), EASYSIMD_FLOAT64_C(  -916.51) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(106),
      { EASYSIMD_FLOAT64_C(  -427.55), EASYSIMD_FLOAT64_C(  -649.18) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { UINT8_C(182),
      { EASYSIMD_FLOAT64_C(  -741.20), EASYSIMD_FLOAT64_C(  -856.81) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { UINT8_C( 97),
      { EASYSIMD_FLOAT64_C(  -671.80), EASYSIMD_FLOAT64_C(  -840.42) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(128),
      { EASYSIMD_FLOAT64_C(   842.89), EASYSIMD_FLOAT64_C(   -97.67) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(195),
      { EASYSIMD_FLOAT64_C(  -667.42), EASYSIMD_FLOAT64_C(   400.83) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(198),
      { EASYSIMD_FLOAT64_C(   570.19), EASYSIMD_FLOAT64_C(   770.58) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(  4),
      { EASYSIMD_FLOAT64_C(  -197.71), EASYSIMD_FLOAT64_C(   720.58) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_rcp14_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_rcp14_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_rcp14_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_rcp14_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_rcp14_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_rcp14_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_rcp14_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_rcp14_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_rcp14_pd)

EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>