#define EASYSIMD_TEST_ARM_NEON_INSN fma

#include "test-neon.h"
#include "../../../easysimd/arm/neon/fma.h"

static int
test_easysimd_vfma_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 c[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -418.49), EASYSIMD_FLOAT32_C(  -138.55) },
      { EASYSIMD_FLOAT32_C(  -524.10), EASYSIMD_FLOAT32_C(   787.36) },
      { EASYSIMD_FLOAT32_C(   147.35), EASYSIMD_FLOAT32_C(  -958.10) },
      { EASYSIMD_FLOAT32_C(-77644.62), EASYSIMD_FLOAT32_C(-754508.12) } },
    { { EASYSIMD_FLOAT32_C(  -853.74), EASYSIMD_FLOAT32_C(  -358.34) },
      { EASYSIMD_FLOAT32_C(   531.45), EASYSIMD_FLOAT32_C(  -127.29) },
      { EASYSIMD_FLOAT32_C(  -403.29), EASYSIMD_FLOAT32_C(  -763.98) },
      { EASYSIMD_FLOAT32_C(-215182.22), EASYSIMD_FLOAT32_C( 96888.67) } },
    { { EASYSIMD_FLOAT32_C(   -14.18), EASYSIMD_FLOAT32_C(  -322.90) },
      { EASYSIMD_FLOAT32_C(   334.11), EASYSIMD_FLOAT32_C(    84.20) },
      { EASYSIMD_FLOAT32_C(   798.51), EASYSIMD_FLOAT32_C(   508.71) },
      { EASYSIMD_FLOAT32_C(266776.00), EASYSIMD_FLOAT32_C( 42510.48) } },
    { { EASYSIMD_FLOAT32_C(   -96.91), EASYSIMD_FLOAT32_C(  -205.64) },
      { EASYSIMD_FLOAT32_C(   562.33), EASYSIMD_FLOAT32_C(  -123.00) },
      { EASYSIMD_FLOAT32_C(    50.61), EASYSIMD_FLOAT32_C(  -890.35) },
      { EASYSIMD_FLOAT32_C( 28362.61), EASYSIMD_FLOAT32_C(109307.41) } },
    { { EASYSIMD_FLOAT32_C(   916.53), EASYSIMD_FLOAT32_C(   885.37) },
      { EASYSIMD_FLOAT32_C(   109.83), EASYSIMD_FLOAT32_C(   977.63) },
      { EASYSIMD_FLOAT32_C(  -333.79), EASYSIMD_FLOAT32_C(   850.31) },
      { EASYSIMD_FLOAT32_C(-35743.63), EASYSIMD_FLOAT32_C(832173.94) } },
    { { EASYSIMD_FLOAT32_C(   819.13), EASYSIMD_FLOAT32_C(   247.72) },
      { EASYSIMD_FLOAT32_C(  -288.24), EASYSIMD_FLOAT32_C(  -704.97) },
      { EASYSIMD_FLOAT32_C(    35.08), EASYSIMD_FLOAT32_C(   859.11) },
      { EASYSIMD_FLOAT32_C( -9292.33), EASYSIMD_FLOAT32_C(-605399.00) } },
    { { EASYSIMD_FLOAT32_C(  -663.07), EASYSIMD_FLOAT32_C(   181.34) },
      { EASYSIMD_FLOAT32_C(  -499.23), EASYSIMD_FLOAT32_C(   868.39) },
      { EASYSIMD_FLOAT32_C(  -945.95), EASYSIMD_FLOAT32_C(    97.47) },
      { EASYSIMD_FLOAT32_C(471583.56), EASYSIMD_FLOAT32_C( 84823.31) } },
    { { EASYSIMD_FLOAT32_C(  -895.59), EASYSIMD_FLOAT32_C(    39.87) },
      { EASYSIMD_FLOAT32_C(   774.58), EASYSIMD_FLOAT32_C(   438.52) },
      { EASYSIMD_FLOAT32_C(  -875.93), EASYSIMD_FLOAT32_C(   573.09) },
      { EASYSIMD_FLOAT32_C(-679373.44), EASYSIMD_FLOAT32_C(251351.30) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t c = easysimd_vld1_f32(test_vec[i].c);
    easysimd_float32x2_t r = easysimd_vfma_f32(a, b, c);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t c = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vfma_f32(a, b, c);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmaq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 c[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   346.50), EASYSIMD_FLOAT32_C(  -115.47), EASYSIMD_FLOAT32_C(  -771.21), EASYSIMD_FLOAT32_C(  -566.62) },
      { EASYSIMD_FLOAT32_C(    92.23), EASYSIMD_FLOAT32_C(   821.24), EASYSIMD_FLOAT32_C(   471.19), EASYSIMD_FLOAT32_C(   163.68) },
      { EASYSIMD_FLOAT32_C(  -528.73), EASYSIMD_FLOAT32_C(   685.78), EASYSIMD_FLOAT32_C(  -994.64), EASYSIMD_FLOAT32_C(  -615.49) },
      { EASYSIMD_FLOAT32_C(-48418.27), EASYSIMD_FLOAT32_C(563074.50), EASYSIMD_FLOAT32_C(-469435.66), EASYSIMD_FLOAT32_C(-101310.02) } },
    { { EASYSIMD_FLOAT32_C(   129.60), EASYSIMD_FLOAT32_C(   478.63), EASYSIMD_FLOAT32_C(   540.40), EASYSIMD_FLOAT32_C(   562.04) },
      { EASYSIMD_FLOAT32_C(   417.81), EASYSIMD_FLOAT32_C(   605.14), EASYSIMD_FLOAT32_C(   -70.65), EASYSIMD_FLOAT32_C(   427.33) },
      { EASYSIMD_FLOAT32_C(   128.97), EASYSIMD_FLOAT32_C(   832.34), EASYSIMD_FLOAT32_C(  -323.86), EASYSIMD_FLOAT32_C(  -987.62) },
      { EASYSIMD_FLOAT32_C( 54014.55), EASYSIMD_FLOAT32_C(504160.88), EASYSIMD_FLOAT32_C( 23421.11), EASYSIMD_FLOAT32_C(-421477.59) } },
    { { EASYSIMD_FLOAT32_C(  -601.15), EASYSIMD_FLOAT32_C(   243.37), EASYSIMD_FLOAT32_C(  -503.84), EASYSIMD_FLOAT32_C(   449.09) },
      { EASYSIMD_FLOAT32_C(   290.97), EASYSIMD_FLOAT32_C(  -850.95), EASYSIMD_FLOAT32_C(  -410.04), EASYSIMD_FLOAT32_C(  -362.53) },
      { EASYSIMD_FLOAT32_C(    33.58), EASYSIMD_FLOAT32_C(  -181.25), EASYSIMD_FLOAT32_C(    70.85), EASYSIMD_FLOAT32_C(  -874.19) },
      { EASYSIMD_FLOAT32_C(  9169.62), EASYSIMD_FLOAT32_C(154478.06), EASYSIMD_FLOAT32_C(-29555.17), EASYSIMD_FLOAT32_C(317369.19) } },
    { { EASYSIMD_FLOAT32_C(  -360.00), EASYSIMD_FLOAT32_C(  -457.96), EASYSIMD_FLOAT32_C(   289.49), EASYSIMD_FLOAT32_C(   111.26) },
      { EASYSIMD_FLOAT32_C(  -772.18), EASYSIMD_FLOAT32_C(   294.86), EASYSIMD_FLOAT32_C(   495.77), EASYSIMD_FLOAT32_C(   357.42) },
      { EASYSIMD_FLOAT32_C(  -226.51), EASYSIMD_FLOAT32_C(    36.17), EASYSIMD_FLOAT32_C(   -80.54), EASYSIMD_FLOAT32_C(  -808.70) },
      { EASYSIMD_FLOAT32_C(174546.48), EASYSIMD_FLOAT32_C( 10207.12), EASYSIMD_FLOAT32_C(-39639.82), EASYSIMD_FLOAT32_C(-288934.31) } },
    { { EASYSIMD_FLOAT32_C(  -358.68), EASYSIMD_FLOAT32_C(   848.81), EASYSIMD_FLOAT32_C(   618.63), EASYSIMD_FLOAT32_C(   770.28) },
      { EASYSIMD_FLOAT32_C(   681.15), EASYSIMD_FLOAT32_C(  -705.23), EASYSIMD_FLOAT32_C(   782.67), EASYSIMD_FLOAT32_C(  -920.00) },
      { EASYSIMD_FLOAT32_C(   538.14), EASYSIMD_FLOAT32_C(  -721.17), EASYSIMD_FLOAT32_C(   529.09), EASYSIMD_FLOAT32_C(  -170.89) },
      { EASYSIMD_FLOAT32_C(366195.41), EASYSIMD_FLOAT32_C(509439.50), EASYSIMD_FLOAT32_C(414721.50), EASYSIMD_FLOAT32_C(157989.08) } },
    { { EASYSIMD_FLOAT32_C(  -572.13), EASYSIMD_FLOAT32_C(  -880.95), EASYSIMD_FLOAT32_C(   466.59), EASYSIMD_FLOAT32_C(   461.46) },
      { EASYSIMD_FLOAT32_C(   -62.19), EASYSIMD_FLOAT32_C(  -462.57), EASYSIMD_FLOAT32_C(   587.26), EASYSIMD_FLOAT32_C(   577.80) },
      { EASYSIMD_FLOAT32_C(    79.47), EASYSIMD_FLOAT32_C(  -123.24), EASYSIMD_FLOAT32_C(  -310.93), EASYSIMD_FLOAT32_C(   307.29) },
      { EASYSIMD_FLOAT32_C( -5514.37), EASYSIMD_FLOAT32_C( 56126.18), EASYSIMD_FLOAT32_C(-182130.16), EASYSIMD_FLOAT32_C(178013.62) } },
    { { EASYSIMD_FLOAT32_C(  -828.39), EASYSIMD_FLOAT32_C(  -815.16), EASYSIMD_FLOAT32_C(  -335.29), EASYSIMD_FLOAT32_C(   -54.90) },
      { EASYSIMD_FLOAT32_C(   221.02), EASYSIMD_FLOAT32_C(   584.17), EASYSIMD_FLOAT32_C(   136.40), EASYSIMD_FLOAT32_C(   862.33) },
      { EASYSIMD_FLOAT32_C(   432.98), EASYSIMD_FLOAT32_C(  -244.97), EASYSIMD_FLOAT32_C(   632.62), EASYSIMD_FLOAT32_C(   114.13) },
      { EASYSIMD_FLOAT32_C( 94868.85), EASYSIMD_FLOAT32_C(-143919.28), EASYSIMD_FLOAT32_C( 85954.07), EASYSIMD_FLOAT32_C( 98362.82) } },
    { { EASYSIMD_FLOAT32_C(    49.80), EASYSIMD_FLOAT32_C(   415.28), EASYSIMD_FLOAT32_C(   194.13), EASYSIMD_FLOAT32_C(  -412.06) },
      { EASYSIMD_FLOAT32_C(   694.11), EASYSIMD_FLOAT32_C(  -276.78), EASYSIMD_FLOAT32_C(   417.06), EASYSIMD_FLOAT32_C(  -878.02) },
      { EASYSIMD_FLOAT32_C(  -157.72), EASYSIMD_FLOAT32_C(  -116.36), EASYSIMD_FLOAT32_C(   583.44), EASYSIMD_FLOAT32_C(   780.08) },
      { EASYSIMD_FLOAT32_C(-109425.23), EASYSIMD_FLOAT32_C( 32621.40), EASYSIMD_FLOAT32_C(243523.61), EASYSIMD_FLOAT32_C(-685337.94) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t c = easysimd_vld1q_f32(test_vec[i].c);
    easysimd_float32x4_t r = easysimd_vfmaq_f32(a, b, c);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t c = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vfmaq_f32(a, b, c);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfma_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 b[1];
    easysimd_float64 c[1];
    easysimd_float64 r[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   476.98) },
      { EASYSIMD_FLOAT64_C(  -170.73) },
      { EASYSIMD_FLOAT64_C(   555.43) },
      { EASYSIMD_FLOAT64_C(-94351.58) } },
    { { EASYSIMD_FLOAT64_C(  -581.61) },
      { EASYSIMD_FLOAT64_C(  -282.78) },
      { EASYSIMD_FLOAT64_C(   375.59) },
      { EASYSIMD_FLOAT64_C(-106790.95) } },
    { { EASYSIMD_FLOAT64_C(  -775.88) },
      { EASYSIMD_FLOAT64_C(  -941.47) },
      { EASYSIMD_FLOAT64_C(   -93.98) },
      { EASYSIMD_FLOAT64_C( 87703.47) } },
    { { EASYSIMD_FLOAT64_C(  -601.19) },
      { EASYSIMD_FLOAT64_C(   198.40) },
      { EASYSIMD_FLOAT64_C(  -419.02) },
      { EASYSIMD_FLOAT64_C(-83734.76) } },
    { { EASYSIMD_FLOAT64_C(  -361.96) },
      { EASYSIMD_FLOAT64_C(  -805.69) },
      { EASYSIMD_FLOAT64_C(   777.69) },
      { EASYSIMD_FLOAT64_C(-626939.02) } },
    { { EASYSIMD_FLOAT64_C(  -232.90) },
      { EASYSIMD_FLOAT64_C(  -937.07) },
      { EASYSIMD_FLOAT64_C(    65.95) },
      { EASYSIMD_FLOAT64_C(-62032.67) } },
    { { EASYSIMD_FLOAT64_C(   636.76) },
      { EASYSIMD_FLOAT64_C(   544.44) },
      { EASYSIMD_FLOAT64_C(  -572.38) },
      { EASYSIMD_FLOAT64_C(-310989.81) } },
    { { EASYSIMD_FLOAT64_C(   285.39) },
      { EASYSIMD_FLOAT64_C(  -657.86) },
      { EASYSIMD_FLOAT64_C(  -418.32) },
      { EASYSIMD_FLOAT64_C(275481.39) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_float64x1_t c = easysimd_vld1_f64(test_vec[i].c);
    easysimd_float64x1_t r = easysimd_vfma_f64(a, b, c);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0f, 1000.0f);
    easysimd_float64x1_t b = easysimd_test_arm_neon_random_f64x1(-1000.0f, 1000.0f);
    easysimd_float64x1_t c = easysimd_test_arm_neon_random_f64x1(-1000.0f, 1000.0f);
    easysimd_float64x1_t r = easysimd_vfma_f64(a, b, c);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmaq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 c[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -518.21), EASYSIMD_FLOAT64_C(   476.64) },
      { EASYSIMD_FLOAT64_C(   440.79), EASYSIMD_FLOAT64_C(   136.55) },
      { EASYSIMD_FLOAT64_C(  -245.37), EASYSIMD_FLOAT64_C(  -809.93) },
      { EASYSIMD_FLOAT64_C(-108674.85), EASYSIMD_FLOAT64_C(-110119.30) } },
    { { EASYSIMD_FLOAT64_C(   189.43), EASYSIMD_FLOAT64_C(  -306.05) },
      { EASYSIMD_FLOAT64_C(  -193.42), EASYSIMD_FLOAT64_C(   154.66) },
      { EASYSIMD_FLOAT64_C(   240.36), EASYSIMD_FLOAT64_C(   543.48) },
      { EASYSIMD_FLOAT64_C(-46301.00), EASYSIMD_FLOAT64_C( 83748.57) } },
    { { EASYSIMD_FLOAT64_C(  -750.62), EASYSIMD_FLOAT64_C(  -963.03) },
      { EASYSIMD_FLOAT64_C(   472.07), EASYSIMD_FLOAT64_C(   853.23) },
      { EASYSIMD_FLOAT64_C(   412.19), EASYSIMD_FLOAT64_C(  -376.01) },
      { EASYSIMD_FLOAT64_C(193831.91), EASYSIMD_FLOAT64_C(-321786.04) } },
    { { EASYSIMD_FLOAT64_C(   870.90), EASYSIMD_FLOAT64_C(  -245.97) },
      { EASYSIMD_FLOAT64_C(   851.28), EASYSIMD_FLOAT64_C(   177.73) },
      { EASYSIMD_FLOAT64_C(   173.41), EASYSIMD_FLOAT64_C(   817.77) },
      { EASYSIMD_FLOAT64_C(148491.36), EASYSIMD_FLOAT64_C(145096.29) } },
    { { EASYSIMD_FLOAT64_C(  -790.75), EASYSIMD_FLOAT64_C(   629.67) },
      { EASYSIMD_FLOAT64_C(  -863.72), EASYSIMD_FLOAT64_C(   676.86) },
      { EASYSIMD_FLOAT64_C(  -698.40), EASYSIMD_FLOAT64_C(  -386.38) },
      { EASYSIMD_FLOAT64_C(602431.30), EASYSIMD_FLOAT64_C(-260895.50) } },
    { { EASYSIMD_FLOAT64_C(   601.21), EASYSIMD_FLOAT64_C(  -216.61) },
      { EASYSIMD_FLOAT64_C(  -909.75), EASYSIMD_FLOAT64_C(    42.01) },
      { EASYSIMD_FLOAT64_C(   919.95), EASYSIMD_FLOAT64_C(  -155.12) },
      { EASYSIMD_FLOAT64_C(-836323.30), EASYSIMD_FLOAT64_C( -6733.20) } },
    { { EASYSIMD_FLOAT64_C(   232.08), EASYSIMD_FLOAT64_C(   109.38) },
      { EASYSIMD_FLOAT64_C(   538.83), EASYSIMD_FLOAT64_C(  -961.34) },
      { EASYSIMD_FLOAT64_C(  -735.96), EASYSIMD_FLOAT64_C(  -220.81) },
      { EASYSIMD_FLOAT64_C(-396325.25), EASYSIMD_FLOAT64_C(212382.87) } },
    { { EASYSIMD_FLOAT64_C(   582.14), EASYSIMD_FLOAT64_C(  -486.58) },
      { EASYSIMD_FLOAT64_C(  -183.84), EASYSIMD_FLOAT64_C(    54.21) },
      { EASYSIMD_FLOAT64_C(  -633.35), EASYSIMD_FLOAT64_C(  -771.64) },
      { EASYSIMD_FLOAT64_C(117017.20), EASYSIMD_FLOAT64_C(-42317.18) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t c = easysimd_vld1q_f64(test_vec[i].c);
    easysimd_float64x2_t r = easysimd_vfmaq_f64(a, b, c);
    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t c = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r = easysimd_vfmaq_f64(a, b, c);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfma_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmaq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfma_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmaq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
