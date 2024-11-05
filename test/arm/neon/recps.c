
#define EASYSIMD_TEST_ARM_NEON_INSN recps

#include "test-neon.h"
#include "../../../easysimd/arm/neon/recps.h"

static int
test_easysimd_vrecpss_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a;
    easysimd_float32 b;
    easysimd_float32 r;
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(  -748.53),
      EASYSIMD_FLOAT32_C(  -323.44),
      EASYSIMD_FLOAT32_C(-242102.55) },
    { EASYSIMD_FLOAT32_C(  -863.84),
      EASYSIMD_FLOAT32_C(  -111.67),
      EASYSIMD_FLOAT32_C(-96463.02) },
    { EASYSIMD_FLOAT32_C(  -771.32),
      EASYSIMD_FLOAT32_C(  -741.55),
      EASYSIMD_FLOAT32_C(-571970.31) },
    { EASYSIMD_FLOAT32_C(  -342.64),
      EASYSIMD_FLOAT32_C(   444.85),
      EASYSIMD_FLOAT32_C(152425.41) },
    { EASYSIMD_FLOAT32_C(  -655.40),
      EASYSIMD_FLOAT32_C(  -879.55),
      EASYSIMD_FLOAT32_C(-576455.06) },
    { EASYSIMD_FLOAT32_C(  -372.59),
      EASYSIMD_FLOAT32_C(  -918.90),
      EASYSIMD_FLOAT32_C(-342370.97) },
    { EASYSIMD_FLOAT32_C(   213.91),
      EASYSIMD_FLOAT32_C(  -931.13),
      EASYSIMD_FLOAT32_C(199180.02) },
    { EASYSIMD_FLOAT32_C(    10.43),
      EASYSIMD_FLOAT32_C(   956.65),
      EASYSIMD_FLOAT32_C( -9975.86) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32_t a = test_vec[i].a;
    easysimd_float32_t b = test_vec[i].b;
    easysimd_float32_t r = easysimd_vrecpss_f32(a, b);

    easysimd_assert_equal_f32(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    easysimd_float32_t b = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    easysimd_float32_t r = easysimd_vrecpss_f32(a, b);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpsd_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a;
    easysimd_float64 b;
    easysimd_float64 r;
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(  -162.57),
      EASYSIMD_FLOAT64_C(   -74.56),
      EASYSIMD_FLOAT64_C(-12119.22) },
    { EASYSIMD_FLOAT64_C(  -728.37),
      EASYSIMD_FLOAT64_C(  -444.51),
      EASYSIMD_FLOAT64_C(-323765.75) },
    { EASYSIMD_FLOAT64_C(  -661.56),
      EASYSIMD_FLOAT64_C(  -963.04),
      EASYSIMD_FLOAT64_C(-637106.74) },
    { EASYSIMD_FLOAT64_C(    94.90),
      EASYSIMD_FLOAT64_C(  -441.19),
      EASYSIMD_FLOAT64_C( 41870.93) },
    { EASYSIMD_FLOAT64_C(   695.96),
      EASYSIMD_FLOAT64_C(   222.69),
      EASYSIMD_FLOAT64_C(-154981.33) },
    { EASYSIMD_FLOAT64_C(  -265.05),
      EASYSIMD_FLOAT64_C(   704.53),
      EASYSIMD_FLOAT64_C(186737.68) },
    { EASYSIMD_FLOAT64_C(   740.69),
      EASYSIMD_FLOAT64_C(    -6.61),
      EASYSIMD_FLOAT64_C(  4897.96) },
    { EASYSIMD_FLOAT64_C(   225.75),
      EASYSIMD_FLOAT64_C(   454.29),
      EASYSIMD_FLOAT64_C(-102553.97) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64_t a = test_vec[i].a;
    easysimd_float64_t b = test_vec[i].b;
    easysimd_float64_t r = easysimd_vrecpsd_f64(a, b);

    easysimd_assert_equal_f64(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64_t a = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    easysimd_float64_t b = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    easysimd_float64_t r = easysimd_vrecpsd_f64(a, b);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecps_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 b[1];
    easysimd_float64 r[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -543.91) },
      { EASYSIMD_FLOAT64_C(  -340.35) },
      { EASYSIMD_FLOAT64_C(-185117.77) } },
    { { EASYSIMD_FLOAT64_C(    33.22) },
      { EASYSIMD_FLOAT64_C(  -459.68) },
      { EASYSIMD_FLOAT64_C( 15272.57) } },
    { { EASYSIMD_FLOAT64_C(    33.15) },
      { EASYSIMD_FLOAT64_C(   387.38) },
      { EASYSIMD_FLOAT64_C(-12839.65) } },
    { { EASYSIMD_FLOAT64_C(  -549.95) },
      { EASYSIMD_FLOAT64_C(   836.96) },
      { EASYSIMD_FLOAT64_C(460288.15) } },
    { { EASYSIMD_FLOAT64_C(  -329.05) },
      { EASYSIMD_FLOAT64_C(  -483.44) },
      { EASYSIMD_FLOAT64_C(-159073.93) } },
    { { EASYSIMD_FLOAT64_C(  -667.77) },
      { EASYSIMD_FLOAT64_C(  -367.80) },
      { EASYSIMD_FLOAT64_C(-245603.81) } },
    { { EASYSIMD_FLOAT64_C(  -347.20) },
      { EASYSIMD_FLOAT64_C(   672.37) },
      { EASYSIMD_FLOAT64_C(233448.86) } },
    { { EASYSIMD_FLOAT64_C(  -463.31) },
      { EASYSIMD_FLOAT64_C(  -856.45) },
      { EASYSIMD_FLOAT64_C(-396799.85) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_float64x1_t r = easysimd_vrecps_f64(a, b);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t b = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vrecps_f64(a, b);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecps_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 b[2];
    easysimd_float32 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    47.87), EASYSIMD_FLOAT32_C(   269.61) },
      { EASYSIMD_FLOAT32_C(    96.45), EASYSIMD_FLOAT32_C(  -250.23) },
      { EASYSIMD_FLOAT32_C( -4615.06), EASYSIMD_FLOAT32_C( 67466.51) } },
    { { EASYSIMD_FLOAT32_C(  -555.90), EASYSIMD_FLOAT32_C(  -609.45) },
      { EASYSIMD_FLOAT32_C(   322.74), EASYSIMD_FLOAT32_C(  -176.70) },
      { EASYSIMD_FLOAT32_C(179413.17), EASYSIMD_FLOAT32_C(-107687.81) } },
    { { EASYSIMD_FLOAT32_C(  -980.02), EASYSIMD_FLOAT32_C(   183.49) },
      { EASYSIMD_FLOAT32_C(  -974.10), EASYSIMD_FLOAT32_C(   331.54) },
      { EASYSIMD_FLOAT32_C(-954635.50), EASYSIMD_FLOAT32_C(-60832.28) } },
    { { EASYSIMD_FLOAT32_C(   388.02), EASYSIMD_FLOAT32_C(  -288.08) },
      { EASYSIMD_FLOAT32_C(  -277.25), EASYSIMD_FLOAT32_C(   632.81) },
      { EASYSIMD_FLOAT32_C(107580.54), EASYSIMD_FLOAT32_C(182301.89) } },
    { { EASYSIMD_FLOAT32_C(  -326.10), EASYSIMD_FLOAT32_C(   760.56) },
      { EASYSIMD_FLOAT32_C(  -335.74), EASYSIMD_FLOAT32_C(  -532.85) },
      { EASYSIMD_FLOAT32_C(-109482.81), EASYSIMD_FLOAT32_C(405266.38) } },
    { { EASYSIMD_FLOAT32_C(  -255.56), EASYSIMD_FLOAT32_C(   505.21) },
      { EASYSIMD_FLOAT32_C(   -48.40), EASYSIMD_FLOAT32_C(   933.85) },
      { EASYSIMD_FLOAT32_C(-12367.10), EASYSIMD_FLOAT32_C(-471788.34) } },
    { { EASYSIMD_FLOAT32_C(   785.46), EASYSIMD_FLOAT32_C(   215.93) },
      { EASYSIMD_FLOAT32_C(   289.25), EASYSIMD_FLOAT32_C(   644.14) },
      { EASYSIMD_FLOAT32_C(-227192.31), EASYSIMD_FLOAT32_C(-139087.16) } },
    { { EASYSIMD_FLOAT32_C(   763.47), EASYSIMD_FLOAT32_C(   586.70) },
      { EASYSIMD_FLOAT32_C(    40.67), EASYSIMD_FLOAT32_C(  -188.67) },
      { EASYSIMD_FLOAT32_C(-31048.32), EASYSIMD_FLOAT32_C(110694.69) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vrecps_f32(a, b);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vrecps_f32(a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpsq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   385.11), EASYSIMD_FLOAT64_C(   -71.96) },
      { EASYSIMD_FLOAT64_C(  -168.56), EASYSIMD_FLOAT64_C(   809.26) },
      { EASYSIMD_FLOAT64_C( 64916.14), EASYSIMD_FLOAT64_C( 58236.35) } },
    { { EASYSIMD_FLOAT64_C(  -902.27), EASYSIMD_FLOAT64_C(    27.68) },
      { EASYSIMD_FLOAT64_C(   106.94), EASYSIMD_FLOAT64_C(   161.22) },
      { EASYSIMD_FLOAT64_C( 96490.75), EASYSIMD_FLOAT64_C( -4460.57) } },
    { { EASYSIMD_FLOAT64_C(   596.08), EASYSIMD_FLOAT64_C(   896.02) },
      { EASYSIMD_FLOAT64_C(    -2.55), EASYSIMD_FLOAT64_C(  -787.75) },
      { EASYSIMD_FLOAT64_C(  1522.00), EASYSIMD_FLOAT64_C(705841.76) } },
    { { EASYSIMD_FLOAT64_C(   890.31), EASYSIMD_FLOAT64_C(  -665.19) },
      { EASYSIMD_FLOAT64_C(   419.11), EASYSIMD_FLOAT64_C(   384.44) },
      { EASYSIMD_FLOAT64_C(-373135.82), EASYSIMD_FLOAT64_C(255727.64) } },
    { { EASYSIMD_FLOAT64_C(  -973.24), EASYSIMD_FLOAT64_C(  -191.28) },
      { EASYSIMD_FLOAT64_C(   845.80), EASYSIMD_FLOAT64_C(  -594.55) },
      { EASYSIMD_FLOAT64_C(823168.39), EASYSIMD_FLOAT64_C(-113723.52) } },
    { { EASYSIMD_FLOAT64_C(   -12.90), EASYSIMD_FLOAT64_C(    90.72) },
      { EASYSIMD_FLOAT64_C(   573.26), EASYSIMD_FLOAT64_C(  -769.29) },
      { EASYSIMD_FLOAT64_C(  7397.05), EASYSIMD_FLOAT64_C( 69791.99) } },
    { { EASYSIMD_FLOAT64_C(   663.06), EASYSIMD_FLOAT64_C(   783.64) },
      { EASYSIMD_FLOAT64_C(   791.80), EASYSIMD_FLOAT64_C(  -184.17) },
      { EASYSIMD_FLOAT64_C(-525008.91), EASYSIMD_FLOAT64_C(144324.98) } },
    { { EASYSIMD_FLOAT64_C(  -853.09), EASYSIMD_FLOAT64_C(   -31.87) },
      { EASYSIMD_FLOAT64_C(   169.12), EASYSIMD_FLOAT64_C(   532.02) },
      { EASYSIMD_FLOAT64_C(144276.58), EASYSIMD_FLOAT64_C( 16957.48) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vrecpsq_f64(a, b);
    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vrecpsq_f64(a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpsq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -143.68), EASYSIMD_FLOAT32_C(  -862.89), EASYSIMD_FLOAT32_C(   561.10), EASYSIMD_FLOAT32_C(   300.42) },
      { EASYSIMD_FLOAT32_C(  -472.33), EASYSIMD_FLOAT32_C(  -116.16), EASYSIMD_FLOAT32_C(  -876.28), EASYSIMD_FLOAT32_C(  -452.35) },
      { EASYSIMD_FLOAT32_C(-67862.37), EASYSIMD_FLOAT32_C(-100231.30), EASYSIMD_FLOAT32_C(491682.69), EASYSIMD_FLOAT32_C(135897.00) } },
    { { EASYSIMD_FLOAT32_C(  -932.67), EASYSIMD_FLOAT32_C(  -850.38), EASYSIMD_FLOAT32_C(   879.19), EASYSIMD_FLOAT32_C(   455.35) },
      { EASYSIMD_FLOAT32_C(  -138.46), EASYSIMD_FLOAT32_C(  -398.06), EASYSIMD_FLOAT32_C(    88.15), EASYSIMD_FLOAT32_C(   535.43) },
      { EASYSIMD_FLOAT32_C(-129135.49), EASYSIMD_FLOAT32_C(-338500.25), EASYSIMD_FLOAT32_C(-77498.60), EASYSIMD_FLOAT32_C(-243806.05) } },
    { { EASYSIMD_FLOAT32_C(  -637.51), EASYSIMD_FLOAT32_C(   752.41), EASYSIMD_FLOAT32_C(  -997.42), EASYSIMD_FLOAT32_C(   106.94) },
      { EASYSIMD_FLOAT32_C(   257.62), EASYSIMD_FLOAT32_C(   -45.82), EASYSIMD_FLOAT32_C(    40.79), EASYSIMD_FLOAT32_C(    43.09) },
      { EASYSIMD_FLOAT32_C(164237.33), EASYSIMD_FLOAT32_C( 34477.43), EASYSIMD_FLOAT32_C( 40686.76), EASYSIMD_FLOAT32_C( -4606.04) } },
    { { EASYSIMD_FLOAT32_C(  -829.89), EASYSIMD_FLOAT32_C(  -669.96), EASYSIMD_FLOAT32_C(  -312.77), EASYSIMD_FLOAT32_C(   933.58) },
      { EASYSIMD_FLOAT32_C(   916.74), EASYSIMD_FLOAT32_C(   727.89), EASYSIMD_FLOAT32_C(  -255.09), EASYSIMD_FLOAT32_C(  -226.94) },
      { EASYSIMD_FLOAT32_C(760795.38), EASYSIMD_FLOAT32_C(487659.22), EASYSIMD_FLOAT32_C(-79782.49), EASYSIMD_FLOAT32_C(211868.66) } },
    { { EASYSIMD_FLOAT32_C(   865.00), EASYSIMD_FLOAT32_C(  -693.98), EASYSIMD_FLOAT32_C(  -926.52), EASYSIMD_FLOAT32_C(  -607.33) },
      { EASYSIMD_FLOAT32_C(   189.86), EASYSIMD_FLOAT32_C(  -802.81), EASYSIMD_FLOAT32_C(   -59.68), EASYSIMD_FLOAT32_C(   257.19) },
      { EASYSIMD_FLOAT32_C(-164226.91), EASYSIMD_FLOAT32_C(-557132.06), EASYSIMD_FLOAT32_C(-55292.71), EASYSIMD_FLOAT32_C(156201.20) } },
    { { EASYSIMD_FLOAT32_C(  -653.19), EASYSIMD_FLOAT32_C(  -180.49), EASYSIMD_FLOAT32_C(  -287.46), EASYSIMD_FLOAT32_C(   208.35) },
      { EASYSIMD_FLOAT32_C(   421.45), EASYSIMD_FLOAT32_C(   800.69), EASYSIMD_FLOAT32_C(  -256.22), EASYSIMD_FLOAT32_C(   783.94) },
      { EASYSIMD_FLOAT32_C(275288.94), EASYSIMD_FLOAT32_C(144518.55), EASYSIMD_FLOAT32_C(-73651.00), EASYSIMD_FLOAT32_C(-163331.91) } },
    { { EASYSIMD_FLOAT32_C(   553.11), EASYSIMD_FLOAT32_C(  -253.63), EASYSIMD_FLOAT32_C(  -109.12), EASYSIMD_FLOAT32_C(  -189.27) },
      { EASYSIMD_FLOAT32_C(   700.55), EASYSIMD_FLOAT32_C(   931.66), EASYSIMD_FLOAT32_C(   853.81), EASYSIMD_FLOAT32_C(   870.66) },
      { EASYSIMD_FLOAT32_C(-387479.19), EASYSIMD_FLOAT32_C(236298.92), EASYSIMD_FLOAT32_C( 93169.75), EASYSIMD_FLOAT32_C(164791.81) } },
    { { EASYSIMD_FLOAT32_C(  -738.30), EASYSIMD_FLOAT32_C(  -458.96), EASYSIMD_FLOAT32_C(   804.24), EASYSIMD_FLOAT32_C(  -821.55) },
      { EASYSIMD_FLOAT32_C(  -731.07), EASYSIMD_FLOAT32_C(  -450.84), EASYSIMD_FLOAT32_C(   -48.49), EASYSIMD_FLOAT32_C(  -866.06) },
      { EASYSIMD_FLOAT32_C(-539747.00), EASYSIMD_FLOAT32_C(-206915.52), EASYSIMD_FLOAT32_C( 38999.60), EASYSIMD_FLOAT32_C(-711509.56) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vrecpsq_f32(a, b);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vrecpsq_f32(a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpss_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpsd_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecps_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecps_f32)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpsq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpsq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
