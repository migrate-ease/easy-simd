#define EASYSIMD_TEST_ARM_NEON_INSN rsqrts

#include "test-neon.h"
#include "../../../easysimd/arm/neon/rsqrts.h"

static int
test_easysimd_vrsqrts_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 b[2];
    easysimd_float32 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    -1.81), EASYSIMD_FLOAT32_C(     6.08) },
      { EASYSIMD_FLOAT32_C(    -0.75), EASYSIMD_FLOAT32_C(     3.06) },
      { EASYSIMD_FLOAT32_C(     0.82), EASYSIMD_FLOAT32_C(    -7.80) } },
    { { EASYSIMD_FLOAT32_C(     4.44), EASYSIMD_FLOAT32_C(     6.63) },
      { EASYSIMD_FLOAT32_C(     5.97), EASYSIMD_FLOAT32_C(    -5.82) },
      { EASYSIMD_FLOAT32_C(   -11.75), EASYSIMD_FLOAT32_C(    20.79) } },
    { { EASYSIMD_FLOAT32_C(    -4.07), EASYSIMD_FLOAT32_C(     2.30) },
      { EASYSIMD_FLOAT32_C(    -2.84), EASYSIMD_FLOAT32_C(    -2.53) },
      { EASYSIMD_FLOAT32_C(    -4.28), EASYSIMD_FLOAT32_C(     4.41) } },
    { { EASYSIMD_FLOAT32_C(    -7.18), EASYSIMD_FLOAT32_C(    -5.76) },
      { EASYSIMD_FLOAT32_C(     0.29), EASYSIMD_FLOAT32_C(    -6.65) },
      { EASYSIMD_FLOAT32_C(     2.54), EASYSIMD_FLOAT32_C(   -17.65) } },
    { { EASYSIMD_FLOAT32_C(    -6.30), EASYSIMD_FLOAT32_C(    -9.44) },
      { EASYSIMD_FLOAT32_C(    -1.04), EASYSIMD_FLOAT32_C(     1.58) },
      { EASYSIMD_FLOAT32_C(    -1.78), EASYSIMD_FLOAT32_C(     8.96) } },
    { { EASYSIMD_FLOAT32_C(     9.05), EASYSIMD_FLOAT32_C(     4.99) },
      { EASYSIMD_FLOAT32_C(     3.75), EASYSIMD_FLOAT32_C(    -1.14) },
      { EASYSIMD_FLOAT32_C(   -15.47), EASYSIMD_FLOAT32_C(     4.34) } },
    { { EASYSIMD_FLOAT32_C(    -6.57), EASYSIMD_FLOAT32_C(    -6.46) },
      { EASYSIMD_FLOAT32_C(    -0.70), EASYSIMD_FLOAT32_C(     8.06) },
      { EASYSIMD_FLOAT32_C(    -0.80), EASYSIMD_FLOAT32_C(    27.53) } },
    { { EASYSIMD_FLOAT32_C(     4.91), EASYSIMD_FLOAT32_C(     1.78) },
      { EASYSIMD_FLOAT32_C(     6.18), EASYSIMD_FLOAT32_C(    -6.90) },
      { EASYSIMD_FLOAT32_C(   -13.67), EASYSIMD_FLOAT32_C(     7.64) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vrsqrts_f32(a, b);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-10.0f, 10.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-10.0f, 10.0f);
    easysimd_float32x2_t r = easysimd_vrsqrts_f32(a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrts_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 b[1];
    easysimd_float64 r[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(     6.87) },
  { EASYSIMD_FLOAT64_C(     8.55) },
  { EASYSIMD_FLOAT64_C(   -27.87) } },
{ { EASYSIMD_FLOAT64_C(    -1.98) },
  { EASYSIMD_FLOAT64_C(     2.09) },
  { EASYSIMD_FLOAT64_C(     3.57) } },
{ { EASYSIMD_FLOAT64_C(     7.13) },
  { EASYSIMD_FLOAT64_C(    -9.63) },
  { EASYSIMD_FLOAT64_C(    35.83) } },
{ { EASYSIMD_FLOAT64_C(    -3.43) },
  { EASYSIMD_FLOAT64_C(    -5.37) },
  { EASYSIMD_FLOAT64_C(    -7.71) } },
{ { EASYSIMD_FLOAT64_C(    -1.34) },
  { EASYSIMD_FLOAT64_C(    -4.98) },
  { EASYSIMD_FLOAT64_C(    -1.84) } },
{ { EASYSIMD_FLOAT64_C(     9.08) },
  { EASYSIMD_FLOAT64_C(    -0.19) },
  { EASYSIMD_FLOAT64_C(     2.36) } },
{ { EASYSIMD_FLOAT64_C(    -7.89) },
  { EASYSIMD_FLOAT64_C(     9.75) },
  { EASYSIMD_FLOAT64_C(    39.96) } },
{ { EASYSIMD_FLOAT64_C(     2.41) },
  { EASYSIMD_FLOAT64_C(     8.30) },
  { EASYSIMD_FLOAT64_C(    -8.50) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_float64x1_t r = easysimd_vrsqrts_f64(a, b);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-10.0, 10.0);
    easysimd_float64x1_t b = easysimd_test_arm_neon_random_f64x1(-10.0, 10.0);
    easysimd_float64x1_t r = easysimd_vrsqrts_f64(a, b);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrtsq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    -2.14), EASYSIMD_FLOAT32_C(    -4.57), EASYSIMD_FLOAT32_C(     6.16), EASYSIMD_FLOAT32_C(    -7.69) },
      { EASYSIMD_FLOAT32_C(    -7.94), EASYSIMD_FLOAT32_C(     2.13), EASYSIMD_FLOAT32_C(    -3.52), EASYSIMD_FLOAT32_C(    -2.01) },
      { EASYSIMD_FLOAT32_C(    -7.00), EASYSIMD_FLOAT32_C(     6.37), EASYSIMD_FLOAT32_C(    12.34), EASYSIMD_FLOAT32_C(    -6.23) } },
    { { EASYSIMD_FLOAT32_C(    -5.58), EASYSIMD_FLOAT32_C(     3.64), EASYSIMD_FLOAT32_C(     5.46), EASYSIMD_FLOAT32_C(    -2.76) },
      { EASYSIMD_FLOAT32_C(     7.89), EASYSIMD_FLOAT32_C(    -4.25), EASYSIMD_FLOAT32_C(     0.59), EASYSIMD_FLOAT32_C(    -8.41) },
      { EASYSIMD_FLOAT32_C(    23.51), EASYSIMD_FLOAT32_C(     9.24), EASYSIMD_FLOAT32_C(    -0.11), EASYSIMD_FLOAT32_C(   -10.11) } },
    { { EASYSIMD_FLOAT32_C(    -3.70), EASYSIMD_FLOAT32_C(     9.56), EASYSIMD_FLOAT32_C(     3.17), EASYSIMD_FLOAT32_C(    -4.64) },
      { EASYSIMD_FLOAT32_C(     4.55), EASYSIMD_FLOAT32_C(    -3.08), EASYSIMD_FLOAT32_C(     4.22), EASYSIMD_FLOAT32_C(     7.98) },
      { EASYSIMD_FLOAT32_C(     9.92), EASYSIMD_FLOAT32_C(    16.22), EASYSIMD_FLOAT32_C(    -5.19), EASYSIMD_FLOAT32_C(    20.01) } },
    { { EASYSIMD_FLOAT32_C(     0.47), EASYSIMD_FLOAT32_C(    -6.48), EASYSIMD_FLOAT32_C(     6.04), EASYSIMD_FLOAT32_C(    -4.62) },
      { EASYSIMD_FLOAT32_C(     5.30), EASYSIMD_FLOAT32_C(     2.22), EASYSIMD_FLOAT32_C(    -1.53), EASYSIMD_FLOAT32_C(    -6.84) },
      { EASYSIMD_FLOAT32_C(     0.25), EASYSIMD_FLOAT32_C(     8.69), EASYSIMD_FLOAT32_C(     6.12), EASYSIMD_FLOAT32_C(   -14.30) } },
    { { EASYSIMD_FLOAT32_C(     7.64), EASYSIMD_FLOAT32_C(    -5.37), EASYSIMD_FLOAT32_C(    -4.53), EASYSIMD_FLOAT32_C(     9.70) },
      { EASYSIMD_FLOAT32_C(     6.76), EASYSIMD_FLOAT32_C(     1.95), EASYSIMD_FLOAT32_C(    -2.31), EASYSIMD_FLOAT32_C(    -8.82) },
      { EASYSIMD_FLOAT32_C(   -24.32), EASYSIMD_FLOAT32_C(     6.74), EASYSIMD_FLOAT32_C(    -3.73), EASYSIMD_FLOAT32_C(    44.28) } },
    { { EASYSIMD_FLOAT32_C(    -4.41), EASYSIMD_FLOAT32_C(    -6.85), EASYSIMD_FLOAT32_C(    -1.58), EASYSIMD_FLOAT32_C(    -6.52) },
      { EASYSIMD_FLOAT32_C(    -1.10), EASYSIMD_FLOAT32_C(     9.02), EASYSIMD_FLOAT32_C(    -4.93), EASYSIMD_FLOAT32_C(     5.20) },
      { EASYSIMD_FLOAT32_C(    -0.93), EASYSIMD_FLOAT32_C(    32.39), EASYSIMD_FLOAT32_C(    -2.39), EASYSIMD_FLOAT32_C(    18.45) } },
    { { EASYSIMD_FLOAT32_C(     8.57), EASYSIMD_FLOAT32_C(     8.24), EASYSIMD_FLOAT32_C(    -9.44), EASYSIMD_FLOAT32_C(     3.12) },
      { EASYSIMD_FLOAT32_C(    -4.83), EASYSIMD_FLOAT32_C(     4.78), EASYSIMD_FLOAT32_C(     1.10), EASYSIMD_FLOAT32_C(     5.63) },
      { EASYSIMD_FLOAT32_C(    22.20), EASYSIMD_FLOAT32_C(   -18.19), EASYSIMD_FLOAT32_C(     6.69), EASYSIMD_FLOAT32_C(    -7.28) } },
    { { EASYSIMD_FLOAT32_C(     8.30), EASYSIMD_FLOAT32_C(    -2.87), EASYSIMD_FLOAT32_C(    -8.99), EASYSIMD_FLOAT32_C(     3.60) },
      { EASYSIMD_FLOAT32_C(     9.35), EASYSIMD_FLOAT32_C(    -0.52), EASYSIMD_FLOAT32_C(     6.77), EASYSIMD_FLOAT32_C(     6.99) },
      { EASYSIMD_FLOAT32_C(   -37.30), EASYSIMD_FLOAT32_C(     0.75), EASYSIMD_FLOAT32_C(    31.93), EASYSIMD_FLOAT32_C(   -11.08) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vrsqrtsq_f32(a, b);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-10.0f, 10.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-10.0f, 10.0f);
    easysimd_float32x4_t r = easysimd_vrsqrtsq_f32(a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrtsq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(     5.69), EASYSIMD_FLOAT64_C(     9.51) },
      { EASYSIMD_FLOAT64_C(     7.48), EASYSIMD_FLOAT64_C(    -9.06) },
      { EASYSIMD_FLOAT64_C(   -19.78), EASYSIMD_FLOAT64_C(    44.58) } },
    { { EASYSIMD_FLOAT64_C(     3.21), EASYSIMD_FLOAT64_C(    -0.11) },
      { EASYSIMD_FLOAT64_C(     4.18), EASYSIMD_FLOAT64_C(    -0.48) },
      { EASYSIMD_FLOAT64_C(    -5.21), EASYSIMD_FLOAT64_C(     1.47) } },
    { { EASYSIMD_FLOAT64_C(    -8.62), EASYSIMD_FLOAT64_C(     1.53) },
      { EASYSIMD_FLOAT64_C(     1.11), EASYSIMD_FLOAT64_C(    -6.72) },
      { EASYSIMD_FLOAT64_C(     6.28), EASYSIMD_FLOAT64_C(     6.64) } },
    { { EASYSIMD_FLOAT64_C(    -9.36), EASYSIMD_FLOAT64_C(    -8.80) },
      { EASYSIMD_FLOAT64_C(     7.91), EASYSIMD_FLOAT64_C(    -3.14) },
      { EASYSIMD_FLOAT64_C(    38.52), EASYSIMD_FLOAT64_C(   -12.32) } },
    { { EASYSIMD_FLOAT64_C(    -6.74), EASYSIMD_FLOAT64_C(     1.44) },
      { EASYSIMD_FLOAT64_C(    -9.99), EASYSIMD_FLOAT64_C(    -4.84) },
      { EASYSIMD_FLOAT64_C(   -32.17), EASYSIMD_FLOAT64_C(     4.98) } },
    { { EASYSIMD_FLOAT64_C(    -1.82), EASYSIMD_FLOAT64_C(     4.17) },
      { EASYSIMD_FLOAT64_C(    -1.16), EASYSIMD_FLOAT64_C(    -1.96) },
      { EASYSIMD_FLOAT64_C(     0.44), EASYSIMD_FLOAT64_C(     5.59) } },
    { { EASYSIMD_FLOAT64_C(     5.33), EASYSIMD_FLOAT64_C(     5.84) },
      { EASYSIMD_FLOAT64_C(    -3.06), EASYSIMD_FLOAT64_C(     7.73) },
      { EASYSIMD_FLOAT64_C(     9.65), EASYSIMD_FLOAT64_C(   -21.07) } },
    { { EASYSIMD_FLOAT64_C(    -7.36), EASYSIMD_FLOAT64_C(     9.00) },
      { EASYSIMD_FLOAT64_C(     4.68), EASYSIMD_FLOAT64_C(     8.34) },
      { EASYSIMD_FLOAT64_C(    18.72), EASYSIMD_FLOAT64_C(   -36.03) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vrsqrtsq_f64(a, b);
    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-10.0, 10.0);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-10.0, 10.0);
    easysimd_float64x2_t r = easysimd_vrsqrtsq_f64(a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrtss_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32_t a;
    easysimd_float32_t b;
    easysimd_float32_t r;
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(   -68.85),
      EASYSIMD_FLOAT32_C(  -965.73),
      EASYSIMD_FLOAT32_C(-33243.75) },
    { EASYSIMD_FLOAT32_C(  -888.41),
      EASYSIMD_FLOAT32_C(  -408.81),
      EASYSIMD_FLOAT32_C(-181593.94) },
    { EASYSIMD_FLOAT32_C(  -560.30),
      EASYSIMD_FLOAT32_C(  -722.30),
      EASYSIMD_FLOAT32_C(-202350.84) },
    { EASYSIMD_FLOAT32_C(   234.04),
      EASYSIMD_FLOAT32_C(  -798.13),
      EASYSIMD_FLOAT32_C( 93398.67) },
    { EASYSIMD_FLOAT32_C(  -880.28),
      EASYSIMD_FLOAT32_C(  -643.05),
      EASYSIMD_FLOAT32_C(-283030.53) },
    { EASYSIMD_FLOAT32_C(  -257.26),
      EASYSIMD_FLOAT32_C(   445.45),
      EASYSIMD_FLOAT32_C( 57299.74) },
    { EASYSIMD_FLOAT32_C(   162.67),
      EASYSIMD_FLOAT32_C(   786.15),
      EASYSIMD_FLOAT32_C(-63940.01) },
    { EASYSIMD_FLOAT32_C(  -927.81),
      EASYSIMD_FLOAT32_C(   416.86),
      EASYSIMD_FLOAT32_C(193384.94) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32_t r = easysimd_vrsqrtss_f32(test_vec[i].a, test_vec[i].b);
    easysimd_assert_equal_f32(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-1000, 1000);
    easysimd_float32_t b = easysimd_test_codegen_random_f32(-1000, 1000);
    easysimd_float32_t r = easysimd_vrsqrtss_f32(a, b);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrtsd_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a;
    easysimd_float64_t b;
    easysimd_float64_t r;
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(   733.74),
      EASYSIMD_FLOAT64_C(  -247.63),
      EASYSIMD_FLOAT64_C( 90849.52) },
    { EASYSIMD_FLOAT64_C(  -888.92),
      EASYSIMD_FLOAT64_C(  -818.58),
      EASYSIMD_FLOAT64_C(-363824.57) },
    { EASYSIMD_FLOAT64_C(  -846.64),
      EASYSIMD_FLOAT64_C(   961.55),
      EASYSIMD_FLOAT64_C(407044.85) },
    { EASYSIMD_FLOAT64_C(  -865.36),
      EASYSIMD_FLOAT64_C(   170.96),
      EASYSIMD_FLOAT64_C( 73972.47) },
    { EASYSIMD_FLOAT64_C(  -307.65),
      EASYSIMD_FLOAT64_C(  -831.35),
      EASYSIMD_FLOAT64_C(-127880.91) },
    { EASYSIMD_FLOAT64_C(  -150.80),
      EASYSIMD_FLOAT64_C(   619.79),
      EASYSIMD_FLOAT64_C( 46733.67) },
    { EASYSIMD_FLOAT64_C(   764.67),
      EASYSIMD_FLOAT64_C(  -382.92),
      EASYSIMD_FLOAT64_C(146405.22) },
    { EASYSIMD_FLOAT64_C(  -804.26),
      EASYSIMD_FLOAT64_C(  -267.13),
      EASYSIMD_FLOAT64_C(-107419.49) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64_t r = easysimd_vrsqrtsd_f64(test_vec[i].a, test_vec[i].b);
    easysimd_assert_equal_f64(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64_t a = easysimd_test_codegen_random_f64(-1000, 1000);
    easysimd_float64_t b = easysimd_test_codegen_random_f64(-1000, 1000);
    easysimd_float64_t r = easysimd_vrsqrtsd_f64(a, b);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrts_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrts_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrtsq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrtsq_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrtss_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrtsd_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
