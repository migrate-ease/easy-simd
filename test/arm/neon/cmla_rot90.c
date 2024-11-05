#define EASYSIMD_TEST_ARM_NEON_INSN cmla_rot90

#include "test-neon.h"
#include "../../../easysimd/arm/neon/cmla_rot90.h"

static int
test_easysimd_vcmla_rot90_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -736.61), EASYSIMD_FLOAT32_C(  -886.60) },
      { EASYSIMD_FLOAT32_C(   281.46), EASYSIMD_FLOAT32_C(   182.30) },
      { EASYSIMD_FLOAT32_C(   429.99), EASYSIMD_FLOAT32_C(  -897.20) },
      { EASYSIMD_FLOAT32_C(162057.17), EASYSIMD_FLOAT32_C(-250439.62) } },
    { { EASYSIMD_FLOAT32_C(   776.49), EASYSIMD_FLOAT32_C(   342.21) },
      { EASYSIMD_FLOAT32_C(  -299.64), EASYSIMD_FLOAT32_C(   786.30) },
      { EASYSIMD_FLOAT32_C(  -603.93), EASYSIMD_FLOAT32_C(   -76.17) },
      { EASYSIMD_FLOAT32_C(-269683.66), EASYSIMD_FLOAT32_C(-102615.98) } },
    { { EASYSIMD_FLOAT32_C(   513.95), EASYSIMD_FLOAT32_C(  -299.36) },
      { EASYSIMD_FLOAT32_C(  -719.19), EASYSIMD_FLOAT32_C(   762.89) },
      { EASYSIMD_FLOAT32_C(   552.54), EASYSIMD_FLOAT32_C(    14.06) },
      { EASYSIMD_FLOAT32_C(228931.28), EASYSIMD_FLOAT32_C(215310.77) } },
    { { EASYSIMD_FLOAT32_C(   529.94), EASYSIMD_FLOAT32_C(  -725.03) },
      { EASYSIMD_FLOAT32_C(   627.20), EASYSIMD_FLOAT32_C(   167.43) },
      { EASYSIMD_FLOAT32_C(   101.22), EASYSIMD_FLOAT32_C(   -38.51) },
      { EASYSIMD_FLOAT32_C(121492.99), EASYSIMD_FLOAT32_C(-454777.34) } },
    { { EASYSIMD_FLOAT32_C(  -773.64), EASYSIMD_FLOAT32_C(   578.09) },
      { EASYSIMD_FLOAT32_C(  -557.39), EASYSIMD_FLOAT32_C(  -616.25) },
      { EASYSIMD_FLOAT32_C(   304.61), EASYSIMD_FLOAT32_C(   452.27) },
      { EASYSIMD_FLOAT32_C(356552.59), EASYSIMD_FLOAT32_C(-321769.34) } },
    { { EASYSIMD_FLOAT32_C(   530.02), EASYSIMD_FLOAT32_C(   568.00) },
      { EASYSIMD_FLOAT32_C(   565.67), EASYSIMD_FLOAT32_C(  -188.52) },
      { EASYSIMD_FLOAT32_C(  -249.70), EASYSIMD_FLOAT32_C(    -4.33) },
      { EASYSIMD_FLOAT32_C(106829.66), EASYSIMD_FLOAT32_C(321296.22) } },
    { { EASYSIMD_FLOAT32_C(   -85.72), EASYSIMD_FLOAT32_C(  -473.21) },
      { EASYSIMD_FLOAT32_C(  -662.12), EASYSIMD_FLOAT32_C(   614.64) },
      { EASYSIMD_FLOAT32_C(  -686.90), EASYSIMD_FLOAT32_C(  -266.06) },
      { EASYSIMD_FLOAT32_C(290166.91), EASYSIMD_FLOAT32_C(313055.75) } },
    { { EASYSIMD_FLOAT32_C(  -461.53), EASYSIMD_FLOAT32_C(   827.04) },
      { EASYSIMD_FLOAT32_C(   434.59), EASYSIMD_FLOAT32_C(  -180.71) },
      { EASYSIMD_FLOAT32_C(   589.93), EASYSIMD_FLOAT32_C(   -12.87) },
      { EASYSIMD_FLOAT32_C(150044.33), EASYSIMD_FLOAT32_C(359410.44) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vld1_f32(test_vec[i].r);
    easysimd_float32x2_t r_ = easysimd_vcmla_rot90_f32(r, a, b);

    easysimd_test_arm_neon_assert_equal_f32x2(r_, easysimd_vld1_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r_ = easysimd_vcmla_rot90_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_rot90_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   809.81), EASYSIMD_FLOAT32_C(   702.36), EASYSIMD_FLOAT32_C(  -440.66), EASYSIMD_FLOAT32_C(    26.41) },
      { EASYSIMD_FLOAT32_C(   196.20), EASYSIMD_FLOAT32_C(   580.52), EASYSIMD_FLOAT32_C(   918.89), EASYSIMD_FLOAT32_C(  -116.23) },
      { EASYSIMD_FLOAT32_C(   484.07), EASYSIMD_FLOAT32_C(   711.80), EASYSIMD_FLOAT32_C(   891.23), EASYSIMD_FLOAT32_C(    40.72) },
      { EASYSIMD_FLOAT32_C(-407249.97), EASYSIMD_FLOAT32_C(138514.83), EASYSIMD_FLOAT32_C(  3960.86), EASYSIMD_FLOAT32_C( 24308.61) } },
    { { EASYSIMD_FLOAT32_C(   509.40), EASYSIMD_FLOAT32_C(  -504.44), EASYSIMD_FLOAT32_C(    19.20), EASYSIMD_FLOAT32_C(   170.91) },
      { EASYSIMD_FLOAT32_C(   886.34), EASYSIMD_FLOAT32_C(   555.20), EASYSIMD_FLOAT32_C(  -753.29), EASYSIMD_FLOAT32_C(   -56.21) },
      { EASYSIMD_FLOAT32_C(  -843.97), EASYSIMD_FLOAT32_C(  -640.31), EASYSIMD_FLOAT32_C(   469.39), EASYSIMD_FLOAT32_C(  -535.58) },
      { EASYSIMD_FLOAT32_C(279221.12), EASYSIMD_FLOAT32_C(-447745.69), EASYSIMD_FLOAT32_C( 10076.24), EASYSIMD_FLOAT32_C(-129280.38) } },
    { { EASYSIMD_FLOAT32_C(  -886.22), EASYSIMD_FLOAT32_C(   250.51), EASYSIMD_FLOAT32_C(   591.50), EASYSIMD_FLOAT32_C(   599.25) },
      { EASYSIMD_FLOAT32_C(   260.71), EASYSIMD_FLOAT32_C(  -574.11), EASYSIMD_FLOAT32_C(  -734.49), EASYSIMD_FLOAT32_C(    70.52) },
      { EASYSIMD_FLOAT32_C(  -871.75), EASYSIMD_FLOAT32_C(  -175.15), EASYSIMD_FLOAT32_C(  -903.07), EASYSIMD_FLOAT32_C(   324.45) },
      { EASYSIMD_FLOAT32_C(142948.55), EASYSIMD_FLOAT32_C( 65135.31), EASYSIMD_FLOAT32_C(-43162.18), EASYSIMD_FLOAT32_C(-439818.69) } },
    { { EASYSIMD_FLOAT32_C(  -594.62), EASYSIMD_FLOAT32_C(  -984.18), EASYSIMD_FLOAT32_C(  -791.78), EASYSIMD_FLOAT32_C(   889.45) },
      { EASYSIMD_FLOAT32_C(   727.61), EASYSIMD_FLOAT32_C(  -900.55), EASYSIMD_FLOAT32_C(   -69.84), EASYSIMD_FLOAT32_C(   237.02) },
      { EASYSIMD_FLOAT32_C(  -404.99), EASYSIMD_FLOAT32_C(   949.36), EASYSIMD_FLOAT32_C(  -592.07), EASYSIMD_FLOAT32_C(  -518.65) },
      { EASYSIMD_FLOAT32_C(-886708.25), EASYSIMD_FLOAT32_C(-715149.81), EASYSIMD_FLOAT32_C(-211409.52), EASYSIMD_FLOAT32_C(-62637.84) } },
    { { EASYSIMD_FLOAT32_C(   504.57), EASYSIMD_FLOAT32_C(  -345.36), EASYSIMD_FLOAT32_C(   425.14), EASYSIMD_FLOAT32_C(   660.60) },
      { EASYSIMD_FLOAT32_C(    14.33), EASYSIMD_FLOAT32_C(  -105.47), EASYSIMD_FLOAT32_C(  -874.99), EASYSIMD_FLOAT32_C(   128.10) },
      { EASYSIMD_FLOAT32_C(  -854.96), EASYSIMD_FLOAT32_C(   716.51), EASYSIMD_FLOAT32_C(  -272.64), EASYSIMD_FLOAT32_C(   405.75) },
      { EASYSIMD_FLOAT32_C(-37280.08), EASYSIMD_FLOAT32_C( -4232.50), EASYSIMD_FLOAT32_C(-84895.50), EASYSIMD_FLOAT32_C(-577612.62) } },
    { { EASYSIMD_FLOAT32_C(  -857.59), EASYSIMD_FLOAT32_C(    -7.13), EASYSIMD_FLOAT32_C(  -523.73), EASYSIMD_FLOAT32_C(  -729.34) },
      { EASYSIMD_FLOAT32_C(   817.72), EASYSIMD_FLOAT32_C(  -426.81), EASYSIMD_FLOAT32_C(   595.11), EASYSIMD_FLOAT32_C(  -776.90) },
      { EASYSIMD_FLOAT32_C(  -410.99), EASYSIMD_FLOAT32_C(   803.33), EASYSIMD_FLOAT32_C(  -887.46), EASYSIMD_FLOAT32_C(  -683.37) },
      { EASYSIMD_FLOAT32_C( -3454.15), EASYSIMD_FLOAT32_C( -5027.01), EASYSIMD_FLOAT32_C(-567511.75), EASYSIMD_FLOAT32_C(-434720.91) } },
    { { EASYSIMD_FLOAT32_C(   902.78), EASYSIMD_FLOAT32_C(    42.71), EASYSIMD_FLOAT32_C(   553.64), EASYSIMD_FLOAT32_C(  -502.21) },
      { EASYSIMD_FLOAT32_C(    -7.93), EASYSIMD_FLOAT32_C(   961.57), EASYSIMD_FLOAT32_C(   -20.86), EASYSIMD_FLOAT32_C(  -503.36) },
      { EASYSIMD_FLOAT32_C(  -383.79), EASYSIMD_FLOAT32_C(  -595.71), EASYSIMD_FLOAT32_C(  -842.76), EASYSIMD_FLOAT32_C(   630.53) },
      { EASYSIMD_FLOAT32_C(-41452.45), EASYSIMD_FLOAT32_C(  -934.40), EASYSIMD_FLOAT32_C(-253635.17), EASYSIMD_FLOAT32_C( 11106.63) } },
    { { EASYSIMD_FLOAT32_C(   298.82), EASYSIMD_FLOAT32_C(  -717.75), EASYSIMD_FLOAT32_C(  -241.36), EASYSIMD_FLOAT32_C(   443.86) },
      { EASYSIMD_FLOAT32_C(   998.76), EASYSIMD_FLOAT32_C(   486.00), EASYSIMD_FLOAT32_C(  -150.39), EASYSIMD_FLOAT32_C(  -858.84) },
      { EASYSIMD_FLOAT32_C(  -521.14), EASYSIMD_FLOAT32_C(   325.87), EASYSIMD_FLOAT32_C(  -588.18), EASYSIMD_FLOAT32_C(  -703.42) },
      { EASYSIMD_FLOAT32_C(348305.38), EASYSIMD_FLOAT32_C(-716534.12), EASYSIMD_FLOAT32_C(380616.53), EASYSIMD_FLOAT32_C(-67455.52) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vld1q_f32(test_vec[i].r);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_rot90_f32(r, a, b);
    easysimd_test_arm_neon_assert_equal_f32x4(r_, easysimd_vld1q_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_rot90_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_rot90_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
    easysimd_float64 r_[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -913.35), EASYSIMD_FLOAT64_C(   651.70) },
      { EASYSIMD_FLOAT64_C(  -202.18), EASYSIMD_FLOAT64_C(   611.89) },
      { EASYSIMD_FLOAT64_C(  -924.95), EASYSIMD_FLOAT64_C(   395.75) },
      { EASYSIMD_FLOAT64_C(-399693.66), EASYSIMD_FLOAT64_C(-131364.96) } },
    { { EASYSIMD_FLOAT64_C(  -917.75), EASYSIMD_FLOAT64_C(  -438.33) },
      { EASYSIMD_FLOAT64_C(    -0.32), EASYSIMD_FLOAT64_C(   810.69) },
      { EASYSIMD_FLOAT64_C(    11.74), EASYSIMD_FLOAT64_C(  -666.19) },
      { EASYSIMD_FLOAT64_C(355361.49), EASYSIMD_FLOAT64_C(  -525.92) } },
    { { EASYSIMD_FLOAT64_C(  -434.30), EASYSIMD_FLOAT64_C(  -744.37) },
      { EASYSIMD_FLOAT64_C(  -564.35), EASYSIMD_FLOAT64_C(   -95.18) },
      { EASYSIMD_FLOAT64_C(   956.50), EASYSIMD_FLOAT64_C(   334.31) },
      { EASYSIMD_FLOAT64_C(-69892.64), EASYSIMD_FLOAT64_C(420419.52) } },
    { { EASYSIMD_FLOAT64_C(   595.20), EASYSIMD_FLOAT64_C(   271.18) },
      { EASYSIMD_FLOAT64_C(   967.27), EASYSIMD_FLOAT64_C(   979.29) },
      { EASYSIMD_FLOAT64_C(   428.77), EASYSIMD_FLOAT64_C(   858.33) },
      { EASYSIMD_FLOAT64_C(-265135.09), EASYSIMD_FLOAT64_C(263162.61) } },
    { { EASYSIMD_FLOAT64_C(  -931.38), EASYSIMD_FLOAT64_C(   720.93) },
      { EASYSIMD_FLOAT64_C(   860.06), EASYSIMD_FLOAT64_C(  -977.02) },
      { EASYSIMD_FLOAT64_C(  -592.65), EASYSIMD_FLOAT64_C(   976.85) },
      { EASYSIMD_FLOAT64_C(703770.38), EASYSIMD_FLOAT64_C(621019.91) } },
    { { EASYSIMD_FLOAT64_C(   776.69), EASYSIMD_FLOAT64_C(  -506.00) },
      { EASYSIMD_FLOAT64_C(   628.55), EASYSIMD_FLOAT64_C(  -425.49) },
      { EASYSIMD_FLOAT64_C(  -894.11), EASYSIMD_FLOAT64_C(   703.60) },
      { EASYSIMD_FLOAT64_C(-216192.05), EASYSIMD_FLOAT64_C(-317342.70) } },
    { { EASYSIMD_FLOAT64_C(   970.25), EASYSIMD_FLOAT64_C(  -811.86) },
      { EASYSIMD_FLOAT64_C(  -734.73), EASYSIMD_FLOAT64_C(   -30.06) },
      { EASYSIMD_FLOAT64_C(   998.84), EASYSIMD_FLOAT64_C(   277.01) },
      { EASYSIMD_FLOAT64_C(-23405.67), EASYSIMD_FLOAT64_C(596774.91) } },
    { { EASYSIMD_FLOAT64_C(   303.75), EASYSIMD_FLOAT64_C(  -435.46) },
      { EASYSIMD_FLOAT64_C(   532.64), EASYSIMD_FLOAT64_C(   739.40) },
      { EASYSIMD_FLOAT64_C(   469.36), EASYSIMD_FLOAT64_C(   489.13) },
      { EASYSIMD_FLOAT64_C(322448.48), EASYSIMD_FLOAT64_C(-231454.28) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vld1q_f64(test_vec[i].r);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_rot90_f64(r, a, b);
    easysimd_test_arm_neon_assert_equal_f64x2(r_, easysimd_vld1q_f64(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_rot90_f64(r, a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmla_rot90_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_rot90_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_rot90_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
