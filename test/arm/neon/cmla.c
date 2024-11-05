#define EASYSIMD_TEST_ARM_NEON_INSN cmla

#include "test-neon.h"
#include "../../../easysimd/arm/neon/cmla.h"

static int
test_easysimd_vcmla_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   212.87), EASYSIMD_FLOAT32_C(    70.46) },
      { EASYSIMD_FLOAT32_C(  -900.86), EASYSIMD_FLOAT32_C(  -975.82) },
      { EASYSIMD_FLOAT32_C(  -978.80), EASYSIMD_FLOAT32_C(   764.65) },
      { EASYSIMD_FLOAT32_C(-192744.86), EASYSIMD_FLOAT32_C(-206958.16) } },
    { { EASYSIMD_FLOAT32_C(  -187.99), EASYSIMD_FLOAT32_C(    49.57) },
      { EASYSIMD_FLOAT32_C(  -750.73), EASYSIMD_FLOAT32_C(   590.72) },
      { EASYSIMD_FLOAT32_C(   251.58), EASYSIMD_FLOAT32_C(   997.06) },
      { EASYSIMD_FLOAT32_C(141381.31), EASYSIMD_FLOAT32_C(-110052.39) } },
    { { EASYSIMD_FLOAT32_C(  -131.09), EASYSIMD_FLOAT32_C(  -878.84) },
      { EASYSIMD_FLOAT32_C(   800.24), EASYSIMD_FLOAT32_C(   223.42) },
      { EASYSIMD_FLOAT32_C(  -487.42), EASYSIMD_FLOAT32_C(  -464.44) },
      { EASYSIMD_FLOAT32_C(-105390.88), EASYSIMD_FLOAT32_C(-29752.57) } },
    { { EASYSIMD_FLOAT32_C(   669.97), EASYSIMD_FLOAT32_C(  -863.16) },
      { EASYSIMD_FLOAT32_C(   580.78), EASYSIMD_FLOAT32_C(  -195.58) },
      { EASYSIMD_FLOAT32_C(  -125.02), EASYSIMD_FLOAT32_C(  -125.29) },
      { EASYSIMD_FLOAT32_C(388980.16), EASYSIMD_FLOAT32_C(-131158.02) } },
    { { EASYSIMD_FLOAT32_C(    11.00), EASYSIMD_FLOAT32_C(  -886.22) },
      { EASYSIMD_FLOAT32_C(  -210.36), EASYSIMD_FLOAT32_C(  -525.26) },
      { EASYSIMD_FLOAT32_C(  -275.11), EASYSIMD_FLOAT32_C(  -377.71) },
      { EASYSIMD_FLOAT32_C( -2589.07), EASYSIMD_FLOAT32_C( -6155.57) } },
    { { EASYSIMD_FLOAT32_C(   383.29), EASYSIMD_FLOAT32_C(   937.75) },
      { EASYSIMD_FLOAT32_C(   692.75), EASYSIMD_FLOAT32_C(   482.42) },
      { EASYSIMD_FLOAT32_C(   961.94), EASYSIMD_FLOAT32_C(   713.95) },
      { EASYSIMD_FLOAT32_C(266486.09), EASYSIMD_FLOAT32_C(185620.72) } },
    { { EASYSIMD_FLOAT32_C(   247.08), EASYSIMD_FLOAT32_C(  -226.05) },
      { EASYSIMD_FLOAT32_C(  -236.48), EASYSIMD_FLOAT32_C(   496.35) },
      { EASYSIMD_FLOAT32_C(  -635.33), EASYSIMD_FLOAT32_C(  -984.90) },
      { EASYSIMD_FLOAT32_C(-59064.81), EASYSIMD_FLOAT32_C(121653.26) } },
    { { EASYSIMD_FLOAT32_C(   493.40), EASYSIMD_FLOAT32_C(   233.58) },
      { EASYSIMD_FLOAT32_C(  -863.74), EASYSIMD_FLOAT32_C(   293.64) },
      { EASYSIMD_FLOAT32_C(  -543.00), EASYSIMD_FLOAT32_C(  -351.16) },
      { EASYSIMD_FLOAT32_C(-426712.31), EASYSIMD_FLOAT32_C(144530.83) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vld1_f32(test_vec[i].r);
    easysimd_float32x2_t r_ = easysimd_vcmla_f32(r, a, b);

    easysimd_test_arm_neon_assert_equal_f32x2(r_, easysimd_vld1_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r_ = easysimd_vcmla_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -44.15), EASYSIMD_FLOAT32_C(   162.35), EASYSIMD_FLOAT32_C(  -263.81), EASYSIMD_FLOAT32_C(  -419.51) },
      { EASYSIMD_FLOAT32_C(   557.47), EASYSIMD_FLOAT32_C(   231.31), EASYSIMD_FLOAT32_C(  -146.84), EASYSIMD_FLOAT32_C(  -373.27) },
      { EASYSIMD_FLOAT32_C(   890.34), EASYSIMD_FLOAT32_C(   811.38), EASYSIMD_FLOAT32_C(  -300.85), EASYSIMD_FLOAT32_C(   -15.80) },
      { EASYSIMD_FLOAT32_C(-23721.96), EASYSIMD_FLOAT32_C( -9400.96), EASYSIMD_FLOAT32_C( 38437.01), EASYSIMD_FLOAT32_C( 98456.55) } },
    { { EASYSIMD_FLOAT32_C(  -992.19), EASYSIMD_FLOAT32_C(   519.83), EASYSIMD_FLOAT32_C(  -699.21), EASYSIMD_FLOAT32_C(   988.58) },
      { EASYSIMD_FLOAT32_C(   744.14), EASYSIMD_FLOAT32_C(  -947.63), EASYSIMD_FLOAT32_C(   992.55), EASYSIMD_FLOAT32_C(   654.06) },
      { EASYSIMD_FLOAT32_C(  -430.76), EASYSIMD_FLOAT32_C(  -498.64), EASYSIMD_FLOAT32_C(  -584.99), EASYSIMD_FLOAT32_C(   338.77) },
      { EASYSIMD_FLOAT32_C(-738759.06), EASYSIMD_FLOAT32_C(939730.38), EASYSIMD_FLOAT32_C(-694585.88), EASYSIMD_FLOAT32_C(-456986.53) } },
    { { EASYSIMD_FLOAT32_C(  -355.89), EASYSIMD_FLOAT32_C(  -589.60), EASYSIMD_FLOAT32_C(   579.00), EASYSIMD_FLOAT32_C(  -954.28) },
      { EASYSIMD_FLOAT32_C(  -272.65), EASYSIMD_FLOAT32_C(   977.05), EASYSIMD_FLOAT32_C(  -572.00), EASYSIMD_FLOAT32_C(   683.20) },
      { EASYSIMD_FLOAT32_C(   139.40), EASYSIMD_FLOAT32_C(   164.19), EASYSIMD_FLOAT32_C(  -736.30), EASYSIMD_FLOAT32_C(  -303.12) },
      { EASYSIMD_FLOAT32_C( 97172.81), EASYSIMD_FLOAT32_C(-347558.16), EASYSIMD_FLOAT32_C(-331924.31), EASYSIMD_FLOAT32_C(395269.69) } },
    { { EASYSIMD_FLOAT32_C(  -604.50), EASYSIMD_FLOAT32_C(   116.85), EASYSIMD_FLOAT32_C(   323.61), EASYSIMD_FLOAT32_C(  -714.16) },
      { EASYSIMD_FLOAT32_C(   -71.76), EASYSIMD_FLOAT32_C(  -977.24), EASYSIMD_FLOAT32_C(   270.05), EASYSIMD_FLOAT32_C(   -63.95) },
      { EASYSIMD_FLOAT32_C(   542.59), EASYSIMD_FLOAT32_C(   570.84), EASYSIMD_FLOAT32_C(   -75.38), EASYSIMD_FLOAT32_C(   286.73) },
      { EASYSIMD_FLOAT32_C( 43921.51), EASYSIMD_FLOAT32_C(591312.44), EASYSIMD_FLOAT32_C( 87315.49), EASYSIMD_FLOAT32_C(-20408.13) } },
    { { EASYSIMD_FLOAT32_C(   623.21), EASYSIMD_FLOAT32_C(   -82.82), EASYSIMD_FLOAT32_C(   -59.21), EASYSIMD_FLOAT32_C(  -807.55) },
      { EASYSIMD_FLOAT32_C(   418.54), EASYSIMD_FLOAT32_C(   355.80), EASYSIMD_FLOAT32_C(   531.22), EASYSIMD_FLOAT32_C(  -937.34) },
      { EASYSIMD_FLOAT32_C(   766.20), EASYSIMD_FLOAT32_C(   110.21), EASYSIMD_FLOAT32_C(  -891.63), EASYSIMD_FLOAT32_C(  -506.46) },
      { EASYSIMD_FLOAT32_C(261604.53), EASYSIMD_FLOAT32_C(221848.33), EASYSIMD_FLOAT32_C(-32345.16), EASYSIMD_FLOAT32_C( 54993.44) } },
    { { EASYSIMD_FLOAT32_C(    87.27), EASYSIMD_FLOAT32_C(  -463.63), EASYSIMD_FLOAT32_C(  -823.25), EASYSIMD_FLOAT32_C(  -773.33) },
      { EASYSIMD_FLOAT32_C(   700.56), EASYSIMD_FLOAT32_C(  -559.56), EASYSIMD_FLOAT32_C(   -76.45), EASYSIMD_FLOAT32_C(  -903.93) },
      { EASYSIMD_FLOAT32_C(   557.30), EASYSIMD_FLOAT32_C(  -752.84), EASYSIMD_FLOAT32_C(  -618.09), EASYSIMD_FLOAT32_C(  -514.46) },
      { EASYSIMD_FLOAT32_C( 61695.17), EASYSIMD_FLOAT32_C(-49585.64), EASYSIMD_FLOAT32_C( 62319.37), EASYSIMD_FLOAT32_C(743645.94) } },
    { { EASYSIMD_FLOAT32_C(  -730.08), EASYSIMD_FLOAT32_C(   651.96), EASYSIMD_FLOAT32_C(   421.58), EASYSIMD_FLOAT32_C(   812.51) },
      { EASYSIMD_FLOAT32_C(   222.80), EASYSIMD_FLOAT32_C(  -653.79), EASYSIMD_FLOAT32_C(    99.24), EASYSIMD_FLOAT32_C(  -153.99) },
      { EASYSIMD_FLOAT32_C(   263.39), EASYSIMD_FLOAT32_C(  -959.97), EASYSIMD_FLOAT32_C(    38.46), EASYSIMD_FLOAT32_C(  -318.07) },
      { EASYSIMD_FLOAT32_C(-162398.44), EASYSIMD_FLOAT32_C(476359.03), EASYSIMD_FLOAT32_C( 41876.06), EASYSIMD_FLOAT32_C(-65237.18) } },
    { { EASYSIMD_FLOAT32_C(   395.83), EASYSIMD_FLOAT32_C(  -430.33), EASYSIMD_FLOAT32_C(  -255.42), EASYSIMD_FLOAT32_C(   162.03) },
      { EASYSIMD_FLOAT32_C(   679.89), EASYSIMD_FLOAT32_C(  -147.05), EASYSIMD_FLOAT32_C(   655.57), EASYSIMD_FLOAT32_C(  -232.84) },
      { EASYSIMD_FLOAT32_C(   389.33), EASYSIMD_FLOAT32_C(   832.32), EASYSIMD_FLOAT32_C(    -6.17), EASYSIMD_FLOAT32_C(    89.89) },
      { EASYSIMD_FLOAT32_C(269510.19), EASYSIMD_FLOAT32_C(-57374.48), EASYSIMD_FLOAT32_C(-167451.86), EASYSIMD_FLOAT32_C( 59561.88) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vld1q_f32(test_vec[i].r);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_f32(r, a, b);
    easysimd_test_arm_neon_assert_equal_f32x4(r_, easysimd_vld1q_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
    easysimd_float64 r_[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -57.99), EASYSIMD_FLOAT64_C(   135.55) },
      { EASYSIMD_FLOAT64_C(  -596.04), EASYSIMD_FLOAT64_C(  -782.75) },
      { EASYSIMD_FLOAT64_C(  -359.68), EASYSIMD_FLOAT64_C(   599.02) },
      { EASYSIMD_FLOAT64_C( 34204.68), EASYSIMD_FLOAT64_C( 45990.69) } },
    { { EASYSIMD_FLOAT64_C(  -264.72), EASYSIMD_FLOAT64_C(  -256.75) },
      { EASYSIMD_FLOAT64_C(   447.46), EASYSIMD_FLOAT64_C(  -783.75) },
      { EASYSIMD_FLOAT64_C(   191.17), EASYSIMD_FLOAT64_C(  -558.46) },
      { EASYSIMD_FLOAT64_C(-118260.44), EASYSIMD_FLOAT64_C(206915.84) } },
    { { EASYSIMD_FLOAT64_C(  -770.71), EASYSIMD_FLOAT64_C(  -784.24) },
      { EASYSIMD_FLOAT64_C(  -699.41), EASYSIMD_FLOAT64_C(   563.40) },
      { EASYSIMD_FLOAT64_C(   833.00), EASYSIMD_FLOAT64_C(   587.87) },
      { EASYSIMD_FLOAT64_C(539875.28), EASYSIMD_FLOAT64_C(-433630.14) } },
    { { EASYSIMD_FLOAT64_C(  -937.25), EASYSIMD_FLOAT64_C(  -903.59) },
      { EASYSIMD_FLOAT64_C(  -150.40), EASYSIMD_FLOAT64_C(  -333.08) },
      { EASYSIMD_FLOAT64_C(  -232.36), EASYSIMD_FLOAT64_C(   783.49) },
      { EASYSIMD_FLOAT64_C(140730.04), EASYSIMD_FLOAT64_C(312962.72) } },
    { { EASYSIMD_FLOAT64_C(   694.12), EASYSIMD_FLOAT64_C(   454.67) },
      { EASYSIMD_FLOAT64_C(   921.25), EASYSIMD_FLOAT64_C(  -888.91) },
      { EASYSIMD_FLOAT64_C(  -246.87), EASYSIMD_FLOAT64_C(  -960.96) },
      { EASYSIMD_FLOAT64_C(639211.18), EASYSIMD_FLOAT64_C(-617971.17) } },
    { { EASYSIMD_FLOAT64_C(    69.74), EASYSIMD_FLOAT64_C(   695.13) },
      { EASYSIMD_FLOAT64_C(   174.59), EASYSIMD_FLOAT64_C(   473.70) },
      { EASYSIMD_FLOAT64_C(   912.38), EASYSIMD_FLOAT64_C(   814.91) },
      { EASYSIMD_FLOAT64_C( 13088.29), EASYSIMD_FLOAT64_C( 33850.75) } },
    { { EASYSIMD_FLOAT64_C(    72.72), EASYSIMD_FLOAT64_C(  -352.34) },
      { EASYSIMD_FLOAT64_C(  -441.84), EASYSIMD_FLOAT64_C(  -479.82) },
      { EASYSIMD_FLOAT64_C(  -136.09), EASYSIMD_FLOAT64_C(   749.33) },
      { EASYSIMD_FLOAT64_C(-32266.69), EASYSIMD_FLOAT64_C(-34143.18) } },
    { { EASYSIMD_FLOAT64_C(   -38.28), EASYSIMD_FLOAT64_C(    93.20) },
      { EASYSIMD_FLOAT64_C(   965.09), EASYSIMD_FLOAT64_C(   262.31) },
      { EASYSIMD_FLOAT64_C(  -343.40), EASYSIMD_FLOAT64_C(   798.09) },
      { EASYSIMD_FLOAT64_C(-37287.05), EASYSIMD_FLOAT64_C( -9243.14) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vld1q_f64(test_vec[i].r);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_f64(r, a, b);
    easysimd_test_arm_neon_assert_equal_f64x2(r_, easysimd_vld1q_f64(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_f64(r, a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmla_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
