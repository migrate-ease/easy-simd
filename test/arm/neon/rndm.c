#define EASYSIMD_TEST_ARM_NEON_INSN rndm

#include "test-neon.h"
#include "../../../easysimd/arm/neon/rndm.h"

static int
test_easysimd_vrndm_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 r[2];
  } test_vec[] = {
    #if defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF,           -EASYSIMD_MATH_NANF },
      {            EASYSIMD_MATH_NANF,           -EASYSIMD_MATH_NANF } },
    #endif
    { { EASYSIMD_FLOAT32_C(    -1.50), EASYSIMD_FLOAT32_C(     1.50) },
      { EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     1.00) } },
    { { EASYSIMD_FLOAT32_C(    -2.50), EASYSIMD_FLOAT32_C(     2.50) },
      { EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     2.00) } },

    { { EASYSIMD_FLOAT32_C(  -897.30), EASYSIMD_FLOAT32_C(   351.51) },
      { EASYSIMD_FLOAT32_C(  -898.00), EASYSIMD_FLOAT32_C(   351.00) } },
    { { EASYSIMD_FLOAT32_C(  -396.24), EASYSIMD_FLOAT32_C(  -136.90) },
      { EASYSIMD_FLOAT32_C(  -397.00), EASYSIMD_FLOAT32_C(  -137.00) } },
    { { EASYSIMD_FLOAT32_C(  -966.64), EASYSIMD_FLOAT32_C(   805.58) },
      { EASYSIMD_FLOAT32_C(  -967.00), EASYSIMD_FLOAT32_C(   805.00) } },
    { { EASYSIMD_FLOAT32_C(   848.81), EASYSIMD_FLOAT32_C(  -910.27) },
      { EASYSIMD_FLOAT32_C(   848.00), EASYSIMD_FLOAT32_C(  -911.00) } },
    { { EASYSIMD_FLOAT32_C(  -262.75), EASYSIMD_FLOAT32_C(   779.23) },
      { EASYSIMD_FLOAT32_C(  -263.00), EASYSIMD_FLOAT32_C(   779.00) } },
    { { EASYSIMD_FLOAT32_C(   824.19), EASYSIMD_FLOAT32_C(  -986.07) },
      { EASYSIMD_FLOAT32_C(   824.00), EASYSIMD_FLOAT32_C(  -987.00) } },
    { { EASYSIMD_FLOAT32_C(   272.13), EASYSIMD_FLOAT32_C(   812.56) },
      { EASYSIMD_FLOAT32_C(   272.00), EASYSIMD_FLOAT32_C(   812.00) } },
    { { EASYSIMD_FLOAT32_C(  -763.50), EASYSIMD_FLOAT32_C(   477.59) },
      { EASYSIMD_FLOAT32_C(  -764.00), EASYSIMD_FLOAT32_C(   477.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t r = easysimd_vrndm_f32(a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vrndm_f32(a);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndm_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 r[1];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {             EASYSIMD_MATH_NAN },
      {             EASYSIMD_MATH_NAN } },
    { {            -EASYSIMD_MATH_NAN },
      {            -EASYSIMD_MATH_NAN } },
    #endif
    { { EASYSIMD_FLOAT64_C(    -1.50) },
      { EASYSIMD_FLOAT64_C(    -2.00) } },
    { { EASYSIMD_FLOAT64_C(     1.50) },
      { EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50) },
      { EASYSIMD_FLOAT64_C(    -3.00) } },
    { { EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(     2.00) } },

    { { EASYSIMD_FLOAT64_C(   333.88) },
      { EASYSIMD_FLOAT64_C(   333.00) } },
    { { EASYSIMD_FLOAT64_C(   629.40) },
      { EASYSIMD_FLOAT64_C(   629.00) } },
    { { EASYSIMD_FLOAT64_C(  -124.31) },
      { EASYSIMD_FLOAT64_C(  -125.00) } },
    { { EASYSIMD_FLOAT64_C(   133.65) },
      { EASYSIMD_FLOAT64_C(   133.00) } },
    { { EASYSIMD_FLOAT64_C(  -307.19) },
      { EASYSIMD_FLOAT64_C(  -308.00) } },
    { { EASYSIMD_FLOAT64_C(   596.65) },
      { EASYSIMD_FLOAT64_C(   596.00) } },
    { { EASYSIMD_FLOAT64_C(   827.64) },
      { EASYSIMD_FLOAT64_C(   827.00) } },
    { { EASYSIMD_FLOAT64_C(   250.89) },
      { EASYSIMD_FLOAT64_C(   250.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t r = easysimd_vrndm_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vrndm_f64(a);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndmq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF,           -EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,           -EASYSIMD_MATH_NANF },
      {            EASYSIMD_MATH_NANF,           -EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,           -EASYSIMD_MATH_NANF } },
    #endif
    { { EASYSIMD_FLOAT32_C(    -1.50), EASYSIMD_FLOAT32_C(     1.50), EASYSIMD_FLOAT32_C(    -2.50), EASYSIMD_FLOAT32_C(     2.50) },
      { EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     2.00) } },

    { { EASYSIMD_FLOAT32_C(   744.58), EASYSIMD_FLOAT32_C(  -175.23), EASYSIMD_FLOAT32_C(  -591.29), EASYSIMD_FLOAT32_C(   759.75) },
      { EASYSIMD_FLOAT32_C(   744.00), EASYSIMD_FLOAT32_C(  -176.00), EASYSIMD_FLOAT32_C(  -592.00), EASYSIMD_FLOAT32_C(   759.00) } },
    { { EASYSIMD_FLOAT32_C(   273.17), EASYSIMD_FLOAT32_C(   118.54), EASYSIMD_FLOAT32_C(  -744.67), EASYSIMD_FLOAT32_C(   375.86) },
      { EASYSIMD_FLOAT32_C(   273.00), EASYSIMD_FLOAT32_C(   118.00), EASYSIMD_FLOAT32_C(  -745.00), EASYSIMD_FLOAT32_C(   375.00) } },
    { { EASYSIMD_FLOAT32_C(  -529.96), EASYSIMD_FLOAT32_C(  -140.92), EASYSIMD_FLOAT32_C(  -761.03), EASYSIMD_FLOAT32_C(  -496.59) },
      { EASYSIMD_FLOAT32_C(  -530.00), EASYSIMD_FLOAT32_C(  -141.00), EASYSIMD_FLOAT32_C(  -762.00), EASYSIMD_FLOAT32_C(  -497.00) } },
    { { EASYSIMD_FLOAT32_C(  -335.34), EASYSIMD_FLOAT32_C(  -912.22), EASYSIMD_FLOAT32_C(  -406.86), EASYSIMD_FLOAT32_C(   401.91) },
      { EASYSIMD_FLOAT32_C(  -336.00), EASYSIMD_FLOAT32_C(  -913.00), EASYSIMD_FLOAT32_C(  -407.00), EASYSIMD_FLOAT32_C(   401.00) } },
    { { EASYSIMD_FLOAT32_C(   867.01), EASYSIMD_FLOAT32_C(  -582.67), EASYSIMD_FLOAT32_C(   415.83), EASYSIMD_FLOAT32_C(   139.14) },
      { EASYSIMD_FLOAT32_C(   867.00), EASYSIMD_FLOAT32_C(  -583.00), EASYSIMD_FLOAT32_C(   415.00), EASYSIMD_FLOAT32_C(   139.00) } },
    { { EASYSIMD_FLOAT32_C(  -770.11), EASYSIMD_FLOAT32_C(   652.33), EASYSIMD_FLOAT32_C(  -383.28), EASYSIMD_FLOAT32_C(   563.77) },
      { EASYSIMD_FLOAT32_C(  -771.00), EASYSIMD_FLOAT32_C(   652.00), EASYSIMD_FLOAT32_C(  -384.00), EASYSIMD_FLOAT32_C(   563.00) } },
    { { EASYSIMD_FLOAT32_C(   281.73), EASYSIMD_FLOAT32_C(   492.41), EASYSIMD_FLOAT32_C(  -302.57), EASYSIMD_FLOAT32_C(   974.54) },
      { EASYSIMD_FLOAT32_C(   281.00), EASYSIMD_FLOAT32_C(   492.00), EASYSIMD_FLOAT32_C(  -303.00), EASYSIMD_FLOAT32_C(   974.00) } },
    { { EASYSIMD_FLOAT32_C(    89.06), EASYSIMD_FLOAT32_C(  -474.93), EASYSIMD_FLOAT32_C(   225.42), EASYSIMD_FLOAT32_C(  -166.36) },
      { EASYSIMD_FLOAT32_C(    89.00), EASYSIMD_FLOAT32_C(  -475.00), EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -167.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t r = easysimd_vrndmq_f32(a);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vrndmq_f32(a);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndmq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {             EASYSIMD_MATH_NAN,            -EASYSIMD_MATH_NAN },
      {             EASYSIMD_MATH_NAN,            -EASYSIMD_MATH_NAN } },
    #endif
    { { EASYSIMD_FLOAT64_C(    -1.50), EASYSIMD_FLOAT64_C(     1.50) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50), EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(     2.00) } },

    { { EASYSIMD_FLOAT64_C(   349.83), EASYSIMD_FLOAT64_C(   634.13) },
      { EASYSIMD_FLOAT64_C(   349.00), EASYSIMD_FLOAT64_C(   634.00) } },
    { { EASYSIMD_FLOAT64_C(  -406.61), EASYSIMD_FLOAT64_C(  -377.00) },
      { EASYSIMD_FLOAT64_C(  -407.00), EASYSIMD_FLOAT64_C(  -377.00) } },
    { { EASYSIMD_FLOAT64_C(  -247.33), EASYSIMD_FLOAT64_C(  -151.28) },
      { EASYSIMD_FLOAT64_C(  -248.00), EASYSIMD_FLOAT64_C(  -152.00) } },
    { { EASYSIMD_FLOAT64_C(   998.86), EASYSIMD_FLOAT64_C(   222.71) },
      { EASYSIMD_FLOAT64_C(   998.00), EASYSIMD_FLOAT64_C(   222.00) } },
    { { EASYSIMD_FLOAT64_C(   707.80), EASYSIMD_FLOAT64_C(  -762.17) },
      { EASYSIMD_FLOAT64_C(   707.00), EASYSIMD_FLOAT64_C(  -763.00) } },
    { { EASYSIMD_FLOAT64_C(   726.12), EASYSIMD_FLOAT64_C(  -627.54) },
      { EASYSIMD_FLOAT64_C(   726.00), EASYSIMD_FLOAT64_C(  -628.00) } },
    { { EASYSIMD_FLOAT64_C(  -674.40), EASYSIMD_FLOAT64_C(  -680.74) },
      { EASYSIMD_FLOAT64_C(  -675.00), EASYSIMD_FLOAT64_C(  -681.00) } },
    { { EASYSIMD_FLOAT64_C(   774.37), EASYSIMD_FLOAT64_C(  -807.39) },
      { EASYSIMD_FLOAT64_C(   774.00), EASYSIMD_FLOAT64_C(  -808.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t r = easysimd_vrndmq_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vrndmq_f64(a);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndm_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndm_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndmq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndmq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
