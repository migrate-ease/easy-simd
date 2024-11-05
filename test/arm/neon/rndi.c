#define EASYSIMD_TEST_ARM_NEON_INSN rndi

#include "test-neon.h"
#include "../../../easysimd/arm/neon/rndi.h"

static int
test_easysimd_vrndi_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00) } },
    { { EASYSIMD_FLOAT32_C(    -2.50), EASYSIMD_FLOAT32_C(     2.50) },
      { EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00) } },

    { { EASYSIMD_FLOAT32_C(  -787.65), EASYSIMD_FLOAT32_C(  -795.95) },
      { EASYSIMD_FLOAT32_C(  -788.00), EASYSIMD_FLOAT32_C(  -796.00) } },
    { { EASYSIMD_FLOAT32_C(   899.31), EASYSIMD_FLOAT32_C(   -93.42) },
      { EASYSIMD_FLOAT32_C(   899.00), EASYSIMD_FLOAT32_C(   -93.00) } },
    { { EASYSIMD_FLOAT32_C(  -436.50), EASYSIMD_FLOAT32_C(  -165.94) },
      { EASYSIMD_FLOAT32_C(  -436.00), EASYSIMD_FLOAT32_C(  -166.00) } },
    { { EASYSIMD_FLOAT32_C(  -516.13), EASYSIMD_FLOAT32_C(  -288.52) },
      { EASYSIMD_FLOAT32_C(  -516.00), EASYSIMD_FLOAT32_C(  -289.00) } },
    { { EASYSIMD_FLOAT32_C(  -568.31), EASYSIMD_FLOAT32_C(  -937.97) },
      { EASYSIMD_FLOAT32_C(  -568.00), EASYSIMD_FLOAT32_C(  -938.00) } },
    { { EASYSIMD_FLOAT32_C(   827.64), EASYSIMD_FLOAT32_C(   984.63) },
      { EASYSIMD_FLOAT32_C(   828.00), EASYSIMD_FLOAT32_C(   985.00) } },
    { { EASYSIMD_FLOAT32_C(   261.25), EASYSIMD_FLOAT32_C(   -11.30) },
      { EASYSIMD_FLOAT32_C(   261.00), EASYSIMD_FLOAT32_C(   -11.00) } },
    { { EASYSIMD_FLOAT32_C(    97.38), EASYSIMD_FLOAT32_C(  -824.40) },
      { EASYSIMD_FLOAT32_C(    97.00), EASYSIMD_FLOAT32_C(  -824.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t r = easysimd_vrndi_f32(a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vrndi_f32(a);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndi_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT64_C(     2.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50) },
      { EASYSIMD_FLOAT64_C(    -2.00) } },
    { { EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(     2.00) } },

    { { EASYSIMD_FLOAT64_C(  -405.63) },
      { EASYSIMD_FLOAT64_C(  -406.00) } },
    { { EASYSIMD_FLOAT64_C(   554.36) },
      { EASYSIMD_FLOAT64_C(   554.00) } },
    { { EASYSIMD_FLOAT64_C(  -286.09) },
      { EASYSIMD_FLOAT64_C(  -286.00) } },
    { { EASYSIMD_FLOAT64_C(  -583.84) },
      { EASYSIMD_FLOAT64_C(  -584.00) } },
    { { EASYSIMD_FLOAT64_C(  -389.59) },
      { EASYSIMD_FLOAT64_C(  -390.00) } },
    { { EASYSIMD_FLOAT64_C(  -683.98) },
      { EASYSIMD_FLOAT64_C(  -684.00) } },
    { { EASYSIMD_FLOAT64_C(   628.93) },
      { EASYSIMD_FLOAT64_C(   629.00) } },
    { { EASYSIMD_FLOAT64_C(  -112.86) },
      { EASYSIMD_FLOAT64_C(  -113.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t r = easysimd_vrndi_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vrndi_f64(a);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndiq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00) } },

    { { EASYSIMD_FLOAT32_C(  -938.67), EASYSIMD_FLOAT32_C(  -583.30), EASYSIMD_FLOAT32_C(  -219.07), EASYSIMD_FLOAT32_C(   510.59) },
      { EASYSIMD_FLOAT32_C(  -939.00), EASYSIMD_FLOAT32_C(  -583.00), EASYSIMD_FLOAT32_C(  -219.00), EASYSIMD_FLOAT32_C(   511.00) } },
    { { EASYSIMD_FLOAT32_C(  -715.91), EASYSIMD_FLOAT32_C(  -372.75), EASYSIMD_FLOAT32_C(  -712.38), EASYSIMD_FLOAT32_C(  -503.56) },
      { EASYSIMD_FLOAT32_C(  -716.00), EASYSIMD_FLOAT32_C(  -373.00), EASYSIMD_FLOAT32_C(  -712.00), EASYSIMD_FLOAT32_C(  -504.00) } },
    { { EASYSIMD_FLOAT32_C(  -168.71), EASYSIMD_FLOAT32_C(  -813.07), EASYSIMD_FLOAT32_C(   403.02), EASYSIMD_FLOAT32_C(   394.80) },
      { EASYSIMD_FLOAT32_C(  -169.00), EASYSIMD_FLOAT32_C(  -813.00), EASYSIMD_FLOAT32_C(   403.00), EASYSIMD_FLOAT32_C(   395.00) } },
    { { EASYSIMD_FLOAT32_C(    21.00), EASYSIMD_FLOAT32_C(   886.89), EASYSIMD_FLOAT32_C(  -893.72), EASYSIMD_FLOAT32_C(   452.69) },
      { EASYSIMD_FLOAT32_C(    21.00), EASYSIMD_FLOAT32_C(   887.00), EASYSIMD_FLOAT32_C(  -894.00), EASYSIMD_FLOAT32_C(   453.00) } },
    { { EASYSIMD_FLOAT32_C(   948.91), EASYSIMD_FLOAT32_C(   933.92), EASYSIMD_FLOAT32_C(   437.32), EASYSIMD_FLOAT32_C(   210.16) },
      { EASYSIMD_FLOAT32_C(   949.00), EASYSIMD_FLOAT32_C(   934.00), EASYSIMD_FLOAT32_C(   437.00), EASYSIMD_FLOAT32_C(   210.00) } },
    { { EASYSIMD_FLOAT32_C(   -77.38), EASYSIMD_FLOAT32_C(  -465.30), EASYSIMD_FLOAT32_C(   385.77), EASYSIMD_FLOAT32_C(   516.99) },
      { EASYSIMD_FLOAT32_C(   -77.00), EASYSIMD_FLOAT32_C(  -465.00), EASYSIMD_FLOAT32_C(   386.00), EASYSIMD_FLOAT32_C(   517.00) } },
    { { EASYSIMD_FLOAT32_C(  -910.94), EASYSIMD_FLOAT32_C(  -900.33), EASYSIMD_FLOAT32_C(   933.15), EASYSIMD_FLOAT32_C(  -300.52) },
      { EASYSIMD_FLOAT32_C(  -911.00), EASYSIMD_FLOAT32_C(  -900.00), EASYSIMD_FLOAT32_C(   933.00), EASYSIMD_FLOAT32_C(  -301.00) } },
    { { EASYSIMD_FLOAT32_C(  -584.31), EASYSIMD_FLOAT32_C(   562.08), EASYSIMD_FLOAT32_C(   586.62), EASYSIMD_FLOAT32_C(  -522.98) },
      { EASYSIMD_FLOAT32_C(  -584.00), EASYSIMD_FLOAT32_C(   562.00), EASYSIMD_FLOAT32_C(   587.00), EASYSIMD_FLOAT32_C(  -523.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t r = easysimd_vrndiq_f32(a);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vrndiq_f32(a);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndiq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     2.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50), EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     2.00) } },

    { { EASYSIMD_FLOAT64_C(   978.78), EASYSIMD_FLOAT64_C(  -632.45) },
      { EASYSIMD_FLOAT64_C(   979.00), EASYSIMD_FLOAT64_C(  -632.00) } },
    { { EASYSIMD_FLOAT64_C(   987.61), EASYSIMD_FLOAT64_C(  -737.13) },
      { EASYSIMD_FLOAT64_C(   988.00), EASYSIMD_FLOAT64_C(  -737.00) } },
    { { EASYSIMD_FLOAT64_C(    -5.20), EASYSIMD_FLOAT64_C(  -724.77) },
      { EASYSIMD_FLOAT64_C(    -5.00), EASYSIMD_FLOAT64_C(  -725.00) } },
    { { EASYSIMD_FLOAT64_C(  -240.69), EASYSIMD_FLOAT64_C(   826.09) },
      { EASYSIMD_FLOAT64_C(  -241.00), EASYSIMD_FLOAT64_C(   826.00) } },
    { { EASYSIMD_FLOAT64_C(  -537.84), EASYSIMD_FLOAT64_C(  -837.67) },
      { EASYSIMD_FLOAT64_C(  -538.00), EASYSIMD_FLOAT64_C(  -838.00) } },
    { { EASYSIMD_FLOAT64_C(   220.89), EASYSIMD_FLOAT64_C(   483.16) },
      { EASYSIMD_FLOAT64_C(   221.00), EASYSIMD_FLOAT64_C(   483.00) } },
    { { EASYSIMD_FLOAT64_C(  -950.78), EASYSIMD_FLOAT64_C(   327.17) },
      { EASYSIMD_FLOAT64_C(  -951.00), EASYSIMD_FLOAT64_C(   327.00) } },
    { { EASYSIMD_FLOAT64_C(   -64.15), EASYSIMD_FLOAT64_C(   998.14) },
      { EASYSIMD_FLOAT64_C(   -64.00), EASYSIMD_FLOAT64_C(   998.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t r = easysimd_vrndiq_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vrndiq_f64(a);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndi_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndi_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndiq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndiq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
