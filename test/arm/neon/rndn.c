#define EASYSIMD_TEST_ARM_NEON_INSN rndn

#include "test-neon.h"
#include "../../../easysimd/arm/neon/rndn.h"

static int
test_easysimd_vrndn_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT32_C(  -593.90), EASYSIMD_FLOAT32_C(   196.84) },
      { EASYSIMD_FLOAT32_C(  -594.00), EASYSIMD_FLOAT32_C(   197.00) } },
    { { EASYSIMD_FLOAT32_C(   569.79), EASYSIMD_FLOAT32_C(   336.27) },
      { EASYSIMD_FLOAT32_C(   570.00), EASYSIMD_FLOAT32_C(   336.00) } },
    { { EASYSIMD_FLOAT32_C(  -670.11), EASYSIMD_FLOAT32_C(   299.96) },
      { EASYSIMD_FLOAT32_C(  -670.00), EASYSIMD_FLOAT32_C(   300.00) } },
    { { EASYSIMD_FLOAT32_C(    -4.27), EASYSIMD_FLOAT32_C(  -333.31) },
      { EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(  -333.00) } },
    { { EASYSIMD_FLOAT32_C(  -389.20), EASYSIMD_FLOAT32_C(   338.21) },
      { EASYSIMD_FLOAT32_C(  -389.00), EASYSIMD_FLOAT32_C(   338.00) } },
    { { EASYSIMD_FLOAT32_C(   172.22), EASYSIMD_FLOAT32_C(   764.71) },
      { EASYSIMD_FLOAT32_C(   172.00), EASYSIMD_FLOAT32_C(   765.00) } },
    { { EASYSIMD_FLOAT32_C(   789.38), EASYSIMD_FLOAT32_C(  -740.62) },
      { EASYSIMD_FLOAT32_C(   789.00), EASYSIMD_FLOAT32_C(  -741.00) } },
    { { EASYSIMD_FLOAT32_C(   713.87), EASYSIMD_FLOAT32_C(   -75.96) },
      { EASYSIMD_FLOAT32_C(   714.00), EASYSIMD_FLOAT32_C(   -76.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t r = easysimd_vrndn_f32(a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vrndn_f32(a);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndn_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT64_C(   956.89) },
      { EASYSIMD_FLOAT64_C(   957.00) } },
    { { EASYSIMD_FLOAT64_C(   240.71) },
      { EASYSIMD_FLOAT64_C(   241.00) } },
    { { EASYSIMD_FLOAT64_C(  -255.78) },
      { EASYSIMD_FLOAT64_C(  -256.00) } },
    { { EASYSIMD_FLOAT64_C(   583.46) },
      { EASYSIMD_FLOAT64_C(   583.00) } },
    { { EASYSIMD_FLOAT64_C(   184.46) },
      { EASYSIMD_FLOAT64_C(   184.00) } },
    { { EASYSIMD_FLOAT64_C(  -123.90) },
      { EASYSIMD_FLOAT64_C(  -124.00) } },
    { { EASYSIMD_FLOAT64_C(   757.51) },
      { EASYSIMD_FLOAT64_C(   758.00) } },
    { { EASYSIMD_FLOAT64_C(   200.47) },
      { EASYSIMD_FLOAT64_C(   200.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t r = easysimd_vrndn_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vrndn_f64(a);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndnq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT32_C(   826.17), EASYSIMD_FLOAT32_C(  -229.41), EASYSIMD_FLOAT32_C(  -487.35), EASYSIMD_FLOAT32_C(    89.00) },
      { EASYSIMD_FLOAT32_C(   826.00), EASYSIMD_FLOAT32_C(  -229.00), EASYSIMD_FLOAT32_C(  -487.00), EASYSIMD_FLOAT32_C(    89.00) } },
    { { EASYSIMD_FLOAT32_C(  -306.79), EASYSIMD_FLOAT32_C(  -855.59), EASYSIMD_FLOAT32_C(   532.14), EASYSIMD_FLOAT32_C(    99.31) },
      { EASYSIMD_FLOAT32_C(  -307.00), EASYSIMD_FLOAT32_C(  -856.00), EASYSIMD_FLOAT32_C(   532.00), EASYSIMD_FLOAT32_C(    99.00) } },
    { { EASYSIMD_FLOAT32_C(   341.26), EASYSIMD_FLOAT32_C(   101.93), EASYSIMD_FLOAT32_C(  -564.42), EASYSIMD_FLOAT32_C(   671.15) },
      { EASYSIMD_FLOAT32_C(   341.00), EASYSIMD_FLOAT32_C(   102.00), EASYSIMD_FLOAT32_C(  -564.00), EASYSIMD_FLOAT32_C(   671.00) } },
    { { EASYSIMD_FLOAT32_C(  -598.11), EASYSIMD_FLOAT32_C(   431.31), EASYSIMD_FLOAT32_C(  -662.17), EASYSIMD_FLOAT32_C(    12.69) },
      { EASYSIMD_FLOAT32_C(  -598.00), EASYSIMD_FLOAT32_C(   431.00), EASYSIMD_FLOAT32_C(  -662.00), EASYSIMD_FLOAT32_C(    13.00) } },
    { { EASYSIMD_FLOAT32_C(  -230.48), EASYSIMD_FLOAT32_C(   510.05), EASYSIMD_FLOAT32_C(  -222.60), EASYSIMD_FLOAT32_C(  -441.10) },
      { EASYSIMD_FLOAT32_C(  -230.00), EASYSIMD_FLOAT32_C(   510.00), EASYSIMD_FLOAT32_C(  -223.00), EASYSIMD_FLOAT32_C(  -441.00) } },
    { { EASYSIMD_FLOAT32_C(   769.43), EASYSIMD_FLOAT32_C(  -508.73), EASYSIMD_FLOAT32_C(   482.94), EASYSIMD_FLOAT32_C(   726.32) },
      { EASYSIMD_FLOAT32_C(   769.00), EASYSIMD_FLOAT32_C(  -509.00), EASYSIMD_FLOAT32_C(   483.00), EASYSIMD_FLOAT32_C(   726.00) } },
    { { EASYSIMD_FLOAT32_C(   731.99), EASYSIMD_FLOAT32_C(  -772.85), EASYSIMD_FLOAT32_C(   309.78), EASYSIMD_FLOAT32_C(   -83.55) },
      { EASYSIMD_FLOAT32_C(   732.00), EASYSIMD_FLOAT32_C(  -773.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(   -84.00) } },
    { { EASYSIMD_FLOAT32_C(   103.25), EASYSIMD_FLOAT32_C(    67.29), EASYSIMD_FLOAT32_C(  -883.08), EASYSIMD_FLOAT32_C(   -70.58) },
      { EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(    67.00), EASYSIMD_FLOAT32_C(  -883.00), EASYSIMD_FLOAT32_C(   -71.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t r = easysimd_vrndnq_f32(a);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vrndnq_f32(a);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndnq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT64_C(   837.88), EASYSIMD_FLOAT64_C(  -370.43) },
      { EASYSIMD_FLOAT64_C(   838.00), EASYSIMD_FLOAT64_C(  -370.00) } },
    { { EASYSIMD_FLOAT64_C(  -981.58), EASYSIMD_FLOAT64_C(  -468.91) },
      { EASYSIMD_FLOAT64_C(  -982.00), EASYSIMD_FLOAT64_C(  -469.00) } },
    { { EASYSIMD_FLOAT64_C(  -226.02), EASYSIMD_FLOAT64_C(   550.56) },
      { EASYSIMD_FLOAT64_C(  -226.00), EASYSIMD_FLOAT64_C(   551.00) } },
    { { EASYSIMD_FLOAT64_C(   630.40), EASYSIMD_FLOAT64_C(  -884.76) },
      { EASYSIMD_FLOAT64_C(   630.00), EASYSIMD_FLOAT64_C(  -885.00) } },
    { { EASYSIMD_FLOAT64_C(  -347.50), EASYSIMD_FLOAT64_C(  -934.02) },
      { EASYSIMD_FLOAT64_C(  -348.00), EASYSIMD_FLOAT64_C(  -934.00) } },
    { { EASYSIMD_FLOAT64_C(   786.38), EASYSIMD_FLOAT64_C(    54.39) },
      { EASYSIMD_FLOAT64_C(   786.00), EASYSIMD_FLOAT64_C(    54.00) } },
    { { EASYSIMD_FLOAT64_C(   497.29), EASYSIMD_FLOAT64_C(  -875.79) },
      { EASYSIMD_FLOAT64_C(   497.00), EASYSIMD_FLOAT64_C(  -876.00) } },
    { { EASYSIMD_FLOAT64_C(  -932.92), EASYSIMD_FLOAT64_C(  -733.19) },
      { EASYSIMD_FLOAT64_C(  -933.00), EASYSIMD_FLOAT64_C(  -733.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t r = easysimd_vrndnq_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vrndnq_f64(a);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndn_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndn_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndnq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndnq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
