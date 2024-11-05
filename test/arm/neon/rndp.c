#define EASYSIMD_TEST_ARM_NEON_INSN rndp

#include "test-neon.h"
#include "../../../easysimd/arm/neon/rndp.h"

static int
test_easysimd_vrndp_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     2.00) } },
    { { EASYSIMD_FLOAT32_C(    -2.50), EASYSIMD_FLOAT32_C(     2.50) },
      { EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     3.00) } },
    { { EASYSIMD_FLOAT32_C(  -980.04), EASYSIMD_FLOAT32_C(   939.96) },
      { EASYSIMD_FLOAT32_C(  -980.00), EASYSIMD_FLOAT32_C(   940.00) } },
    { { EASYSIMD_FLOAT32_C(   208.31), EASYSIMD_FLOAT32_C(  -110.24) },
      { EASYSIMD_FLOAT32_C(   209.00), EASYSIMD_FLOAT32_C(  -110.00) } },
    { { EASYSIMD_FLOAT32_C(  -288.01), EASYSIMD_FLOAT32_C(   612.61) },
      { EASYSIMD_FLOAT32_C(  -288.00), EASYSIMD_FLOAT32_C(   613.00) } },
    { { EASYSIMD_FLOAT32_C(   975.34), EASYSIMD_FLOAT32_C(   999.38) },
      { EASYSIMD_FLOAT32_C(   976.00), EASYSIMD_FLOAT32_C(  1000.00) } },
    { { EASYSIMD_FLOAT32_C(  -633.20), EASYSIMD_FLOAT32_C(  -603.45) },
      { EASYSIMD_FLOAT32_C(  -633.00), EASYSIMD_FLOAT32_C(  -603.00) } },
    { { EASYSIMD_FLOAT32_C(    29.78), EASYSIMD_FLOAT32_C(   554.21) },
      { EASYSIMD_FLOAT32_C(    30.00), EASYSIMD_FLOAT32_C(   555.00) } },
    { { EASYSIMD_FLOAT32_C(  -734.21), EASYSIMD_FLOAT32_C(   840.44) },
      { EASYSIMD_FLOAT32_C(  -734.00), EASYSIMD_FLOAT32_C(   841.00) } },
    { { EASYSIMD_FLOAT32_C(   418.90), EASYSIMD_FLOAT32_C(   259.02) },
      { EASYSIMD_FLOAT32_C(   419.00), EASYSIMD_FLOAT32_C(   260.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t r = easysimd_vrndp_f32(a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vrndp_f32(a);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndp_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT64_C(    -1.00) } },
    { { EASYSIMD_FLOAT64_C(     1.50) },
      { EASYSIMD_FLOAT64_C(     2.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50) },
      { EASYSIMD_FLOAT64_C(    -2.00) } },
    { { EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(     3.00) } },
    { { EASYSIMD_FLOAT64_C(  -235.01) },
      { EASYSIMD_FLOAT64_C(  -235.00) } },
    { { EASYSIMD_FLOAT64_C(  -729.80) },
      { EASYSIMD_FLOAT64_C(  -729.00) } },
    { { EASYSIMD_FLOAT64_C(  -569.69) },
      { EASYSIMD_FLOAT64_C(  -569.00) } },
    { { EASYSIMD_FLOAT64_C(  -128.47) },
      { EASYSIMD_FLOAT64_C(  -128.00) } },
    { { EASYSIMD_FLOAT64_C(  -404.59) },
      { EASYSIMD_FLOAT64_C(  -404.00) } },
    { { EASYSIMD_FLOAT64_C(   535.14) },
      { EASYSIMD_FLOAT64_C(   536.00) } },
    { { EASYSIMD_FLOAT64_C(  -863.09) },
      { EASYSIMD_FLOAT64_C(  -863.00) } },
    { { EASYSIMD_FLOAT64_C(   977.14) },
      { EASYSIMD_FLOAT64_C(   978.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t r = easysimd_vrndp_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vrndp_f64(a);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndpq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     3.00) } },
    { { EASYSIMD_FLOAT32_C(    86.24), EASYSIMD_FLOAT32_C(   581.00), EASYSIMD_FLOAT32_C(   890.92), EASYSIMD_FLOAT32_C(   188.30) },
      { EASYSIMD_FLOAT32_C(    87.00), EASYSIMD_FLOAT32_C(   581.00), EASYSIMD_FLOAT32_C(   891.00), EASYSIMD_FLOAT32_C(   189.00) } },
    { { EASYSIMD_FLOAT32_C(   162.02), EASYSIMD_FLOAT32_C(   633.16), EASYSIMD_FLOAT32_C(  -103.71), EASYSIMD_FLOAT32_C(   181.98) },
      { EASYSIMD_FLOAT32_C(   163.00), EASYSIMD_FLOAT32_C(   634.00), EASYSIMD_FLOAT32_C(  -103.00), EASYSIMD_FLOAT32_C(   182.00) } },
    { { EASYSIMD_FLOAT32_C(   573.12), EASYSIMD_FLOAT32_C(  -895.40), EASYSIMD_FLOAT32_C(  -928.26), EASYSIMD_FLOAT32_C(  -714.90) },
      { EASYSIMD_FLOAT32_C(   574.00), EASYSIMD_FLOAT32_C(  -895.00), EASYSIMD_FLOAT32_C(  -928.00), EASYSIMD_FLOAT32_C(  -714.00) } },
    { { EASYSIMD_FLOAT32_C(   717.20), EASYSIMD_FLOAT32_C(  -952.92), EASYSIMD_FLOAT32_C(  -715.52), EASYSIMD_FLOAT32_C(  -915.99) },
      { EASYSIMD_FLOAT32_C(   718.00), EASYSIMD_FLOAT32_C(  -952.00), EASYSIMD_FLOAT32_C(  -715.00), EASYSIMD_FLOAT32_C(  -915.00) } },
    { { EASYSIMD_FLOAT32_C(  -556.37), EASYSIMD_FLOAT32_C(   314.25), EASYSIMD_FLOAT32_C(   638.22), EASYSIMD_FLOAT32_C(  -290.58) },
      { EASYSIMD_FLOAT32_C(  -556.00), EASYSIMD_FLOAT32_C(   315.00), EASYSIMD_FLOAT32_C(   639.00), EASYSIMD_FLOAT32_C(  -290.00) } },
    { { EASYSIMD_FLOAT32_C(   154.70), EASYSIMD_FLOAT32_C(    57.12), EASYSIMD_FLOAT32_C(   968.43), EASYSIMD_FLOAT32_C(   919.68) },
      { EASYSIMD_FLOAT32_C(   155.00), EASYSIMD_FLOAT32_C(    58.00), EASYSIMD_FLOAT32_C(   969.00), EASYSIMD_FLOAT32_C(   920.00) } },
    { { EASYSIMD_FLOAT32_C(   327.32), EASYSIMD_FLOAT32_C(  -601.25), EASYSIMD_FLOAT32_C(  -208.79), EASYSIMD_FLOAT32_C(   922.73) },
      { EASYSIMD_FLOAT32_C(   328.00), EASYSIMD_FLOAT32_C(  -601.00), EASYSIMD_FLOAT32_C(  -208.00), EASYSIMD_FLOAT32_C(   923.00) } },
    { { EASYSIMD_FLOAT32_C(   933.89), EASYSIMD_FLOAT32_C(   -71.87), EASYSIMD_FLOAT32_C(   899.87), EASYSIMD_FLOAT32_C(    20.13) },
      { EASYSIMD_FLOAT32_C(   934.00), EASYSIMD_FLOAT32_C(   -71.00), EASYSIMD_FLOAT32_C(   900.00), EASYSIMD_FLOAT32_C(    21.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t r = easysimd_vrndpq_f32(a);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vrndpq_f32(a);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndpq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     2.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50), EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     3.00) } },
    { { EASYSIMD_FLOAT64_C(  -490.87), EASYSIMD_FLOAT64_C(   790.79) },
      { EASYSIMD_FLOAT64_C(  -490.00), EASYSIMD_FLOAT64_C(   791.00) } },
    { { EASYSIMD_FLOAT64_C(  -791.57), EASYSIMD_FLOAT64_C(   671.15) },
      { EASYSIMD_FLOAT64_C(  -791.00), EASYSIMD_FLOAT64_C(   672.00) } },
    { { EASYSIMD_FLOAT64_C(   423.95), EASYSIMD_FLOAT64_C(   104.72) },
      { EASYSIMD_FLOAT64_C(   424.00), EASYSIMD_FLOAT64_C(   105.00) } },
    { { EASYSIMD_FLOAT64_C(  -146.87), EASYSIMD_FLOAT64_C(    -2.94) },
      { EASYSIMD_FLOAT64_C(  -146.00), EASYSIMD_FLOAT64_C(    -2.00) } },
    { { EASYSIMD_FLOAT64_C(   209.32), EASYSIMD_FLOAT64_C(   -75.14) },
      { EASYSIMD_FLOAT64_C(   210.00), EASYSIMD_FLOAT64_C(   -75.00) } },
    { { EASYSIMD_FLOAT64_C(   282.16), EASYSIMD_FLOAT64_C(   -73.47) },
      { EASYSIMD_FLOAT64_C(   283.00), EASYSIMD_FLOAT64_C(   -73.00) } },
    { { EASYSIMD_FLOAT64_C(   -28.06), EASYSIMD_FLOAT64_C(   566.64) },
      { EASYSIMD_FLOAT64_C(   -28.00), EASYSIMD_FLOAT64_C(   567.00) } },
    { { EASYSIMD_FLOAT64_C(    10.53), EASYSIMD_FLOAT64_C(   415.57) },
      { EASYSIMD_FLOAT64_C(    11.00), EASYSIMD_FLOAT64_C(   416.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t r = easysimd_vrndpq_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vrndpq_f64(a);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndp_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndp_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndpq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndpq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
