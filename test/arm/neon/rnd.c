#define EASYSIMD_TEST_ARM_NEON_INSN rnd

#include "test-neon.h"
#include "../../../easysimd/arm/neon/rnd.h"

static int
test_easysimd_vrnd_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00) } },
    { { EASYSIMD_FLOAT32_C(    -2.50), EASYSIMD_FLOAT32_C(     2.50) },
      { EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00) } },
    { { EASYSIMD_FLOAT32_C(   782.33), EASYSIMD_FLOAT32_C(    23.83) },
      { EASYSIMD_FLOAT32_C(   782.00), EASYSIMD_FLOAT32_C(    23.00) } },
    { { EASYSIMD_FLOAT32_C(  -231.98), EASYSIMD_FLOAT32_C(  -121.26) },
      { EASYSIMD_FLOAT32_C(  -231.00), EASYSIMD_FLOAT32_C(  -121.00) } },
    { { EASYSIMD_FLOAT32_C(   524.61), EASYSIMD_FLOAT32_C(   500.02) },
      { EASYSIMD_FLOAT32_C(   524.00), EASYSIMD_FLOAT32_C(   500.00) } },
    { { EASYSIMD_FLOAT32_C(    80.15), EASYSIMD_FLOAT32_C(   517.44) },
      { EASYSIMD_FLOAT32_C(    80.00), EASYSIMD_FLOAT32_C(   517.00) } },
    { { EASYSIMD_FLOAT32_C(  -754.87), EASYSIMD_FLOAT32_C(   128.37) },
      { EASYSIMD_FLOAT32_C(  -754.00), EASYSIMD_FLOAT32_C(   128.00) } },
    { { EASYSIMD_FLOAT32_C(   182.53), EASYSIMD_FLOAT32_C(   136.96) },
      { EASYSIMD_FLOAT32_C(   182.00), EASYSIMD_FLOAT32_C(   136.00) } },
    { { EASYSIMD_FLOAT32_C(   605.41), EASYSIMD_FLOAT32_C(  -833.56) },
      { EASYSIMD_FLOAT32_C(   605.00), EASYSIMD_FLOAT32_C(  -833.00) } },
    { { EASYSIMD_FLOAT32_C(   774.26), EASYSIMD_FLOAT32_C(  -578.69) },
      { EASYSIMD_FLOAT32_C(   774.00), EASYSIMD_FLOAT32_C(  -578.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t r = easysimd_vrnd_f32(a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vrnd_f32(a);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrnd_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50) },
      { EASYSIMD_FLOAT64_C(    -2.00) } },
    { { EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(     2.00) } },
    { { EASYSIMD_FLOAT64_C(   667.17) },
      { EASYSIMD_FLOAT64_C(   667.00) } },
    { { EASYSIMD_FLOAT64_C(   472.88) },
      { EASYSIMD_FLOAT64_C(   472.00) } },
    { { EASYSIMD_FLOAT64_C(   161.95) },
      { EASYSIMD_FLOAT64_C(   161.00) } },
    { { EASYSIMD_FLOAT64_C(  -277.95) },
      { EASYSIMD_FLOAT64_C(  -277.00) } },
    { { EASYSIMD_FLOAT64_C(   876.07) },
      { EASYSIMD_FLOAT64_C(   876.00) } },
    { { EASYSIMD_FLOAT64_C(   151.96) },
      { EASYSIMD_FLOAT64_C(   151.00) } },
    { { EASYSIMD_FLOAT64_C(  -135.07) },
      { EASYSIMD_FLOAT64_C(  -135.00) } },
    { { EASYSIMD_FLOAT64_C(  -815.34) },
      { EASYSIMD_FLOAT64_C(  -815.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t r = easysimd_vrnd_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vrnd_f64(a);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00) } },
    { { EASYSIMD_FLOAT32_C(  -722.64), EASYSIMD_FLOAT32_C(   549.67), EASYSIMD_FLOAT32_C(   360.83), EASYSIMD_FLOAT32_C(   702.11) },
      { EASYSIMD_FLOAT32_C(  -722.00), EASYSIMD_FLOAT32_C(   549.00), EASYSIMD_FLOAT32_C(   360.00), EASYSIMD_FLOAT32_C(   702.00) } },
    { { EASYSIMD_FLOAT32_C(   923.48), EASYSIMD_FLOAT32_C(   285.32), EASYSIMD_FLOAT32_C(    55.43), EASYSIMD_FLOAT32_C(   705.81) },
      { EASYSIMD_FLOAT32_C(   923.00), EASYSIMD_FLOAT32_C(   285.00), EASYSIMD_FLOAT32_C(    55.00), EASYSIMD_FLOAT32_C(   705.00) } },
    { { EASYSIMD_FLOAT32_C(  -690.85), EASYSIMD_FLOAT32_C(   823.44), EASYSIMD_FLOAT32_C(  -415.44), EASYSIMD_FLOAT32_C(   833.76) },
      { EASYSIMD_FLOAT32_C(  -690.00), EASYSIMD_FLOAT32_C(   823.00), EASYSIMD_FLOAT32_C(  -415.00), EASYSIMD_FLOAT32_C(   833.00) } },
    { { EASYSIMD_FLOAT32_C(   323.46), EASYSIMD_FLOAT32_C(   664.70), EASYSIMD_FLOAT32_C(   351.21), EASYSIMD_FLOAT32_C(   568.59) },
      { EASYSIMD_FLOAT32_C(   323.00), EASYSIMD_FLOAT32_C(   664.00), EASYSIMD_FLOAT32_C(   351.00), EASYSIMD_FLOAT32_C(   568.00) } },
    { { EASYSIMD_FLOAT32_C(  -206.93), EASYSIMD_FLOAT32_C(  -466.27), EASYSIMD_FLOAT32_C(  -294.45), EASYSIMD_FLOAT32_C(  -601.52) },
      { EASYSIMD_FLOAT32_C(  -206.00), EASYSIMD_FLOAT32_C(  -466.00), EASYSIMD_FLOAT32_C(  -294.00), EASYSIMD_FLOAT32_C(  -601.00) } },
    { { EASYSIMD_FLOAT32_C(  -299.83), EASYSIMD_FLOAT32_C(  -520.19), EASYSIMD_FLOAT32_C(  -180.21), EASYSIMD_FLOAT32_C(  -632.66) },
      { EASYSIMD_FLOAT32_C(  -299.00), EASYSIMD_FLOAT32_C(  -520.00), EASYSIMD_FLOAT32_C(  -180.00), EASYSIMD_FLOAT32_C(  -632.00) } },
    { { EASYSIMD_FLOAT32_C(   952.69), EASYSIMD_FLOAT32_C(   981.74), EASYSIMD_FLOAT32_C(    89.39), EASYSIMD_FLOAT32_C(   828.76) },
      { EASYSIMD_FLOAT32_C(   952.00), EASYSIMD_FLOAT32_C(   981.00), EASYSIMD_FLOAT32_C(    89.00), EASYSIMD_FLOAT32_C(   828.00) } },
    { { EASYSIMD_FLOAT32_C(   133.70), EASYSIMD_FLOAT32_C(   954.32), EASYSIMD_FLOAT32_C(  -986.58), EASYSIMD_FLOAT32_C(   411.06) },
      { EASYSIMD_FLOAT32_C(   133.00), EASYSIMD_FLOAT32_C(   954.00), EASYSIMD_FLOAT32_C(  -986.00), EASYSIMD_FLOAT32_C(   411.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t r = easysimd_vrndq_f32(a);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vrndq_f32(a);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrndq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(    -2.50), EASYSIMD_FLOAT64_C(     2.50) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     2.00) } },
    { { EASYSIMD_FLOAT64_C(   503.99), EASYSIMD_FLOAT64_C(   374.26) },
      { EASYSIMD_FLOAT64_C(   503.00), EASYSIMD_FLOAT64_C(   374.00) } },
    { { EASYSIMD_FLOAT64_C(   113.17), EASYSIMD_FLOAT64_C(   427.47) },
      { EASYSIMD_FLOAT64_C(   113.00), EASYSIMD_FLOAT64_C(   427.00) } },
    { { EASYSIMD_FLOAT64_C(  -340.42), EASYSIMD_FLOAT64_C(  -831.40) },
      { EASYSIMD_FLOAT64_C(  -340.00), EASYSIMD_FLOAT64_C(  -831.00) } },
    { { EASYSIMD_FLOAT64_C(   133.28), EASYSIMD_FLOAT64_C(   -31.27) },
      { EASYSIMD_FLOAT64_C(   133.00), EASYSIMD_FLOAT64_C(   -31.00) } },
    { { EASYSIMD_FLOAT64_C(   992.04), EASYSIMD_FLOAT64_C(   717.84) },
      { EASYSIMD_FLOAT64_C(   992.00), EASYSIMD_FLOAT64_C(   717.00) } },
    { { EASYSIMD_FLOAT64_C(  -197.51), EASYSIMD_FLOAT64_C(   315.50) },
      { EASYSIMD_FLOAT64_C(  -197.00), EASYSIMD_FLOAT64_C(   315.00) } },
    { { EASYSIMD_FLOAT64_C(   382.54), EASYSIMD_FLOAT64_C(  -846.31) },
      { EASYSIMD_FLOAT64_C(   382.00), EASYSIMD_FLOAT64_C(  -846.00) } },
    { { EASYSIMD_FLOAT64_C(  -115.91), EASYSIMD_FLOAT64_C(  -824.39) },
      { EASYSIMD_FLOAT64_C(  -115.00), EASYSIMD_FLOAT64_C(  -824.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t r = easysimd_vrndq_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vrndq_f64(a);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrnd_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrnd_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrndq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
