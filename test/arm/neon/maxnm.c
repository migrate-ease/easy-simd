#define EASYSIMD_TEST_ARM_NEON_INSN maxnm

#include "test-neon.h"
#include "../../../easysimd/arm/neon/maxnm.h"

static int
test_easysimd_vmaxnm_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 b[2];
    easysimd_float32 r[2];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   656.90) },
      { EASYSIMD_FLOAT32_C(   427.79),            EASYSIMD_MATH_NANF },
      { EASYSIMD_FLOAT32_C(   427.79), EASYSIMD_FLOAT32_C(   656.90) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   116.96) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -999.94) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   116.96) } },
    #endif
    { { EASYSIMD_FLOAT32_C(  -619.20), EASYSIMD_FLOAT32_C(  -413.47) },
      { EASYSIMD_FLOAT32_C(   871.28), EASYSIMD_FLOAT32_C(  -660.33) },
      { EASYSIMD_FLOAT32_C(   871.28), EASYSIMD_FLOAT32_C(  -413.47) } },
    { { EASYSIMD_FLOAT32_C(   422.55), EASYSIMD_FLOAT32_C(   160.51) },
      { EASYSIMD_FLOAT32_C(   148.88), EASYSIMD_FLOAT32_C(   905.13) },
      { EASYSIMD_FLOAT32_C(   422.55), EASYSIMD_FLOAT32_C(   905.13) } },
    { { EASYSIMD_FLOAT32_C(  -605.53), EASYSIMD_FLOAT32_C(  -971.47) },
      { EASYSIMD_FLOAT32_C(   182.75), EASYSIMD_FLOAT32_C(  -737.07) },
      { EASYSIMD_FLOAT32_C(   182.75), EASYSIMD_FLOAT32_C(  -737.07) } },
    { { EASYSIMD_FLOAT32_C(  -182.06), EASYSIMD_FLOAT32_C(  -678.54) },
      { EASYSIMD_FLOAT32_C(   165.68), EASYSIMD_FLOAT32_C(   413.12) },
      { EASYSIMD_FLOAT32_C(   165.68), EASYSIMD_FLOAT32_C(   413.12) } },
    { { EASYSIMD_FLOAT32_C(    20.28), EASYSIMD_FLOAT32_C(  -770.49) },
      { EASYSIMD_FLOAT32_C(   647.00), EASYSIMD_FLOAT32_C(  -632.40) },
      { EASYSIMD_FLOAT32_C(   647.00), EASYSIMD_FLOAT32_C(  -632.40) } },
    { { EASYSIMD_FLOAT32_C(   949.17), EASYSIMD_FLOAT32_C(   616.00) },
      { EASYSIMD_FLOAT32_C(  -967.88), EASYSIMD_FLOAT32_C(  -301.85) },
      { EASYSIMD_FLOAT32_C(   949.17), EASYSIMD_FLOAT32_C(   616.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vmaxnm_f32(a, b);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32_t values[8 * 2 * sizeof(easysimd_float32x2_t)];
  easysimd_test_arm_neon_random_f32x2_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_extract_f32x2(i, 2, 0, values);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_extract_f32x2(i, 2, 1, values);
    easysimd_float32x2_t r = easysimd_vmaxnm_f32(a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vmaxnm_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 b[1];
    easysimd_float64 r[1];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {             EASYSIMD_MATH_NAN },
      { EASYSIMD_FLOAT64_C(   169.64) },
      { EASYSIMD_FLOAT64_C(   169.64) } },
    { { EASYSIMD_FLOAT64_C(  -986.47) },
      {             EASYSIMD_MATH_NAN },
      { EASYSIMD_FLOAT64_C(  -986.47) } },
    { {             EASYSIMD_MATH_NAN },
      {             EASYSIMD_MATH_NAN },
      {             EASYSIMD_MATH_NAN } },
    #endif
    { { EASYSIMD_FLOAT64_C(   827.71) },
      { EASYSIMD_FLOAT64_C(   191.90) },
      { EASYSIMD_FLOAT64_C(   827.71) } },
    { { EASYSIMD_FLOAT64_C(  -275.70) },
      { EASYSIMD_FLOAT64_C(   295.23) },
      { EASYSIMD_FLOAT64_C(   295.23) } },
    { { EASYSIMD_FLOAT64_C(   188.72) },
      { EASYSIMD_FLOAT64_C(   429.15) },
      { EASYSIMD_FLOAT64_C(   429.15) } },
    { { EASYSIMD_FLOAT64_C(  -147.66) },
      { EASYSIMD_FLOAT64_C(   487.54) },
      { EASYSIMD_FLOAT64_C(   487.54) } },
    { { EASYSIMD_FLOAT64_C(  -528.84) },
      { EASYSIMD_FLOAT64_C(  -797.49) },
      { EASYSIMD_FLOAT64_C(  -528.84) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_float64x1_t r = easysimd_vmaxnm_f64(a, b);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64_t values[8 * 2 * sizeof(easysimd_float64x1_t)];
  easysimd_test_arm_neon_random_f64x1_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_extract_f64x1(i, 2, 0, values);
    easysimd_float64x1_t b = easysimd_test_arm_neon_random_extract_f64x1(i, 2, 1, values);
    easysimd_float64x1_t r = easysimd_vmaxnm_f64(a, b);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vmaxnmq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -830.15),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   116.42) },
      { EASYSIMD_FLOAT32_C(  -786.61),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   704.38) },
      { EASYSIMD_FLOAT32_C(  -786.61), EASYSIMD_FLOAT32_C(  -830.15),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   704.38) } },
    #endif
    { { EASYSIMD_FLOAT32_C(    42.56), EASYSIMD_FLOAT32_C(  -762.07), EASYSIMD_FLOAT32_C(   243.80), EASYSIMD_FLOAT32_C(   127.20) },
      { EASYSIMD_FLOAT32_C(  -554.99), EASYSIMD_FLOAT32_C(   818.92), EASYSIMD_FLOAT32_C(  -693.45), EASYSIMD_FLOAT32_C(   417.65) },
      { EASYSIMD_FLOAT32_C(    42.56), EASYSIMD_FLOAT32_C(   818.92), EASYSIMD_FLOAT32_C(   243.80), EASYSIMD_FLOAT32_C(   417.65) } },
    { { EASYSIMD_FLOAT32_C(   -84.44), EASYSIMD_FLOAT32_C(   320.00), EASYSIMD_FLOAT32_C(   451.21), EASYSIMD_FLOAT32_C(    71.41) },
      { EASYSIMD_FLOAT32_C(  -104.53), EASYSIMD_FLOAT32_C(   615.74), EASYSIMD_FLOAT32_C(  -465.38), EASYSIMD_FLOAT32_C(    92.29) },
      { EASYSIMD_FLOAT32_C(   -84.44), EASYSIMD_FLOAT32_C(   615.74), EASYSIMD_FLOAT32_C(   451.21), EASYSIMD_FLOAT32_C(    92.29) } },
    { { EASYSIMD_FLOAT32_C(  -599.78), EASYSIMD_FLOAT32_C(   592.36), EASYSIMD_FLOAT32_C(  -378.07), EASYSIMD_FLOAT32_C(  -109.06) },
      { EASYSIMD_FLOAT32_C(   386.28), EASYSIMD_FLOAT32_C(  -742.91), EASYSIMD_FLOAT32_C(   270.77), EASYSIMD_FLOAT32_C(   106.07) },
      { EASYSIMD_FLOAT32_C(   386.28), EASYSIMD_FLOAT32_C(   592.36), EASYSIMD_FLOAT32_C(   270.77), EASYSIMD_FLOAT32_C(   106.07) } },
    { { EASYSIMD_FLOAT32_C(  -573.06), EASYSIMD_FLOAT32_C(   931.15), EASYSIMD_FLOAT32_C(  -777.52), EASYSIMD_FLOAT32_C(  -359.67) },
      { EASYSIMD_FLOAT32_C(  -618.04), EASYSIMD_FLOAT32_C(  -590.04), EASYSIMD_FLOAT32_C(  -655.29), EASYSIMD_FLOAT32_C(   424.52) },
      { EASYSIMD_FLOAT32_C(  -573.06), EASYSIMD_FLOAT32_C(   931.15), EASYSIMD_FLOAT32_C(  -655.29), EASYSIMD_FLOAT32_C(   424.52) } },
    { { EASYSIMD_FLOAT32_C(  -352.11), EASYSIMD_FLOAT32_C(   588.52), EASYSIMD_FLOAT32_C(  -448.29), EASYSIMD_FLOAT32_C(    92.89) },
      { EASYSIMD_FLOAT32_C(   407.44), EASYSIMD_FLOAT32_C(  -141.74), EASYSIMD_FLOAT32_C(  -489.46), EASYSIMD_FLOAT32_C(  -677.00) },
      { EASYSIMD_FLOAT32_C(   407.44), EASYSIMD_FLOAT32_C(   588.52), EASYSIMD_FLOAT32_C(  -448.29), EASYSIMD_FLOAT32_C(    92.89) } },
    { { EASYSIMD_FLOAT32_C(  -821.73), EASYSIMD_FLOAT32_C(   961.75), EASYSIMD_FLOAT32_C(   394.41), EASYSIMD_FLOAT32_C(    73.73) },
      { EASYSIMD_FLOAT32_C(   577.49), EASYSIMD_FLOAT32_C(   929.03), EASYSIMD_FLOAT32_C(  -833.98), EASYSIMD_FLOAT32_C(   977.71) },
      { EASYSIMD_FLOAT32_C(   577.49), EASYSIMD_FLOAT32_C(   961.75), EASYSIMD_FLOAT32_C(   394.41), EASYSIMD_FLOAT32_C(   977.71) } },
    { { EASYSIMD_FLOAT32_C(   521.39), EASYSIMD_FLOAT32_C(  -212.06), EASYSIMD_FLOAT32_C(  -131.35), EASYSIMD_FLOAT32_C(   -92.34) },
      { EASYSIMD_FLOAT32_C(    45.03), EASYSIMD_FLOAT32_C(  -860.58), EASYSIMD_FLOAT32_C(  -986.27), EASYSIMD_FLOAT32_C(   471.98) },
      { EASYSIMD_FLOAT32_C(   521.39), EASYSIMD_FLOAT32_C(  -212.06), EASYSIMD_FLOAT32_C(  -131.35), EASYSIMD_FLOAT32_C(   471.98) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vmaxnmq_f32(a, b);

    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32_t values[8 * 2 * sizeof(easysimd_float32x4_t)];
  easysimd_test_arm_neon_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_extract_f32x4(i, 2, 0, values);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_extract_f32x4(i, 2, 1, values);
    easysimd_float32x4_t r = easysimd_vmaxnmq_f32(a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vmaxnmq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
        { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -182.58) },
      { EASYSIMD_FLOAT64_C(   743.82),             EASYSIMD_MATH_NAN },
      { EASYSIMD_FLOAT64_C(   743.82), EASYSIMD_FLOAT64_C(  -182.58) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   493.92) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   934.94) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   934.94) } },
    #endif
    { { EASYSIMD_FLOAT64_C(  -559.75), EASYSIMD_FLOAT64_C(  -168.42) },
      { EASYSIMD_FLOAT64_C(   193.83), EASYSIMD_FLOAT64_C(  -311.43) },
      { EASYSIMD_FLOAT64_C(   193.83), EASYSIMD_FLOAT64_C(  -168.42) } },
    { { EASYSIMD_FLOAT64_C(   685.60), EASYSIMD_FLOAT64_C(   642.39) },
      { EASYSIMD_FLOAT64_C(  -129.77), EASYSIMD_FLOAT64_C(  -575.43) },
      { EASYSIMD_FLOAT64_C(   685.60), EASYSIMD_FLOAT64_C(   642.39) } },
    { { EASYSIMD_FLOAT64_C(   166.90), EASYSIMD_FLOAT64_C(  -869.88) },
      { EASYSIMD_FLOAT64_C(    87.77), EASYSIMD_FLOAT64_C(  -554.15) },
      { EASYSIMD_FLOAT64_C(   166.90), EASYSIMD_FLOAT64_C(  -554.15) } },
    { { EASYSIMD_FLOAT64_C(  -667.35), EASYSIMD_FLOAT64_C(  -294.71) },
      { EASYSIMD_FLOAT64_C(   134.11), EASYSIMD_FLOAT64_C(   615.74) },
      { EASYSIMD_FLOAT64_C(   134.11), EASYSIMD_FLOAT64_C(   615.74) } },
    { { EASYSIMD_FLOAT64_C(   -85.63), EASYSIMD_FLOAT64_C(  -649.42) },
      { EASYSIMD_FLOAT64_C(  -536.78), EASYSIMD_FLOAT64_C(   843.96) },
      { EASYSIMD_FLOAT64_C(   -85.63), EASYSIMD_FLOAT64_C(   843.96) } },
    { { EASYSIMD_FLOAT64_C(   349.79), EASYSIMD_FLOAT64_C(   234.11) },
      { EASYSIMD_FLOAT64_C(  -713.81), EASYSIMD_FLOAT64_C(   557.65) },
      { EASYSIMD_FLOAT64_C(   349.79), EASYSIMD_FLOAT64_C(   557.65) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vmaxnmq_f64(a, b);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64_t values[8 * 2 * sizeof(easysimd_float64x2_t)];
  easysimd_test_arm_neon_random_f64x2_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_extract_f64x2(i, 2, 0, values);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_extract_f64x2(i, 2, 1, values);
    easysimd_float64x2_t r = easysimd_vmaxnmq_f64(a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vmaxnm_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vmaxnm_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vmaxnmq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vmaxnmq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
