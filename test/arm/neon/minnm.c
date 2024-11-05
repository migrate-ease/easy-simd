#define EASYSIMD_TEST_ARM_NEON_INSN minnm

#include "test-neon.h"
#include "../../../easysimd/arm/neon/minnm.h"

static int
test_easysimd_vminnm_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 b[2];
    easysimd_float32 r[2];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   944.82) },
      { EASYSIMD_FLOAT32_C(   575.31),            EASYSIMD_MATH_NANF },
      { EASYSIMD_FLOAT32_C(   575.31), EASYSIMD_FLOAT32_C(   944.82) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -343.95) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   317.39) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -343.95) } },
    #endif
    { { EASYSIMD_FLOAT32_C(  -696.17), EASYSIMD_FLOAT32_C(   907.59) },
      { EASYSIMD_FLOAT32_C(  -623.94), EASYSIMD_FLOAT32_C(   625.50) },
      { EASYSIMD_FLOAT32_C(  -696.17), EASYSIMD_FLOAT32_C(   625.50) } },
    { { EASYSIMD_FLOAT32_C(  -705.76), EASYSIMD_FLOAT32_C(  -732.20) },
      { EASYSIMD_FLOAT32_C(  -126.64), EASYSIMD_FLOAT32_C(  -660.16) },
      { EASYSIMD_FLOAT32_C(  -705.76), EASYSIMD_FLOAT32_C(  -732.20) } },
    { { EASYSIMD_FLOAT32_C(  -661.61), EASYSIMD_FLOAT32_C(  -734.04) },
      { EASYSIMD_FLOAT32_C(   847.38), EASYSIMD_FLOAT32_C(   816.85) },
      { EASYSIMD_FLOAT32_C(  -661.61), EASYSIMD_FLOAT32_C(  -734.04) } },
    { { EASYSIMD_FLOAT32_C(   945.94), EASYSIMD_FLOAT32_C(  -136.95) },
      { EASYSIMD_FLOAT32_C(    70.32), EASYSIMD_FLOAT32_C(   820.87) },
      { EASYSIMD_FLOAT32_C(    70.32), EASYSIMD_FLOAT32_C(  -136.95) } },
    { { EASYSIMD_FLOAT32_C(   441.43), EASYSIMD_FLOAT32_C(  -694.16) },
      { EASYSIMD_FLOAT32_C(   343.41), EASYSIMD_FLOAT32_C(    88.05) },
      { EASYSIMD_FLOAT32_C(   343.41), EASYSIMD_FLOAT32_C(  -694.16) } },
    { { EASYSIMD_FLOAT32_C(   175.22), EASYSIMD_FLOAT32_C(  -756.19) },
      { EASYSIMD_FLOAT32_C(  -558.30), EASYSIMD_FLOAT32_C(   795.61) },
      { EASYSIMD_FLOAT32_C(  -558.30), EASYSIMD_FLOAT32_C(  -756.19) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vminnm_f32(a, b);

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
    easysimd_float32x2_t r = easysimd_vminnm_f32(a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vminnm_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 b[1];
    easysimd_float64 r[1];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {             EASYSIMD_MATH_NAN },
      { EASYSIMD_FLOAT64_C(   814.09) },
      { EASYSIMD_FLOAT64_C(   814.09) } },
    { { EASYSIMD_FLOAT64_C(   857.46) },
      {             EASYSIMD_MATH_NAN },
      { EASYSIMD_FLOAT64_C(   857.46) } },
    { {             EASYSIMD_MATH_NAN },
      {             EASYSIMD_MATH_NAN },
      {             EASYSIMD_MATH_NAN } },
    #endif
    { { EASYSIMD_FLOAT64_C(   611.47) },
      { EASYSIMD_FLOAT64_C(   938.24) },
      { EASYSIMD_FLOAT64_C(   611.47) } },
    { { EASYSIMD_FLOAT64_C(  -733.28) },
      { EASYSIMD_FLOAT64_C(  -430.87) },
      { EASYSIMD_FLOAT64_C(  -733.28) } },
    { { EASYSIMD_FLOAT64_C(   558.71) },
      { EASYSIMD_FLOAT64_C(   197.76) },
      { EASYSIMD_FLOAT64_C(   197.76) } },
    { { EASYSIMD_FLOAT64_C(   -73.48) },
      { EASYSIMD_FLOAT64_C(  -904.42) },
      { EASYSIMD_FLOAT64_C(  -904.42) } },
    { { EASYSIMD_FLOAT64_C(   443.92) },
      { EASYSIMD_FLOAT64_C(   926.58) },
      { EASYSIMD_FLOAT64_C(   443.92) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_float64x1_t r = easysimd_vminnm_f64(a, b);

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
    easysimd_float64x1_t r = easysimd_vminnm_f64(a, b);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vminnmq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   819.39),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   912.19) },
      { EASYSIMD_FLOAT32_C(  -631.16),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   587.97) },
      { EASYSIMD_FLOAT32_C(  -631.16), EASYSIMD_FLOAT32_C(   819.39),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   587.97) } },
    #endif
    { { EASYSIMD_FLOAT32_C(   979.32), EASYSIMD_FLOAT32_C(  -967.75), EASYSIMD_FLOAT32_C(  -462.78), EASYSIMD_FLOAT32_C(  -270.14) },
      { EASYSIMD_FLOAT32_C(  -821.32), EASYSIMD_FLOAT32_C(  -724.47), EASYSIMD_FLOAT32_C(  -442.09), EASYSIMD_FLOAT32_C(   -73.38) },
      { EASYSIMD_FLOAT32_C(  -821.32), EASYSIMD_FLOAT32_C(  -967.75), EASYSIMD_FLOAT32_C(  -462.78), EASYSIMD_FLOAT32_C(  -270.14) } },
    { { EASYSIMD_FLOAT32_C(  -910.38), EASYSIMD_FLOAT32_C(  -584.63), EASYSIMD_FLOAT32_C(   694.05), EASYSIMD_FLOAT32_C(  -314.00) },
      { EASYSIMD_FLOAT32_C(   781.88), EASYSIMD_FLOAT32_C(   305.53), EASYSIMD_FLOAT32_C(  -375.75), EASYSIMD_FLOAT32_C(  -951.40) },
      { EASYSIMD_FLOAT32_C(  -910.38), EASYSIMD_FLOAT32_C(  -584.63), EASYSIMD_FLOAT32_C(  -375.75), EASYSIMD_FLOAT32_C(  -951.40) } },
    { { EASYSIMD_FLOAT32_C(   874.66), EASYSIMD_FLOAT32_C(  -817.04), EASYSIMD_FLOAT32_C(   246.35), EASYSIMD_FLOAT32_C(  -198.82) },
      { EASYSIMD_FLOAT32_C(  -721.46), EASYSIMD_FLOAT32_C(  -309.72), EASYSIMD_FLOAT32_C(  -272.24), EASYSIMD_FLOAT32_C(  -582.08) },
      { EASYSIMD_FLOAT32_C(  -721.46), EASYSIMD_FLOAT32_C(  -817.04), EASYSIMD_FLOAT32_C(  -272.24), EASYSIMD_FLOAT32_C(  -582.08) } },
    { { EASYSIMD_FLOAT32_C(  -490.34), EASYSIMD_FLOAT32_C(  -147.19), EASYSIMD_FLOAT32_C(  -669.89), EASYSIMD_FLOAT32_C(  -121.49) },
      { EASYSIMD_FLOAT32_C(  -220.92), EASYSIMD_FLOAT32_C(   -59.54), EASYSIMD_FLOAT32_C(  -533.53), EASYSIMD_FLOAT32_C(  -241.60) },
      { EASYSIMD_FLOAT32_C(  -490.34), EASYSIMD_FLOAT32_C(  -147.19), EASYSIMD_FLOAT32_C(  -669.89), EASYSIMD_FLOAT32_C(  -241.60) } },
    { { EASYSIMD_FLOAT32_C(   -27.29), EASYSIMD_FLOAT32_C(     3.69), EASYSIMD_FLOAT32_C(   488.26), EASYSIMD_FLOAT32_C(   151.39) },
      { EASYSIMD_FLOAT32_C(   279.22), EASYSIMD_FLOAT32_C(  -953.83), EASYSIMD_FLOAT32_C(  -922.00), EASYSIMD_FLOAT32_C(   368.84) },
      { EASYSIMD_FLOAT32_C(   -27.29), EASYSIMD_FLOAT32_C(  -953.83), EASYSIMD_FLOAT32_C(  -922.00), EASYSIMD_FLOAT32_C(   151.39) } },
    { { EASYSIMD_FLOAT32_C(  -538.47), EASYSIMD_FLOAT32_C(   772.06), EASYSIMD_FLOAT32_C(  -945.16), EASYSIMD_FLOAT32_C(  -756.59) },
      { EASYSIMD_FLOAT32_C(    77.58), EASYSIMD_FLOAT32_C(  -320.91), EASYSIMD_FLOAT32_C(  -708.00), EASYSIMD_FLOAT32_C(   -47.76) },
      { EASYSIMD_FLOAT32_C(  -538.47), EASYSIMD_FLOAT32_C(  -320.91), EASYSIMD_FLOAT32_C(  -945.16), EASYSIMD_FLOAT32_C(  -756.59) } },
    { { EASYSIMD_FLOAT32_C(  -137.95), EASYSIMD_FLOAT32_C(   538.36), EASYSIMD_FLOAT32_C(   753.42), EASYSIMD_FLOAT32_C(   140.59) },
      { EASYSIMD_FLOAT32_C(  -771.36), EASYSIMD_FLOAT32_C(  -518.82), EASYSIMD_FLOAT32_C(   558.51), EASYSIMD_FLOAT32_C(  -261.70) },
      { EASYSIMD_FLOAT32_C(  -771.36), EASYSIMD_FLOAT32_C(  -518.82), EASYSIMD_FLOAT32_C(   558.51), EASYSIMD_FLOAT32_C(  -261.70) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vminnmq_f32(a, b);

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
    easysimd_float32x4_t r = easysimd_vminnmq_f32(a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vminnmq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   888.63) },
      { EASYSIMD_FLOAT64_C(   616.81),             EASYSIMD_MATH_NAN },
      { EASYSIMD_FLOAT64_C(   616.81), EASYSIMD_FLOAT64_C(   888.63) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -916.72) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   801.79) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -916.72) } },
    #endif
    { { EASYSIMD_FLOAT64_C(    86.97), EASYSIMD_FLOAT64_C(  -640.28) },
      { EASYSIMD_FLOAT64_C(   -46.82), EASYSIMD_FLOAT64_C(  -633.81) },
      { EASYSIMD_FLOAT64_C(   -46.82), EASYSIMD_FLOAT64_C(  -640.28) } },
    { { EASYSIMD_FLOAT64_C(  -594.11), EASYSIMD_FLOAT64_C(    31.18) },
      { EASYSIMD_FLOAT64_C(   735.03), EASYSIMD_FLOAT64_C(  -132.58) },
      { EASYSIMD_FLOAT64_C(  -594.11), EASYSIMD_FLOAT64_C(  -132.58) } },
    { { EASYSIMD_FLOAT64_C(  -196.76), EASYSIMD_FLOAT64_C(   789.88) },
      { EASYSIMD_FLOAT64_C(   110.83), EASYSIMD_FLOAT64_C(   880.82) },
      { EASYSIMD_FLOAT64_C(  -196.76), EASYSIMD_FLOAT64_C(   789.88) } },
    { { EASYSIMD_FLOAT64_C(  -531.03), EASYSIMD_FLOAT64_C(   402.83) },
      { EASYSIMD_FLOAT64_C(  -166.93), EASYSIMD_FLOAT64_C(   331.02) },
      { EASYSIMD_FLOAT64_C(  -531.03), EASYSIMD_FLOAT64_C(   331.02) } },
    { { EASYSIMD_FLOAT64_C(   -58.81), EASYSIMD_FLOAT64_C(  -413.51) },
      { EASYSIMD_FLOAT64_C(  -528.39), EASYSIMD_FLOAT64_C(   169.82) },
      { EASYSIMD_FLOAT64_C(  -528.39), EASYSIMD_FLOAT64_C(  -413.51) } },
    { { EASYSIMD_FLOAT64_C(    67.67), EASYSIMD_FLOAT64_C(  -969.88) },
      { EASYSIMD_FLOAT64_C(   908.12), EASYSIMD_FLOAT64_C(  -598.34) },
      { EASYSIMD_FLOAT64_C(    67.67), EASYSIMD_FLOAT64_C(  -969.88) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vminnmq_f64(a, b);

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
    easysimd_float64x2_t r = easysimd_vminnmq_f64(a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vminnm_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vminnm_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vminnmq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vminnmq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
