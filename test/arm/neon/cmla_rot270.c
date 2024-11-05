#define EASYSIMD_TEST_ARM_NEON_INSN cmla_rot270

#include "test-neon.h"
#include "../../../easysimd/arm/neon/cmla_rot270.h"

static int
test_easysimd_vcmla_rot270_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -50.23), EASYSIMD_FLOAT32_C(   -93.80) },
      { EASYSIMD_FLOAT32_C(   167.59), EASYSIMD_FLOAT32_C(    54.72) },
      { EASYSIMD_FLOAT32_C(   889.39), EASYSIMD_FLOAT32_C(   -46.05) },
      { EASYSIMD_FLOAT32_C( -4243.35), EASYSIMD_FLOAT32_C( 15673.89) } },
    { { EASYSIMD_FLOAT32_C(  -703.48), EASYSIMD_FLOAT32_C(   367.81) },
      { EASYSIMD_FLOAT32_C(   743.72), EASYSIMD_FLOAT32_C(    47.92) },
      { EASYSIMD_FLOAT32_C(   999.37), EASYSIMD_FLOAT32_C(   -87.94) },
      { EASYSIMD_FLOAT32_C( 18624.82), EASYSIMD_FLOAT32_C(-273635.59) } },
    { { EASYSIMD_FLOAT32_C(  -326.37), EASYSIMD_FLOAT32_C(   604.88) },
      { EASYSIMD_FLOAT32_C(   224.18), EASYSIMD_FLOAT32_C(  -994.49) },
      { EASYSIMD_FLOAT32_C(  -175.46), EASYSIMD_FLOAT32_C(  -858.15) },
      { EASYSIMD_FLOAT32_C(-601722.56), EASYSIMD_FLOAT32_C(-136460.14) } },
    { { EASYSIMD_FLOAT32_C(  -942.56), EASYSIMD_FLOAT32_C(    45.37) },
      { EASYSIMD_FLOAT32_C(   163.87), EASYSIMD_FLOAT32_C(   969.32) },
      { EASYSIMD_FLOAT32_C(   778.80), EASYSIMD_FLOAT32_C(   918.99) },
      { EASYSIMD_FLOAT32_C( 44756.85), EASYSIMD_FLOAT32_C( -6515.79) } },
    { { EASYSIMD_FLOAT32_C(   909.88), EASYSIMD_FLOAT32_C(   955.41) },
      { EASYSIMD_FLOAT32_C(   975.43), EASYSIMD_FLOAT32_C(   253.10) },
      { EASYSIMD_FLOAT32_C(  -261.19), EASYSIMD_FLOAT32_C(   233.33) },
      { EASYSIMD_FLOAT32_C(241553.08), EASYSIMD_FLOAT32_C(-931702.19) } },
    { { EASYSIMD_FLOAT32_C(  -953.37), EASYSIMD_FLOAT32_C(   688.58) },
      { EASYSIMD_FLOAT32_C(  -860.47), EASYSIMD_FLOAT32_C(   214.22) },
      { EASYSIMD_FLOAT32_C(  -256.70), EASYSIMD_FLOAT32_C(  -971.08) },
      { EASYSIMD_FLOAT32_C(147250.91), EASYSIMD_FLOAT32_C(591531.38) } },
    { { EASYSIMD_FLOAT32_C(  -831.83), EASYSIMD_FLOAT32_C(    39.82) },
      { EASYSIMD_FLOAT32_C(   396.73), EASYSIMD_FLOAT32_C(   911.89) },
      { EASYSIMD_FLOAT32_C(  -912.26), EASYSIMD_FLOAT32_C(   396.10) },
      { EASYSIMD_FLOAT32_C( 35399.20), EASYSIMD_FLOAT32_C(-15401.69) } },
    { { EASYSIMD_FLOAT32_C(  -176.06), EASYSIMD_FLOAT32_C(  -238.62) },
      { EASYSIMD_FLOAT32_C(     0.99), EASYSIMD_FLOAT32_C(  -951.87) },
      { EASYSIMD_FLOAT32_C(  -233.12), EASYSIMD_FLOAT32_C(   825.53) },
      { EASYSIMD_FLOAT32_C(226902.09), EASYSIMD_FLOAT32_C(  1061.76) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vld1_f32(test_vec[i].r);
    easysimd_float32x2_t r_ = easysimd_vcmla_rot270_f32(r, a, b);

    easysimd_test_arm_neon_assert_equal_f32x2(r_, easysimd_vld1_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r_ = easysimd_vcmla_rot270_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_rot270_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   126.46), EASYSIMD_FLOAT32_C(  -102.84), EASYSIMD_FLOAT32_C(  -564.92), EASYSIMD_FLOAT32_C(  -510.64) },
      { EASYSIMD_FLOAT32_C(   175.98), EASYSIMD_FLOAT32_C(  -552.56), EASYSIMD_FLOAT32_C(  -211.73), EASYSIMD_FLOAT32_C(  -777.22) },
      { EASYSIMD_FLOAT32_C(   582.03), EASYSIMD_FLOAT32_C(   874.49), EASYSIMD_FLOAT32_C(  -164.75), EASYSIMD_FLOAT32_C(  -792.98) },
      { EASYSIMD_FLOAT32_C( 57407.30), EASYSIMD_FLOAT32_C( 18972.27), EASYSIMD_FLOAT32_C(396714.88), EASYSIMD_FLOAT32_C(-108910.79) } },
    { { EASYSIMD_FLOAT32_C(   596.49), EASYSIMD_FLOAT32_C(  -138.97), EASYSIMD_FLOAT32_C(   939.08), EASYSIMD_FLOAT32_C(  -705.25) },
      { EASYSIMD_FLOAT32_C(   449.34), EASYSIMD_FLOAT32_C(  -193.80), EASYSIMD_FLOAT32_C(   928.27), EASYSIMD_FLOAT32_C(  -625.06) },
      { EASYSIMD_FLOAT32_C(  -301.19), EASYSIMD_FLOAT32_C(  -563.73), EASYSIMD_FLOAT32_C(  -710.46), EASYSIMD_FLOAT32_C(  -738.46) },
      { EASYSIMD_FLOAT32_C( 26631.20), EASYSIMD_FLOAT32_C( 61881.05), EASYSIMD_FLOAT32_C(440113.09), EASYSIMD_FLOAT32_C(653924.00) } },
    { { EASYSIMD_FLOAT32_C(   489.04), EASYSIMD_FLOAT32_C(   298.23), EASYSIMD_FLOAT32_C(  -122.18), EASYSIMD_FLOAT32_C(   614.08) },
      { EASYSIMD_FLOAT32_C(   157.80), EASYSIMD_FLOAT32_C(  -378.12), EASYSIMD_FLOAT32_C(   654.24), EASYSIMD_FLOAT32_C(  -715.73) },
      { EASYSIMD_FLOAT32_C(   519.04), EASYSIMD_FLOAT32_C(  -910.68), EASYSIMD_FLOAT32_C(  -226.37), EASYSIMD_FLOAT32_C(  -304.98) },
      { EASYSIMD_FLOAT32_C(-112247.69), EASYSIMD_FLOAT32_C(-47971.38), EASYSIMD_FLOAT32_C(-439741.84), EASYSIMD_FLOAT32_C(-402060.69) } },
    { { EASYSIMD_FLOAT32_C(  -463.25), EASYSIMD_FLOAT32_C(   561.90), EASYSIMD_FLOAT32_C(   -82.20), EASYSIMD_FLOAT32_C(  -881.22) },
      { EASYSIMD_FLOAT32_C(   436.39), EASYSIMD_FLOAT32_C(   753.06), EASYSIMD_FLOAT32_C(  -674.20), EASYSIMD_FLOAT32_C(    32.88) },
      { EASYSIMD_FLOAT32_C(  -385.92), EASYSIMD_FLOAT32_C(  -735.12), EASYSIMD_FLOAT32_C(   327.63), EASYSIMD_FLOAT32_C(  -936.58) },
      { EASYSIMD_FLOAT32_C(422758.50), EASYSIMD_FLOAT32_C(-245942.69), EASYSIMD_FLOAT32_C(-28646.88), EASYSIMD_FLOAT32_C(-595055.12) } },
    { { EASYSIMD_FLOAT32_C(    71.08), EASYSIMD_FLOAT32_C(   255.90), EASYSIMD_FLOAT32_C(  -561.64), EASYSIMD_FLOAT32_C(   769.89) },
      { EASYSIMD_FLOAT32_C(   692.17), EASYSIMD_FLOAT32_C(  -272.09), EASYSIMD_FLOAT32_C(  -968.57), EASYSIMD_FLOAT32_C(   181.21) },
      { EASYSIMD_FLOAT32_C(  -973.86), EASYSIMD_FLOAT32_C(   -90.75), EASYSIMD_FLOAT32_C(  -204.71), EASYSIMD_FLOAT32_C(   183.94) },
      { EASYSIMD_FLOAT32_C(-70601.69), EASYSIMD_FLOAT32_C(-177217.05), EASYSIMD_FLOAT32_C(139307.06), EASYSIMD_FLOAT32_C(745876.31) } },
    { { EASYSIMD_FLOAT32_C(   531.13), EASYSIMD_FLOAT32_C(  -550.47), EASYSIMD_FLOAT32_C(   468.21), EASYSIMD_FLOAT32_C(    50.17) },
      { EASYSIMD_FLOAT32_C(  -461.15), EASYSIMD_FLOAT32_C(  -758.16), EASYSIMD_FLOAT32_C(   745.19), EASYSIMD_FLOAT32_C(    75.60) },
      { EASYSIMD_FLOAT32_C(   803.74), EASYSIMD_FLOAT32_C(  -337.01), EASYSIMD_FLOAT32_C(   194.38), EASYSIMD_FLOAT32_C(   240.13) },
      { EASYSIMD_FLOAT32_C(418148.03), EASYSIMD_FLOAT32_C(-254186.23), EASYSIMD_FLOAT32_C(  3987.23), EASYSIMD_FLOAT32_C(-37146.05) } },
    { { EASYSIMD_FLOAT32_C(  -583.95), EASYSIMD_FLOAT32_C(   520.18), EASYSIMD_FLOAT32_C(  -726.99), EASYSIMD_FLOAT32_C(    30.13) },
      { EASYSIMD_FLOAT32_C(   785.06), EASYSIMD_FLOAT32_C(   600.64), EASYSIMD_FLOAT32_C(    93.55), EASYSIMD_FLOAT32_C(  -143.86) },
      { EASYSIMD_FLOAT32_C(  -143.46), EASYSIMD_FLOAT32_C(   531.91), EASYSIMD_FLOAT32_C(  -373.97), EASYSIMD_FLOAT32_C(  -451.29) },
      { EASYSIMD_FLOAT32_C(312297.47), EASYSIMD_FLOAT32_C(-407840.59), EASYSIMD_FLOAT32_C( -4708.47), EASYSIMD_FLOAT32_C( -3269.95) } },
    { { EASYSIMD_FLOAT32_C(  -740.18), EASYSIMD_FLOAT32_C(  -342.54), EASYSIMD_FLOAT32_C(   729.92), EASYSIMD_FLOAT32_C(  -714.05) },
      { EASYSIMD_FLOAT32_C(   566.71), EASYSIMD_FLOAT32_C(  -474.78), EASYSIMD_FLOAT32_C(   469.90), EASYSIMD_FLOAT32_C(    97.84) },
      { EASYSIMD_FLOAT32_C(   -25.25), EASYSIMD_FLOAT32_C(   -61.89), EASYSIMD_FLOAT32_C(  -851.99), EASYSIMD_FLOAT32_C(   513.60) },
      { EASYSIMD_FLOAT32_C(162605.89), EASYSIMD_FLOAT32_C(194058.97), EASYSIMD_FLOAT32_C(-70714.64), EASYSIMD_FLOAT32_C(336045.69) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vld1q_f32(test_vec[i].r);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_rot270_f32(r, a, b);
    easysimd_test_arm_neon_assert_equal_f32x4(r_, easysimd_vld1q_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_rot270_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_rot270_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
    easysimd_float64 r_[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -772.20), EASYSIMD_FLOAT64_C(  -289.07) },
      { EASYSIMD_FLOAT64_C(  -254.86), EASYSIMD_FLOAT64_C(   875.19) },
      { EASYSIMD_FLOAT64_C(  -291.25), EASYSIMD_FLOAT64_C(  -386.54) },
      { EASYSIMD_FLOAT64_C(-253282.42), EASYSIMD_FLOAT64_C(-74058.92) } },
    { { EASYSIMD_FLOAT64_C(   901.93), EASYSIMD_FLOAT64_C(  -182.45) },
      { EASYSIMD_FLOAT64_C(  -401.46), EASYSIMD_FLOAT64_C(   570.66) },
      { EASYSIMD_FLOAT64_C(  -805.46), EASYSIMD_FLOAT64_C(   495.43) },
      { EASYSIMD_FLOAT64_C(-104922.38), EASYSIMD_FLOAT64_C(-72750.95) } },
    { { EASYSIMD_FLOAT64_C(   735.86), EASYSIMD_FLOAT64_C(   696.07) },
      { EASYSIMD_FLOAT64_C(    67.50), EASYSIMD_FLOAT64_C(  -993.11) },
      { EASYSIMD_FLOAT64_C(  -328.56), EASYSIMD_FLOAT64_C(  -238.82) },
      { EASYSIMD_FLOAT64_C(-691602.64), EASYSIMD_FLOAT64_C(-47223.55) } },
    { { EASYSIMD_FLOAT64_C(  -528.21), EASYSIMD_FLOAT64_C(   -73.41) },
      { EASYSIMD_FLOAT64_C(    25.60), EASYSIMD_FLOAT64_C(  -492.76) },
      { EASYSIMD_FLOAT64_C(  -996.44), EASYSIMD_FLOAT64_C(  -171.25) },
      { EASYSIMD_FLOAT64_C( 35177.07), EASYSIMD_FLOAT64_C(  1708.05) } },
    { { EASYSIMD_FLOAT64_C(   532.55), EASYSIMD_FLOAT64_C(    64.30) },
      { EASYSIMD_FLOAT64_C(  -451.87), EASYSIMD_FLOAT64_C(  -697.40) },
      { EASYSIMD_FLOAT64_C(  -232.31), EASYSIMD_FLOAT64_C(  -793.59) },
      { EASYSIMD_FLOAT64_C(-45075.13), EASYSIMD_FLOAT64_C( 28261.65) } },
    { { EASYSIMD_FLOAT64_C(    99.38), EASYSIMD_FLOAT64_C(    -4.51) },
      { EASYSIMD_FLOAT64_C(   -82.66), EASYSIMD_FLOAT64_C(   844.53) },
      { EASYSIMD_FLOAT64_C(  -129.32), EASYSIMD_FLOAT64_C(   626.09) },
      { EASYSIMD_FLOAT64_C( -3938.15), EASYSIMD_FLOAT64_C(   253.29) } },
    { { EASYSIMD_FLOAT64_C(  -542.02), EASYSIMD_FLOAT64_C(  -227.39) },
      { EASYSIMD_FLOAT64_C(  -556.35), EASYSIMD_FLOAT64_C(    56.52) },
      { EASYSIMD_FLOAT64_C(  -656.73), EASYSIMD_FLOAT64_C(  -361.81) },
      { EASYSIMD_FLOAT64_C(-13508.81), EASYSIMD_FLOAT64_C(-126870.24) } },
    { { EASYSIMD_FLOAT64_C(  -448.05), EASYSIMD_FLOAT64_C(  -920.87) },
      { EASYSIMD_FLOAT64_C(  -665.74), EASYSIMD_FLOAT64_C(   619.45) },
      { EASYSIMD_FLOAT64_C(  -913.98), EASYSIMD_FLOAT64_C(     5.70) },
      { EASYSIMD_FLOAT64_C(-571346.90), EASYSIMD_FLOAT64_C(-613054.29) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vld1q_f64(test_vec[i].r);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_rot270_f64(r, a, b);
    easysimd_test_arm_neon_assert_equal_f64x2(r_, easysimd_vld1q_f64(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_rot270_f64(r, a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmla_rot270_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_rot270_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_rot270_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
