#define EASYSIMD_TEST_ARM_NEON_INSN cagt

#include "test-neon.h"
#include "../../../easysimd/arm/neon/cagt.h"

static int
test_easysimd_vcagth_f16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd_float16 a;
    easysimd_float16 b;
    uint16_t r;
  } test_vec[] = {
    { EASYSIMD_FLOAT16_VALUE(  -774.00),
      EASYSIMD_FLOAT16_VALUE(   279.00),
           UINT16_MAX },
    { EASYSIMD_FLOAT16_VALUE(  -933.00),
      EASYSIMD_FLOAT16_VALUE(   505.00),
           UINT16_MAX },
    { EASYSIMD_FLOAT16_VALUE(   510.00),
      EASYSIMD_FLOAT16_VALUE(    91.44),
           UINT16_MAX },
    { EASYSIMD_FLOAT16_VALUE(  -980.50),
      EASYSIMD_FLOAT16_VALUE(   217.50),
           UINT16_MAX },
    { EASYSIMD_FLOAT16_VALUE(   716.50),
      EASYSIMD_FLOAT16_VALUE(   903.00),
      UINT16_C(    0) },
    { EASYSIMD_FLOAT16_VALUE(   875.00),
      EASYSIMD_FLOAT16_VALUE(  -717.50),
           UINT16_MAX },
    { EASYSIMD_FLOAT16_VALUE(    -9.65),
      EASYSIMD_FLOAT16_VALUE(    45.19),
      UINT16_C(    0) },
    { EASYSIMD_FLOAT16_VALUE(  -580.00),
      EASYSIMD_FLOAT16_VALUE(   148.88),
           UINT16_MAX }   
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint16_t r = easysimd_vcagth_f16(test_vec[i].a, test_vec[i].b);

    easysimd_assert_equal_u16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float16_t a = easysimd_test_codegen_random_f16(-1000.0f, 1000.0f);
    easysimd_float16_t b = easysimd_test_codegen_random_f16(-1000.0f, 1000.0f);
    uint16_t r = easysimd_vcagth_f16(a, b);

    easysimd_test_codegen_write_f16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcagts_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a;
    easysimd_float32 b;
    uint32_t r;
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    {            EASYSIMD_MATH_NANF,
      EASYSIMD_FLOAT32_C(     0.52),
      UINT32_C(         0) },
    { EASYSIMD_FLOAT32_C(   705.02),
                 EASYSIMD_MATH_NANF,
      UINT32_C(         0) },
    {            EASYSIMD_MATH_NANF,
                 EASYSIMD_MATH_NANF,
      UINT32_C(         0) },
    #endif

    { EASYSIMD_FLOAT32_C(     8.79),
      EASYSIMD_FLOAT32_C(   792.83),
      UINT32_C(         0) },
    { EASYSIMD_FLOAT32_C(  -399.97),
      EASYSIMD_FLOAT32_C(  -256.84),
                UINT32_MAX },
    { EASYSIMD_FLOAT32_C(   231.75),
      EASYSIMD_FLOAT32_C(  -411.54),
      UINT32_C(         0) },
    { EASYSIMD_FLOAT32_C(   864.59),
      EASYSIMD_FLOAT32_C(  -881.95),
      UINT32_C(         0) },
    { EASYSIMD_FLOAT32_C(  -814.20),
      EASYSIMD_FLOAT32_C(   479.81),
                UINT32_MAX },
    { EASYSIMD_FLOAT32_C(   263.32),
      EASYSIMD_FLOAT32_C(  -797.51),
      UINT32_C(         0) },
    { EASYSIMD_FLOAT32_C(   321.47),
      EASYSIMD_FLOAT32_C(   -74.97),
                UINT32_MAX },
    { EASYSIMD_FLOAT32_C(   -57.92),
      EASYSIMD_FLOAT32_C(   535.57),
      UINT32_C(         0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint32_t r = easysimd_vcagts_f32(test_vec[i].a, test_vec[i].b);

    easysimd_assert_equal_u32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    easysimd_float32_t b = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    uint32_t r = easysimd_vcagts_f32(a, b);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcagtd_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a;
    easysimd_float64 b;
    uint64_t r;
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    {              EASYSIMD_MATH_NAN,
      EASYSIMD_FLOAT64_C(     0.52),
      UINT64_C(         0) },
    { EASYSIMD_FLOAT64_C(   705.02),
                  EASYSIMD_MATH_NAN,
      UINT64_C(         0) },
    {             EASYSIMD_MATH_NAN,
                  EASYSIMD_MATH_NAN,
      UINT64_C(         0) },
    #endif

    { EASYSIMD_FLOAT64_C(  -111.66),
      EASYSIMD_FLOAT64_C(  -149.68),
      UINT64_C(                   0) },
    { EASYSIMD_FLOAT64_C(  -365.17),
      EASYSIMD_FLOAT64_C(  -219.70),
                         UINT64_MAX },
    { EASYSIMD_FLOAT64_C(   -45.32),
      EASYSIMD_FLOAT64_C(   606.55),
      UINT64_C(                   0) },
    { EASYSIMD_FLOAT64_C(  -324.50),
      EASYSIMD_FLOAT64_C(  -332.43),
      UINT64_C(                   0) },
    { EASYSIMD_FLOAT64_C(   611.77),
      EASYSIMD_FLOAT64_C(   425.54),
                         UINT64_MAX },
    { EASYSIMD_FLOAT64_C(   910.11),
      EASYSIMD_FLOAT64_C(   648.44),
                         UINT64_MAX },
    { EASYSIMD_FLOAT64_C(   572.56),
      EASYSIMD_FLOAT64_C(  -409.05),
                         UINT64_MAX },
    { EASYSIMD_FLOAT64_C(   265.81),
      EASYSIMD_FLOAT64_C(  -418.65),
      UINT64_C(                   0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint64_t r = easysimd_vcagtd_f64(test_vec[i].a, test_vec[i].b);

    easysimd_assert_equal_u64(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64_t a = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    easysimd_float64_t b = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    uint64_t r = easysimd_vcagtd_f64(a, b);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcagt_f16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd_float16 a[4];
    easysimd_float16 b[4];
    uint16_t r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT16_VALUE(   930.50), EASYSIMD_FLOAT16_VALUE(  -703.50), EASYSIMD_FLOAT16_VALUE(  -125.12), EASYSIMD_FLOAT16_VALUE(   783.00) },
      { EASYSIMD_FLOAT16_VALUE(   402.50), EASYSIMD_FLOAT16_VALUE(  -327.25), EASYSIMD_FLOAT16_VALUE(   405.25), EASYSIMD_FLOAT16_VALUE(  -207.75) },
      {      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(   -29.70), EASYSIMD_FLOAT16_VALUE(   210.88), EASYSIMD_FLOAT16_VALUE(  -861.00), EASYSIMD_FLOAT16_VALUE(  -614.50) },
      { EASYSIMD_FLOAT16_VALUE(  -248.62), EASYSIMD_FLOAT16_VALUE(   342.00), EASYSIMD_FLOAT16_VALUE(  -816.50), EASYSIMD_FLOAT16_VALUE(   -39.50) },
      { UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(   419.75), EASYSIMD_FLOAT16_VALUE(  -664.50), EASYSIMD_FLOAT16_VALUE(  -289.75), EASYSIMD_FLOAT16_VALUE(   396.25) },
      { EASYSIMD_FLOAT16_VALUE(  -934.50), EASYSIMD_FLOAT16_VALUE(   -18.20), EASYSIMD_FLOAT16_VALUE(   855.00), EASYSIMD_FLOAT16_VALUE(   748.50) },
      { UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(   145.12), EASYSIMD_FLOAT16_VALUE(  -781.50), EASYSIMD_FLOAT16_VALUE(  -379.50), EASYSIMD_FLOAT16_VALUE(    23.91) },
      { EASYSIMD_FLOAT16_VALUE(   854.00), EASYSIMD_FLOAT16_VALUE(   763.50), EASYSIMD_FLOAT16_VALUE(   -35.88), EASYSIMD_FLOAT16_VALUE(   784.50) },
      { UINT16_C(    0),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -940.00), EASYSIMD_FLOAT16_VALUE(   839.00), EASYSIMD_FLOAT16_VALUE(   568.00), EASYSIMD_FLOAT16_VALUE(   462.25) },
      { EASYSIMD_FLOAT16_VALUE(  -488.25), EASYSIMD_FLOAT16_VALUE(   -26.98), EASYSIMD_FLOAT16_VALUE(  -745.50), EASYSIMD_FLOAT16_VALUE(   482.00) },
      {      UINT16_MAX,      UINT16_MAX, UINT16_C(    0), UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -816.00), EASYSIMD_FLOAT16_VALUE(  -606.50), EASYSIMD_FLOAT16_VALUE(   867.50), EASYSIMD_FLOAT16_VALUE(   -64.75) },
      { EASYSIMD_FLOAT16_VALUE(   735.50), EASYSIMD_FLOAT16_VALUE(  -949.00), EASYSIMD_FLOAT16_VALUE(   895.50), EASYSIMD_FLOAT16_VALUE(   155.25) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -613.50), EASYSIMD_FLOAT16_VALUE(  -394.00), EASYSIMD_FLOAT16_VALUE(  -448.50), EASYSIMD_FLOAT16_VALUE(  -548.00) },
      { EASYSIMD_FLOAT16_VALUE(   587.50), EASYSIMD_FLOAT16_VALUE(  -593.50), EASYSIMD_FLOAT16_VALUE(  -799.00), EASYSIMD_FLOAT16_VALUE(  -267.00) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(  -375.00), EASYSIMD_FLOAT16_VALUE(  -178.62), EASYSIMD_FLOAT16_VALUE(   757.00), EASYSIMD_FLOAT16_VALUE(  -521.00) },
      { EASYSIMD_FLOAT16_VALUE(  -415.25), EASYSIMD_FLOAT16_VALUE(  -279.00), EASYSIMD_FLOAT16_VALUE(  -736.00), EASYSIMD_FLOAT16_VALUE(  -355.25) },
      { UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float16x4_t a = easysimd_vld1_f16(test_vec[i].a);
    easysimd_float16x4_t b = easysimd_vld1_f16(test_vec[i].b);
    easysimd_uint16x4_t r = easysimd_vld1_u16(test_vec[i].r);
    easysimd_uint16x4_t r_ = easysimd_vcagt_f16(a, b);

    easysimd_test_arm_neon_assert_equal_u16x4(r_, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float16x4_t a = easysimd_test_arm_neon_random_f16x4(-1000.0f, 1000.0f);
    easysimd_float16x4_t b = easysimd_test_arm_neon_random_f16x4(-1000.0f, 1000.0f);
    easysimd_uint16x4_t r = easysimd_vcagt_f16(a, b);

    easysimd_test_arm_neon_write_f16x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f16x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcagt_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    easysimd_float32 a[2];
    easysimd_float32 b[2];
    uint32_t r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   311.69), EASYSIMD_FLOAT32_C(  -932.68) },
      { EASYSIMD_FLOAT32_C(    98.33), EASYSIMD_FLOAT32_C(  -552.98) },
      {           UINT32_MAX,           UINT32_MAX } },
    { { EASYSIMD_FLOAT32_C(   959.61), EASYSIMD_FLOAT32_C(   617.75) },
      { EASYSIMD_FLOAT32_C(  -197.11), EASYSIMD_FLOAT32_C(   562.98) },
      {           UINT32_MAX,           UINT32_MAX } },
    { { EASYSIMD_FLOAT32_C(   468.98), EASYSIMD_FLOAT32_C(  -916.49) },
      { EASYSIMD_FLOAT32_C(   965.35), EASYSIMD_FLOAT32_C(   700.25) },
      { UINT32_C(         0),           UINT32_MAX } },
    { { EASYSIMD_FLOAT32_C(  -647.13), EASYSIMD_FLOAT32_C(  -147.35) },
      { EASYSIMD_FLOAT32_C(  -117.68), EASYSIMD_FLOAT32_C(  -241.37) },
      {           UINT32_MAX, UINT32_C(         0) } },
    { { EASYSIMD_FLOAT32_C(  -664.10), EASYSIMD_FLOAT32_C(  -976.12) },
      { EASYSIMD_FLOAT32_C(   874.22), EASYSIMD_FLOAT32_C(   -12.94) },
      { UINT32_C(         0),           UINT32_MAX } },
    { { EASYSIMD_FLOAT32_C(    25.04), EASYSIMD_FLOAT32_C(  -125.75) },
      { EASYSIMD_FLOAT32_C(   212.15), EASYSIMD_FLOAT32_C(   782.89) },
      { UINT32_C(         0), UINT32_C(         0) } },
    { { EASYSIMD_FLOAT32_C(   561.17), EASYSIMD_FLOAT32_C(   217.87) },
      { EASYSIMD_FLOAT32_C(  -238.74), EASYSIMD_FLOAT32_C(   679.32) },
      {           UINT32_MAX, UINT32_C(         0) } },
    { { EASYSIMD_FLOAT32_C(  -965.46), EASYSIMD_FLOAT32_C(  -738.96) },
      { EASYSIMD_FLOAT32_C(  -711.74), EASYSIMD_FLOAT32_C(   346.23) },
      {           UINT32_MAX,           UINT32_MAX } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_uint32x2_t r = easysimd_vcagt_f32(a, b);

    easysimd_test_arm_neon_assert_equal_u32x2(r, easysimd_vld1_u32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_vcagt_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    easysimd_float64 a[1];
    easysimd_float64 b[1];
    uint64_t r[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    85.26) },
      { EASYSIMD_FLOAT64_C(   122.65) },
      { UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(  -500.89) },
      { EASYSIMD_FLOAT64_C(   936.69) },
      { UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(   594.89) },
      { EASYSIMD_FLOAT64_C(   788.77) },
      { UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(   543.70) },
      { EASYSIMD_FLOAT64_C(  -150.09) },
      {                    UINT64_MAX } },
    { { EASYSIMD_FLOAT64_C(  -875.02) },
      { EASYSIMD_FLOAT64_C(   442.69) },
      {                    UINT64_MAX } },
    { { EASYSIMD_FLOAT64_C(   673.76) },
      { EASYSIMD_FLOAT64_C(   217.24) },
      {                    UINT64_MAX } },
    { { EASYSIMD_FLOAT64_C(   789.39) },
      { EASYSIMD_FLOAT64_C(   718.78) },
      {                    UINT64_MAX } },
    { { EASYSIMD_FLOAT64_C(  -511.44) },
      { EASYSIMD_FLOAT64_C(   752.01) },
      { UINT64_C(                   0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_uint64x1_t r = easysimd_vcagt_f64(a, b);

    easysimd_test_arm_neon_assert_equal_u64x1(r, easysimd_vld1_u64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_vcagtq_f16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd_float16 a[8];
    easysimd_float16 b[8];
    uint16_t r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT16_VALUE(   131.50), EASYSIMD_FLOAT16_VALUE(  -289.00), EASYSIMD_FLOAT16_VALUE(  -100.88), EASYSIMD_FLOAT16_VALUE(  -881.00),
        EASYSIMD_FLOAT16_VALUE(  -149.50), EASYSIMD_FLOAT16_VALUE(   558.00), EASYSIMD_FLOAT16_VALUE(   800.50), EASYSIMD_FLOAT16_VALUE(  -454.00) },
      { EASYSIMD_FLOAT16_VALUE(   227.38), EASYSIMD_FLOAT16_VALUE(   969.00), EASYSIMD_FLOAT16_VALUE(   828.50), EASYSIMD_FLOAT16_VALUE(  -672.50),
        EASYSIMD_FLOAT16_VALUE(  -452.25), EASYSIMD_FLOAT16_VALUE(  -720.50), EASYSIMD_FLOAT16_VALUE(   609.00), EASYSIMD_FLOAT16_VALUE(   -97.19) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(   400.00), EASYSIMD_FLOAT16_VALUE(  -230.38), EASYSIMD_FLOAT16_VALUE(  -477.50), EASYSIMD_FLOAT16_VALUE(   924.00),
        EASYSIMD_FLOAT16_VALUE(   -85.00), EASYSIMD_FLOAT16_VALUE(   -74.06), EASYSIMD_FLOAT16_VALUE(  -465.50), EASYSIMD_FLOAT16_VALUE(  -573.50) },
      { EASYSIMD_FLOAT16_VALUE(  -854.00), EASYSIMD_FLOAT16_VALUE(   866.00), EASYSIMD_FLOAT16_VALUE(  -726.00), EASYSIMD_FLOAT16_VALUE(  -426.00),
        EASYSIMD_FLOAT16_VALUE(   380.00), EASYSIMD_FLOAT16_VALUE(  -691.00), EASYSIMD_FLOAT16_VALUE(   747.50), EASYSIMD_FLOAT16_VALUE(  -488.50) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(    19.80), EASYSIMD_FLOAT16_VALUE(  -353.25), EASYSIMD_FLOAT16_VALUE(  -369.25), EASYSIMD_FLOAT16_VALUE(   870.50),
        EASYSIMD_FLOAT16_VALUE(  -795.50), EASYSIMD_FLOAT16_VALUE(  -569.00), EASYSIMD_FLOAT16_VALUE(  -584.00), EASYSIMD_FLOAT16_VALUE(   432.00) },
      { EASYSIMD_FLOAT16_VALUE(  -600.00), EASYSIMD_FLOAT16_VALUE(  -755.00), EASYSIMD_FLOAT16_VALUE(   759.50), EASYSIMD_FLOAT16_VALUE(   -52.28),
        EASYSIMD_FLOAT16_VALUE(  -475.25), EASYSIMD_FLOAT16_VALUE(   368.25), EASYSIMD_FLOAT16_VALUE(   850.50), EASYSIMD_FLOAT16_VALUE(   924.50) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0), UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -862.00), EASYSIMD_FLOAT16_VALUE(  -627.00), EASYSIMD_FLOAT16_VALUE(   848.50), EASYSIMD_FLOAT16_VALUE(    52.91),
        EASYSIMD_FLOAT16_VALUE(   299.00), EASYSIMD_FLOAT16_VALUE(  -617.00), EASYSIMD_FLOAT16_VALUE(   479.50), EASYSIMD_FLOAT16_VALUE(   445.25) },
      { EASYSIMD_FLOAT16_VALUE(  -751.00), EASYSIMD_FLOAT16_VALUE(   753.50), EASYSIMD_FLOAT16_VALUE(  -981.00), EASYSIMD_FLOAT16_VALUE(   629.00),
        EASYSIMD_FLOAT16_VALUE(  -937.50), EASYSIMD_FLOAT16_VALUE(   766.50), EASYSIMD_FLOAT16_VALUE(  -859.50), EASYSIMD_FLOAT16_VALUE(    82.19) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(  -586.50), EASYSIMD_FLOAT16_VALUE(  -229.00), EASYSIMD_FLOAT16_VALUE(   -47.53), EASYSIMD_FLOAT16_VALUE(  -382.00),
        EASYSIMD_FLOAT16_VALUE(   202.12), EASYSIMD_FLOAT16_VALUE(   368.75), EASYSIMD_FLOAT16_VALUE(  -950.00), EASYSIMD_FLOAT16_VALUE(   602.00) },
      { EASYSIMD_FLOAT16_VALUE(   613.50), EASYSIMD_FLOAT16_VALUE(   809.50), EASYSIMD_FLOAT16_VALUE(  -450.00), EASYSIMD_FLOAT16_VALUE(  -861.50),
        EASYSIMD_FLOAT16_VALUE(   177.62), EASYSIMD_FLOAT16_VALUE(  -599.50), EASYSIMD_FLOAT16_VALUE(  -937.00), EASYSIMD_FLOAT16_VALUE(   315.50) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(  -226.62), EASYSIMD_FLOAT16_VALUE(   911.50), EASYSIMD_FLOAT16_VALUE(  -631.50), EASYSIMD_FLOAT16_VALUE(  -927.50),
        EASYSIMD_FLOAT16_VALUE(  -705.50), EASYSIMD_FLOAT16_VALUE(   848.00), EASYSIMD_FLOAT16_VALUE(   517.50), EASYSIMD_FLOAT16_VALUE(  -456.50) },
      { EASYSIMD_FLOAT16_VALUE(   601.50), EASYSIMD_FLOAT16_VALUE(   536.50), EASYSIMD_FLOAT16_VALUE(  -827.50), EASYSIMD_FLOAT16_VALUE(   664.00),
        EASYSIMD_FLOAT16_VALUE(   303.25), EASYSIMD_FLOAT16_VALUE(  -687.50), EASYSIMD_FLOAT16_VALUE(  -253.88), EASYSIMD_FLOAT16_VALUE(   717.00) },
      { UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(    83.62), EASYSIMD_FLOAT16_VALUE(   698.50), EASYSIMD_FLOAT16_VALUE(  -665.00), EASYSIMD_FLOAT16_VALUE(  -714.50),
        EASYSIMD_FLOAT16_VALUE(    67.25), EASYSIMD_FLOAT16_VALUE(  -615.00), EASYSIMD_FLOAT16_VALUE(   888.00), EASYSIMD_FLOAT16_VALUE(  -319.25) },
      { EASYSIMD_FLOAT16_VALUE(  -806.00), EASYSIMD_FLOAT16_VALUE(  -562.00), EASYSIMD_FLOAT16_VALUE(  -180.88), EASYSIMD_FLOAT16_VALUE(   371.75),
        EASYSIMD_FLOAT16_VALUE(  -161.75), EASYSIMD_FLOAT16_VALUE(  -117.88), EASYSIMD_FLOAT16_VALUE(  -312.50), EASYSIMD_FLOAT16_VALUE(   611.50) },
      { UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -206.38), EASYSIMD_FLOAT16_VALUE(    55.94), EASYSIMD_FLOAT16_VALUE(   684.00), EASYSIMD_FLOAT16_VALUE(    88.25),
        EASYSIMD_FLOAT16_VALUE(   -96.19), EASYSIMD_FLOAT16_VALUE(   201.50), EASYSIMD_FLOAT16_VALUE(   631.50), EASYSIMD_FLOAT16_VALUE(  -494.75) },
      { EASYSIMD_FLOAT16_VALUE(  -261.75), EASYSIMD_FLOAT16_VALUE(   804.00), EASYSIMD_FLOAT16_VALUE(  -830.50), EASYSIMD_FLOAT16_VALUE(  -958.50),
        EASYSIMD_FLOAT16_VALUE(  -883.50), EASYSIMD_FLOAT16_VALUE(   -84.69), EASYSIMD_FLOAT16_VALUE(   758.50), EASYSIMD_FLOAT16_VALUE(   200.25) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float16x8_t a = easysimd_vld1q_f16(test_vec[i].a);
    easysimd_float16x8_t b = easysimd_vld1q_f16(test_vec[i].b);
    easysimd_uint16x8_t r = easysimd_vld1q_u16(test_vec[i].r);
    easysimd_uint16x8_t r_ = easysimd_vcagtq_f16(a, b);

    easysimd_test_arm_neon_assert_equal_u16x8(r_, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float16x8_t a = easysimd_test_arm_neon_random_f16x8(-1000.0f, 1000.0f);
    easysimd_float16x8_t b = easysimd_test_arm_neon_random_f16x8(-1000.0f, 1000.0f);
    easysimd_uint16x8_t r = easysimd_vcagtq_f16(a, b);

    easysimd_test_arm_neon_write_f16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcagtq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    uint32_t r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   880.25), EASYSIMD_FLOAT32_C(   497.37), EASYSIMD_FLOAT32_C(   188.18), EASYSIMD_FLOAT32_C(  -214.92) },
      { EASYSIMD_FLOAT32_C(  -292.63), EASYSIMD_FLOAT32_C(   165.21), EASYSIMD_FLOAT32_C(  -507.32), EASYSIMD_FLOAT32_C(  -554.07) },
      {           UINT32_MAX,           UINT32_MAX, UINT32_C(         0), UINT32_C(         0) } },
    { { EASYSIMD_FLOAT32_C(    21.58), EASYSIMD_FLOAT32_C(  -187.66), EASYSIMD_FLOAT32_C(    52.34), EASYSIMD_FLOAT32_C(   522.72) },
      { EASYSIMD_FLOAT32_C(   805.10), EASYSIMD_FLOAT32_C(  -357.26), EASYSIMD_FLOAT32_C(   451.59), EASYSIMD_FLOAT32_C(   744.08) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { { EASYSIMD_FLOAT32_C(   113.67), EASYSIMD_FLOAT32_C(   334.71), EASYSIMD_FLOAT32_C(   489.01), EASYSIMD_FLOAT32_C(   347.72) },
      { EASYSIMD_FLOAT32_C(  -991.50), EASYSIMD_FLOAT32_C(  -625.74), EASYSIMD_FLOAT32_C(  -356.50), EASYSIMD_FLOAT32_C(   848.94) },
      { UINT32_C(         0), UINT32_C(         0),           UINT32_MAX, UINT32_C(         0) } },
    { { EASYSIMD_FLOAT32_C(    90.46), EASYSIMD_FLOAT32_C(   858.14), EASYSIMD_FLOAT32_C(  -123.29), EASYSIMD_FLOAT32_C(  -917.86) },
      { EASYSIMD_FLOAT32_C(  -788.14), EASYSIMD_FLOAT32_C(   739.22), EASYSIMD_FLOAT32_C(   572.18), EASYSIMD_FLOAT32_C(  -907.90) },
      { UINT32_C(         0),           UINT32_MAX, UINT32_C(         0),           UINT32_MAX } },
    { { EASYSIMD_FLOAT32_C(   236.59), EASYSIMD_FLOAT32_C(  -239.64), EASYSIMD_FLOAT32_C(  -122.81), EASYSIMD_FLOAT32_C(   943.97) },
      { EASYSIMD_FLOAT32_C(   925.57), EASYSIMD_FLOAT32_C(   369.86), EASYSIMD_FLOAT32_C(  -610.11), EASYSIMD_FLOAT32_C(   -52.85) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0),           UINT32_MAX } },
    { { EASYSIMD_FLOAT32_C(  -817.80), EASYSIMD_FLOAT32_C(   442.23), EASYSIMD_FLOAT32_C(  -530.12), EASYSIMD_FLOAT32_C(   987.30) },
      { EASYSIMD_FLOAT32_C(  -915.03), EASYSIMD_FLOAT32_C(   921.46), EASYSIMD_FLOAT32_C(   731.38), EASYSIMD_FLOAT32_C(   198.64) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0),           UINT32_MAX } },
    { { EASYSIMD_FLOAT32_C(   256.18), EASYSIMD_FLOAT32_C(   220.39), EASYSIMD_FLOAT32_C(  -453.64), EASYSIMD_FLOAT32_C(   264.67) },
      { EASYSIMD_FLOAT32_C(   594.64), EASYSIMD_FLOAT32_C(   189.87), EASYSIMD_FLOAT32_C(   113.62), EASYSIMD_FLOAT32_C(  -314.89) },
      { UINT32_C(         0),           UINT32_MAX,           UINT32_MAX, UINT32_C(         0) } },
    { { EASYSIMD_FLOAT32_C(    48.01), EASYSIMD_FLOAT32_C(   990.32), EASYSIMD_FLOAT32_C(  -232.76), EASYSIMD_FLOAT32_C(   259.86) },
      { EASYSIMD_FLOAT32_C(   729.55), EASYSIMD_FLOAT32_C(  -660.58), EASYSIMD_FLOAT32_C(   351.97), EASYSIMD_FLOAT32_C(   -33.86) },
      { UINT32_C(         0),           UINT32_MAX, UINT32_C(         0),           UINT32_MAX } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_uint32x4_t r = easysimd_vcagtq_f32(a, b);
    easysimd_test_arm_neon_assert_equal_u32x4(r, easysimd_vld1q_u32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_vcagtq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    uint64_t r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   230.31), EASYSIMD_FLOAT64_C(  -618.28) },
      { EASYSIMD_FLOAT64_C(   180.85), EASYSIMD_FLOAT64_C(   444.53) },
      {                    UINT64_MAX,                    UINT64_MAX } },
    { { EASYSIMD_FLOAT64_C(   217.53), EASYSIMD_FLOAT64_C(  -615.67) },
      { EASYSIMD_FLOAT64_C(   629.35), EASYSIMD_FLOAT64_C(  -484.75) },
      { UINT64_C(                   0),                    UINT64_MAX } },
    { { EASYSIMD_FLOAT64_C(   170.44), EASYSIMD_FLOAT64_C(  -454.09) },
      { EASYSIMD_FLOAT64_C(   330.58), EASYSIMD_FLOAT64_C(   520.13) },
      { UINT64_C(                   0), UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(  -764.76), EASYSIMD_FLOAT64_C(  -650.22) },
      { EASYSIMD_FLOAT64_C(   -78.50), EASYSIMD_FLOAT64_C(   683.38) },
      {                    UINT64_MAX, UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(  -812.10), EASYSIMD_FLOAT64_C(   401.95) },
      { EASYSIMD_FLOAT64_C(  -416.07), EASYSIMD_FLOAT64_C(   983.29) },
      {                    UINT64_MAX, UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(  -496.16), EASYSIMD_FLOAT64_C(   249.85) },
      { EASYSIMD_FLOAT64_C(    57.13), EASYSIMD_FLOAT64_C(  -909.73) },
      {                    UINT64_MAX, UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(  -537.53), EASYSIMD_FLOAT64_C(   707.06) },
      { EASYSIMD_FLOAT64_C(   -45.84), EASYSIMD_FLOAT64_C(  -807.07) },
      {                    UINT64_MAX, UINT64_C(                   0) } },
    { { EASYSIMD_FLOAT64_C(   -27.41), EASYSIMD_FLOAT64_C(   231.88) },
      { EASYSIMD_FLOAT64_C(  -442.67), EASYSIMD_FLOAT64_C(  -797.10) },
      { UINT64_C(                   0), UINT64_C(                   0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_uint64x2_t r = easysimd_vcagtq_f64(a, b);

    easysimd_test_arm_neon_assert_equal_u64x2(r, easysimd_vld1q_u64(test_vec[i].r));
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagth_f16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagts_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagtd_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagt_f16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagt_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagt_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagtq_f16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagtq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcagtq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
