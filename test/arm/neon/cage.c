#define EASYSIMD_TEST_ARM_NEON_INSN cage

#include "test-neon.h"
#include "../../../easysimd/arm/neon/cage.h"

static int
test_easysimd_vcageh_f16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a;
    easysimd_float32 b;
    uint16_t r;
  } test_vec[] = {
     { EASYSIMD_FLOAT32_C(   -68.07),
      EASYSIMD_FLOAT32_C(   -41.89),
           UINT16_MAX },
    { EASYSIMD_FLOAT32_C(    94.15),
      EASYSIMD_FLOAT32_C(   -23.64),
           UINT16_MAX },
    { EASYSIMD_FLOAT32_C(   -16.98),
      EASYSIMD_FLOAT32_C(    36.07),
      UINT16_C(    0) },
    { EASYSIMD_FLOAT32_C(     8.15),
      EASYSIMD_FLOAT32_C(   -14.55),
      UINT16_C(    0) },
    { EASYSIMD_FLOAT32_C(   -74.90),
      EASYSIMD_FLOAT32_C(    20.98),
           UINT16_MAX },
    { EASYSIMD_FLOAT32_C(   -93.79),
      EASYSIMD_FLOAT32_C(    28.02),
           UINT16_MAX },
    { EASYSIMD_FLOAT32_C(   -40.82),
      EASYSIMD_FLOAT32_C(    11.37),
           UINT16_MAX },
    { EASYSIMD_FLOAT32_C(    41.99),
      EASYSIMD_FLOAT32_C(    40.71),
           UINT16_MAX }    
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint16_t r = easysimd_vcageh_f16(easysimd_float16_from_float32(test_vec[i].a), easysimd_float16_from_float32(test_vec[i].b));

    easysimd_assert_equal_u16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-100.0f, 100.0f);
    easysimd_float32_t b = easysimd_test_codegen_random_f32(-100.0f, 100.0f);
    uint16_t r = easysimd_vcageh_f16(easysimd_float16_from_float32(a), easysimd_float16_from_float32(b));

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcages_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    uint32_t r = easysimd_vcages_f32(test_vec[i].a, test_vec[i].b);

    easysimd_assert_equal_u32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    easysimd_float32_t b = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    uint32_t r = easysimd_vcages_f32(a, b);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcaged_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    uint64_t r = easysimd_vcaged_f64(test_vec[i].a, test_vec[i].b);

    easysimd_assert_equal_u64(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64_t a = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    easysimd_float64_t b = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    uint64_t r = easysimd_vcaged_f64(a, b);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcage_f16 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
    struct {
      easysimd_float16 a[4];
      easysimd_float16 b[4];
      uint16_t r[4];
    } test_vec[] = {
      { { EASYSIMD_FLOAT16_VALUE(   506.50), EASYSIMD_FLOAT16_VALUE(   580.50), EASYSIMD_FLOAT16_VALUE(   209.88), EASYSIMD_FLOAT16_VALUE(  -273.25) },
      { EASYSIMD_FLOAT16_VALUE(  -451.25), EASYSIMD_FLOAT16_VALUE(  -948.00), EASYSIMD_FLOAT16_VALUE(   325.00), EASYSIMD_FLOAT16_VALUE(   577.50) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -715.50), EASYSIMD_FLOAT16_VALUE(  -305.50), EASYSIMD_FLOAT16_VALUE(  -358.25), EASYSIMD_FLOAT16_VALUE(     5.56) },
      { EASYSIMD_FLOAT16_VALUE(   466.75), EASYSIMD_FLOAT16_VALUE(   482.25), EASYSIMD_FLOAT16_VALUE(  -649.50), EASYSIMD_FLOAT16_VALUE(   274.00) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(   386.00), EASYSIMD_FLOAT16_VALUE(   -44.34), EASYSIMD_FLOAT16_VALUE(   -28.00), EASYSIMD_FLOAT16_VALUE(  -189.50) },
      { EASYSIMD_FLOAT16_VALUE(  -874.00), EASYSIMD_FLOAT16_VALUE(   179.12), EASYSIMD_FLOAT16_VALUE(   498.25), EASYSIMD_FLOAT16_VALUE(    26.06) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(  -109.38), EASYSIMD_FLOAT16_VALUE(  -715.50), EASYSIMD_FLOAT16_VALUE(   598.00), EASYSIMD_FLOAT16_VALUE(    66.88) },
      { EASYSIMD_FLOAT16_VALUE(   604.50), EASYSIMD_FLOAT16_VALUE(  -889.50), EASYSIMD_FLOAT16_VALUE(   -76.75), EASYSIMD_FLOAT16_VALUE(   111.31) },
      { UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(   691.00), EASYSIMD_FLOAT16_VALUE(  -867.00), EASYSIMD_FLOAT16_VALUE(   838.00), EASYSIMD_FLOAT16_VALUE(  -760.00) },
      { EASYSIMD_FLOAT16_VALUE(  -815.00), EASYSIMD_FLOAT16_VALUE(   163.00), EASYSIMD_FLOAT16_VALUE(   817.50), EASYSIMD_FLOAT16_VALUE(  -530.50) },
      { UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(   857.50), EASYSIMD_FLOAT16_VALUE(  -540.50), EASYSIMD_FLOAT16_VALUE(   475.25), EASYSIMD_FLOAT16_VALUE(   324.25) },
      { EASYSIMD_FLOAT16_VALUE(   941.50), EASYSIMD_FLOAT16_VALUE(   826.00), EASYSIMD_FLOAT16_VALUE(  -401.75), EASYSIMD_FLOAT16_VALUE(   327.75) },
      { UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -218.38), EASYSIMD_FLOAT16_VALUE(   570.00), EASYSIMD_FLOAT16_VALUE(  -862.00), EASYSIMD_FLOAT16_VALUE(   -92.25) },
      { EASYSIMD_FLOAT16_VALUE(  -250.75), EASYSIMD_FLOAT16_VALUE(   636.50), EASYSIMD_FLOAT16_VALUE(   934.00), EASYSIMD_FLOAT16_VALUE(   640.00) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(   920.50), EASYSIMD_FLOAT16_VALUE(   532.00), EASYSIMD_FLOAT16_VALUE(  -293.25), EASYSIMD_FLOAT16_VALUE(   525.50) },
      { EASYSIMD_FLOAT16_VALUE(   642.50), EASYSIMD_FLOAT16_VALUE(   630.00), EASYSIMD_FLOAT16_VALUE(  -363.25), EASYSIMD_FLOAT16_VALUE(   333.75) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX } }
    };

    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      easysimd_float16x4_t a = easysimd_vld1_f16(test_vec[i].a);
      easysimd_float16x4_t b = easysimd_vld1_f16(test_vec[i].b);
      easysimd_uint16x4_t r = easysimd_vcage_f16(a, b);

      easysimd_test_arm_neon_assert_equal_u16x4(r, easysimd_vld1_u16(test_vec[i].r));
    }

    return 0;
  #else
    fputc('\n', stdout);
    for (int i = 0 ; i < 8 ; i++) {
      easysimd_float16x4_t a = easysimd_test_arm_neon_random_f16x4(-1000.0f, 1000.0f);
      easysimd_float16x4_t b = easysimd_test_arm_neon_random_f16x4(-1000.0f, 1000.0f);
      easysimd_uint16x4_t r = easysimd_vcage_f16(a, b);

      easysimd_test_arm_neon_write_f16x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_arm_neon_write_f16x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_arm_neon_write_u16x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
    return 1;
  #endif
}

static int
test_easysimd_vcage_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    easysimd_uint32x2_t r = easysimd_vcage_f32(a, b);

    easysimd_test_arm_neon_assert_equal_u32x2(r, easysimd_vld1_u32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_vcage_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    easysimd_uint64x1_t r = easysimd_vcage_f64(a, b);

    easysimd_test_arm_neon_assert_equal_u64x1(r, easysimd_vld1_u64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_vcageq_f16 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
    struct {
      easysimd_float16 a[8];
      easysimd_float16 b[8];
      uint16_t r[8];
    } test_vec[] = {
      { { EASYSIMD_FLOAT16_VALUE(   476.50), EASYSIMD_FLOAT16_VALUE(   975.50), EASYSIMD_FLOAT16_VALUE(   915.00), EASYSIMD_FLOAT16_VALUE(  -632.50),
        EASYSIMD_FLOAT16_VALUE(  -472.75), EASYSIMD_FLOAT16_VALUE(   317.50), EASYSIMD_FLOAT16_VALUE(   621.50), EASYSIMD_FLOAT16_VALUE(   622.50) },
      { EASYSIMD_FLOAT16_VALUE(  -717.00), EASYSIMD_FLOAT16_VALUE(  -404.50), EASYSIMD_FLOAT16_VALUE(   444.00), EASYSIMD_FLOAT16_VALUE(  -493.50),
        EASYSIMD_FLOAT16_VALUE(  -270.75), EASYSIMD_FLOAT16_VALUE(   -59.31), EASYSIMD_FLOAT16_VALUE(  -330.25), EASYSIMD_FLOAT16_VALUE(   212.00) },
      { UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(  -169.50), EASYSIMD_FLOAT16_VALUE(  -542.50), EASYSIMD_FLOAT16_VALUE(  -149.88), EASYSIMD_FLOAT16_VALUE(  -427.25),
        EASYSIMD_FLOAT16_VALUE(  -861.00), EASYSIMD_FLOAT16_VALUE(  -977.50), EASYSIMD_FLOAT16_VALUE(   717.00), EASYSIMD_FLOAT16_VALUE(  -377.75) },
      { EASYSIMD_FLOAT16_VALUE(  -880.50), EASYSIMD_FLOAT16_VALUE(  -855.00), EASYSIMD_FLOAT16_VALUE(  -173.38), EASYSIMD_FLOAT16_VALUE(   725.50),
        EASYSIMD_FLOAT16_VALUE(   -76.69), EASYSIMD_FLOAT16_VALUE(  -541.00), EASYSIMD_FLOAT16_VALUE(   -72.81), EASYSIMD_FLOAT16_VALUE(  -600.00) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -566.00), EASYSIMD_FLOAT16_VALUE(  -157.62), EASYSIMD_FLOAT16_VALUE(  -232.50), EASYSIMD_FLOAT16_VALUE(   -38.72),
        EASYSIMD_FLOAT16_VALUE(  -840.50), EASYSIMD_FLOAT16_VALUE(  -611.00), EASYSIMD_FLOAT16_VALUE(  -416.50), EASYSIMD_FLOAT16_VALUE(  -557.50) },
      { EASYSIMD_FLOAT16_VALUE(   -15.61), EASYSIMD_FLOAT16_VALUE(  -972.50), EASYSIMD_FLOAT16_VALUE(   -50.97), EASYSIMD_FLOAT16_VALUE(   713.50),
        EASYSIMD_FLOAT16_VALUE(   -31.72), EASYSIMD_FLOAT16_VALUE(   619.00), EASYSIMD_FLOAT16_VALUE(   -74.44), EASYSIMD_FLOAT16_VALUE(   799.00) },
      {      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0) } },
    { { EASYSIMD_FLOAT16_VALUE(  -923.50), EASYSIMD_FLOAT16_VALUE(   775.50), EASYSIMD_FLOAT16_VALUE(  -628.50), EASYSIMD_FLOAT16_VALUE(  -784.50),
        EASYSIMD_FLOAT16_VALUE(   798.00), EASYSIMD_FLOAT16_VALUE(  -911.50), EASYSIMD_FLOAT16_VALUE(  -162.25), EASYSIMD_FLOAT16_VALUE(   917.50) },
      { EASYSIMD_FLOAT16_VALUE(  -766.50), EASYSIMD_FLOAT16_VALUE(   664.50), EASYSIMD_FLOAT16_VALUE(   643.50), EASYSIMD_FLOAT16_VALUE(   157.00),
        EASYSIMD_FLOAT16_VALUE(  -877.00), EASYSIMD_FLOAT16_VALUE(  -429.50), EASYSIMD_FLOAT16_VALUE(   557.00), EASYSIMD_FLOAT16_VALUE(  -442.75) },
      {      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(   413.00), EASYSIMD_FLOAT16_VALUE(  -675.50), EASYSIMD_FLOAT16_VALUE(   518.50), EASYSIMD_FLOAT16_VALUE(   572.50),
        EASYSIMD_FLOAT16_VALUE(  -286.75), EASYSIMD_FLOAT16_VALUE(  -898.00), EASYSIMD_FLOAT16_VALUE(  -985.00), EASYSIMD_FLOAT16_VALUE(   697.50) },
      { EASYSIMD_FLOAT16_VALUE(  -870.00), EASYSIMD_FLOAT16_VALUE(   -35.94), EASYSIMD_FLOAT16_VALUE(   411.25), EASYSIMD_FLOAT16_VALUE(    98.06),
        EASYSIMD_FLOAT16_VALUE(  -417.00), EASYSIMD_FLOAT16_VALUE(  -663.00), EASYSIMD_FLOAT16_VALUE(  -103.12), EASYSIMD_FLOAT16_VALUE(  -340.50) },
      { UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(  -887.50), EASYSIMD_FLOAT16_VALUE(   268.50), EASYSIMD_FLOAT16_VALUE(  -125.00), EASYSIMD_FLOAT16_VALUE(   910.50),
        EASYSIMD_FLOAT16_VALUE(   357.00), EASYSIMD_FLOAT16_VALUE(   712.50), EASYSIMD_FLOAT16_VALUE(   828.50), EASYSIMD_FLOAT16_VALUE(   591.00) },
      { EASYSIMD_FLOAT16_VALUE(   377.00), EASYSIMD_FLOAT16_VALUE(   471.75), EASYSIMD_FLOAT16_VALUE(  -252.25), EASYSIMD_FLOAT16_VALUE(   500.25),
        EASYSIMD_FLOAT16_VALUE(  -958.00), EASYSIMD_FLOAT16_VALUE(  -695.50), EASYSIMD_FLOAT16_VALUE(  -942.50), EASYSIMD_FLOAT16_VALUE(   455.25) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(  -371.00), EASYSIMD_FLOAT16_VALUE(   576.00), EASYSIMD_FLOAT16_VALUE(    27.72), EASYSIMD_FLOAT16_VALUE(   342.25),
        EASYSIMD_FLOAT16_VALUE(   678.50), EASYSIMD_FLOAT16_VALUE(    42.72), EASYSIMD_FLOAT16_VALUE(    40.00), EASYSIMD_FLOAT16_VALUE(   808.00) },
      { EASYSIMD_FLOAT16_VALUE(  -993.00), EASYSIMD_FLOAT16_VALUE(  -548.50), EASYSIMD_FLOAT16_VALUE(   -93.81), EASYSIMD_FLOAT16_VALUE(  -410.25),
        EASYSIMD_FLOAT16_VALUE(  -211.88), EASYSIMD_FLOAT16_VALUE(   803.00), EASYSIMD_FLOAT16_VALUE(   249.12), EASYSIMD_FLOAT16_VALUE(   -99.44) },
      { UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX } },
    { { EASYSIMD_FLOAT16_VALUE(    71.50), EASYSIMD_FLOAT16_VALUE(  -876.00), EASYSIMD_FLOAT16_VALUE(  -188.88), EASYSIMD_FLOAT16_VALUE(  -571.50),
        EASYSIMD_FLOAT16_VALUE(   837.00), EASYSIMD_FLOAT16_VALUE(  -360.50), EASYSIMD_FLOAT16_VALUE(  -980.50), EASYSIMD_FLOAT16_VALUE(   213.88) },
      { EASYSIMD_FLOAT16_VALUE(  -889.00), EASYSIMD_FLOAT16_VALUE(  -233.00), EASYSIMD_FLOAT16_VALUE(  -285.75), EASYSIMD_FLOAT16_VALUE(  -846.50),
        EASYSIMD_FLOAT16_VALUE(    71.56), EASYSIMD_FLOAT16_VALUE(  -228.25), EASYSIMD_FLOAT16_VALUE(   608.50), EASYSIMD_FLOAT16_VALUE(   700.50) },
      { UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } }
    };

    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      easysimd_float16x8_t a = easysimd_vld1q_f16(test_vec[i].a);
      easysimd_float16x8_t b = easysimd_vld1q_f16(test_vec[i].b);
      easysimd_uint16x8_t r = easysimd_vcageq_f16(a, b);
      easysimd_test_arm_neon_assert_equal_u16x8(r, easysimd_vld1q_u16(test_vec[i].r));
    }

    return 0;
  #else
    fputc('\n', stdout);
    for (int i = 0 ; i < 8 ; i++) {
      easysimd_float16x8_t a = easysimd_test_arm_neon_random_f16x8(-1000.0f, 1000.0f);
      easysimd_float16x8_t b = easysimd_test_arm_neon_random_f16x8(-1000.0f, 1000.0f);
      easysimd_uint16x8_t r = easysimd_vcageq_f16(a, b);

      easysimd_test_arm_neon_write_f16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_arm_neon_write_f16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_arm_neon_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
    return 1;
  #endif
}

static int
test_easysimd_vcageq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    easysimd_uint32x4_t r = easysimd_vcageq_f32(a, b);
    easysimd_test_arm_neon_assert_equal_u32x4(r, easysimd_vld1q_u32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_vcageq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
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
    easysimd_uint64x2_t r = easysimd_vcageq_f64(a, b);

    easysimd_test_arm_neon_assert_equal_u64x2(r, easysimd_vld1q_u64(test_vec[i].r));
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcageh_f16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcages_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcaged_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vcage_f16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcage_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcage_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vcageq_f16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcageq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcageq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
