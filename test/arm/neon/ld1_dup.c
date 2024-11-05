#define EASYSIMD_TEST_ARM_NEON_INSN ld1_dup

#include "test-neon.h"
#include "../../../easysimd/arm/neon/ld1_dup.h"

#if !defined(EASYSIMD_BUG_INTEL_857088)

static int
test_easysimd_vld1_dup_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    float a;
    float unused;
    float r[2];
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(  -662.67),
      EASYSIMD_FLOAT32_C(    17.22),
      { EASYSIMD_FLOAT32_C(  -662.67), EASYSIMD_FLOAT32_C(  -662.67) } },
    { EASYSIMD_FLOAT32_C(   654.51),
      EASYSIMD_FLOAT32_C(   873.43),
      { EASYSIMD_FLOAT32_C(   654.51), EASYSIMD_FLOAT32_C(   654.51) } },
    { EASYSIMD_FLOAT32_C(  -203.77),
      EASYSIMD_FLOAT32_C(   870.06),
      { EASYSIMD_FLOAT32_C(  -203.77), EASYSIMD_FLOAT32_C(  -203.77) } },
    { EASYSIMD_FLOAT32_C(   727.90),
      EASYSIMD_FLOAT32_C(  -797.12),
      { EASYSIMD_FLOAT32_C(   727.90), EASYSIMD_FLOAT32_C(   727.90) } },
    { EASYSIMD_FLOAT32_C(   -68.97),
      EASYSIMD_FLOAT32_C(  -923.20),
      { EASYSIMD_FLOAT32_C(   -68.97), EASYSIMD_FLOAT32_C(   -68.97) } },
    { EASYSIMD_FLOAT32_C(  -345.60),
      EASYSIMD_FLOAT32_C(  -686.50),
      { EASYSIMD_FLOAT32_C(  -345.60), EASYSIMD_FLOAT32_C(  -345.60) } },
    { EASYSIMD_FLOAT32_C(   638.84),
      EASYSIMD_FLOAT32_C(  -823.74),
      { EASYSIMD_FLOAT32_C(   638.84), EASYSIMD_FLOAT32_C(   638.84) } },
    { EASYSIMD_FLOAT32_C(  -581.13),
      EASYSIMD_FLOAT32_C(   639.86),
      { EASYSIMD_FLOAT32_C(  -581.13), EASYSIMD_FLOAT32_C(  -581.13) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t r = easysimd_vld1_dup_f32(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r),
                                           INT_MAX);
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32 a = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vld1_dup_f32(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    easysimd_float32 unused = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a;
    easysimd_float64 unused;
    easysimd_float64 r[1];
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(  -199.19),
      EASYSIMD_FLOAT64_C(  -652.54),
      { EASYSIMD_FLOAT64_C(  -199.19) } },
    { EASYSIMD_FLOAT64_C(    11.30),
      EASYSIMD_FLOAT64_C(  -621.52),
      { EASYSIMD_FLOAT64_C(    11.30) } },
    { EASYSIMD_FLOAT64_C(  -587.57),
      EASYSIMD_FLOAT64_C(  -210.68),
      { EASYSIMD_FLOAT64_C(  -587.57) } },
    { EASYSIMD_FLOAT64_C(  -223.20),
      EASYSIMD_FLOAT64_C(  -765.77),
      { EASYSIMD_FLOAT64_C(  -223.20) } },
    { EASYSIMD_FLOAT64_C(  -367.49),
      EASYSIMD_FLOAT64_C(  -137.16),
      { EASYSIMD_FLOAT64_C(  -367.49) } },
    { EASYSIMD_FLOAT64_C(  -246.44),
      EASYSIMD_FLOAT64_C(  -591.64),
      { EASYSIMD_FLOAT64_C(  -246.44) } },
    { EASYSIMD_FLOAT64_C(  -126.41),
      EASYSIMD_FLOAT64_C(  -142.21),
      { EASYSIMD_FLOAT64_C(  -126.41) } },
    { EASYSIMD_FLOAT64_C(   386.04),
      EASYSIMD_FLOAT64_C(   914.92),
      { EASYSIMD_FLOAT64_C(   386.04) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t r = easysimd_vld1_dup_f64(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r),
                                           INT_MAX);
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64 a = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vld1_dup_f64(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    easysimd_float64 unused = easysimd_test_codegen_random_f64(-1000.0, 1000.0);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_s8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int8_t a;
    int8_t unused;
    int8_t r[8];
  } test_vec[] = {
    {  INT8_C(  73),
       INT8_C(  15),
      {  INT8_C(  73),  INT8_C(  73),  INT8_C(  73),  INT8_C(  73),  INT8_C(  73),  INT8_C(  73),  INT8_C(  73),  INT8_C(  73) } },
    { -INT8_C( 124),
       INT8_C(  68),
      { -INT8_C( 124), -INT8_C( 124), -INT8_C( 124), -INT8_C( 124), -INT8_C( 124), -INT8_C( 124), -INT8_C( 124), -INT8_C( 124) } },
    {  INT8_C(  26),
       INT8_C(  85),
      {  INT8_C(  26),  INT8_C(  26),  INT8_C(  26),  INT8_C(  26),  INT8_C(  26),  INT8_C(  26),  INT8_C(  26),  INT8_C(  26) } },
    { -INT8_C(  89),
      -INT8_C(  30),
      { -INT8_C(  89), -INT8_C(  89), -INT8_C(  89), -INT8_C(  89), -INT8_C(  89), -INT8_C(  89), -INT8_C(  89), -INT8_C(  89) } },
    {  INT8_C(  76),
       INT8_C( 115),
      {  INT8_C(  76),  INT8_C(  76),  INT8_C(  76),  INT8_C(  76),  INT8_C(  76),  INT8_C(  76),  INT8_C(  76),  INT8_C(  76) } },
    { -INT8_C(  91),
       INT8_C(  77),
      { -INT8_C(  91), -INT8_C(  91), -INT8_C(  91), -INT8_C(  91), -INT8_C(  91), -INT8_C(  91), -INT8_C(  91), -INT8_C(  91) } },
    {  INT8_C(  29),
       INT8_C(  98),
      {  INT8_C(  29),  INT8_C(  29),  INT8_C(  29),  INT8_C(  29),  INT8_C(  29),  INT8_C(  29),  INT8_C(  29),  INT8_C(  29) } },
    { -INT8_C(  28),
       INT8_C(  80),
      { -INT8_C(  28), -INT8_C(  28), -INT8_C(  28), -INT8_C(  28), -INT8_C(  28), -INT8_C(  28), -INT8_C(  28), -INT8_C(  28) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int8x8_t r = easysimd_vld1_dup_s8(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i8x8(r,
                                           easysimd_vld1_s8(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a = easysimd_test_codegen_random_i8();
    easysimd_int8x8_t r = easysimd_vld1_dup_s8(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int8_t unused = easysimd_test_codegen_random_i8();

    easysimd_test_codegen_write_i8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i8(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i8x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_s16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int16_t a;
    int16_t unused;
    int16_t r[4];
  } test_vec[] = {
    { -INT16_C( 27597),
      -INT16_C(  9545),
      { -INT16_C( 27597), -INT16_C( 27597), -INT16_C( 27597), -INT16_C( 27597) } },
    {  INT16_C( 28663),
      -INT16_C( 25243),
      {  INT16_C( 28663),  INT16_C( 28663),  INT16_C( 28663),  INT16_C( 28663) } },
    { -INT16_C(  2220),
      -INT16_C( 10872),
      { -INT16_C(  2220), -INT16_C(  2220), -INT16_C(  2220), -INT16_C(  2220) } },
    { -INT16_C( 15825),
      -INT16_C( 27888),
      { -INT16_C( 15825), -INT16_C( 15825), -INT16_C( 15825), -INT16_C( 15825) } },
    {  INT16_C( 31110),
      -INT16_C(   164),
      {  INT16_C( 31110),  INT16_C( 31110),  INT16_C( 31110),  INT16_C( 31110) } },
    { -INT16_C(  1723),
       INT16_C( 30925),
      { -INT16_C(  1723), -INT16_C(  1723), -INT16_C(  1723), -INT16_C(  1723) } },
    {  INT16_C( 13821),
      -INT16_C( 11774),
      {  INT16_C( 13821),  INT16_C( 13821),  INT16_C( 13821),  INT16_C( 13821) } },
    { -INT16_C( 17883),
       INT16_C( 23030),
      { -INT16_C( 17883), -INT16_C( 17883), -INT16_C( 17883), -INT16_C( 17883) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int16x4_t r = easysimd_vld1_dup_s16(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i16x4(r,
                                           easysimd_vld1_s16(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a = easysimd_test_codegen_random_i16();
    easysimd_int16x4_t r = easysimd_vld1_dup_s16(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int16_t unused = easysimd_test_codegen_random_i16();

    easysimd_test_codegen_write_i16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i16(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i16x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_s32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int32_t a;
    int32_t unused;
    int32_t r[4];
  } test_vec[] = {
    {  INT32_C(  1464770501),
       INT32_C(  1347243241),
      {  INT32_C(  1464770501),  INT32_C(  1464770501) } },
    { -INT32_C(   966582799),
       INT32_C(  1680225793),
      { -INT32_C(   966582799), -INT32_C(   966582799) } },
    {  INT32_C(   208435532),
       INT32_C(  1415871482),
      {  INT32_C(   208435532),  INT32_C(   208435532) } },
    { -INT32_C(   568185185),
       INT32_C(  2063026613),
      { -INT32_C(   568185185), -INT32_C(   568185185) } },
    { -INT32_C(   774814232),
      -INT32_C(  2094915951),
      { -INT32_C(   774814232), -INT32_C(   774814232) } },
    {  INT32_C(  1028228412),
       INT32_C(   128020411),
      {  INT32_C(  1028228412),  INT32_C(  1028228412) } },
    { -INT32_C(   485290520),
       INT32_C(   674723976),
      { -INT32_C(   485290520), -INT32_C(   485290520) } },
    {  INT32_C(  1527142822),
      -INT32_C(  1948910174),
      {  INT32_C(  1527142822),  INT32_C(  1527142822) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int32x2_t r = easysimd_vld1_dup_s32(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i32x2(r,
                                           easysimd_vld1_s32(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a = easysimd_test_codegen_random_i32();
    easysimd_int32x2_t r = easysimd_vld1_dup_s32(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int32_t unused = easysimd_test_codegen_random_i32();

    easysimd_test_codegen_write_i32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i32(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_s64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int64_t a;
    int64_t unused;
    int64_t r[2];
  } test_vec[] = {
    {  INT64_C( 8743357196936080404),
      -INT64_C( 8148682543142269101),
      {  INT64_C( 8743357196936080404) } },
    {  INT64_C( 8533853554141708737),
      -INT64_C( 5683316659150636598),
      {  INT64_C( 8533853554141708737) } },
    { -INT64_C( 3224891244747629894),
      -INT64_C( 6564478166056840207),
      { -INT64_C( 3224891244747629894) } },
    { -INT64_C( 2767481697241876086),
       INT64_C( 7166672494563351591),
      { -INT64_C( 2767481697241876086) } },
    { -INT64_C( 3076125793732525571),
      -INT64_C( 9096967820244308422),
      { -INT64_C( 3076125793732525571) } },
    {  INT64_C( 6319740158625333708),
      -INT64_C( 3016408060519532778),
      {  INT64_C( 6319740158625333708) } },
    { -INT64_C( 1420143845125698671),
       INT64_C( 6326092421369788241),
      { -INT64_C( 1420143845125698671) } },
    { -INT64_C( 4858003396865397748),
      -INT64_C( 3364576772367798370),
      { -INT64_C( 4858003396865397748) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int64x1_t r = easysimd_vld1_dup_s64(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i64x1(r,
                                           easysimd_vld1_s64(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a = easysimd_test_codegen_random_i64();
    easysimd_int64x1_t r = easysimd_vld1_dup_s64(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int64_t unused = easysimd_test_codegen_random_i64();

    easysimd_test_codegen_write_i64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i64(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_u8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint8_t a;
    uint8_t unused;
    uint8_t r[8];
  } test_vec[] = {
    { UINT8_C(231),
      UINT8_C(217),
      { UINT8_C(231), UINT8_C(231), UINT8_C(231), UINT8_C(231), UINT8_C(231), UINT8_C(231), UINT8_C(231), UINT8_C(231) } },
    { UINT8_C(245),
      UINT8_C(157),
      { UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245) } },
    { UINT8_C(245),
      UINT8_C(131),
      { UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245), UINT8_C(245) } },
    { UINT8_C(185),
      UINT8_C(129),
      { UINT8_C(185), UINT8_C(185), UINT8_C(185), UINT8_C(185), UINT8_C(185), UINT8_C(185), UINT8_C(185), UINT8_C(185) } },
    { UINT8_C(125),
      UINT8_C(175),
      { UINT8_C(125), UINT8_C(125), UINT8_C(125), UINT8_C(125), UINT8_C(125), UINT8_C(125), UINT8_C(125), UINT8_C(125) } },
    { UINT8_C(177),
      UINT8_C( 91),
      { UINT8_C(177), UINT8_C(177), UINT8_C(177), UINT8_C(177), UINT8_C(177), UINT8_C(177), UINT8_C(177), UINT8_C(177) } },
    { UINT8_C(  1),
      UINT8_C( 55),
      { UINT8_C(  1), UINT8_C(  1), UINT8_C(  1), UINT8_C(  1), UINT8_C(  1), UINT8_C(  1), UINT8_C(  1), UINT8_C(  1) } },
    { UINT8_C(218),
      UINT8_C( 31),
      { UINT8_C(218), UINT8_C(218), UINT8_C(218), UINT8_C(218), UINT8_C(218), UINT8_C(218), UINT8_C(218), UINT8_C(218) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint8x8_t r = easysimd_vld1_dup_u8(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u8x8(r,
                                           easysimd_vld1_u8(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint8_t a = easysimd_test_codegen_random_u8();
    easysimd_uint8x8_t r = easysimd_vld1_dup_u8(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint8_t unused = easysimd_test_codegen_random_i8();

    easysimd_test_codegen_write_u8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u8x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_u16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint16_t a;
    uint16_t unused;
    uint16_t r[4];
  } test_vec[] = {
    { UINT16_C(31796),
      UINT16_C(36233),
      { UINT16_C(31796), UINT16_C(31796), UINT16_C(31796), UINT16_C(31796) } },
    { UINT16_C(51277),
      UINT16_C(42972),
      { UINT16_C(51277), UINT16_C(51277), UINT16_C(51277), UINT16_C(51277) } },
    { UINT16_C(39412),
      UINT16_C(27257),
      { UINT16_C(39412), UINT16_C(39412), UINT16_C(39412), UINT16_C(39412) } },
    { UINT16_C(51769),
      UINT16_C(15179),
      { UINT16_C(51769), UINT16_C(51769), UINT16_C(51769), UINT16_C(51769) } },
    { UINT16_C(35128),
      UINT16_C(46029),
      { UINT16_C(35128), UINT16_C(35128), UINT16_C(35128), UINT16_C(35128) } },
    { UINT16_C( 9030),
      UINT16_C( 9851),
      { UINT16_C( 9030), UINT16_C( 9030), UINT16_C( 9030), UINT16_C( 9030) } },
    { UINT16_C(20351),
      UINT16_C( 7861),
      { UINT16_C(20351), UINT16_C(20351), UINT16_C(20351), UINT16_C(20351) } },
    { UINT16_C( 8317),
      UINT16_C(45707),
      { UINT16_C( 8317), UINT16_C( 8317), UINT16_C( 8317), UINT16_C( 8317) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint16x4_t r = easysimd_vld1_dup_u16(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u16x4(r,
                                           easysimd_vld1_u16(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint16_t a = easysimd_test_codegen_random_u16();
    easysimd_uint16x4_t r = easysimd_vld1_dup_u16(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint16_t unused = easysimd_test_codegen_random_u16();

    easysimd_test_codegen_write_u16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint32_t a;
    uint32_t unused;
    uint32_t r[4];
  } test_vec[] = {
    { UINT32_C(3825720010),
      UINT32_C(3826244170),
      { UINT32_C(3825720010), UINT32_C(3825720010) } },
    { UINT32_C(1087586190),
      UINT32_C(3816690413),
      { UINT32_C(1087586190), UINT32_C(1087586190) } },
    { UINT32_C(2085795807),
      UINT32_C(3597975878),
      { UINT32_C(2085795807), UINT32_C(2085795807) } },
    { UINT32_C( 666002012),
      UINT32_C( 825250919),
      { UINT32_C( 666002012), UINT32_C( 666002012) } },
    { UINT32_C(2014721838),
      UINT32_C(2690393361),
      { UINT32_C(2014721838), UINT32_C(2014721838) } },
    { UINT32_C(1373646948),
      UINT32_C( 557145666),
      { UINT32_C(1373646948), UINT32_C(1373646948) } },
    { UINT32_C(1553893142),
      UINT32_C(2754744904),
      { UINT32_C(1553893142), UINT32_C(1553893142) } },
    { UINT32_C(3687572596),
      UINT32_C(1728903992),
      { UINT32_C(3687572596), UINT32_C(3687572596) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x2_t r = easysimd_vld1_dup_u32(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u32x2(r,
                                           easysimd_vld1_u32(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint32_t a = easysimd_test_codegen_random_u32();
    easysimd_uint32x2_t r = easysimd_vld1_dup_u32(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint32_t unused = easysimd_test_codegen_random_u32();

    easysimd_test_codegen_write_u32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u32(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1_dup_u64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint64_t a;
    uint64_t unused;
    uint64_t r[2];
  } test_vec[] = {
    { UINT64_C( 6656865874902462022),
      UINT64_C(16891003216436391159),
      { UINT64_C( 6656865874902462022) } },
    { UINT64_C( 2467426079629300125),
      UINT64_C( 9361357704487281973),
      { UINT64_C( 2467426079629300125) } },
    { UINT64_C(14700383420062623079),
      UINT64_C( 3388127016823008557),
      { UINT64_C(14700383420062623079) } },
    { UINT64_C( 2302103479159151581),
      UINT64_C(16580521545662613084),
      { UINT64_C( 2302103479159151581) } },
    { UINT64_C( 7377052567572641982),
      UINT64_C(17076952764489204539),
      { UINT64_C( 7377052567572641982) } },
    { UINT64_C( 9830141017426982241),
      UINT64_C(17318935631344130547),
      { UINT64_C( 9830141017426982241) } },
    { UINT64_C(12668875840935945521),
      UINT64_C( 4098973678832538087),
      { UINT64_C(12668875840935945521) } },
    { UINT64_C( 4513714357418630639),
      UINT64_C( 3200382560411697992),
      { UINT64_C( 4513714357418630639) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint64x1_t r = easysimd_vld1_dup_u64(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u64x1(r,
                                           easysimd_vld1_u64(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint64_t a = easysimd_test_codegen_random_u64();
    easysimd_uint64x1_t r = easysimd_vld1_dup_u64(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint64_t unused = easysimd_test_codegen_random_u64();

    easysimd_test_codegen_write_u64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u64(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    float a;
    float unused;
    float r[16];
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(  -275.48),
      EASYSIMD_FLOAT32_C(   563.30),
      { EASYSIMD_FLOAT32_C(  -275.48), EASYSIMD_FLOAT32_C(  -275.48), EASYSIMD_FLOAT32_C(  -275.48), EASYSIMD_FLOAT32_C(  -275.48) } },
    { EASYSIMD_FLOAT32_C(  -478.65),
      EASYSIMD_FLOAT32_C(   688.84),
      { EASYSIMD_FLOAT32_C(  -478.65), EASYSIMD_FLOAT32_C(  -478.65), EASYSIMD_FLOAT32_C(  -478.65), EASYSIMD_FLOAT32_C(  -478.65) } },
    { EASYSIMD_FLOAT32_C(   810.78),
      EASYSIMD_FLOAT32_C(  -456.08),
      { EASYSIMD_FLOAT32_C(   810.78), EASYSIMD_FLOAT32_C(   810.78), EASYSIMD_FLOAT32_C(   810.78), EASYSIMD_FLOAT32_C(   810.78) } },
    { EASYSIMD_FLOAT32_C(  -139.83),
      EASYSIMD_FLOAT32_C(  -302.96),
      { EASYSIMD_FLOAT32_C(  -139.83), EASYSIMD_FLOAT32_C(  -139.83), EASYSIMD_FLOAT32_C(  -139.83), EASYSIMD_FLOAT32_C(  -139.83) } },
    { EASYSIMD_FLOAT32_C(    16.91),
      EASYSIMD_FLOAT32_C(  -128.20),
      { EASYSIMD_FLOAT32_C(    16.91), EASYSIMD_FLOAT32_C(    16.91), EASYSIMD_FLOAT32_C(    16.91), EASYSIMD_FLOAT32_C(    16.91) } },
    { EASYSIMD_FLOAT32_C(   315.19),
      EASYSIMD_FLOAT32_C(  -659.25),
      { EASYSIMD_FLOAT32_C(   315.19), EASYSIMD_FLOAT32_C(   315.19), EASYSIMD_FLOAT32_C(   315.19), EASYSIMD_FLOAT32_C(   315.19) } },
    { EASYSIMD_FLOAT32_C(  -303.00),
      EASYSIMD_FLOAT32_C(  -865.13),
      { EASYSIMD_FLOAT32_C(  -303.00), EASYSIMD_FLOAT32_C(  -303.00), EASYSIMD_FLOAT32_C(  -303.00), EASYSIMD_FLOAT32_C(  -303.00) } },
    { EASYSIMD_FLOAT32_C(   391.93),
      EASYSIMD_FLOAT32_C(  -958.40),
      { EASYSIMD_FLOAT32_C(   391.93), EASYSIMD_FLOAT32_C(   391.93), EASYSIMD_FLOAT32_C(   391.93), EASYSIMD_FLOAT32_C(   391.93) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t r = easysimd_vld1q_dup_f32(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r),
                                           INT_MAX);
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32 a = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vld1q_dup_f32(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    easysimd_float32 unused = easysimd_test_codegen_random_f32(-1000.0f, 1000.0f);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a;
    easysimd_float64 unused;
    easysimd_float64 r[2];
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(   945.56),
      EASYSIMD_FLOAT64_C(   952.66),
      { EASYSIMD_FLOAT64_C(   945.56), EASYSIMD_FLOAT64_C(   945.56) } },
    { EASYSIMD_FLOAT64_C(   348.50),
      EASYSIMD_FLOAT64_C(   702.22),
      { EASYSIMD_FLOAT64_C(   348.50), EASYSIMD_FLOAT64_C(   348.50) } },
    { EASYSIMD_FLOAT64_C(   923.73),
      EASYSIMD_FLOAT64_C(   862.68),
      { EASYSIMD_FLOAT64_C(   923.73), EASYSIMD_FLOAT64_C(   923.73) } },
    { EASYSIMD_FLOAT64_C(  -143.77),
      EASYSIMD_FLOAT64_C(  -692.29),
      { EASYSIMD_FLOAT64_C(  -143.77), EASYSIMD_FLOAT64_C(  -143.77) } },
    { EASYSIMD_FLOAT64_C(   416.05),
      EASYSIMD_FLOAT64_C(   338.33),
      { EASYSIMD_FLOAT64_C(   416.05), EASYSIMD_FLOAT64_C(   416.05) } },
    { EASYSIMD_FLOAT64_C(  -126.60),
      EASYSIMD_FLOAT64_C(  -878.65),
      { EASYSIMD_FLOAT64_C(  -126.60), EASYSIMD_FLOAT64_C(  -126.60) } },
    { EASYSIMD_FLOAT64_C(   790.83),
      EASYSIMD_FLOAT64_C(   220.64),
      { EASYSIMD_FLOAT64_C(   790.83), EASYSIMD_FLOAT64_C(   790.83) } },
    { EASYSIMD_FLOAT64_C(   -50.24),
      EASYSIMD_FLOAT64_C(   252.99),
      { EASYSIMD_FLOAT64_C(   -50.24), EASYSIMD_FLOAT64_C(   -50.24) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t r = easysimd_vld1q_dup_f64(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r),
                                           INT_MAX);
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64 a = easysimd_test_codegen_random_f64(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vld1q_dup_f64(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    easysimd_float64 unused = easysimd_test_codegen_random_f64(-1000.0, 1000.0);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_s8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int8_t a;
    int8_t unused;
    int8_t r[16];
  } test_vec[] = {
    { -INT8_C(   6),
      -INT8_C(  98),
      { -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6),
        -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6), -INT8_C(   6) } },
    {  INT8_C(  10),
      -INT8_C(  19),
      {  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),
         INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10),  INT8_C(  10) } },
    {  INT8_C(  51),
      -INT8_C(  99),
      {  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),
         INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51),  INT8_C(  51) } },
    { -INT8_C(  62),
       INT8_C(  75),
      { -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62),
        -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62), -INT8_C(  62) } },
    { -INT8_C(  38),
      -INT8_C( 112),
      { -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38),
        -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38), -INT8_C(  38) } },
    { -INT8_C(  94),
      -INT8_C(  52),
      { -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94),
        -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94) } },
    { -INT8_C(  74),
       INT8_C( 103),
      { -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74),
        -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74), -INT8_C(  74) } },
    { -INT8_C(  77),
      -INT8_C(  89),
      { -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77),
        -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77), -INT8_C(  77) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int8x16_t r = easysimd_vld1q_dup_s8(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i8x16(r,
                                           easysimd_vld1q_s8(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a = easysimd_test_codegen_random_i8();
    easysimd_int8x16_t r = easysimd_vld1q_dup_s8(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int8_t unused = easysimd_test_codegen_random_i8();

    easysimd_test_codegen_write_i8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i8(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_s16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int16_t a;
    int16_t unused;
    int16_t r[16];
  } test_vec[] = {
    {  INT16_C( 25140),
       INT16_C( 27159),
      {  INT16_C( 25140),  INT16_C( 25140),  INT16_C( 25140),  INT16_C( 25140),  INT16_C( 25140),  INT16_C( 25140),  INT16_C( 25140),  INT16_C( 25140) } },
    {  INT16_C( 17366),
       INT16_C(  7763),
      {  INT16_C( 17366),  INT16_C( 17366),  INT16_C( 17366),  INT16_C( 17366),  INT16_C( 17366),  INT16_C( 17366),  INT16_C( 17366),  INT16_C( 17366) } },
    {  INT16_C( 28655),
       INT16_C( 17684),
      {  INT16_C( 28655),  INT16_C( 28655),  INT16_C( 28655),  INT16_C( 28655),  INT16_C( 28655),  INT16_C( 28655),  INT16_C( 28655),  INT16_C( 28655) } },
    { -INT16_C( 24572),
       INT16_C( 23778),
      { -INT16_C( 24572), -INT16_C( 24572), -INT16_C( 24572), -INT16_C( 24572), -INT16_C( 24572), -INT16_C( 24572), -INT16_C( 24572), -INT16_C( 24572) } },
    {  INT16_C( 11818),
       INT16_C( 21424),
      {  INT16_C( 11818),  INT16_C( 11818),  INT16_C( 11818),  INT16_C( 11818),  INT16_C( 11818),  INT16_C( 11818),  INT16_C( 11818),  INT16_C( 11818) } },
    {  INT16_C( 11802),
       INT16_C( 23800),
      {  INT16_C( 11802),  INT16_C( 11802),  INT16_C( 11802),  INT16_C( 11802),  INT16_C( 11802),  INT16_C( 11802),  INT16_C( 11802),  INT16_C( 11802) } },
    {  INT16_C( 15325),
       INT16_C( 30593),
      {  INT16_C( 15325),  INT16_C( 15325),  INT16_C( 15325),  INT16_C( 15325),  INT16_C( 15325),  INT16_C( 15325),  INT16_C( 15325),  INT16_C( 15325) } },
    { -INT16_C(  1781),
       INT16_C( 16340),
      { -INT16_C(  1781), -INT16_C(  1781), -INT16_C(  1781), -INT16_C(  1781), -INT16_C(  1781), -INT16_C(  1781), -INT16_C(  1781), -INT16_C(  1781) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int16x8_t r = easysimd_vld1q_dup_s16(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i16x8(r,
                                           easysimd_vld1q_s16(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a = easysimd_test_codegen_random_i16();
    easysimd_int16x8_t r = easysimd_vld1q_dup_s16(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int16_t unused = easysimd_test_codegen_random_i16();

    easysimd_test_codegen_write_i16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i16(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_s32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int32_t a;
    int32_t unused;
    int32_t r[16];
  } test_vec[] = {
    { -INT32_C(   184150047),
       INT32_C(   616751798),
      { -INT32_C(   184150047), -INT32_C(   184150047), -INT32_C(   184150047), -INT32_C(   184150047) } },
    {  INT32_C(  2117210528),
       INT32_C(  1709426408),
      {  INT32_C(  2117210528),  INT32_C(  2117210528),  INT32_C(  2117210528),  INT32_C(  2117210528) } },
    { -INT32_C(   300479163),
       INT32_C(   569423300),
      { -INT32_C(   300479163), -INT32_C(   300479163), -INT32_C(   300479163), -INT32_C(   300479163) } },
    { -INT32_C(   818696462),
       INT32_C(  1251895145),
      { -INT32_C(   818696462), -INT32_C(   818696462), -INT32_C(   818696462), -INT32_C(   818696462) } },
    {  INT32_C(   876651646),
       INT32_C(   643302022),
      {  INT32_C(   876651646),  INT32_C(   876651646),  INT32_C(   876651646),  INT32_C(   876651646) } },
    { -INT32_C(    56259820),
      -INT32_C(  1755215791),
      { -INT32_C(    56259820), -INT32_C(    56259820), -INT32_C(    56259820), -INT32_C(    56259820) } },
    {  INT32_C(  1518696597),
       INT32_C(   544961838),
      {  INT32_C(  1518696597),  INT32_C(  1518696597),  INT32_C(  1518696597),  INT32_C(  1518696597) } },
    { -INT32_C(  1913672157),
      -INT32_C(  1814590187),
      { -INT32_C(  1913672157), -INT32_C(  1913672157), -INT32_C(  1913672157), -INT32_C(  1913672157) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int32x4_t r = easysimd_vld1q_dup_s32(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i32x4(r,
                                           easysimd_vld1q_s32(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a = easysimd_test_codegen_random_i32();
    easysimd_int32x4_t r = easysimd_vld1q_dup_s32(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int32_t unused = easysimd_test_codegen_random_i32();

    easysimd_test_codegen_write_i32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i32(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_s64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    int64_t a;
    int64_t unused;
    int64_t r[16];
  } test_vec[] = {
    {  INT64_C( 8263121084250523414),
      -INT64_C(  133247727922472292),
      {  INT64_C( 8263121084250523414),  INT64_C( 8263121084250523414) } },
    { -INT64_C( 6063255998107405179),
       INT64_C( 2810408970900099590),
      { -INT64_C( 6063255998107405179), -INT64_C( 6063255998107405179) } },
    { -INT64_C( 1592455347479851877),
       INT64_C( 1510058888478061328),
      { -INT64_C( 1592455347479851877), -INT64_C( 1592455347479851877) } },
    { -INT64_C( 8901012275148193534),
      -INT64_C( 4915075221215212654),
      { -INT64_C( 8901012275148193534), -INT64_C( 8901012275148193534) } },
    {  INT64_C( 1884216627821516017),
      -INT64_C( 7013506924103545335),
      {  INT64_C( 1884216627821516017),  INT64_C( 1884216627821516017) } },
    { -INT64_C( 2051417869619818107),
      -INT64_C( 4445807296613274511),
      { -INT64_C( 2051417869619818107), -INT64_C( 2051417869619818107) } },
    { -INT64_C( 7274959835513258779),
      -INT64_C( 6616106579711091980),
      { -INT64_C( 7274959835513258779), -INT64_C( 7274959835513258779) } },
    { -INT64_C(  998620985170617361),
      -INT64_C( 2202592284581254181),
      { -INT64_C(  998620985170617361), -INT64_C(  998620985170617361) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int64x2_t r = easysimd_vld1q_dup_s64(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_i64x2(r,
                                           easysimd_vld1q_s64(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a = easysimd_test_codegen_random_i64();
    easysimd_int64x2_t r = easysimd_vld1q_dup_s64(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    int64_t unused = easysimd_test_codegen_random_i64();

    easysimd_test_codegen_write_i64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i64(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_u8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint8_t a;
    uint8_t unused;
    uint8_t r[16];
  } test_vec[] = {
    { UINT8_C( 49),
      UINT8_C(147),
      { UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49),
        UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49), UINT8_C( 49) } },
    { UINT8_C(153),
      UINT8_C(198),
      { UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153),
        UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153), UINT8_C(153) } },
    { UINT8_C(159),
      UINT8_C(134),
      { UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159),
        UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159), UINT8_C(159) } },
    { UINT8_C( 72),
      UINT8_C(166),
      { UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72),
        UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72), UINT8_C( 72) } },
    { UINT8_C(102),
      UINT8_C( 68),
      { UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102),
        UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102), UINT8_C(102) } },
    { UINT8_C(112),
      UINT8_C(230),
      { UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112),
        UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112), UINT8_C(112) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint8x16_t r = easysimd_vld1q_dup_u8(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u8x16(r,
                                           easysimd_vld1q_u8(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint8_t a = easysimd_test_codegen_random_u8();
    easysimd_uint8x16_t r = easysimd_vld1q_dup_u8(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint8_t unused = easysimd_test_codegen_random_u8();

    easysimd_test_codegen_write_u8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_u16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint16_t a;
    uint16_t unused;
    uint16_t r[16];
  } test_vec[] = {
    { UINT16_C( 4043),
      UINT16_C(61235),
      { UINT16_C( 4043), UINT16_C( 4043), UINT16_C( 4043), UINT16_C( 4043), UINT16_C( 4043), UINT16_C( 4043), UINT16_C( 4043), UINT16_C( 4043) } },
    { UINT16_C(48168),
      UINT16_C(22282),
      { UINT16_C(48168), UINT16_C(48168), UINT16_C(48168), UINT16_C(48168), UINT16_C(48168), UINT16_C(48168), UINT16_C(48168), UINT16_C(48168) } },
    { UINT16_C( 4777),
      UINT16_C( 5331),
      { UINT16_C( 4777), UINT16_C( 4777), UINT16_C( 4777), UINT16_C( 4777), UINT16_C( 4777), UINT16_C( 4777), UINT16_C( 4777), UINT16_C( 4777) } },
    { UINT16_C(45057),
      UINT16_C(39588),
      { UINT16_C(45057), UINT16_C(45057), UINT16_C(45057), UINT16_C(45057), UINT16_C(45057), UINT16_C(45057), UINT16_C(45057), UINT16_C(45057) } },
    { UINT16_C( 6709),
      UINT16_C(26419),
      { UINT16_C( 6709), UINT16_C( 6709), UINT16_C( 6709), UINT16_C( 6709), UINT16_C( 6709), UINT16_C( 6709), UINT16_C( 6709), UINT16_C( 6709) } },
    { UINT16_C(52397),
      UINT16_C(19757),
      { UINT16_C(52397), UINT16_C(52397), UINT16_C(52397), UINT16_C(52397), UINT16_C(52397), UINT16_C(52397), UINT16_C(52397), UINT16_C(52397) } },
    { UINT16_C(30290),
      UINT16_C(47603),
      { UINT16_C(30290), UINT16_C(30290), UINT16_C(30290), UINT16_C(30290), UINT16_C(30290), UINT16_C(30290), UINT16_C(30290), UINT16_C(30290) } },
    { UINT16_C(25530),
      UINT16_C(34207),
      { UINT16_C(25530), UINT16_C(25530), UINT16_C(25530), UINT16_C(25530), UINT16_C(25530), UINT16_C(25530), UINT16_C(25530), UINT16_C(25530) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint16x8_t r = easysimd_vld1q_dup_u16(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u16x8(r,
                                           easysimd_vld1q_u16(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint16_t a = easysimd_test_codegen_random_u16();
    easysimd_uint16x8_t r = easysimd_vld1q_dup_u16(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint16_t unused = easysimd_test_codegen_random_u16();

    easysimd_test_codegen_write_u16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint32_t a;
    uint32_t unused;
    uint32_t r[16];
  } test_vec[] = {
    { UINT32_C(2591347570),
      UINT32_C( 955416207),
      { UINT32_C(2591347570), UINT32_C(2591347570), UINT32_C(2591347570), UINT32_C(2591347570) } },
    { UINT32_C(2454570385),
      UINT32_C(2871849333),
      { UINT32_C(2454570385), UINT32_C(2454570385), UINT32_C(2454570385), UINT32_C(2454570385) } },
    { UINT32_C(3104988939),
      UINT32_C(2114338603),
      { UINT32_C(3104988939), UINT32_C(3104988939), UINT32_C(3104988939), UINT32_C(3104988939) } },
    { UINT32_C(1865939381),
      UINT32_C(3472217692),
      { UINT32_C(1865939381), UINT32_C(1865939381), UINT32_C(1865939381), UINT32_C(1865939381) } },
    { UINT32_C( 963144105),
      UINT32_C(2037471976),
      { UINT32_C( 963144105), UINT32_C( 963144105), UINT32_C( 963144105), UINT32_C( 963144105) } },
    { UINT32_C(2500574751),
      UINT32_C(3141547951),
      { UINT32_C(2500574751), UINT32_C(2500574751), UINT32_C(2500574751), UINT32_C(2500574751) } },
    { UINT32_C(3245626006),
      UINT32_C(1195342481),
      { UINT32_C(3245626006), UINT32_C(3245626006), UINT32_C(3245626006), UINT32_C(3245626006) } },
    { UINT32_C(3484841587),
      UINT32_C(4137528141),
      { UINT32_C(3484841587), UINT32_C(3484841587), UINT32_C(3484841587), UINT32_C(3484841587) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x4_t r = easysimd_vld1q_dup_u32(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u32x4(r,
                                           easysimd_vld1q_u32(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint32_t a = easysimd_test_codegen_random_u32();
    easysimd_uint32x4_t r = easysimd_vld1q_dup_u32(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint32_t unused = easysimd_test_codegen_random_u32();

    easysimd_test_codegen_write_u32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u32(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vld1q_dup_u64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint64_t a;
    uint64_t unused;
    uint64_t r[16];
  } test_vec[] = {
    { UINT64_C( 9184705928005354773),
      UINT64_C( 5677443429757649247),
      { UINT64_C( 9184705928005354773), UINT64_C( 9184705928005354773) } },
    { UINT64_C( 3134311518306254502),
      UINT64_C(17728868401062491589),
      { UINT64_C( 3134311518306254502), UINT64_C( 3134311518306254502) } },
    { UINT64_C( 4142302142246631580),
      UINT64_C(10083580343162015978),
      { UINT64_C( 4142302142246631580), UINT64_C( 4142302142246631580) } },
    { UINT64_C( 1455580007492747088),
      UINT64_C( 6939818071156993400),
      { UINT64_C( 1455580007492747088), UINT64_C( 1455580007492747088) } },
    { UINT64_C(10771160788369228392),
      UINT64_C( 2655478723087155817),
      { UINT64_C(10771160788369228392), UINT64_C(10771160788369228392) } },
    { UINT64_C( 6379735764790189350),
      UINT64_C( 5165911206702198668),
      { UINT64_C( 6379735764790189350), UINT64_C( 6379735764790189350) } },
    { UINT64_C( 5801203564664065602),
      UINT64_C(  485959815376386501),
      { UINT64_C( 5801203564664065602), UINT64_C( 5801203564664065602) } },
    { UINT64_C(15514342486347344662),
      UINT64_C( 4745343749585371060),
      { UINT64_C(15514342486347344662), UINT64_C(15514342486347344662) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint64x2_t r = easysimd_vld1q_dup_u64(&test_vec[i].a);

    easysimd_test_arm_neon_assert_equal_u64x2(r,
                                           easysimd_vld1q_u64(test_vec[i].r));
  }

  return 0;
#else
  for (int i = 0 ; i < 8 ; i++) {
    uint64_t a = easysimd_test_codegen_random_u64();
    easysimd_uint64x2_t r = easysimd_vld1q_dup_u64(&a);
    // Random buffer between pointer to load and expected results to check for
    // loading out of bounds.
    uint64_t unused = easysimd_test_codegen_random_u64();

    easysimd_test_codegen_write_u64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u64(2, unused, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */

EASYSIMD_TEST_FUNC_LIST_BEGIN
#if !defined(EASYSIMD_BUG_INTEL_857088)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_s8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_s16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_s32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_s64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_u8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_u16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_u32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1_dup_u64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_s8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_s16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_s32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_s64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_u8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_u16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_u32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vld1q_dup_u64)
#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
