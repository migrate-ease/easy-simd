
#define EASYSIMD_TEST_ARM_NEON_INSN rsqrte

#include "test-neon.h"
#include "../../../easysimd/arm/neon/rsqrte.h"

static int
test_easysimd_vrsqrtes_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a;
    easysimd_float32 r;
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(    33.60),
      EASYSIMD_FLOAT32_C(     0.17) },
    { EASYSIMD_FLOAT32_C(    85.84),
      EASYSIMD_FLOAT32_C(     0.11) },
    { EASYSIMD_FLOAT32_C(     3.60),
      EASYSIMD_FLOAT32_C(     0.53) },
    { EASYSIMD_FLOAT32_C(    82.23),
      EASYSIMD_FLOAT32_C(     0.11) },
    { EASYSIMD_FLOAT32_C(    13.21),
      EASYSIMD_FLOAT32_C(     0.28) },
    { EASYSIMD_FLOAT32_C(    98.57),
      EASYSIMD_FLOAT32_C(     0.10) },
    { EASYSIMD_FLOAT32_C(    20.89),
      EASYSIMD_FLOAT32_C(     0.22) },
    { EASYSIMD_FLOAT32_C(    58.72),
      EASYSIMD_FLOAT32_C(     0.13) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32_t a = test_vec[i].a;
    easysimd_float32_t r = easysimd_vrsqrtes_f32(a);

    easysimd_assert_equal_f32(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(0.0f, 100.0f);
    easysimd_float32_t r = easysimd_vrsqrtes_f32(a);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrted_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a;
    easysimd_float64 r;
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(    74.41),
      EASYSIMD_FLOAT64_C(     0.12) },
    { EASYSIMD_FLOAT64_C(    17.90),
      EASYSIMD_FLOAT64_C(     0.24) },
    { EASYSIMD_FLOAT64_C(     3.86),
      EASYSIMD_FLOAT64_C(     0.51) },
    { EASYSIMD_FLOAT64_C(    93.39),
      EASYSIMD_FLOAT64_C(     0.10) },
    { EASYSIMD_FLOAT64_C(    27.68),
      EASYSIMD_FLOAT64_C(     0.19) },
    { EASYSIMD_FLOAT64_C(    38.04),
      EASYSIMD_FLOAT64_C(     0.16) },
    { EASYSIMD_FLOAT64_C(     8.09),
      EASYSIMD_FLOAT64_C(     0.35) },
    { EASYSIMD_FLOAT64_C(     2.91),
      EASYSIMD_FLOAT64_C(     0.59) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64_t a = test_vec[i].a;
    easysimd_float64_t r = easysimd_vrsqrted_f64(a);

    easysimd_assert_equal_f64(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64_t a = easysimd_test_codegen_random_f64(0.0, 100.0);
    easysimd_float64_t r = easysimd_vrsqrted_f64(a);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrte_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    63.71), EASYSIMD_FLOAT32_C(    90.61) },
      { EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.11) } },
    { { EASYSIMD_FLOAT32_C(     6.73), EASYSIMD_FLOAT32_C(     1.48) },
      { EASYSIMD_FLOAT32_C(     0.39), EASYSIMD_FLOAT32_C(     0.82) } },
    { { EASYSIMD_FLOAT32_C(    25.93), EASYSIMD_FLOAT32_C(     0.58) },
      { EASYSIMD_FLOAT32_C(     0.20), EASYSIMD_FLOAT32_C(     1.31) } },
    { { EASYSIMD_FLOAT32_C(    61.27), EASYSIMD_FLOAT32_C(    46.52) },
      { EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.15) } },
    { { EASYSIMD_FLOAT32_C(    51.18), EASYSIMD_FLOAT32_C(    61.62) },
      { EASYSIMD_FLOAT32_C(     0.14), EASYSIMD_FLOAT32_C(     0.13) } },
    { { EASYSIMD_FLOAT32_C(    14.31), EASYSIMD_FLOAT32_C(    66.58) },
      { EASYSIMD_FLOAT32_C(     0.26), EASYSIMD_FLOAT32_C(     0.12) } },
    { { EASYSIMD_FLOAT32_C(    73.91), EASYSIMD_FLOAT32_C(    16.73) },
      { EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(     0.24) } },
    { { EASYSIMD_FLOAT32_C(    86.97), EASYSIMD_FLOAT32_C(    33.51) },
      { EASYSIMD_FLOAT32_C(     0.11), EASYSIMD_FLOAT32_C(     0.17) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t r = easysimd_vrsqrte_f32(a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(0.0f, 100.0f);
    easysimd_float32x2_t r = easysimd_vrsqrte_f32(a);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrte_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 r[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    45.08) },
      { EASYSIMD_FLOAT64_C(     0.15) } },
    { { EASYSIMD_FLOAT64_C(    22.53) },
      { EASYSIMD_FLOAT64_C(     0.21) } },
    { { EASYSIMD_FLOAT64_C(    77.74) },
      { EASYSIMD_FLOAT64_C(     0.11) } },
    { { EASYSIMD_FLOAT64_C(    86.85) },
      { EASYSIMD_FLOAT64_C(     0.11) } },
    { { EASYSIMD_FLOAT64_C(    42.05) },
      { EASYSIMD_FLOAT64_C(     0.15) } },
    { { EASYSIMD_FLOAT64_C(    56.86) },
      { EASYSIMD_FLOAT64_C(     0.13) } },
    { { EASYSIMD_FLOAT64_C(    29.17) },
      { EASYSIMD_FLOAT64_C(     0.19) } },
    { { EASYSIMD_FLOAT64_C(    24.93) },
      { EASYSIMD_FLOAT64_C(     0.20) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t r = easysimd_vrsqrte_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(0.0, 100.0);
    easysimd_float64x1_t r = easysimd_vrsqrte_f64(a);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrte_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint32_t a[2];
    uint32_t r[2];
  } test_vec[] = {
    { { UINT32_C( 124792316), UINT32_C( 707972624) },
      {           UINT32_MAX,           UINT32_MAX } },
    { { UINT32_C(2075341289), UINT32_C(3964530840) },
      { UINT32_C(3087007744), UINT32_C(2231369728) } },
    { { UINT32_C(3739131242), UINT32_C(1536833269) },
      { UINT32_C(2306867200), UINT32_C(3590324224) } },
    { { UINT32_C(3047417575), UINT32_C( 907668538) },
      { UINT32_C(2550136832),           UINT32_MAX } },
    { { UINT32_C( 708675865), UINT32_C(1096052568) },
      {           UINT32_MAX, UINT32_C(4253024256) } },
    { { UINT32_C( 951846816), UINT32_C(1629752055) },
      {           UINT32_MAX, UINT32_C(3481272320) } },
    { { UINT32_C(2638217895), UINT32_C( 553179705) },
      { UINT32_C(2734686208),           UINT32_MAX } },
    { { UINT32_C(4208303040), UINT32_C(2704338568) },
      { UINT32_C(2172649472), UINT32_C(2701131776) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x2_t a = easysimd_vld1_u32(test_vec[i].a);
    easysimd_uint32x2_t r = easysimd_vrsqrte_u32(a);

    easysimd_test_arm_neon_assert_equal_u32x2(r, easysimd_vld1_u32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint32x2_t a = easysimd_test_arm_neon_random_u32x2();
    easysimd_uint32x2_t r = easysimd_vrsqrte_u32(a);

    easysimd_test_arm_neon_write_u32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrteq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    54.34), EASYSIMD_FLOAT32_C(    78.91), EASYSIMD_FLOAT32_C(    73.82), EASYSIMD_FLOAT32_C(     9.39) },
      { EASYSIMD_FLOAT32_C(     0.14), EASYSIMD_FLOAT32_C(     0.11), EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(     0.33) } },
    { { EASYSIMD_FLOAT32_C(    73.60), EASYSIMD_FLOAT32_C(    67.42), EASYSIMD_FLOAT32_C(    10.95), EASYSIMD_FLOAT32_C(     2.53) },
      { EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(     0.30), EASYSIMD_FLOAT32_C(     0.63) } },
    { { EASYSIMD_FLOAT32_C(    57.12), EASYSIMD_FLOAT32_C(    47.91), EASYSIMD_FLOAT32_C(    10.51), EASYSIMD_FLOAT32_C(    90.53) },
      { EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.14), EASYSIMD_FLOAT32_C(     0.31), EASYSIMD_FLOAT32_C(     0.11) } },
    { { EASYSIMD_FLOAT32_C(    74.61), EASYSIMD_FLOAT32_C(    29.21), EASYSIMD_FLOAT32_C(    26.44), EASYSIMD_FLOAT32_C(    38.33) },
      { EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(     0.19), EASYSIMD_FLOAT32_C(     0.19), EASYSIMD_FLOAT32_C(     0.16) } },
    { { EASYSIMD_FLOAT32_C(    19.82), EASYSIMD_FLOAT32_C(    33.17), EASYSIMD_FLOAT32_C(    39.80), EASYSIMD_FLOAT32_C(    45.75) },
      { EASYSIMD_FLOAT32_C(     0.22), EASYSIMD_FLOAT32_C(     0.17), EASYSIMD_FLOAT32_C(     0.16), EASYSIMD_FLOAT32_C(     0.15) } },
    { { EASYSIMD_FLOAT32_C(    33.74), EASYSIMD_FLOAT32_C(     1.07), EASYSIMD_FLOAT32_C(    92.27), EASYSIMD_FLOAT32_C(    84.92) },
      { EASYSIMD_FLOAT32_C(     0.17), EASYSIMD_FLOAT32_C(     0.97), EASYSIMD_FLOAT32_C(     0.10), EASYSIMD_FLOAT32_C(     0.11) } },
    { { EASYSIMD_FLOAT32_C(    62.69), EASYSIMD_FLOAT32_C(     6.57), EASYSIMD_FLOAT32_C(    51.50), EASYSIMD_FLOAT32_C(    36.60) },
      { EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.39), EASYSIMD_FLOAT32_C(     0.14), EASYSIMD_FLOAT32_C(     0.17) } },
    { { EASYSIMD_FLOAT32_C(    23.31), EASYSIMD_FLOAT32_C(    38.48), EASYSIMD_FLOAT32_C(    70.11), EASYSIMD_FLOAT32_C(    77.65) },
      { EASYSIMD_FLOAT32_C(     0.21), EASYSIMD_FLOAT32_C(     0.16), EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(     0.11) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t r = easysimd_vrsqrteq_f32(a);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(0.0f, 100.0f);
    easysimd_float32x4_t r = easysimd_vrsqrteq_f32(a);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrteq_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint32_t a[4];
    uint32_t r[4];
  } test_vec[] = {
    { { UINT32_C(2479621035), UINT32_C(3682523218), UINT32_C(1969086320), UINT32_C(3328336721) },
      { UINT32_C(2826960896), UINT32_C(2315255808), UINT32_C(3170893824), UINT32_C(2441084928) } },
    { { UINT32_C(1171993373), UINT32_C( 360332709), UINT32_C( 965093525), UINT32_C(4206131278) },
      { UINT32_C(4110417920),           UINT32_MAX,           UINT32_MAX, UINT32_C(2172649472) } },
    { { UINT32_C(3448603003), UINT32_C(3383233369), UINT32_C( 993920746), UINT32_C(2265096553) },
      { UINT32_C(2399141888), UINT32_C(2424307712),           UINT32_MAX, UINT32_C(2952790016) } },
    { { UINT32_C(2043469268), UINT32_C(2962114074), UINT32_C(3186168943), UINT32_C( 129474188) },
      { UINT32_C(3112173568), UINT32_C(2583691264), UINT32_C(2499805184),           UINT32_MAX } },
    { { UINT32_C(2027176991), UINT32_C( 977435983), UINT32_C(3967123587), UINT32_C(4117985057) },
      { UINT32_C(3128950784),           UINT32_MAX, UINT32_C(2231369728), UINT32_C(2189426688) } },
    { { UINT32_C(1869496148), UINT32_C(4112514182), UINT32_C(2628913168), UINT32_C(3315821222) },
      { UINT32_C(3254779904), UINT32_C(2189426688), UINT32_C(2743074816), UINT32_C(2441084928) } },
    { { UINT32_C(4265506990), UINT32_C(2016968949), UINT32_C( 577023232), UINT32_C(2031605797) },
      { UINT32_C(2155872256), UINT32_C(3137339392),           UINT32_MAX, UINT32_C(3120562176) } },
    { { UINT32_C(2649261591), UINT32_C(2475820930), UINT32_C(3056551184), UINT32_C(1568461743) },
      { UINT32_C(2734686208), UINT32_C(2826960896), UINT32_C(2541748224), UINT32_C(3556769792) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x4_t a = easysimd_vld1q_u32(test_vec[i].a);
    easysimd_uint32x4_t r = easysimd_vrsqrteq_u32(a);

    easysimd_test_arm_neon_assert_equal_u32x4(r, easysimd_vld1q_u32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint32x4_t a = easysimd_test_arm_neon_random_u32x4();
    easysimd_uint32x4_t r = easysimd_vrsqrteq_u32(a);

    easysimd_test_arm_neon_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrsqrteq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    23.63), EASYSIMD_FLOAT64_C(    16.02) },
      { EASYSIMD_FLOAT64_C(     0.21), EASYSIMD_FLOAT64_C(     0.25) } },
    { { EASYSIMD_FLOAT64_C(    58.09), EASYSIMD_FLOAT64_C(    45.27) },
      { EASYSIMD_FLOAT64_C(     0.13), EASYSIMD_FLOAT64_C(     0.15) } },
    { { EASYSIMD_FLOAT64_C(    10.31), EASYSIMD_FLOAT64_C(    45.18) },
      { EASYSIMD_FLOAT64_C(     0.31), EASYSIMD_FLOAT64_C(     0.15) } },
    { { EASYSIMD_FLOAT64_C(    34.52), EASYSIMD_FLOAT64_C(    70.06) },
      { EASYSIMD_FLOAT64_C(     0.17), EASYSIMD_FLOAT64_C(     0.12) } },
    { { EASYSIMD_FLOAT64_C(    49.78), EASYSIMD_FLOAT64_C(    54.92) },
      { EASYSIMD_FLOAT64_C(     0.14), EASYSIMD_FLOAT64_C(     0.13) } },
    { { EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(    65.66) },
      { EASYSIMD_FLOAT64_C(     0.11), EASYSIMD_FLOAT64_C(     0.12) } },
    { { EASYSIMD_FLOAT64_C(    30.85), EASYSIMD_FLOAT64_C(    79.17) },
      { EASYSIMD_FLOAT64_C(     0.18), EASYSIMD_FLOAT64_C(     0.11) } },
    { { EASYSIMD_FLOAT64_C(    71.16), EASYSIMD_FLOAT64_C(    24.47) },
      { EASYSIMD_FLOAT64_C(     0.12), EASYSIMD_FLOAT64_C(     0.20) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t r = easysimd_vrsqrteq_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(0.0, 100.0);
    easysimd_float64x2_t r = easysimd_vrsqrteq_f64(a);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrtes_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrted_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrte_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrte_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrte_u32)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrteq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrteq_u32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrsqrteq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
