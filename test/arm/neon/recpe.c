#define EASYSIMD_TEST_ARM_NEON_INSN recpe

#include "test-neon.h"
#include "../../../easysimd/arm/neon/recpe.h"

static int
test_easysimd_vrecpes_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a;
    easysimd_float32 r;
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(    -1.18),
      EASYSIMD_FLOAT32_C(    -0.85) },
    { EASYSIMD_FLOAT32_C(    -7.49),
      EASYSIMD_FLOAT32_C(    -0.13) },
    { EASYSIMD_FLOAT32_C(     4.04),
      EASYSIMD_FLOAT32_C(     0.25) },
    { EASYSIMD_FLOAT32_C(     1.26),
      EASYSIMD_FLOAT32_C(     0.79) },
    { EASYSIMD_FLOAT32_C(     8.86),
      EASYSIMD_FLOAT32_C(     0.11) },
    { EASYSIMD_FLOAT32_C(    -2.49),
      EASYSIMD_FLOAT32_C(    -0.40) },
    { EASYSIMD_FLOAT32_C(     2.79),
      EASYSIMD_FLOAT32_C(     0.36) },
    { EASYSIMD_FLOAT32_C(    -2.20),
      EASYSIMD_FLOAT32_C(    -0.46) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32_t a = test_vec[i].a;
    easysimd_float32_t r = easysimd_vrecpes_f32(a);

    easysimd_assert_equal_f32(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-10.0f, 10.0f);
    easysimd_float32_t r = easysimd_vrecpes_f32(a);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecped_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a;
    easysimd_float64 r;
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(     3.83),
      EASYSIMD_FLOAT64_C(     0.26) },
    { EASYSIMD_FLOAT64_C(     8.43),
      EASYSIMD_FLOAT64_C(     0.12) },
    { EASYSIMD_FLOAT64_C(    -8.88),
      EASYSIMD_FLOAT64_C(    -0.11) },
    { EASYSIMD_FLOAT64_C(     0.81),
      EASYSIMD_FLOAT64_C(     1.23) },
    { EASYSIMD_FLOAT64_C(     7.00),
      EASYSIMD_FLOAT64_C(     0.14) },
    { EASYSIMD_FLOAT64_C(     5.50),
      EASYSIMD_FLOAT64_C(     0.18) },
    { EASYSIMD_FLOAT64_C(     9.65),
      EASYSIMD_FLOAT64_C(     0.10) },
    { EASYSIMD_FLOAT64_C(    -1.78),
      EASYSIMD_FLOAT64_C(    -0.56) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64_t a = test_vec[i].a;
    easysimd_float64_t r = easysimd_vrecped_f64(a);

    easysimd_assert_equal_f64(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_float64_t a = easysimd_test_codegen_random_f64(-10.0, 10.0);
    easysimd_float64_t r = easysimd_vrecped_f64(a);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpe_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    -3.61), EASYSIMD_FLOAT32_C(    -8.68) },
      { EASYSIMD_FLOAT32_C(    -0.28), EASYSIMD_FLOAT32_C(    -0.12) } },
    { { EASYSIMD_FLOAT32_C(    -6.51), EASYSIMD_FLOAT32_C(    -7.63) },
      { EASYSIMD_FLOAT32_C(    -0.15), EASYSIMD_FLOAT32_C(    -0.13) } },
    { { EASYSIMD_FLOAT32_C(    -2.80), EASYSIMD_FLOAT32_C(    -7.27) },
      { EASYSIMD_FLOAT32_C(    -0.36), EASYSIMD_FLOAT32_C(    -0.14) } },
    { { EASYSIMD_FLOAT32_C(    -6.49), EASYSIMD_FLOAT32_C(    -7.56) },
      { EASYSIMD_FLOAT32_C(    -0.15), EASYSIMD_FLOAT32_C(    -0.13) } },
    { { EASYSIMD_FLOAT32_C(    -5.41), EASYSIMD_FLOAT32_C(    -0.72) },
      { EASYSIMD_FLOAT32_C(    -0.18), EASYSIMD_FLOAT32_C(    -1.39) } },
    { { EASYSIMD_FLOAT32_C(     8.89), EASYSIMD_FLOAT32_C(     2.37) },
      { EASYSIMD_FLOAT32_C(     0.11), EASYSIMD_FLOAT32_C(     0.42) } },
    { { EASYSIMD_FLOAT32_C(    -6.54), EASYSIMD_FLOAT32_C(     4.78) },
      { EASYSIMD_FLOAT32_C(    -0.15), EASYSIMD_FLOAT32_C(     0.21) } },
    { { EASYSIMD_FLOAT32_C(     7.48), EASYSIMD_FLOAT32_C(     5.64) },
      { EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.18) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t r = easysimd_vrecpe_f32(a);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-10.0f, 10.0f);
    easysimd_float32x2_t r = easysimd_vrecpe_f32(a);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpe_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a[1];
    easysimd_float64_t r[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(     0.29) },
      { EASYSIMD_FLOAT64_C(     3.45) } },
    { { EASYSIMD_FLOAT64_C(    -6.80) },
      { EASYSIMD_FLOAT64_C(    -0.15) } },
    { { EASYSIMD_FLOAT64_C(     4.19) },
      { EASYSIMD_FLOAT64_C(     0.24) } },
    { { EASYSIMD_FLOAT64_C(    -7.19) },
      { EASYSIMD_FLOAT64_C(    -0.14) } },
    { { EASYSIMD_FLOAT64_C(     7.23) },
      { EASYSIMD_FLOAT64_C(     0.14) } },
    { { EASYSIMD_FLOAT64_C(     9.23) },
      { EASYSIMD_FLOAT64_C(     0.11) } },
    { { EASYSIMD_FLOAT64_C(     8.28) },
      { EASYSIMD_FLOAT64_C(     0.12) } },
    { { EASYSIMD_FLOAT64_C(    -5.16) },
      { EASYSIMD_FLOAT64_C(    -0.19) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t r = easysimd_vrecpe_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-10.0, 10.0);
    easysimd_float64x1_t r = easysimd_vrecpe_f64(a);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpeq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a[2];
    easysimd_float64_t r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    -7.14), EASYSIMD_FLOAT64_C(     8.22) },
      { EASYSIMD_FLOAT64_C(    -0.14), EASYSIMD_FLOAT64_C(     0.12) } },
    { { EASYSIMD_FLOAT64_C(     5.72), EASYSIMD_FLOAT64_C(     4.54) },
      { EASYSIMD_FLOAT64_C(     0.17), EASYSIMD_FLOAT64_C(     0.22) } },
    { { EASYSIMD_FLOAT64_C(    -8.03), EASYSIMD_FLOAT64_C(    -1.08) },
      { EASYSIMD_FLOAT64_C(    -0.12), EASYSIMD_FLOAT64_C(    -0.93) } },
    { { EASYSIMD_FLOAT64_C(    -1.87), EASYSIMD_FLOAT64_C(     9.58) },
      { EASYSIMD_FLOAT64_C(    -0.54), EASYSIMD_FLOAT64_C(     0.10) } },
    { { EASYSIMD_FLOAT64_C(    -5.75), EASYSIMD_FLOAT64_C(     8.95) },
      { EASYSIMD_FLOAT64_C(    -0.17), EASYSIMD_FLOAT64_C(     0.11) } },
    { { EASYSIMD_FLOAT64_C(    -1.22), EASYSIMD_FLOAT64_C(     7.75) },
      { EASYSIMD_FLOAT64_C(    -0.82), EASYSIMD_FLOAT64_C(     0.13) } },
    { { EASYSIMD_FLOAT64_C(     3.96), EASYSIMD_FLOAT64_C(    -9.77) },
      { EASYSIMD_FLOAT64_C(     0.25), EASYSIMD_FLOAT64_C(    -0.10) } },
    { { EASYSIMD_FLOAT64_C(     3.77), EASYSIMD_FLOAT64_C(    -3.60) },
      { EASYSIMD_FLOAT64_C(     0.27), EASYSIMD_FLOAT64_C(    -0.28) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t r = easysimd_vrecpeq_f64(a);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-10.0, 10.0);
    easysimd_float64x2_t r = easysimd_vrecpeq_f64(a);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpeq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    -9.52), EASYSIMD_FLOAT32_C(     2.49), EASYSIMD_FLOAT32_C(     4.41), EASYSIMD_FLOAT32_C(     9.23) },
      { EASYSIMD_FLOAT32_C(    -0.10), EASYSIMD_FLOAT32_C(     0.40), EASYSIMD_FLOAT32_C(     0.23), EASYSIMD_FLOAT32_C(     0.11) } },
    { { EASYSIMD_FLOAT32_C(    -5.40), EASYSIMD_FLOAT32_C(    -6.01), EASYSIMD_FLOAT32_C(     4.48), EASYSIMD_FLOAT32_C(    -0.45) },
      { EASYSIMD_FLOAT32_C(    -0.19), EASYSIMD_FLOAT32_C(    -0.17), EASYSIMD_FLOAT32_C(     0.22), EASYSIMD_FLOAT32_C(    -2.23) } },
    { { EASYSIMD_FLOAT32_C(    -4.27), EASYSIMD_FLOAT32_C(    -5.87), EASYSIMD_FLOAT32_C(     4.11), EASYSIMD_FLOAT32_C(     0.21) },
      { EASYSIMD_FLOAT32_C(    -0.23), EASYSIMD_FLOAT32_C(    -0.17), EASYSIMD_FLOAT32_C(     0.24), EASYSIMD_FLOAT32_C(     4.75) } },
    { { EASYSIMD_FLOAT32_C(     6.69), EASYSIMD_FLOAT32_C(     7.78), EASYSIMD_FLOAT32_C(     8.38), EASYSIMD_FLOAT32_C(    -6.92) },
      { EASYSIMD_FLOAT32_C(     0.15), EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(    -0.14) } },
    { { EASYSIMD_FLOAT32_C(     9.10), EASYSIMD_FLOAT32_C(    -8.13), EASYSIMD_FLOAT32_C(    -4.56), EASYSIMD_FLOAT32_C(    -3.69) },
      { EASYSIMD_FLOAT32_C(     0.11), EASYSIMD_FLOAT32_C(    -0.12), EASYSIMD_FLOAT32_C(    -0.22), EASYSIMD_FLOAT32_C(    -0.27) } },
    { { EASYSIMD_FLOAT32_C(    -5.40), EASYSIMD_FLOAT32_C(    -1.04), EASYSIMD_FLOAT32_C(    -1.25), EASYSIMD_FLOAT32_C(    -0.81) },
      { EASYSIMD_FLOAT32_C(    -0.19), EASYSIMD_FLOAT32_C(    -0.96), EASYSIMD_FLOAT32_C(    -0.80), EASYSIMD_FLOAT32_C(    -1.23) } },
    { { EASYSIMD_FLOAT32_C(     8.23), EASYSIMD_FLOAT32_C(    -2.36), EASYSIMD_FLOAT32_C(    -8.44), EASYSIMD_FLOAT32_C(    -8.31) },
      { EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(    -0.42), EASYSIMD_FLOAT32_C(    -0.12), EASYSIMD_FLOAT32_C(    -0.12) } },
    { { EASYSIMD_FLOAT32_C(    -7.58), EASYSIMD_FLOAT32_C(     9.03), EASYSIMD_FLOAT32_C(     7.33), EASYSIMD_FLOAT32_C(    -7.10) },
      { EASYSIMD_FLOAT32_C(    -0.13), EASYSIMD_FLOAT32_C(     0.11), EASYSIMD_FLOAT32_C(     0.14), EASYSIMD_FLOAT32_C(    -0.14) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t r = easysimd_vrecpeq_f32(a);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-10.0f, 10.0f);
    easysimd_float32x4_t r = easysimd_vrecpeq_f32(a);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpe_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint32_t a[2];
    uint32_t r[2];
  } test_vec[] = {
    { { UINT32_C(3564426969), UINT32_C( 462037507) },
     { UINT32_C(2592079872),           UINT32_MAX } },
   { { UINT32_C(3406220423), UINT32_C(1732561961) },
     { UINT32_C(2701131776),           UINT32_MAX } },
   { { UINT32_C(3184077464), UINT32_C(2948966078) },
     { UINT32_C(2894069760), UINT32_C(3128950784) } },
   { { UINT32_C(1649294201), UINT32_C(4149110557) },
     {           UINT32_MAX, UINT32_C(2222981120) } },
   { { UINT32_C(1103872574), UINT32_C(1818056164) },
     {           UINT32_MAX,           UINT32_MAX } },
   { { UINT32_C(1261921057), UINT32_C(3367140144) },
     {           UINT32_MAX, UINT32_C(2734686208) } },
   { { UINT32_C(1669692325), UINT32_C(2584889889) },
     {           UINT32_MAX, UINT32_C(3565158400) } },
   { { UINT32_C(2734449029), UINT32_C( 110709448) },
     { UINT32_C(3380609024),           UINT32_MAX } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x2_t a = easysimd_vld1_u32(test_vec[i].a);
    easysimd_uint32x2_t r = easysimd_vrecpe_u32(a);

    easysimd_test_arm_neon_assert_equal_u32x2(r, easysimd_vld1_u32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_uint32x2_t a = easysimd_test_arm_neon_random_u32x2();
    easysimd_uint32x2_t r = easysimd_vrecpe_u32(a);

    easysimd_test_arm_neon_write_u32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vrecpeq_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    uint32_t a[4];
    uint32_t r[4];
  } test_vec[] = {
    { { UINT32_C(1460138119), UINT32_C(2503456530), UINT32_C(1265957790), UINT32_C(3090174878) },
      {           UINT32_MAX, UINT32_C(3682598912),           UINT32_MAX, UINT32_C(2986344448) } },
    { { UINT32_C( 620841409), UINT32_C(3775439020), UINT32_C(1866396079), UINT32_C(2739570204) },
      {           UINT32_MAX, UINT32_C(2441084928),           UINT32_MAX, UINT32_C(3363831808) } },
    { { UINT32_C(2298106486), UINT32_C(2820551177), UINT32_C(3438514733), UINT32_C(2810455013) },
      { UINT32_C(4018143232), UINT32_C(3271557120), UINT32_C(2684354560), UINT32_C(3279945728) } },
    { { UINT32_C( 416056684), UINT32_C(3573142565), UINT32_C(3779344325), UINT32_C( 881102526) },
      {           UINT32_MAX, UINT32_C(2583691264), UINT32_C(2441084928),           UINT32_MAX } },
    { { UINT32_C(3938287584), UINT32_C(3734166449), UINT32_C(1403684205), UINT32_C( 351940520) },
      { UINT32_C(2340421632), UINT32_C(2466250752),           UINT32_MAX,           UINT32_MAX } },
    { { UINT32_C(3660433076), UINT32_C(1622025883), UINT32_C( 474149470), UINT32_C(1632683649) },
      { UINT32_C(2516582400),           UINT32_MAX,           UINT32_MAX,           UINT32_MAX } },
    { { UINT32_C(4132113733), UINT32_C(1456856552), UINT32_C( 178880354), UINT32_C(1663017902) },
      { UINT32_C(2231369728),           UINT32_MAX,           UINT32_MAX,           UINT32_MAX } },
    { { UINT32_C(  71126121), UINT32_C(3496340338), UINT32_C(1609344990), UINT32_C(3015720301) },
      {           UINT32_MAX, UINT32_C(2642411520),           UINT32_MAX, UINT32_C(3061841920) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x4_t a = easysimd_vld1q_u32(test_vec[i].a);
    easysimd_uint32x4_t r = easysimd_vrecpeq_u32(a);

    easysimd_test_arm_neon_assert_equal_u32x4(r, easysimd_vld1q_u32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    // Keep the numbers small otherwise taking the reciprocal just gives 0.
    easysimd_uint32x4_t a = easysimd_test_arm_neon_random_u32x4();
    easysimd_uint32x4_t r = easysimd_vrecpeq_u32(a);

    easysimd_test_arm_neon_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpes_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecped_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpe_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpe_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpeq_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpe_u32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpeq_u32)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vrecpeq_f32)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
