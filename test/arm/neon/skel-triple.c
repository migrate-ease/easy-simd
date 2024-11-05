#define EASYSIMD_TEST_ARM_NEON_INSN xxx

#include "test-neon.h"
#include "../../../easysimd/arm/neon/xxx.h"

static int
test_easysimd_vxxx_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    easysimd_float32 a[2];
    easysimd_float32 b[2];
    easysimd_float32 c[2];
    easysimd_float32 r[2];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t c = easysimd_vld1_f32(test_vec[i].c);
    easysimd_float32x2_t r = easysimd_vxxx_f32(a, b, c);

    easysimd_test_arm_neon_assert_equal_f32x2(r, easysimd_vld1_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t c = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_vxxx_f32(a, b, c);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    easysimd_float64 a[1];
    easysimd_float64 b[1];
    easysimd_float64 c[1];
    easysimd_float64 r[1];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_float64x1_t c = easysimd_vld1_f64(test_vec[i].c);
    easysimd_float64x1_t r = easysimd_vxxx_f64(a, b, c);

    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t b = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t c = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vxxx_f64(a, b, c);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_s8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int8_t a[8];
    int8_t b[8];
    int8_t c[8];
    int8_t r[8];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int8x8_t a = easysimd_vld1_s8(test_vec[i].a);
    easysimd_int8x8_t b = easysimd_vld1_s8(test_vec[i].b);
    easysimd_int8x8_t c = easysimd_vld1_s8(test_vec[i].c);
    easysimd_int8x8_t r = easysimd_vxxx_s8(a, b, c);

    easysimd_test_arm_neon_assert_equal_i8x8(r, easysimd_vld1_s8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int8x8_t a = easysimd_test_arm_neon_random_i8x8();
    easysimd_int8x8_t b = easysimd_test_arm_neon_random_i8x8();
    easysimd_int8x8_t c = easysimd_test_arm_neon_random_i8x8();
    easysimd_int8x8_t r = easysimd_vxxx_s8(a, b, c);

    easysimd_test_arm_neon_write_i8x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i8x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i8x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i8x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_s16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int16_t a[4];
    int16_t b[4];
    int16_t c[4];
    int16_t r[4];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int16x4_t a = easysimd_vld1_s16(test_vec[i].a);
    easysimd_int16x4_t b = easysimd_vld1_s16(test_vec[i].b);
    easysimd_int16x4_t c = easysimd_vld1_s16(test_vec[i].c);
    easysimd_int16x4_t r = easysimd_vxxx_s16(a, b, c);

    easysimd_test_arm_neon_assert_equal_i16x4(r, easysimd_vld1_s16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int16x4_t a = easysimd_test_arm_neon_random_i16x4();
    easysimd_int16x4_t b = easysimd_test_arm_neon_random_i16x4();
    easysimd_int16x4_t c = easysimd_test_arm_neon_random_i16x4();
    easysimd_int16x4_t r = easysimd_vxxx_s16(a, b, c);

    easysimd_test_arm_neon_write_i16x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i16x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i16x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i16x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_s32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int32_t a[2];
    int32_t b[2];
    int32_t c[2];
    int32_t r[2];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int32x2_t a = easysimd_vld1_s32(test_vec[i].a);
    easysimd_int32x2_t b = easysimd_vld1_s32(test_vec[i].b);
    easysimd_int32x2_t c = easysimd_vld1_s32(test_vec[i].c);
    easysimd_int32x2_t r = easysimd_vxxx_s32(a, b, c);

    easysimd_test_arm_neon_assert_equal_i32x2(r, easysimd_vld1_s32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int32x2_t a = easysimd_test_arm_neon_random_i32x2();
    easysimd_int32x2_t b = easysimd_test_arm_neon_random_i32x2();
    easysimd_int32x2_t c = easysimd_test_arm_neon_random_i32x2();
    easysimd_int32x2_t r = easysimd_vxxx_s32(a, b, c);

    easysimd_test_arm_neon_write_i32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i32x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_s64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int64_t a[1];
    int64_t b[1];
    int64_t c[1];
    int64_t r[1];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int64x1_t a = easysimd_vld1_s64(test_vec[i].a);
    easysimd_int64x1_t b = easysimd_vld1_s64(test_vec[i].b);
    easysimd_int64x1_t c = easysimd_vld1_s64(test_vec[i].c);
    easysimd_int64x1_t r = easysimd_vxxx_s64(a, b, c);

    easysimd_test_arm_neon_assert_equal_i64x1(r, easysimd_vld1_s64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int64x1_t a = easysimd_test_arm_neon_random_i64x1();
    easysimd_int64x1_t b = easysimd_test_arm_neon_random_i64x1();
    easysimd_int64x1_t c = easysimd_test_arm_neon_random_i64x1();
    easysimd_int64x1_t r = easysimd_vxxx_s64(a, b, c);

    easysimd_test_arm_neon_write_i64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i64x1(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_u8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint8_t a[8];
    uint8_t b[8];
    uint8_t c[8];
    uint8_t r[8];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint8x8_t a = easysimd_vld1_u8(test_vec[i].a);
    easysimd_uint8x8_t b = easysimd_vld1_u8(test_vec[i].b);
    easysimd_uint8x8_t c = easysimd_vld1_u8(test_vec[i].c);
    easysimd_uint8x8_t r = easysimd_vxxx_u8(a, b, c);

    easysimd_test_arm_neon_assert_equal_u8x8(r, easysimd_vld1_u8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint8x8_t a = easysimd_test_arm_neon_random_u8x8();
    easysimd_uint8x8_t b = easysimd_test_arm_neon_random_u8x8();
    easysimd_uint8x8_t c = easysimd_test_arm_neon_random_u8x8();
    easysimd_uint8x8_t r = easysimd_vxxx_u8(a, b, c);

    easysimd_test_arm_neon_write_u8x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u8x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u8x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u8x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_u16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint16_t a[4];
    uint16_t b[4];
    uint16_t c[4];
    uint16_t r[4];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint16x4_t a = easysimd_vld1_u16(test_vec[i].a);
    easysimd_uint16x4_t b = easysimd_vld1_u16(test_vec[i].b);
    easysimd_uint16x4_t c = easysimd_vld1_u16(test_vec[i].c);
    easysimd_uint16x4_t r = easysimd_vxxx_u16(a, b, c);
    easysimd_test_arm_neon_assert_equal_u16x4(r, easysimd_vld1_u16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint16x4_t a = easysimd_test_arm_neon_random_u16x4();
    easysimd_uint16x4_t b = easysimd_test_arm_neon_random_u16x4();
    easysimd_uint16x4_t c = easysimd_test_arm_neon_random_u16x4();
    easysimd_uint16x4_t r = easysimd_vxxx_u16(a, b, c);

    easysimd_test_arm_neon_write_u16x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u16x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint32_t a[2];
    uint32_t b[2];
    uint32_t c[2];
    uint32_t r[2];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x2_t a = easysimd_vld1_u32(test_vec[i].a);
    easysimd_uint32x2_t b = easysimd_vld1_u32(test_vec[i].b);
    easysimd_uint32x2_t c = easysimd_vld1_u32(test_vec[i].c);
    easysimd_uint32x2_t r = easysimd_vxxx_u32(a, b, c);
    easysimd_test_arm_neon_assert_equal_u32x2(r, easysimd_vld1_u32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint32x2_t a = easysimd_test_arm_neon_random_u32x2();
    easysimd_uint32x2_t b = easysimd_test_arm_neon_random_u32x2();
    easysimd_uint32x2_t c = easysimd_test_arm_neon_random_u32x2();
    easysimd_uint32x2_t r = easysimd_vxxx_u32(a, b, c);

    easysimd_test_arm_neon_write_u32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u32x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u32x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxx_u64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint64_t a[1];
    uint64_t b[1];
    uint64_t c[1];
    uint64_t r[1];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint64x1_t a = easysimd_vld1_u64(test_vec[i].a);
    easysimd_uint64x1_t b = easysimd_vld1_u64(test_vec[i].b);
    easysimd_uint64x1_t c = easysimd_vld1_u64(test_vec[i].c);
    easysimd_uint64x1_t r = easysimd_vxxx_u64(a, b, c);
    easysimd_test_arm_neon_assert_equal_u64x1(r, easysimd_vld1_u64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint64x1_t a = easysimd_test_arm_neon_random_u64x1();
    easysimd_uint64x1_t b = easysimd_test_arm_neon_random_u64x1();
    easysimd_uint64x1_t c = easysimd_test_arm_neon_random_u64x1();
    easysimd_uint64x1_t r = easysimd_vxxx_u64(a, b, c);

    easysimd_test_arm_neon_write_u64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u64x1(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 c[4];
    easysimd_float32 r[4];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t c = easysimd_vld1q_f32(test_vec[i].c);
    easysimd_float32x4_t r = easysimd_vxxxq_f32(a, b, c);
    easysimd_test_arm_neon_assert_equal_f32x4(r, easysimd_vld1q_f32(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t c = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_vxxxq_f32(a, b, c);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 c[2];
    easysimd_float64 r[2];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t c = easysimd_vld1q_f64(test_vec[i].c);
    easysimd_float64x2_t r = easysimd_vxxxq_f64(a, b, c);

    easysimd_test_arm_neon_assert_equal_f64x2(r, easysimd_vld1q_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t c = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r = easysimd_vxxxq_f64(a, b, c);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_s8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int8_t a[16];
    int8_t b[16];
    int8_t c[16];
    int8_t r[16];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int8x16_t a = easysimd_vld1q_s8(test_vec[i].a);
    easysimd_int8x16_t b = easysimd_vld1q_s8(test_vec[i].b);
    easysimd_int8x16_t c = easysimd_vld1q_s8(test_vec[i].c);
    easysimd_int8x16_t r = easysimd_vxxxq_s8(a, b, c);

    easysimd_test_arm_neon_assert_equal_i8x16(r, easysimd_vld1q_s8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int8x16_t a = easysimd_test_arm_neon_random_i8x16();
    easysimd_int8x16_t b = easysimd_test_arm_neon_random_i8x16();
    easysimd_int8x16_t c = easysimd_test_arm_neon_random_i8x16();
    easysimd_int8x16_t r = easysimd_vxxxq_s8(a, b, c);

    easysimd_test_arm_neon_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i8x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_s16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int16_t a[8];
    int16_t b[8];
    int16_t c[8];
    int16_t r[8];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int16x8_t a = easysimd_vld1q_s16(test_vec[i].a);
    easysimd_int16x8_t b = easysimd_vld1q_s16(test_vec[i].b);
    easysimd_int16x8_t c = easysimd_vld1q_s16(test_vec[i].c);
    easysimd_int16x8_t r = easysimd_vxxxq_s16(a, b, c);

    easysimd_test_arm_neon_assert_equal_i16x8(r, easysimd_vld1q_s16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int16x8_t a = easysimd_test_arm_neon_random_i16x8();
    easysimd_int16x8_t b = easysimd_test_arm_neon_random_i16x8();
    easysimd_int16x8_t c = easysimd_test_arm_neon_random_i16x8();
    easysimd_int16x8_t r = easysimd_vxxxq_s16(a, b, c);

    easysimd_test_arm_neon_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i16x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_s32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int32_t a[4];
    int32_t b[4];
    int32_t c[4];
    int32_t r[4];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int32x4_t a = easysimd_vld1q_s32(test_vec[i].a);
    easysimd_int32x4_t b = easysimd_vld1q_s32(test_vec[i].b);
    easysimd_int32x4_t c = easysimd_vld1q_s32(test_vec[i].c);
    easysimd_int32x4_t r = easysimd_vxxxq_s32(a, b, c);
    easysimd_test_arm_neon_assert_equal_i32x4(r, easysimd_vld1q_s32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int32x4_t a = easysimd_test_arm_neon_random_i32x4();
    easysimd_int32x4_t b = easysimd_test_arm_neon_random_i32x4();
    easysimd_int32x4_t c = easysimd_test_arm_neon_random_i32x4();
    easysimd_int32x4_t r = easysimd_vxxxq_s32(a, b, c);

    easysimd_test_arm_neon_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_s64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    int64_t a[2];
    int64_t b[2];
    int64_t c[2];
    int64_t r[2];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_int64x2_t a = easysimd_vld1q_s64(test_vec[i].a);
    easysimd_int64x2_t b = easysimd_vld1q_s64(test_vec[i].b);
    easysimd_int64x2_t c = easysimd_vld1q_s64(test_vec[i].c);
    easysimd_int64x2_t r = easysimd_vxxxq_s64(a, b, c);
    easysimd_test_arm_neon_assert_equal_i64x2(r, easysimd_vld1q_s64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_int64x2_t a = easysimd_test_arm_neon_random_i64x2();
    easysimd_int64x2_t b = easysimd_test_arm_neon_random_i64x2();
    easysimd_int64x2_t c = easysimd_test_arm_neon_random_i64x2();
    easysimd_int64x2_t r = easysimd_vxxxq_s64(a, b, c);

    easysimd_test_arm_neon_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_u8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint8_t a[16];
    uint8_t b[16];
    uint8_t c[16];
    uint8_t r[16];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint8x16_t a = easysimd_vld1q_u8(test_vec[i].a);
    easysimd_uint8x16_t b = easysimd_vld1q_u8(test_vec[i].b);
    easysimd_uint8x16_t c = easysimd_vld1q_u8(test_vec[i].c);
    easysimd_uint8x16_t r = easysimd_vxxxq_u8(a, b, c);

    easysimd_test_arm_neon_assert_equal_u8x16(r, easysimd_vld1q_u8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint8x16_t a = easysimd_test_arm_neon_random_u8x16();
    easysimd_uint8x16_t b = easysimd_test_arm_neon_random_u8x16();
    easysimd_uint8x16_t c = easysimd_test_arm_neon_random_u8x16();
    easysimd_uint8x16_t r = easysimd_vxxxq_u8(a, b, c);

    easysimd_test_arm_neon_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u8x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_u16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint16_t a[8];
    uint16_t b[8];
    uint16_t c[8];
    uint16_t r[8];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint16x8_t a = easysimd_vld1q_u16(test_vec[i].a);
    easysimd_uint16x8_t b = easysimd_vld1q_u16(test_vec[i].b);
    easysimd_uint16x8_t c = easysimd_vld1q_u16(test_vec[i].c);
    easysimd_uint16x8_t r = easysimd_vxxxq_u16(a, b, c);
    easysimd_test_arm_neon_assert_equal_u16x8(r, easysimd_vld1q_u16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint16x8_t a = easysimd_test_arm_neon_random_u16x8();
    easysimd_uint16x8_t b = easysimd_test_arm_neon_random_u16x8();
    easysimd_uint16x8_t c = easysimd_test_arm_neon_random_u16x8();
    easysimd_uint16x8_t r = easysimd_vxxxq_u16(a, b, c);

    easysimd_test_arm_neon_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint32_t a[4];
    uint32_t b[4];
    uint32_t c[4];
    uint32_t r[4];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint32x4_t a = easysimd_vld1q_u32(test_vec[i].a);
    easysimd_uint32x4_t b = easysimd_vld1q_u32(test_vec[i].b);
    easysimd_uint32x4_t c = easysimd_vld1q_u32(test_vec[i].c);
    easysimd_uint32x4_t r = easysimd_vxxxq_u32(a, b, c);

    easysimd_test_arm_neon_assert_equal_u32x4(r, easysimd_vld1q_u32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint32x4_t a = easysimd_test_arm_neon_random_u32x4();
    easysimd_uint32x4_t b = easysimd_test_arm_neon_random_u32x4();
    easysimd_uint32x4_t c = easysimd_test_arm_neon_random_u32x4();
    easysimd_uint32x4_t r = easysimd_vxxxq_u32(a, b, c);

    easysimd_test_arm_neon_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vxxxq_u64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 0
  static const struct {
    uint64_t a[2];
    uint64_t b[2];
    uint64_t c[2];
    uint64_t r[2];
  } test_vec[] = {

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_uint64x2_t a = easysimd_vld1q_u64(test_vec[i].a);
    easysimd_uint64x2_t b = easysimd_vld1q_u64(test_vec[i].b);
    easysimd_uint64x2_t c = easysimd_vld1q_u64(test_vec[i].c);
    easysimd_uint64x2_t r = easysimd_vxxxq_u64(a, b, c);
    easysimd_test_arm_neon_assert_equal_u64x2(r, easysimd_vld1q_u64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_uint64x2_t a = easysimd_test_arm_neon_random_u64x2();
    easysimd_uint64x2_t b = easysimd_test_arm_neon_random_u64x2();
    easysimd_uint64x2_t c = easysimd_test_arm_neon_random_u64x2();
    easysimd_uint64x2_t r = easysimd_vxxxq_u64(a, b, c);

    easysimd_test_arm_neon_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_s8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_s16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_s32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_s64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_u8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_u16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_u32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxx_u64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_s8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_s16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_s32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_s64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_u8)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_u16)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_u32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vxxxq_u64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
