#if !defined(EASYSIMD_TEST_ARM_NEON_TEST_NEON_H)
#define EASYSIMD_TEST_ARM_NEON_TEST_NEON_H

#include "../../test.h"
#include "../../../easysimd/arm/neon/ld1.h"
#include "../../../easysimd/arm/neon/st1.h"

#define EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(NT, ET, element_count, modifier, symbol_identifier, neon_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count(void) { \
    easysimd_##NT v; \
    easysimd_test_codegen_random_memory(sizeof(v), HEDLEY_REINTERPRET_CAST(uint8_t*, &v)); \
    return v; \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    ET value_[sizeof(value) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(value_, value); \
 \
    easysimd_test_codegen_write_v##symbol_identifier(indent, sizeof(value) / sizeof(ET), value_, pos); \
 \
  } \
 \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##_(easysimd_##NT a, easysimd_##NT b, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    ET a_[sizeof(a) / sizeof(ET)], b_[sizeof(b) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(a_, a); \
    easysimd_vst1##modifier##_##neon_identifier(b_, b); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a_) / sizeof(a_[0]), a_, b_, filename, line, astr, bstr); \
  }

#define EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_EQUAL_FUNC_(NT, ET, SET, element_count, modifier, symbol_identifier) \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##_(easysimd_##NT a, easysimd_##NT b, ET slop, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    SET a_[sizeof(a) / sizeof(ET)], b_[sizeof(b) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##symbol_identifier(a_, a); \
    easysimd_vst1##modifier##_##symbol_identifier(b_, b); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a_), HEDLEY_REINTERPRET_CAST(SET*, b_), slop, filename, line, astr, bstr); \
  } \

#define EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_FUNCS_(NT, ET, SET, element_count, modifier, symbol_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count(ET min, ET max) { \
    SET v[sizeof(easysimd_##NT) / sizeof(ET)]; \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v) / sizeof(v[0]), v, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    return easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v)); \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    ET value_[sizeof(value) / sizeof(ET)]; \
    easysimd_vst1##modifier##_##symbol_identifier(value_, value); \
    easysimd_test_codegen_write_v##symbol_identifier(indent, sizeof(value_) / sizeof(value_[0]), value_, pos); \
  } \
 \
  EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_EQUAL_FUNC_(NT, ET, SET, element_count, modifier, symbol_identifier) \
  \
  static void \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count##_full( \
      size_t test_sets, size_t vectors_per_set, \
      ET values[HEDLEY_ARRAY_PARAM(test_sets * vectors_per_set * (sizeof(easysimd_##NT) / sizeof(ET)))], \
      ET min, ET max, SimdeTestVecFloatType type) { \
    easysimd_test_codegen_random_v##symbol_identifier##_full(test_sets, vectors_per_set, sizeof(easysimd_##NT) / sizeof(ET), values, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max), type); \
  } \
  \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_extract_##symbol_identifier##x##element_count(size_t set_num, size_t vectors_per_set, size_t vector_num, ET* values) { \
    const size_t elements_per_vector = sizeof(easysimd_##NT) / sizeof(ET); \
    const size_t elements_per_set = elements_per_vector * vectors_per_set; \
    const size_t pos = (elements_per_set * set_num) + (elements_per_vector * vector_num); \
    return easysimd_vld1##modifier##_##symbol_identifier(&(values[pos])); \
  }


HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DIAGNOSTIC_DISABLE_CPP98_COMPAT_PEDANTIC_
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(     int8x8_t,   int8_t,  8,  ,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    int16x4_t,  int16_t,  4,  , i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    int32x2_t,  int32_t,  2,  , i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    int64x1_t,  int64_t,  1,  , i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    uint8x8_t,  uint8_t,  8,  ,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(   uint16x4_t, uint16_t,  4,  , u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(   uint32x2_t, uint32_t,  2,  , u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(   uint64x1_t, uint64_t,  1,  , u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_FUNCS_(float32x2_t, easysimd_float32_t, easysimd_float32, 2,  , f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_FUNCS_(float64x1_t, easysimd_float64_t, easysimd_float64, 1,  , f64)

EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    int8x16_t,   int8_t, 16, q,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    int16x8_t,  int16_t,  8, q, i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    int32x4_t,  int32_t,  4, q, i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(    int64x2_t,  int64_t,  2, q, i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(   uint8x16_t,  uint8_t, 16, q,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(   uint16x8_t,  uint16_t,  8, q, u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(   uint32x4_t,  uint32_t,  4, q, u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_INT_TYPE_FUNCS_(   uint64x2_t,  uint64_t,  2, q, u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_FUNCS_(float32x4_t, easysimd_float32_t, easysimd_float32, 4, q, f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_FUNCS_(float64x2_t, easysimd_float64_t, easysimd_float64, 2, q, f64)

#if EASYSIMD_FLOAT16_API == EASYSIMD_FLOAT16_API_FP16
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_FUNCS_(float16x4_t, easysimd_float16_t, easysimd_float16, 4,  , f16)
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_FUNCS_(float16x8_t, easysimd_float16_t, easysimd_float16, 8, q, f16)
#else
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_EQUAL_FUNC_(float16x4_t, easysimd_float16_t, easysimd_float16, 4,  , f16)
EASYSIMD_TEST_ARM_NEON_GENERATE_FLOAT_TYPE_EQUAL_FUNC_(float16x8_t, easysimd_float16_t, easysimd_float16, 8, q, f16)
#endif

HEDLEY_DIAGNOSTIC_POP

#define EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(NT, ET, element_count, modifier, symbol_identifier, neon_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count##x2(void) { \
    easysimd_##NT v; \
    easysimd_test_codegen_random_memory(sizeof(v), HEDLEY_REINTERPRET_CAST(uint8_t*, &v)); \
    return v; \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count##x2(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    if (pos == EASYSIMD_TEST_VEC_POS_FIRST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("{\n", EASYSIMD_CODEGEN_FP); \
    } \
    ET value0_[sizeof(value) / sizeof(ET) / 2]; \
    ET value1_[sizeof(value) / sizeof(ET) / 2]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(value0_, value.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(value1_, value.val[1]); \
 \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value0_) / sizeof(ET), value0_, EASYSIMD_TEST_VEC_POS_FIRST); \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value1_) / sizeof(ET), value1_, EASYSIMD_TEST_VEC_POS_LAST); \
    if (pos == EASYSIMD_TEST_VEC_POS_LAST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("},\n", EASYSIMD_CODEGEN_FP); \
    } \
 \
  } \
 \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##x2_(easysimd_##NT a, easysimd_##NT b, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    ET a0_[sizeof(a.val[0]) / sizeof(ET)], b0_[sizeof(b.val[0]) / sizeof(ET)]; \
    ET a1_[sizeof(a.val[1]) / sizeof(ET)], b1_[sizeof(b.val[1]) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(a0_, a.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(b0_, b.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(a1_, a.val[1]); \
    easysimd_vst1##modifier##_##neon_identifier(b1_, b.val[1]); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a0_) / sizeof(a0_[0]), a0_, b0_, filename, line, astr, bstr) \
      && easysimd_assert_equal_v##symbol_identifier##_(sizeof(a1_) / sizeof(a1_[0]), a1_, b1_, filename, line, astr, bstr); \
  }

#define EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_FLOAT_TYPE_FUNCS_(NT, ET, SET, element_count, modifier, symbol_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count##x2(ET min, ET max) { \
    SET v0[sizeof(easysimd_##NT) / sizeof(ET) / 2]; \
    SET v1[sizeof(easysimd_##NT) / sizeof(ET) / 2]; \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v0) / sizeof(v0[0]), v0, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v1) / sizeof(v1[0]), v1, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_##NT r; \
    r.val[0] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v0)); \
    r.val[1] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v1)); \
    return r; \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count##x2(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    if (pos == EASYSIMD_TEST_VEC_POS_FIRST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("{\n", EASYSIMD_CODEGEN_FP); \
    } \
 \
    ET value0_[sizeof(value) / sizeof(ET) / 2]; \
    ET value1_[sizeof(value) / sizeof(ET) / 2]; \
    easysimd_vst1##modifier##_##symbol_identifier(value0_, value.val[0]); \
    easysimd_vst1##modifier##_##symbol_identifier(value1_, value.val[1]); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value0_) / sizeof(value0_[0]), value0_, EASYSIMD_TEST_VEC_POS_FIRST); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value1_) / sizeof(value1_[0]), value1_, EASYSIMD_TEST_VEC_POS_LAST); \
    if (pos == EASYSIMD_TEST_VEC_POS_LAST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("},\n", EASYSIMD_CODEGEN_FP); \
    } \
  } \
 \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##x2_(easysimd_##NT a, easysimd_##NT b, ET slop, \
     const char* filename, int line, const char* astr, const char* bstr) { \
    SET a0_[sizeof(a) / sizeof(ET)], b0_[sizeof(b) / sizeof(ET)]; \
    SET a1_[sizeof(a) / sizeof(ET)], b1_[sizeof(b) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##symbol_identifier(a0_, a.val[0]); \
    easysimd_vst1##modifier##_##symbol_identifier(b0_, b.val[0]);       \
    easysimd_vst1##modifier##_##symbol_identifier(a1_, a.val[1]); \
    easysimd_vst1##modifier##_##symbol_identifier(b1_, b.val[1]); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a0_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a0_), HEDLEY_REINTERPRET_CAST(SET*, b0_), slop, filename, line, astr, bstr) && \
      easysimd_assert_equal_v##symbol_identifier##_(sizeof(a1_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a1_), HEDLEY_REINTERPRET_CAST(SET*, b1_), slop, filename, line, astr, bstr); \
  }

#if !defined(EASYSIMD_BUG_INTEL_857088)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(     int8x8x2_t,   int8_t,  8,  ,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    int16x4x2_t,  int16_t,  4,  , i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    int32x2x2_t,  int32_t,  2,  , i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    int64x1x2_t,  int64_t,  1,  , i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    uint8x8x2_t,  uint8_t,  8,  ,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(   uint16x4x2_t, uint16_t,  4,  , u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(   uint32x2x2_t, uint32_t,  2,  , u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(   uint64x1x2_t, uint64_t,  1,  , u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_FLOAT_TYPE_FUNCS_(float32x2x2_t, easysimd_float32_t, easysimd_float32, 2, , f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_FLOAT_TYPE_FUNCS_(float64x1x2_t, easysimd_float64_t, easysimd_float64, 1, , f64)

EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    int8x16x2_t,   int8_t, 16, q,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    int16x8x2_t,  int16_t,  8, q, i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    int32x4x2_t,  int32_t,  4, q, i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(    int64x2x2_t,  int64_t,  2, q, i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(   uint8x16x2_t,  uint8_t, 16, q,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(   uint16x8x2_t,  uint16_t,  8, q, u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(   uint32x4x2_t,  uint32_t,  4, q, u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_INT_TYPE_FUNCS_(   uint64x2x2_t,  uint64_t,  2, q, u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_FLOAT_TYPE_FUNCS_(float32x4x2_t, easysimd_float32_t, easysimd_float32, 4, q, f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X2_VECTOR_FLOAT_TYPE_FUNCS_(float64x2x2_t, easysimd_float64_t, easysimd_float64, 2, q, f64)
#endif

#define EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(NT, ET, element_count, modifier, symbol_identifier, neon_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count##x3(void) { \
    easysimd_##NT v; \
    easysimd_test_codegen_random_memory(sizeof(v), HEDLEY_REINTERPRET_CAST(uint8_t*, &v)); \
    return v; \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count##x3(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    if (pos == EASYSIMD_TEST_VEC_POS_FIRST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("{\n", EASYSIMD_CODEGEN_FP); \
    } \
    ET value0_[sizeof(value) / sizeof(ET) / 3]; \
    ET value1_[sizeof(value) / sizeof(ET) / 3]; \
    ET value2_[sizeof(value) / sizeof(ET) / 3]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(value0_, value.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(value1_, value.val[1]); \
    easysimd_vst1##modifier##_##neon_identifier(value2_, value.val[2]); \
 \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value0_) / sizeof(ET), value0_, EASYSIMD_TEST_VEC_POS_FIRST); \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value1_) / sizeof(ET), value1_, EASYSIMD_TEST_VEC_POS_MIDDLE); \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value2_) / sizeof(ET), value2_, EASYSIMD_TEST_VEC_POS_LAST); \
    if (pos == EASYSIMD_TEST_VEC_POS_LAST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("},\n", EASYSIMD_CODEGEN_FP); \
    } \
 \
  } \
 \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##x3_(easysimd_##NT a, easysimd_##NT b, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    ET a0_[sizeof(a.val[0]) / sizeof(ET)], b0_[sizeof(b.val[0]) / sizeof(ET)]; \
    ET a1_[sizeof(a.val[1]) / sizeof(ET)], b1_[sizeof(b.val[1]) / sizeof(ET)]; \
    ET a2_[sizeof(a.val[2]) / sizeof(ET)], b2_[sizeof(b.val[2]) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(a0_, a.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(b0_, b.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(a1_, a.val[1]); \
    easysimd_vst1##modifier##_##neon_identifier(b1_, b.val[1]); \
    easysimd_vst1##modifier##_##neon_identifier(a2_, a.val[2]); \
    easysimd_vst1##modifier##_##neon_identifier(b2_, b.val[2]); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a0_) / sizeof(a0_[0]), a0_, b0_, filename, line, astr, bstr) \
      && easysimd_assert_equal_v##symbol_identifier##_(sizeof(a1_) / sizeof(a1_[0]), a1_, b1_, filename, line, astr, bstr) \
      && easysimd_assert_equal_v##symbol_identifier##_(sizeof(a2_) / sizeof(a2_[0]), a2_, b2_, filename, line, astr, bstr); \
  }

#define EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_FLOAT_TYPE_FUNCS_(NT, ET, SET, element_count, modifier, symbol_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count##x3(ET min, ET max) { \
    SET v0[sizeof(easysimd_##NT) / sizeof(ET) / 3]; \
    SET v1[sizeof(easysimd_##NT) / sizeof(ET) / 3]; \
    SET v2[sizeof(easysimd_##NT) / sizeof(ET) / 3]; \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v0) / sizeof(v0[0]), v0, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v1) / sizeof(v1[0]), v1, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v2) / sizeof(v2[0]), v2, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_##NT r; \
    r.val[0] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v0)); \
    r.val[1] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v1)); \
    r.val[2] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v2)); \
    return r; \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count##x3(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    if (pos == EASYSIMD_TEST_VEC_POS_FIRST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("{\n", EASYSIMD_CODEGEN_FP); \
    } \
 \
    ET value0_[sizeof(value) / sizeof(ET) / 3]; \
    ET value1_[sizeof(value) / sizeof(ET) / 3]; \
    ET value2_[sizeof(value) / sizeof(ET) / 3]; \
    easysimd_vst1##modifier##_##symbol_identifier(value0_, value.val[0]); \
    easysimd_vst1##modifier##_##symbol_identifier(value1_, value.val[1]); \
    easysimd_vst1##modifier##_##symbol_identifier(value2_, value.val[2]); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value0_) / sizeof(value0_[0]), value0_, EASYSIMD_TEST_VEC_POS_FIRST); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value1_) / sizeof(value1_[0]), value1_, EASYSIMD_TEST_VEC_POS_MIDDLE); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value2_) / sizeof(value2_[0]), value2_, EASYSIMD_TEST_VEC_POS_LAST); \
    if (pos == EASYSIMD_TEST_VEC_POS_LAST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("},\n", EASYSIMD_CODEGEN_FP); \
    } \
  } \
 \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##x3_(easysimd_##NT a, easysimd_##NT b, ET slop, \
     const char* filename, int line, const char* astr, const char* bstr) { \
    SET a0_[sizeof(a) / sizeof(ET)], b0_[sizeof(b) / sizeof(ET)]; \
    SET a1_[sizeof(a) / sizeof(ET)], b1_[sizeof(b) / sizeof(ET)]; \
    SET a2_[sizeof(a) / sizeof(ET)], b2_[sizeof(b) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##symbol_identifier(a0_, a.val[0]); \
    easysimd_vst1##modifier##_##symbol_identifier(b0_, b.val[0]);       \
    easysimd_vst1##modifier##_##symbol_identifier(a1_, a.val[1]); \
    easysimd_vst1##modifier##_##symbol_identifier(b1_, b.val[1]); \
    easysimd_vst1##modifier##_##symbol_identifier(a2_, a.val[2]); \
    easysimd_vst1##modifier##_##symbol_identifier(b2_, b.val[2]); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a0_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a0_), HEDLEY_REINTERPRET_CAST(SET*, b0_), slop, filename, line, astr, bstr) && \
      easysimd_assert_equal_v##symbol_identifier##_(sizeof(a1_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a1_), HEDLEY_REINTERPRET_CAST(SET*, b1_), slop, filename, line, astr, bstr) && \
      easysimd_assert_equal_v##symbol_identifier##_(sizeof(a2_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a2_), HEDLEY_REINTERPRET_CAST(SET*, b2_), slop, filename, line, astr, bstr); \
  }

#if !defined(EASYSIMD_BUG_INTEL_857088)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(     int8x8x3_t,   int8_t,  8,  ,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    int16x4x3_t,  int16_t,  4,  , i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    int32x2x3_t,  int32_t,  2,  , i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    int64x1x3_t,  int64_t,  1,  , i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    uint8x8x3_t,  uint8_t,  8,  ,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(   uint16x4x3_t, uint16_t,  4,  , u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(   uint32x2x3_t, uint32_t,  2,  , u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(   uint64x1x3_t, uint64_t,  1,  , u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_FLOAT_TYPE_FUNCS_(float32x2x3_t, easysimd_float32_t, easysimd_float32, 2, , f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_FLOAT_TYPE_FUNCS_(float64x1x3_t, easysimd_float64_t, easysimd_float64, 1, , f64)

EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    int8x16x3_t,   int8_t, 16, q,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    int16x8x3_t,  int16_t,  8, q, i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    int32x4x3_t,  int32_t,  4, q, i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(    int64x2x3_t,  int64_t,  2, q, i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(   uint8x16x3_t,  uint8_t, 16, q,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(   uint16x8x3_t,  uint16_t,  8, q, u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(   uint32x4x3_t,  uint32_t,  4, q, u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_INT_TYPE_FUNCS_(   uint64x2x3_t,  uint64_t,  2, q, u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_FLOAT_TYPE_FUNCS_(float32x4x3_t, easysimd_float32_t, easysimd_float32, 4, q, f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X3_VECTOR_FLOAT_TYPE_FUNCS_(float64x2x3_t, easysimd_float64_t, easysimd_float64, 2, q, f64)
#endif

#define EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(NT, ET, element_count, modifier, symbol_identifier, neon_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count##x4(void) { \
    easysimd_##NT v; \
    easysimd_test_codegen_random_memory(sizeof(v), HEDLEY_REINTERPRET_CAST(uint8_t*, &v)); \
    return v; \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count##x4(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    if (pos == EASYSIMD_TEST_VEC_POS_FIRST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("{\n", EASYSIMD_CODEGEN_FP); \
    } \
    ET value0_[sizeof(value) / sizeof(ET) / 4]; \
    ET value1_[sizeof(value) / sizeof(ET) / 4]; \
    ET value2_[sizeof(value) / sizeof(ET) / 4]; \
    ET value3_[sizeof(value) / sizeof(ET) / 4]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(value0_, value.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(value1_, value.val[1]); \
    easysimd_vst1##modifier##_##neon_identifier(value2_, value.val[2]); \
    easysimd_vst1##modifier##_##neon_identifier(value3_, value.val[3]); \
 \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value0_) / sizeof(ET), value0_, EASYSIMD_TEST_VEC_POS_FIRST); \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value1_) / sizeof(ET), value1_, EASYSIMD_TEST_VEC_POS_MIDDLE); \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value2_) / sizeof(ET), value2_, EASYSIMD_TEST_VEC_POS_MIDDLE); \
    easysimd_test_codegen_write_v##symbol_identifier(indent+2, sizeof(value3_) / sizeof(ET), value3_, EASYSIMD_TEST_VEC_POS_LAST); \
    if (pos == EASYSIMD_TEST_VEC_POS_LAST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("},\n", EASYSIMD_CODEGEN_FP); \
    } \
 \
  } \
 \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##x4_(easysimd_##NT a, easysimd_##NT b, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    ET a0_[sizeof(a.val[0]) / sizeof(ET)], b0_[sizeof(b.val[0]) / sizeof(ET)]; \
    ET a1_[sizeof(a.val[1]) / sizeof(ET)], b1_[sizeof(b.val[1]) / sizeof(ET)]; \
    ET a2_[sizeof(a.val[2]) / sizeof(ET)], b2_[sizeof(b.val[2]) / sizeof(ET)]; \
    ET a3_[sizeof(a.val[3]) / sizeof(ET)], b3_[sizeof(b.val[3]) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##neon_identifier(a0_, a.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(b0_, b.val[0]); \
    easysimd_vst1##modifier##_##neon_identifier(a1_, a.val[1]); \
    easysimd_vst1##modifier##_##neon_identifier(b1_, b.val[1]); \
    easysimd_vst1##modifier##_##neon_identifier(a2_, a.val[2]); \
    easysimd_vst1##modifier##_##neon_identifier(b2_, b.val[2]); \
    easysimd_vst1##modifier##_##neon_identifier(a3_, a.val[3]); \
    easysimd_vst1##modifier##_##neon_identifier(b3_, b.val[3]); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a0_) / sizeof(a0_[0]), a0_, b0_, filename, line, astr, bstr) \
      && easysimd_assert_equal_v##symbol_identifier##_(sizeof(a1_) / sizeof(a1_[0]), a1_, b1_, filename, line, astr, bstr) \
      && easysimd_assert_equal_v##symbol_identifier##_(sizeof(a2_) / sizeof(a2_[0]), a2_, b2_, filename, line, astr, bstr) \
      && easysimd_assert_equal_v##symbol_identifier##_(sizeof(a3_) / sizeof(a3_[0]), a3_, b3_, filename, line, astr, bstr); \
  }

#define EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_FLOAT_TYPE_FUNCS_(NT, ET, SET, element_count, modifier, symbol_identifier) \
  static easysimd_##NT \
  easysimd_test_arm_neon_random_##symbol_identifier##x##element_count##x4(ET min, ET max) { \
    SET v0[sizeof(easysimd_##NT) / sizeof(ET) / 4]; \
    SET v1[sizeof(easysimd_##NT) / sizeof(ET) / 4]; \
    SET v2[sizeof(easysimd_##NT) / sizeof(ET) / 4]; \
    SET v3[sizeof(easysimd_##NT) / sizeof(ET) / 4]; \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v0) / sizeof(v0[0]), v0, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v1) / sizeof(v1[0]), v1, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v2) / sizeof(v2[0]), v2, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_test_codegen_random_v##symbol_identifier(sizeof(v3) / sizeof(v3[0]), v3, HEDLEY_STATIC_CAST(SET, min), HEDLEY_STATIC_CAST(SET, max)); \
    easysimd_##NT r; \
    r.val[0] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v0)); \
    r.val[1] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v1)); \
    r.val[2] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v2)); \
    r.val[3] = easysimd_vld1##modifier##_##symbol_identifier(HEDLEY_REINTERPRET_CAST(ET*, v3)); \
        return r; \
  } \
 \
  static void \
  easysimd_test_arm_neon_write_##symbol_identifier##x##element_count##x4(int indent, easysimd_##NT value, SimdeTestVecPos pos) { \
    if (pos == EASYSIMD_TEST_VEC_POS_FIRST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("{\n", EASYSIMD_CODEGEN_FP); \
    } \
 \
    ET value0_[sizeof(value) / sizeof(ET) / 4]; \
    ET value1_[sizeof(value) / sizeof(ET) / 4]; \
    ET value2_[sizeof(value) / sizeof(ET) / 4]; \
    ET value3_[sizeof(value) / sizeof(ET) / 4]; \
    easysimd_vst1##modifier##_##symbol_identifier(value0_, value.val[0]); \
    easysimd_vst1##modifier##_##symbol_identifier(value1_, value.val[1]); \
    easysimd_vst1##modifier##_##symbol_identifier(value2_, value.val[2]); \
    easysimd_vst1##modifier##_##symbol_identifier(value3_, value.val[3]); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value0_) / sizeof(value0_[0]), value0_, EASYSIMD_TEST_VEC_POS_FIRST); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value1_) / sizeof(value1_[0]), value1_, EASYSIMD_TEST_VEC_POS_MIDDLE); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value2_) / sizeof(value2_[0]), value2_, EASYSIMD_TEST_VEC_POS_MIDDLE); \
    easysimd_test_codegen_write_v##symbol_identifier(indent + 2, sizeof(value3_) / sizeof(value3_[0]), value3_, EASYSIMD_TEST_VEC_POS_LAST); \
    if (pos == EASYSIMD_TEST_VEC_POS_LAST) { \
      easysimd_test_codegen_write_indent(indent); \
      fputs("},\n", EASYSIMD_CODEGEN_FP); \
    } \
  } \
 \
  static int \
  easysimd_test_arm_neon_assert_equal_##symbol_identifier##x##element_count##x4_(easysimd_##NT a, easysimd_##NT b, ET slop, \
     const char* filename, int line, const char* astr, const char* bstr) { \
    SET a0_[sizeof(a) / sizeof(ET)], b0_[sizeof(b) / sizeof(ET)]; \
    SET a1_[sizeof(a) / sizeof(ET)], b1_[sizeof(b) / sizeof(ET)]; \
    SET a2_[sizeof(a) / sizeof(ET)], b2_[sizeof(b) / sizeof(ET)]; \
    SET a3_[sizeof(a) / sizeof(ET)], b3_[sizeof(b) / sizeof(ET)]; \
 \
    easysimd_vst1##modifier##_##symbol_identifier(a0_, a.val[0]); \
    easysimd_vst1##modifier##_##symbol_identifier(b0_, b.val[0]);       \
    easysimd_vst1##modifier##_##symbol_identifier(a1_, a.val[1]); \
    easysimd_vst1##modifier##_##symbol_identifier(b1_, b.val[1]); \
    easysimd_vst1##modifier##_##symbol_identifier(a2_, a.val[2]); \
    easysimd_vst1##modifier##_##symbol_identifier(b2_, b.val[2]); \
    easysimd_vst1##modifier##_##symbol_identifier(a3_, a.val[3]); \
    easysimd_vst1##modifier##_##symbol_identifier(b3_, b.val[3]); \
 \
    return easysimd_assert_equal_v##symbol_identifier##_(sizeof(a0_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a0_), HEDLEY_REINTERPRET_CAST(SET*, b0_), slop, filename, line, astr, bstr) && \
      easysimd_assert_equal_v##symbol_identifier##_(sizeof(a1_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a1_), HEDLEY_REINTERPRET_CAST(SET*, b1_), slop, filename, line, astr, bstr) && \
      easysimd_assert_equal_v##symbol_identifier##_(sizeof(a2_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a2_), HEDLEY_REINTERPRET_CAST(SET*, b2_), slop, filename, line, astr, bstr) && \
      easysimd_assert_equal_v##symbol_identifier##_(sizeof(a3_) / sizeof(ET), HEDLEY_REINTERPRET_CAST(SET*, a3_), HEDLEY_REINTERPRET_CAST(SET*, b3_), slop, filename, line, astr, bstr); \
  }

#if !defined(EASYSIMD_BUG_INTEL_857088)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(     int8x8x4_t,   int8_t,  8,  ,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    int16x4x4_t,  int16_t,  4,  , i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    int32x2x4_t,  int32_t,  2,  , i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    int64x1x4_t,  int64_t,  1,  , i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    uint8x8x4_t,  uint8_t,  8,  ,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(   uint16x4x4_t, uint16_t,  4,  , u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(   uint32x2x4_t, uint32_t,  2,  , u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(   uint64x1x4_t, uint64_t,  1,  , u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_FLOAT_TYPE_FUNCS_(float32x2x4_t, easysimd_float32_t, easysimd_float32, 2, , f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_FLOAT_TYPE_FUNCS_(float64x1x4_t, easysimd_float64_t, easysimd_float64, 1, , f64)

EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    int8x16x4_t,   int8_t, 16, q,  i8,  s8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    int16x8x4_t,  int16_t,  8, q, i16, s16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    int32x4x4_t,  int32_t,  4, q, i32, s32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(    int64x2x4_t,  int64_t,  2, q, i64, s64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(   uint8x16x4_t,  uint8_t, 16, q,  u8,  u8)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(   uint16x8x4_t,  uint16_t,  8, q, u16, u16)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(   uint32x4x4_t,  uint32_t,  4, q, u32, u32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_INT_TYPE_FUNCS_(   uint64x2x4_t,  uint64_t,  2, q, u64, u64)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_FLOAT_TYPE_FUNCS_(float32x4x4_t, easysimd_float32_t, easysimd_float32, 4, q, f32)
EASYSIMD_TEST_ARM_NEON_GENERATE_X4_VECTOR_FLOAT_TYPE_FUNCS_(float64x2x4_t, easysimd_float64_t, easysimd_float64, 2, q, f64)
#endif

#define easysimd_test_arm_neon_assert_equal_i8x8(a, b)   do { if (easysimd_test_arm_neon_assert_equal_i8x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_i16x4(a, b) do { if (easysimd_test_arm_neon_assert_equal_i16x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_i32x2(a, b) do { if (easysimd_test_arm_neon_assert_equal_i32x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_i64x1(a, b) do { if (easysimd_test_arm_neon_assert_equal_i64x1_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u8x8(a, b)   do { if (easysimd_test_arm_neon_assert_equal_u8x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u16x4(a, b) do { if (easysimd_test_arm_neon_assert_equal_u16x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u32x2(a, b) do { if (easysimd_test_arm_neon_assert_equal_u32x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u64x1(a, b) do { if (easysimd_test_arm_neon_assert_equal_u64x1_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_f16x4(a, b, precision) do { if (easysimd_test_arm_neon_assert_equal_f16x4_(a, b, easysimd_test_f16_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_f32x2(a, b, precision) do { if (easysimd_test_arm_neon_assert_equal_f32x2_(a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_f64x1(a, b, precision) do { if (easysimd_test_arm_neon_assert_equal_f64x1_(a, b, easysimd_test_f64_precision_to_slop(precision),    __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#define easysimd_test_arm_neon_assert_equal_i8x16(a, b) do { if (easysimd_test_arm_neon_assert_equal_i8x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_i16x8(a, b) do { if (easysimd_test_arm_neon_assert_equal_i16x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_i32x4(a, b) do { if (easysimd_test_arm_neon_assert_equal_i32x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_i64x2(a, b) do { if (easysimd_test_arm_neon_assert_equal_i64x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u8x16(a, b) do { if (easysimd_test_arm_neon_assert_equal_u8x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u16x8(a, b) do { if (easysimd_test_arm_neon_assert_equal_u16x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u32x4(a, b) do { if (easysimd_test_arm_neon_assert_equal_u32x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u64x2(a, b) do { if (easysimd_test_arm_neon_assert_equal_u64x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_f16x8(a, b, precision) do { if (easysimd_test_arm_neon_assert_equal_f16x8_(a, b, easysimd_test_f16_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_f32x4(a, b, precision) do { if (easysimd_test_arm_neon_assert_equal_f32x4_(a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_f64x2(a, b, precision) do { if (easysimd_test_arm_neon_assert_equal_f64x2_(a, b, easysimd_test_f64_precision_to_slop(precision),    __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#define easysimd_test_arm_neon_assert_equal_u8x16x2(a, b) do { if (easysimd_test_arm_neon_assert_equal_u8x16x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u8x16x3(a, b) do { if (easysimd_test_arm_neon_assert_equal_u8x16x3_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_arm_neon_assert_equal_u8x16x4(a, b) do { if (easysimd_test_arm_neon_assert_equal_u8x16x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#if !defined(EASYSIMD_TEST_BARE)
  #define EASYSIMD_TEST_DECLARE_SUITE(name) EASYSIMD_TEST_SUITE_DECLARE_GETTERS(HEDLEY_CONCAT(easysimd_test_arm_neon_get_suite_,name))
  #include <test/arm/neon/declare-suites.h>
  #undef EASYSIMD_TEST_DECLARE_SUITE
#endif

#endif /* !defined(EASYSIMD_TEST_ARM_NEON_TEST_NEON_H) */
