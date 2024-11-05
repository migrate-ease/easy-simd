#include "../test.h"

/* These tests are basically to verify assumptions we make about the
 * target platform. */

#if defined(EASYSIMD_IEEE754_STORAGE)

static int
test_easysimd_ieee754_storage_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const easysimd_float32 pif_as_f32 = EASYSIMD_MATH_PIF;

  uint32_t pif_as_u32;

  easysimd_memcpy(&pif_as_u32, &pif_as_f32, sizeof(easysimd_float32));

  easysimd_assert_equal_u32(pif_as_u32, UINT32_C(0x40490fdb));

  return 0;
}

static int
test_easysimd_ieee754_storage_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const easysimd_float64 pid_as_f64 = EASYSIMD_MATH_PI;

  uint64_t pid_as_u64;

  easysimd_memcpy(&pid_as_u64, &pid_as_f64, sizeof(easysimd_float64));

  easysimd_assert_equal_u64(pid_as_u64, UINT64_C(0x400921fb54442d18));

  return 0;
}

#endif

/* These next two make sure that all we need to do is flip a single
 * bit in order to flip the sign of a value without altering the
 * absolute value. i.e., we want to make sure the parts of the float
 * aren't stored as two's complement or something. */

static int
test_easysimd_single_bit_sign_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const easysimd_float32 ppif_as_f32 =  EASYSIMD_MATH_PIF;
  static const easysimd_float32 npif_as_f32 = -EASYSIMD_MATH_PIF;
  uint32_t ppif_as_u32, npif_as_u32, v;

  easysimd_memcpy(&ppif_as_u32, &ppif_as_f32, sizeof(uint32_t));
  easysimd_memcpy(&npif_as_u32, &npif_as_f32, sizeof(uint32_t));

  /* is_power_of_two(pi ^ -pi) */
  v = ppif_as_u32 ^ npif_as_u32;
  v = (v & (v - 1)) == 0;

  easysimd_assert_equal_u32(v, UINT32_C(1));

  return 0;
}

static int
test_easysimd_single_bit_sign_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const easysimd_float64 ppif_as_f64 =  EASYSIMD_MATH_PI;
  static const easysimd_float64 npif_as_f64 = -EASYSIMD_MATH_PI;
  uint64_t ppif_as_u64, npif_as_u64, v;

  easysimd_memcpy(&ppif_as_u64, &ppif_as_f64, sizeof(uint64_t));
  easysimd_memcpy(&npif_as_u64, &npif_as_f64, sizeof(uint64_t));

  /* is_power_of_two(pi ^ -pi) */
  v = ppif_as_u64 ^ npif_as_u64;
  v = (v & (v - 1)) == 0;

  easysimd_assert_equal_u64(v, UINT64_C(1));

  return 0;
}

/* We can handle little and big endian, but not PDP endian (or any
 * other endianness). */

static int
test_easysimd_endian (EASYSIMD_MUNIT_TEST_ARGS) {
  uint8_t a[] = { 1, 2, 3, 4 };
  uint32_t v;

  easysimd_memcpy(&v, a, sizeof(v));

  switch(v) {
    case UINT32_C(0x01020304): /* Big endian */
    case UINT32_C(0x04030201): /* Little endian */
      return 0;
    default:
      return 1;
  }
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(ieee754_storage_f32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(ieee754_storage_f64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(single_bit_sign_f32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(single_bit_sign_f64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(endian)
EASYSIMD_TEST_FUNC_LIST_END

int main(void) {
  int retval = EXIT_SUCCESS;

  fprintf(stdout, "1..%zu\n", (sizeof(test_suite_tests) / sizeof(test_suite_tests[0])));
  for (size_t i = 0 ; i < (sizeof(test_suite_tests) / sizeof(test_suite_tests[0])) ; i++) {
    int res = test_suite_tests[i].func();
    if (res != 0) {
      retval = EXIT_FAILURE;
      fprintf(stdout, "not ok %zu %s\n", i + 1, test_suite_tests[i].name);
    } else {
      fprintf(stdout, "ok %zu %s\n", i + 1, test_suite_tests[i].name);
    }
  }

  return retval;
}
