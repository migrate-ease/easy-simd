#include "test-neon.h"
#include "run-tests.h"

static MunitSuite suites[] = {
  #define EASYSIMD_TEST_DECLARE_SUITE(name) \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE }, \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE }, \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE }, \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE },
  #include <test/arm/neon/declare-suites.h>
  #undef EASYSIMD_TEST_DECLARE_SUITE
  { NULL, NULL, NULL, 0, MUNIT_SUITE_OPTION_NONE }
};

static MunitSuite suite = { "/neon", NULL, suites, 1, MUNIT_SUITE_OPTION_NONE };

MunitSuite*
easysimd_tests_arm_neon_get_suite(void) {
  int i = 0;

  #define EASYSIMD_TEST_DECLARE_SUITE(name) \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_arm_neon_get_suite_, name, _native_c)(); \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_arm_neon_get_suite_, name, _native_cpp)(); \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_arm_neon_get_suite_, name, _emul_c)(); \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_arm_neon_get_suite_, name, _emul_cpp)();
  #include <test/arm/neon/declare-suites.h>
  #undef EASYSIMD_TEST_DECLARE_SUITE

  return &suite;
}
