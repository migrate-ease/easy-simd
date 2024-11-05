#include "test-x86.h"
#include "run-tests.h"

static MunitSuite suites[] = {
  #define EASYSIMD_TEST_DECLARE_SUITE(name) \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE }, \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE }, \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE }, \
    { NULL, NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE },
  #include <test/x86/declare-suites.h>
  #undef EASYSIMD_TEST_DECLARE_SUITE
  { NULL, NULL, NULL, 0, MUNIT_SUITE_OPTION_NONE },
  { NULL, NULL, NULL, 0, MUNIT_SUITE_OPTION_NONE }
};

static MunitSuite suite = { "/x86", NULL, suites, 1, MUNIT_SUITE_OPTION_NONE };

MunitSuite*
easysimd_tests_x86_get_suite(void) {
  int i = 0;

  #define EASYSIMD_TEST_DECLARE_SUITE(name) \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_x86_get_suite_, name, _native_c)(); \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_x86_get_suite_, name, _native_cpp)(); \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_x86_get_suite_, name, _emul_c)(); \
    suites[i++] = *HEDLEY_CONCAT3(easysimd_test_x86_get_suite_, name, _emul_cpp)();
  #include <test/x86/declare-suites.h>
  #undef EASYSIMD_TEST_DECLARE_SUITE
  suites[i++] = *easysimd_tests_x86_avx512_get_suite();

  return &suite;
}
