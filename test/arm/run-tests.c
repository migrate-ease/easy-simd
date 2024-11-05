#include "run-tests.h"

#include "../../easysimd/hedley.h"

static MunitSuite suites[] = {
  #define EASYSIMD_TEST_DECLARE_SUITE(name) { (char*) "/", NULL, NULL, 1, MUNIT_SUITE_OPTION_NONE },
  #include "declare-suites.h"
  #undef EASYSIMD_TEST_DECLARE_SUITE
  { NULL, NULL, NULL, 0, MUNIT_SUITE_OPTION_NONE }
};

static MunitSuite suite = { "/arm", NULL, suites, 1, MUNIT_SUITE_OPTION_NONE };

MunitSuite*
easysimd_tests_arm_get_suite(void) {
  size_t i = 0;
  #define EASYSIMD_TEST_DECLARE_SUITE(name) suites[i++] = *HEDLEY_CONCAT3(easysimd_tests_arm_, name, _get_suite)();
  #include "declare-suites.h"
  #undef EASYSIMD_TEST_DECLARE_SUITE

  return &suite;
}
