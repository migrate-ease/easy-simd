#if defined(EASYSIMD_TESTS_ARM_RUN_TESTS_H)
  #error File already included.
#endif
#define EASYSIMD_TESTS_ARM_RUN_TESTS_H

#include "../test.h"
#include "neon/run-tests.h"
#include "sve/run-tests.h"

MunitSuite* easysimd_tests_arm_get_suite(void);
