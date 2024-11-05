#if defined(EASYSIMD_TESTS_ARM_SVE_RUN_TESTS_H)
  #error File already included.
#endif
#define EASYSIMD_TESTS_ARM_SVE_RUN_TESTS_H

#include "../../munit/munit.h"

MunitSuite* easysimd_tests_arm_sve_get_suite(void);
