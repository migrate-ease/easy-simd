#if !defined(EASYSIMD_TEST_ARM_SVE_TEST_SVE_H)
#define EASYSIMD_TEST_ARM_SVE_TEST_SVE_H

#include "../../test.h"

static int32_t
easysimd_test_arm_sve_random_length(size_t buf_size, size_t elem_size) {
  return
    (easysimd_test_codegen_random_i32() & HEDLEY_STATIC_CAST(int32_t, (buf_size / elem_size) - 1)) |
    HEDLEY_STATIC_CAST(int32_t, (buf_size / 2) / elem_size);
}

#if !defined(EASYSIMD_TEST_BARE)
  #define EASYSIMD_TEST_DECLARE_SUITE(name) EASYSIMD_TEST_SUITE_DECLARE_GETTERS(HEDLEY_CONCAT(easysimd_test_arm_sve_get_suite_,name))
  #include <test/arm/sve/declare-suites.h>
  #undef EASYSIMD_TEST_DECLARE_SUITE
#endif

#endif /* !defined(EASYSIMD_TEST_ARM_SVE_TEST_SVE_H) */
