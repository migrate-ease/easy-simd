#if defined(EASYSIMD_TESTS_X86_AVX512_RUN_TESTS_H)
  #error File already included.
#endif
#define EASYSIMD_TESTS_X86_AVX512_RUN_TESTS_H

#include "../../munit/munit.h"

MunitSuite* easysimd_tests_x86_avx512_get_suite(void);
