#if !defined(EASYSIMD_TEST_X86_TEST_SSE_H)
#define EASYSIMD_TEST_X86_TEST_SSE_H

#include "test-x86.h"
#include "test-mmx.h"
#include "../../easysimd/x86/sse.h"

EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m128, 32, 4, easysimd_mm_storeu_ps)

#define easysimd_test_x86_assert_equal_f32x4(a, b, precision) do { if (easysimd_test_x86_assert_equal_f32x4_(a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#endif /* !defined(EASYSIMD_TEST_X86_TEST_SSE_H) */
