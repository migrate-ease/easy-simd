#if !defined(EASYSIMD_TEST_X86_TEST_MMX_H)
#define EASYSIMD_TEST_X86_TEST_MMX_H

#include "test-x86.h"
#include "../../easysimd/x86/mmx.h"

EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m64, 32, 2, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m64, 64, 1, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m64, 8, 8, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m64, 16, 4, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m64, 32, 2, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m64, 64, 1, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m64, 8, 8, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m64, 16, 4, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m64, 32, 2, easysimd_x_mm_storeu_si64)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m64, 64, 1, easysimd_x_mm_storeu_si64)

#define easysimd_test_x86_assert_equal_f32x2(a, b, precision) do { if (easysimd_test_x86_assert_equal_f32x2_(a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_f64x1(a, b, precision) do { if (easysimd_test_x86_assert_equal_f64x1_(a, b, easysimd_test_f64_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i8x8(a, b) do { if (easysimd_test_x86_assert_equal_i8x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i16x4(a, b) do { if (easysimd_test_x86_assert_equal_i16x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i32x2(a, b) do { if (easysimd_test_x86_assert_equal_i32x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i64x1(a, b) do { if (easysimd_test_x86_assert_equal_i64x1_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u8x8(a, b) do { if (easysimd_test_x86_assert_equal_u8x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u16x4(a, b) do { if (easysimd_test_x86_assert_equal_u16x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u32x2(a, b) do { if (easysimd_test_x86_assert_equal_u32x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u64x1(a, b) do { if (easysimd_test_x86_assert_equal_u64x1_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#endif /* !defined(EASYSIMD_TEST_X86_TEST_MMX_H) */
