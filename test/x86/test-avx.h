#if !defined(EASYSIMD_TEST_X86_TEST_AVX_H)
#define EASYSIMD_TEST_X86_TEST_AVX_H

#include "test-x86.h"
#include "test-sse2.h"
#include "../../easysimd/x86/avx.h"

EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m256, 32, 8, easysimd_mm256_storeu_ps)
EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m256d, 64, 4, easysimd_mm256_storeu_pd)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m256i, 8, 32, easysimd_mm256_storeu_si256)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m256i, 16, 16, easysimd_mm256_storeu_si256)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m256i, 32, 8, easysimd_mm256_storeu_si256)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m256i, 64, 4, easysimd_mm256_storeu_si256)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m256i, 8, 32, easysimd_mm256_storeu_si256)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m256i, 16, 16, easysimd_mm256_storeu_si256)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m256i, 32, 8, easysimd_mm256_storeu_si256)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m256i, 64, 4, easysimd_mm256_storeu_si256)

#define easysimd_test_x86_assert_equal_f32x8(a, b, precision) do { if (easysimd_test_x86_assert_equal_f32x8_(a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_f64x4(a, b, precision) do { if (easysimd_test_x86_assert_equal_f64x4_(a, b, easysimd_test_f64_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i8x32(a, b) do { if (easysimd_test_x86_assert_equal_i8x32_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i16x16(a, b) do { if (easysimd_test_x86_assert_equal_i16x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i32x8(a, b) do { if (easysimd_test_x86_assert_equal_i32x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i64x4(a, b) do { if (easysimd_test_x86_assert_equal_i64x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u8x32(a, b) do { if (easysimd_test_x86_assert_equal_u8x32_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u16x16(a, b) do { if (easysimd_test_x86_assert_equal_u16x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u32x8(a, b) do { if (easysimd_test_x86_assert_equal_u32x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u64x4(a, b) do { if (easysimd_test_x86_assert_equal_u64x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#endif /* !defined(EASYSIMD_TEST_X86_TEST_AVX_H) */
