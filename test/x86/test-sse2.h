#if !defined(EASYSIMD_TEST_X86_TEST_SSE2_H)
#define EASYSIMD_TEST_X86_TEST_SSE2_H

#include "test-x86.h"
#include "test-sse.h"
#include "../../easysimd/x86/sse2.h"

EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m128d, 64, 2, easysimd_mm_storeu_pd)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m128i, 8, 16, easysimd_mm_storeu_si128)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m128i, 16, 8, easysimd_mm_storeu_si128)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m128i, 32, 4, easysimd_mm_storeu_si128)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m128i, 64, 2, easysimd_mm_storeu_si128)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m128i, 8, 16, easysimd_mm_storeu_si128)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m128i, 16, 8, easysimd_mm_storeu_si128)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m128i, 32, 4, easysimd_mm_storeu_si128)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m128i, 64, 2, easysimd_mm_storeu_si128)

#if !defined(EASYSIMD_ENABLE_TEST_PERF)
#define easysimd_test_x86_assert_equal_f64x2(a, b, precision) do { if (easysimd_test_x86_assert_equal_f64x2_(a, b, easysimd_test_f64_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i8x16(a, b) do { if (easysimd_test_x86_assert_equal_i8x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i16x8(a, b) do { if (easysimd_test_x86_assert_equal_i16x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i32x4(a, b) do { if (easysimd_test_x86_assert_equal_i32x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i64x2(a, b) do { if (easysimd_test_x86_assert_equal_i64x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u8x16(a, b) do { if (easysimd_test_x86_assert_equal_u8x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u16x8(a, b) do { if (easysimd_test_x86_assert_equal_u16x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u32x4(a, b) do { if (easysimd_test_x86_assert_equal_u32x4_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u64x2(a, b) do { if (easysimd_test_x86_assert_equal_u64x2_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#else
#define easysimd_test_x86_assert_equal_f64x2(a, b, precision) (void)(a); (void)(b); (void)(precision);
#define easysimd_test_x86_assert_equal_i8x16(a, b)  (void)(a); (void)(b);
#define easysimd_test_x86_assert_equal_i16x8(a, b)  (void)(a); (void)(b);
#define easysimd_test_x86_assert_equal_i32x4(a, b)  (void)(a); (void)(b);
#define easysimd_test_x86_assert_equal_i64x2(a, b)  (void)(a); (void)(b);
#define easysimd_test_x86_assert_equal_u8x16(a, b)  (void)(a); (void)(b);
#define easysimd_test_x86_assert_equal_u16x8(a, b)  (void)(a); (void)(b);
#define easysimd_test_x86_assert_equal_u32x4(a, b)  (void)(a); (void)(b);
#define easysimd_test_x86_assert_equal_u64x2(a, b)  (void)(a); (void)(b);
#endif

#endif /* !defined(EASYSIMD_TEST_X86_TEST_SSE2_H) */
