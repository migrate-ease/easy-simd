#if !defined(EASYSIMD_TEST_X86_TEST_AVX512_H)
#define EASYSIMD_TEST_X86_TEST_AVX512_H

#include "../test-x86.h"
#include "../test-avx.h"
#include "../../../easysimd/x86/avx512/storeu.h"
#include "../../../easysimd/x86/avx512/loadu.h"

EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m512, 32, 16, easysimd_mm512_storeu_ps)
EASYSIMD_TEST_X86_GENERATE_FLOAT_TYPE_FUNCS_(__m512d, 64, 8, easysimd_mm512_storeu_pd)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m512i, 8, 64, easysimd_mm512_storeu_si512)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m512i, 16, 32, easysimd_mm512_storeu_si512)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m512i, 32, 16, easysimd_mm512_storeu_si512)
EASYSIMD_TEST_X86_GENERATE_INT_TYPE_FUNCS_(__m512i, 64, 8, easysimd_mm512_storeu_si512)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m512i, 8, 64, easysimd_mm512_storeu_si512)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m512i, 16, 32, easysimd_mm512_storeu_si512)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m512i, 32, 16, easysimd_mm512_storeu_si512)
EASYSIMD_TEST_X86_GENERATE_UINT_TYPE_FUNCS_(__m512i, 64, 8, easysimd_mm512_storeu_si512)

#define EASYSIMD_TEST_X86_GENERATE_MASK_FUNCS_(EL) \
  static easysimd__mmask##EL \
  easysimd_test_x86_random_mmask##EL(void) { \
    return HEDLEY_STATIC_CAST(easysimd__mmask##EL, easysimd_test_codegen_random_u##EL()); \
  } \
 \
  static void \
  easysimd_test_x86_write_mmask##EL(int indent, easysimd__mmask##EL value, SimdeTestVecPos pos) { \
    easysimd_test_codegen_write_u##EL(indent, HEDLEY_STATIC_CAST(uint##EL##_t, value), pos); \
  }

EASYSIMD_TEST_X86_GENERATE_MASK_FUNCS_(8)
EASYSIMD_TEST_X86_GENERATE_MASK_FUNCS_(16)
EASYSIMD_TEST_X86_GENERATE_MASK_FUNCS_(32)
EASYSIMD_TEST_X86_GENERATE_MASK_FUNCS_(64)

#define easysimd_test_x86_assert_equal_f32x16(a, b, precision) do { if (easysimd_test_x86_assert_equal_f32x16_(a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_f64x8(a, b, precision) do { if (easysimd_test_x86_assert_equal_f64x8_(a, b, easysimd_test_f64_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i8x64(a, b) do { if (easysimd_test_x86_assert_equal_i8x64_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i16x32(a, b) do { if (easysimd_test_x86_assert_equal_i16x32_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i32x16(a, b) do { if (easysimd_test_x86_assert_equal_i32x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_i64x8(a, b) do { if (easysimd_test_x86_assert_equal_i64x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u8x64(a, b) do { if (easysimd_test_x86_assert_equal_u8x64_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u16x32(a, b) do { if (easysimd_test_x86_assert_equal_u16x32_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u32x16(a, b) do { if (easysimd_test_x86_assert_equal_u32x16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_test_x86_assert_equal_u64x8(a, b) do { if (easysimd_test_x86_assert_equal_u64x8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#if !defined(EASYSIMD_TEST_BARE)
  #define EASYSIMD_TEST_DECLARE_SUITE(name) EASYSIMD_TEST_SUITE_DECLARE_GETTERS(HEDLEY_CONCAT(easysimd_test_x86_avx512_get_suite_,name))
  #include <test/x86/avx512/declare-suites.h>
  #undef EASYSIMD_TEST_DECLARE_SUITE
#endif

#endif /* !defined(EASYSIMD_TEST_X86_TEST_AVX512_H) */
