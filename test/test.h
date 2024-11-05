#if !defined(EASYSIMD_TESTS_H)
#define EASYSIMD_TESTS_H

#include "../easysimd/easysimd-common.h"
#include "../easysimd/easysimd-f16.h"

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <stdarg.h>
#include <sys/time.h>

typedef enum SimdeTestVecPos {
  EASYSIMD_TEST_VEC_POS_SINGLE =  2,
  EASYSIMD_TEST_VEC_POS_FIRST  =  1,
  EASYSIMD_TEST_VEC_POS_MIDDLE =  0,
  EASYSIMD_TEST_VEC_POS_LAST   = -1
} SimdeTestVecPos;

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DIAGNOSTIC_DISABLE_VLA_
HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION
EASYSIMD_DIAGNOSTIC_DISABLE_PADDED_
EASYSIMD_DIAGNOSTIC_DISABLE_ZERO_AS_NULL_POINTER_CONSTANT_
EASYSIMD_DIAGNOSTIC_DISABLE_CAST_FUNCTION_TYPE_
EASYSIMD_DIAGNOSTIC_DISABLE_NON_CONSTANT_AGGREGATE_INITIALIZER_
EASYSIMD_DIAGNOSTIC_DISABLE_C99_EXTENSIONS_
EASYSIMD_DIAGNOSTIC_DISABLE_DECLARATION_AFTER_STATEMENT_
EASYSIMD_DIAGNOSTIC_DISABLE_NO_EMMS_INSTRUCTION_
EASYSIMD_DIAGNOSTIC_DISABLE_CPP98_COMPAT_PEDANTIC_
EASYSIMD_DIAGNOSTIC_DISABLE_ANNEX_K_
EASYSIMD_DIAGNOSTIC_DISABLE_DISABLED_MACRO_EXPANSION_

#if \
    HEDLEY_HAS_BUILTIN(__builtin_abort) || \
    HEDLEY_GCC_VERSION_CHECK(3,4,6) || \
    HEDLEY_ARM_VERSION_CHECK(4,1,0)
  #define easysimd_abort() __builtin_abort()
#elif defined(EASYSIMD_HAVE_STDLIB_H)
  #define easysimd_abort() abort()
#endif

#define EASYSIMD_TEST_ASSERT_CONTINUE 0
#define EASYSIMD_TEST_ASSERT_TRAP 1
#define EASYSIMD_TEST_ASSERT_ABORT 2
#if !defined(EASYSIMD_TEST_ASSERT_FAILURE)
  #if defined(EASYSIMD_TEST_BARE)
    #define EASYSIMD_TEST_ASSERT_FAILURE EASYSIMD_TEST_ASSERT_CONTINUE
  #else
    #define EASYSIMD_TEST_ASSERT_FAILURE EASYSIMD_TEST_ASSERT_ABORT
  #endif
#endif

#if !defined(EASYSIMD_TEST_ASSERT_ABORT) && !defined(EASYSIMD_TEST_ASSERT_CONTINUE) && !defined(EASYSIMD_TEST_ASSERT_TRAP)
  #if defined(EASYSIMD_TEST_BARE)
    #define EASYSIMD_TEST_ASSERT_CONTINUE
  #else
    #define EASYSIMD_TEST_ASSERT_ABORT
  #endif
#endif

#if EASYSIMD_TEST_ASSERT_FAILURE == EASYSIMD_TEST_ASSERT_ABORT
  #define EASYSIMD_TEST_ASSERT_RETURN(value) ((void) 0)
#else
  #define EASYSIMD_TEST_ASSERT_RETURN(value) return value
#endif

#if defined(EASYSIMD_TEST_BARE)
  #define EASYSIMD_CODEGEN_FP stderr
#else
  #define EASYSIMD_CODEGEN_FP stdout
#endif

#if EASYSIMD_TEST_ASSERT_FAILURE == 2
  HEDLEY_NO_RETURN
#endif
HEDLEY_PRINTF_FORMAT(1, 2)
static void
easysimd_test_debug_printf_(const char* format, ...) {
  va_list ap;

  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fflush(stderr);

  /* Debug trap is great for local development where you can attach a
   * debugger, but processes exiting with a SIGTRAP seem to be rather
   * confusing for CI. */
  #if EASYSIMD_TEST_ASSERT_FAILURE == 1
    easysimd_trap();
  #elif EASYSIMD_TEST_ASSERT_FAILURE == 2
    easysimd_abort();
  #endif
}

#if !defined(EASYSIMD_TEST_STRUCT_MODIFIERS)
  #define EASYSIMD_TEST_STRUCT_MODIFIERS static const
#endif

HEDLEY_PRINTF_FORMAT(3, 4)
static void
easysimd_test_codegen_snprintf_(char* str, size_t size, const char* format, ...) {
  va_list ap;
  int w;

  va_start(ap, format);
  w = vsnprintf(str, size, format, ap);
  va_end(ap);

  if (w > HEDLEY_STATIC_CAST(int, size)) {
    easysimd_test_debug_printf_("Not enough space to write value (given %zu bytes, need %d bytes)\n", size, w + 1);
  }
}

static void
easysimd_test_codegen_f16(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], easysimd_float16 value) {
  easysimd_float32 valuef = easysimd_float16_to_float32(value);
  if (easysimd_math_isnanf(valuef)) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "           EASYSIMD_NANHF");
  } else if (easysimd_math_isinf(valuef)) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%5cEASYSIMD_INFINITYHF", valuef < 0 ? '-' : ' ');
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "EASYSIMD_FLOAT16_VALUE(%9.2f)", HEDLEY_STATIC_CAST(double, valuef));
  }
}

static void
easysimd_test_codegen_f32(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], easysimd_float32 value) {
  if (easysimd_math_isnan(value)) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "           EASYSIMD_MATH_NANF");
  } else if (easysimd_math_isinf(value)) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%5cEASYSIMD_MATH_INFINITYF", value < 0 ? '-' : ' ');
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "EASYSIMD_FLOAT32_C(%9.2f)", HEDLEY_STATIC_CAST(double, value));
  }
}

static void
easysimd_test_codegen_f64(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], easysimd_float64 value) {
  if (easysimd_math_isnan(value)) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "            EASYSIMD_MATH_NAN");
  } else if (easysimd_math_isinf(value)) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%7cEASYSIMD_MATH_INFINITY", value < 0 ? '-' : ' ');
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "EASYSIMD_FLOAT64_C(%9.2f)", HEDLEY_STATIC_CAST(double, value));
  }
}

static void
easysimd_test_codegen_i8(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], int8_t value) {
  if (value == INT8_MIN) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "     INT8_MIN");
  } else if (value == INT8_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "     INT8_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%cINT8_C(%4" PRId8 ")", (value < 0) ? '-' : ' ', HEDLEY_STATIC_CAST(int8_t, (value < 0) ? -value : value));
  }
}

static void
easysimd_test_codegen_i16(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], int16_t value) {
  if (value == INT16_MIN) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%16s", "INT16_MIN");
  } else if (value == INT16_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%16s", "INT16_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%cINT16_C(%6" PRId16 ")", (value < 0) ? '-' : ' ', HEDLEY_STATIC_CAST(int16_t, (value < 0) ? -value : value));
  }
}

static void
easysimd_test_codegen_i32(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], int32_t value) {
  if (value == INT32_MIN) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%22s", "INT32_MIN");
  } else if (value == INT32_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%22s", "INT32_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%cINT32_C(%12" PRId32 ")", (value < 0) ? '-' : ' ', HEDLEY_STATIC_CAST(int32_t, (value < 0) ? -value : value));
  }
}

static void
easysimd_test_codegen_i64(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], int64_t value) {
  if (value == INT64_MIN) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%30s", "INT64_MIN");
  } else if (value == INT64_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%30s", "INT64_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%cINT64_C(%20" PRId64 ")", (value < 0) ? '-' : ' ', HEDLEY_STATIC_CAST(int64_t, (value < 0) ? -value : value));
  }
}

static void
easysimd_test_codegen_u8(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], uint8_t value) {
  if (value == UINT8_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "   UINT8_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "UINT8_C(%3" PRIu8 ")", value);
  }
}

static void
easysimd_test_codegen_u16(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], uint16_t value) {
  if (value == UINT16_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%15s", "UINT16_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "UINT16_C(%5" PRIu16 ")", value);
  }
}

static void
easysimd_test_codegen_u32(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], uint32_t value) {
  if (value == UINT32_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%20s", "UINT32_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "UINT32_C(%10" PRIu32 ")", value);
  }
}

static void
easysimd_test_codegen_u64(size_t buf_len, char buf[HEDLEY_ARRAY_PARAM(buf_len)], uint64_t value) {
  if (value == UINT64_MAX) {
    easysimd_test_codegen_snprintf_(buf, buf_len, "%30s", "UINT64_MAX");
  } else {
    easysimd_test_codegen_snprintf_(buf, buf_len, "UINT64_C(%20" PRIu64 ")", value);
  }
}

static void
easysimd_test_codegen_write_indent(int indent) {
  for (int i = 0 ; i < indent ; i++) {
    fputs("  ", EASYSIMD_CODEGEN_FP);
  }
}

static int easysimd_test_codegen_rand(void) {
  /* Single-threaded programs are so nice */
  static int is_init = 0;
  if (HEDLEY_UNLIKELY(!is_init)) {
    #if !defined(HEDLEY_EMSCRIPTEN_VERSION)
      FILE* fp = fopen("/dev/urandom", "r");
      if (fp == NULL)
        fp = fopen("/dev/random", "r");

      if (fp != NULL) {
        unsigned int seed;
        size_t nread = fread(&seed, sizeof(seed), 1, fp);
        fclose(fp);
        if (nread == 1) {
          srand(seed);
          is_init = 1;
        }
      }
    #endif

    if (!is_init) {
      srand(HEDLEY_STATIC_CAST(unsigned int, time(NULL)));
      is_init = 1;
    }
  }

  return rand();
}

static void
easysimd_test_codegen_random_memory(size_t buf_len, uint8_t buf[HEDLEY_ARRAY_PARAM(buf_len)]) {
  for (size_t i = 0 ; i < buf_len ; i++) {
    buf[i] = HEDLEY_STATIC_CAST(uint8_t, easysimd_test_codegen_rand() & 0xff);
  }
}

static easysimd_float32
easysimd_test_codegen_random_f32(easysimd_float32 min, easysimd_float32 max) {
  easysimd_float32 v = (HEDLEY_STATIC_CAST(easysimd_float32, easysimd_test_codegen_rand()) / (HEDLEY_STATIC_CAST(easysimd_float32, RAND_MAX) / (max - min))) + min;
  return easysimd_math_roundf(v * EASYSIMD_FLOAT32_C(100.0)) / EASYSIMD_FLOAT32_C(100.0);
}

static easysimd_float16
easysimd_test_codegen_random_f16(easysimd_float16 min, easysimd_float16 max) {
  return
    easysimd_float16_from_float32(
      easysimd_test_codegen_random_f32(
        easysimd_float16_to_float32(min),
        easysimd_float16_to_float32(max)
      )
    );
}

static easysimd_float64
easysimd_test_codegen_random_f64(easysimd_float64 min, easysimd_float64 max) {
  easysimd_float64 v = (HEDLEY_STATIC_CAST(easysimd_float64, easysimd_test_codegen_rand()) / (HEDLEY_STATIC_CAST(easysimd_float64, RAND_MAX) / (max - min))) + min;
  return easysimd_math_round(v * EASYSIMD_FLOAT64_C(100.0)) / EASYSIMD_FLOAT64_C(100.0);
}

typedef enum SimdeTestVecFloatMask {
  EASYSIMD_TEST_VEC_FLOAT_DEFAULT  = 0,
  EASYSIMD_TEST_VEC_FLOAT_PAIR     = 1,
  EASYSIMD_TEST_VEC_FLOAT_NAN      = 2,
  EASYSIMD_TEST_VEC_FLOAT_EQUAL    = 4,
  EASYSIMD_TEST_VEC_FLOAT_ROUND    = 8
}
#if \
    (HEDLEY_HAS_ATTRIBUTE(flag_enum) && !defined(HEDLEY_IBM_VERSION)) && \
    (!defined(__cplusplus) || EASYSIMD_DETECT_CLANG_VERSION_CHECK(5,0,0))
  __attribute__((__flag_enum__))
#endif
SimdeTestVecFloatType;

/* This is a bit messy, sorry.  And I haven't really tested with
 * anything greater than 4-element vectors, there is no input
 * validation, etc.  I'm not going to lose any sleep since it's
 * just a test harness, but you probably shouldn't use this API
 * directly since there is a good chance it will change. */

static void
easysimd_test_codegen_calc_pair(int pairwise, size_t test_sets, size_t vectors_per_set, size_t elements_per_vector, size_t pos, size_t* a, size_t* b) {
  (void) test_sets; // <- for validating ranges

  if (pairwise) {
    *a = (((pos * 2) + 0) % elements_per_vector) + ((((pos * 2) + 0) / elements_per_vector) * elements_per_vector);
    *b = (((pos * 2) + 1) % elements_per_vector) + ((((pos * 2) + 1) / elements_per_vector) * elements_per_vector);
  } else {
    size_t elements_per_set = elements_per_vector * vectors_per_set;
    size_t set_num = pos / elements_per_vector;
    size_t pos_in_set = pos % elements_per_vector;

    *a = (elements_per_set * set_num) + pos_in_set;
    *b = *a + elements_per_vector;
  }
}

static void
easysimd_test_codegen_float_set_value_(size_t element_size, size_t pos, void* values, easysimd_float32 f32_val, easysimd_float64 f64_val) {
  switch (element_size) {
    case sizeof(easysimd_float16):
      HEDLEY_REINTERPRET_CAST(easysimd_float16*, values)[pos] = easysimd_float16_from_float32(f32_val);
      break;
    case sizeof(easysimd_float32):
      HEDLEY_REINTERPRET_CAST(easysimd_float32*, values)[pos] = f32_val;
      break;
    case sizeof(easysimd_float64):
      HEDLEY_REINTERPRET_CAST(easysimd_float64*, values)[pos] = f64_val;
      break;
  }
}

static void
easysimd_test_codegen_random_vfX_full_(
    size_t test_sets, size_t vectors_per_set, size_t elements_per_vector,
    size_t elem_size, void* values,
    easysimd_float64 min, easysimd_float64 max,
    SimdeTestVecFloatType vec_type) {
  for (size_t i = 0 ; i < (test_sets * vectors_per_set * elements_per_vector) ; i++) {
    easysimd_float64 v = easysimd_test_codegen_random_f64(min, max);
    if (vec_type & EASYSIMD_TEST_VEC_FLOAT_ROUND) {
      if (easysimd_test_codegen_rand() & 7) {
        do {
          v = HEDLEY_STATIC_CAST(easysimd_float64, HEDLEY_STATIC_CAST(int64_t, v));
          if (easysimd_test_codegen_rand() & 7)
            v += 0.5;
        } while (v > max || v < min);
      }
    }
    easysimd_test_codegen_float_set_value_(elem_size, i, values, HEDLEY_STATIC_CAST(easysimd_float32, v), v);
  }

  int pairwise = !!(vec_type & EASYSIMD_TEST_VEC_FLOAT_PAIR);
  size_t pos = 0;
  size_t a, b;

  if (vec_type & EASYSIMD_TEST_VEC_FLOAT_NAN) {
    easysimd_test_codegen_calc_pair(pairwise, test_sets, vectors_per_set, elements_per_vector, pos++, &a, &b);
    easysimd_test_codegen_float_set_value_(elem_size, a, values, EASYSIMD_MATH_NANF, EASYSIMD_MATH_NAN);

    easysimd_test_codegen_calc_pair(pairwise, test_sets, vectors_per_set, elements_per_vector, pos++, &a, &b);
    easysimd_test_codegen_float_set_value_(elem_size, b, values, EASYSIMD_MATH_NANF, EASYSIMD_MATH_NAN);

    easysimd_test_codegen_calc_pair(pairwise, test_sets, vectors_per_set, elements_per_vector, pos++, &a, &b);
    easysimd_test_codegen_float_set_value_(elem_size, a, values, EASYSIMD_MATH_NANF, EASYSIMD_MATH_NAN);
    easysimd_test_codegen_float_set_value_(elem_size, b, values, EASYSIMD_MATH_NANF, EASYSIMD_MATH_NAN);
  }

  if (vec_type & EASYSIMD_TEST_VEC_FLOAT_EQUAL) {
    easysimd_test_codegen_calc_pair(pairwise, test_sets, vectors_per_set, elements_per_vector, pos++, &a, &b);
    easysimd_float64 v = easysimd_test_codegen_random_f64(min, max);
    easysimd_test_codegen_float_set_value_(elem_size, a, values, HEDLEY_STATIC_CAST(easysimd_float32, v), v);
    easysimd_test_codegen_float_set_value_(elem_size, b, values, HEDLEY_STATIC_CAST(easysimd_float32, v), v);
  }
}

static void
easysimd_test_codegen_random_vf16_full(
    size_t test_sets, size_t vectors_per_set, size_t elements_per_vector,
    easysimd_float16 values[HEDLEY_ARRAY_PARAM(test_sets * vectors_per_set * elements_per_vector)],
    easysimd_float16 min, easysimd_float16 max,
    SimdeTestVecFloatType vec_type) {
  easysimd_test_codegen_random_vfX_full_(test_sets, vectors_per_set, elements_per_vector,
      sizeof(easysimd_float16), values,
      HEDLEY_STATIC_CAST(easysimd_float64, easysimd_float16_to_float32(min)),
      HEDLEY_STATIC_CAST(easysimd_float64, easysimd_float16_to_float32(max)),
      vec_type);
}

static void
easysimd_test_codegen_random_vf32_full(
    size_t test_sets, size_t vectors_per_set, size_t elements_per_vector,
    easysimd_float32 values[HEDLEY_ARRAY_PARAM(test_sets * vectors_per_set * elements_per_vector)],
    easysimd_float32 min, easysimd_float32 max,
    SimdeTestVecFloatType vec_type) {
  easysimd_test_codegen_random_vfX_full_(test_sets, vectors_per_set, elements_per_vector,
      sizeof(easysimd_float32), values,
      HEDLEY_STATIC_CAST(easysimd_float64, min), HEDLEY_STATIC_CAST(easysimd_float64, max),
      vec_type);
}

static void
easysimd_test_codegen_random_vf64_full(
    size_t test_sets, size_t vectors_per_set, size_t elements_per_vector,
    easysimd_float64 values[HEDLEY_ARRAY_PARAM(test_sets * vectors_per_set * elements_per_vector)],
    easysimd_float64 min, easysimd_float64 max,
    SimdeTestVecFloatType vec_type) {
  easysimd_test_codegen_random_vfX_full_(test_sets, vectors_per_set, elements_per_vector,
      sizeof(easysimd_float64), values,
      min, max,
      vec_type);
}

static void
easysimd_test_codegen_random_vf16(size_t elem_count, easysimd_float16 values[HEDLEY_ARRAY_PARAM(elem_count)], easysimd_float16 min, easysimd_float16 max) {
  for (size_t i = 0 ; i < elem_count ; i++) {
    values[i] = easysimd_test_codegen_random_f16(min, max);
  }
}

static void
easysimd_test_codegen_random_vf32(size_t elem_count, easysimd_float32 values[HEDLEY_ARRAY_PARAM(elem_count)], easysimd_float32 min, easysimd_float32 max) {
  for (size_t i = 0 ; i < elem_count ; i++) {
    values[i] = easysimd_test_codegen_random_f32(min, max);
  }
}

static void
easysimd_test_codegen_random_vf64(size_t elem_count, easysimd_float64 values[HEDLEY_ARRAY_PARAM(elem_count)], easysimd_float64 min, easysimd_float64 max) {
  for (size_t i = 0 ; i < elem_count ; i++) {
    values[i] = easysimd_test_codegen_random_f64(min, max);
  }
}

#define EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(T, symbol_identifier) \
  static T easysimd_test_codegen_random_##symbol_identifier(void) { \
    T r; \
    easysimd_test_codegen_random_memory(sizeof(r), HEDLEY_REINTERPRET_CAST(uint8_t*, &r)); \
    return r; \
  }

EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(int8_t,    i8)
EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(int16_t,  i16)
EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(int32_t,  i32)
EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(int64_t,  i64)
EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(uint8_t,   u8)
EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(uint16_t, u16)
EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(uint32_t, u32)
EASYSIMD_TEST_CODEGEN_GENERATE_RANDOM_INT_FUNC_(uint64_t, u64)

#define EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(T, symbol_identifier, elements_per_line) \
  static void \
  easysimd_test_codegen_write_v##symbol_identifier##_full(int indent, size_t elem_count, const char* name, T values[HEDLEY_ARRAY_PARAM(elem_count)], SimdeTestVecPos pos) { \
    switch (pos) { \
      case EASYSIMD_TEST_VEC_POS_FIRST: \
        easysimd_test_codegen_write_indent(indent); \
        indent++; \
        fputs("{ ", EASYSIMD_CODEGEN_FP); \
        break; \
      case EASYSIMD_TEST_VEC_POS_MIDDLE: \
      case EASYSIMD_TEST_VEC_POS_LAST: \
        indent++; \
        easysimd_test_codegen_write_indent(indent); \
        break; \
      case EASYSIMD_TEST_VEC_POS_SINGLE: \
        easysimd_test_codegen_write_indent(indent++); \
        fprintf(EASYSIMD_CODEGEN_FP, "static const " #T " %s[] = \n", name); \
        easysimd_test_codegen_write_indent(indent); \
        break; \
    } \
    \
    fputs("{ ", EASYSIMD_CODEGEN_FP); \
    for (size_t i = 0 ; i < elem_count ; i++) { \
      if (i != 0) { \
        fputc(',', EASYSIMD_CODEGEN_FP); \
        if ((i % elements_per_line) == 0) { \
          fputc('\n', EASYSIMD_CODEGEN_FP); \
          easysimd_test_codegen_write_indent(indent + 1); \
        } else { \
          fputc(' ', EASYSIMD_CODEGEN_FP); \
        } \
      } \
    \
      char buf[53]; \
      easysimd_test_codegen_##symbol_identifier(sizeof(buf), buf, values[i]); \
      fputs(buf, EASYSIMD_CODEGEN_FP); \
    } \
    fputs(" }", EASYSIMD_CODEGEN_FP); \
    \
    switch (pos) { \
      case EASYSIMD_TEST_VEC_POS_FIRST: \
      case EASYSIMD_TEST_VEC_POS_MIDDLE: \
        fputc(',', EASYSIMD_CODEGEN_FP); \
        break; \
      case EASYSIMD_TEST_VEC_POS_LAST: \
        fputs(" },", EASYSIMD_CODEGEN_FP); \
        break; \
      case EASYSIMD_TEST_VEC_POS_SINGLE: \
        fputs(";", EASYSIMD_CODEGEN_FP); \
        break; \
    } \
    \
    fputc('\n', EASYSIMD_CODEGEN_FP); \
  } \
  \
  static void \
  easysimd_test_codegen_write_v##symbol_identifier(int indent, size_t elem_count, T values[HEDLEY_ARRAY_PARAM(elem_count)], SimdeTestVecPos pos) { \
    easysimd_test_codegen_write_v##symbol_identifier##_full(indent, elem_count, "???", values, pos); \
  }

EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(easysimd_float16, f16, 4)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(easysimd_float32, f32, 4)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(easysimd_float64, f64, 4)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(int8_t, i8, 8)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(int16_t, i16, 8)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(int32_t, i32, 8)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(int64_t, i64, 4)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(uint8_t, u8, 8)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(uint16_t, u16, 8)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(uint32_t, u32, 8)
EASYSIMD_TEST_CODEGEN_GENERATE_WRITE_VECTOR_FUNC_(uint64_t, u64, 4)

#define easysimd_test_codegen_write_1vi8(indent, elem_count, values)  easysimd_test_codegen_write_vi8_full( (indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vi16(indent, elem_count, values) easysimd_test_codegen_write_vi16_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vi32(indent, elem_count, values) easysimd_test_codegen_write_vi32_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vi64(indent, elem_count, values) easysimd_test_codegen_write_vi64_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vu8(indent, elem_count, values)  easysimd_test_codegen_write_vu8_full( (indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vu16(indent, elem_count, values) easysimd_test_codegen_write_vu16_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vu32(indent, elem_count, values) easysimd_test_codegen_write_vu32_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vu64(indent, elem_count, values) easysimd_test_codegen_write_vu64_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vf16(indent, elem_count, values) easysimd_test_codegen_write_vf16_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vf32(indent, elem_count, values) easysimd_test_codegen_write_vf32_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1vf64(indent, elem_count, values) easysimd_test_codegen_write_vf64_full((indent), (elem_count), #values, (values), EASYSIMD_TEST_VEC_POS_SINGLE)

#define EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(T, symbol_identifier) \
  static void \
  easysimd_test_codegen_write_##symbol_identifier##_full(int indent, const char* name, T value, SimdeTestVecPos pos) { \
    switch (pos) { \
      case EASYSIMD_TEST_VEC_POS_FIRST: \
        easysimd_test_codegen_write_indent(indent); \
        indent++; \
        fputs("{ ", EASYSIMD_CODEGEN_FP); \
        break; \
      case EASYSIMD_TEST_VEC_POS_MIDDLE: \
      case EASYSIMD_TEST_VEC_POS_LAST: \
        indent++; \
        easysimd_test_codegen_write_indent(indent); \
        break; \
      case EASYSIMD_TEST_VEC_POS_SINGLE: \
        easysimd_test_codegen_write_indent(indent++); \
        fprintf(EASYSIMD_CODEGEN_FP, "static const " #T " %s = ", name); \
        break; \
    } \
 \
    { \
      char buf[53]; \
      easysimd_test_codegen_##symbol_identifier(sizeof(buf), buf, value); \
      fputs(buf, EASYSIMD_CODEGEN_FP); \
    } \
 \
    switch (pos) { \
      case EASYSIMD_TEST_VEC_POS_FIRST: \
      case EASYSIMD_TEST_VEC_POS_MIDDLE: \
        fputc(',', EASYSIMD_CODEGEN_FP); \
        break; \
      case EASYSIMD_TEST_VEC_POS_LAST: \
        fputs(" },", EASYSIMD_CODEGEN_FP); \
        break; \
      case EASYSIMD_TEST_VEC_POS_SINGLE: \
        fputs(";", EASYSIMD_CODEGEN_FP); \
        break; \
    } \
    \
    fputc('\n', EASYSIMD_CODEGEN_FP); \
  } \
  \
  static void \
  easysimd_test_codegen_write_##symbol_identifier(int indent, T value, SimdeTestVecPos pos) { \
    easysimd_test_codegen_write_##symbol_identifier##_full(indent, "???", value, pos); \
  }

EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(int8_t,    i8)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(int16_t,  i16)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(int32_t,  i32)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(int64_t,  i64)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(uint8_t,   u8)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(uint16_t, u16)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(uint32_t, u32)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(uint64_t, u64)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(easysimd_float16, f16)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(easysimd_float32, f32)
EASYSIMD_TEST_CODEGEN_WRITE_SCALAR_FUNC_(easysimd_float64, f64)

#define easysimd_test_codegen_write_1i8(indent, value)  easysimd_test_codegen_write_i8_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1i16(indent, value) easysimd_test_codegen_write_i16_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1i32(indent, value) easysimd_test_codegen_write_i32_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1i64(indent, value) easysimd_test_codegen_write_i64_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1u8(indent, value)  easysimd_test_codegen_write_u8_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1u16(indent, value) easysimd_test_codegen_write_u16_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1u32(indent, value) easysimd_test_codegen_write_u32_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1u64(indent, value) easysimd_test_codegen_write_u64_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1f16(indent, value) easysimd_test_codegen_write_f16_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1f32(indent, value) easysimd_test_codegen_write_f32_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)
#define easysimd_test_codegen_write_1f64(indent, value) easysimd_test_codegen_write_f64_full((indent), #value, (value), EASYSIMD_TEST_VEC_POS_SINGLE)

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DIAGNOSTIC_DISABLE_FLOAT_EQUAL_

static int
easysimd_test_equal_f32(easysimd_float32 a, easysimd_float32 b, easysimd_float32 slop) {
  if (easysimd_math_isnan(a)) {
    return easysimd_math_isnan(b);
  } else if (easysimd_math_isinf(a)) {
    return !((a < b) || (a > b));
  } else if (slop == EASYSIMD_FLOAT32_C(0.0)) {
    return a == b;
  } else {
    easysimd_float32 lo = a - slop;
    if (HEDLEY_UNLIKELY(lo == a))
      lo = easysimd_math_nextafterf(a, -EASYSIMD_MATH_INFINITYF);

    easysimd_float32 hi = a + slop;
    if (HEDLEY_UNLIKELY(hi == a))
      hi = easysimd_math_nextafterf(a, EASYSIMD_MATH_INFINITYF);

    return ((b >= lo) && (b <= hi));
  }
}

static int
easysimd_test_equal_f16(easysimd_float16 a, easysimd_float16 b, easysimd_float16 slop) {
  easysimd_float32
    af = easysimd_float16_to_float32(a),
    bf = easysimd_float16_to_float32(b),
    slopf = easysimd_float16_to_float32(slop);
  return easysimd_test_equal_f32(af, bf, slopf);
}

static int
easysimd_test_equal_f64(easysimd_float64 a, easysimd_float64 b, easysimd_float64 slop) {
  if (easysimd_math_isnan(a)) {
    return easysimd_math_isnan(b);
  } else if (easysimd_math_isinf(a)) {
    return !((a < b) || (a > b));
  } else if (slop == EASYSIMD_FLOAT64_C(0.0)) {
    return a == b;
  } else {
    easysimd_float64 lo = a - slop;
    if (HEDLEY_UNLIKELY(lo == a))
      lo = easysimd_math_nextafter(a, -EASYSIMD_MATH_INFINITY);

    easysimd_float64 hi = a + slop;
    if (HEDLEY_UNLIKELY(hi == a))
      hi = easysimd_math_nextafter(a, EASYSIMD_MATH_INFINITY);

    return ((b >= lo) && (b <= hi));
  }
}

HEDLEY_DIAGNOSTIC_POP

static easysimd_float16
easysimd_test_f16_precision_to_slop(int precision) {
  return HEDLEY_UNLIKELY(precision == INT_MAX) ? EASYSIMD_FLOAT16_VALUE(0.0) : easysimd_float16_from_float32(easysimd_math_powf(EASYSIMD_FLOAT32_C(10.0), -HEDLEY_STATIC_CAST(float, precision)));
}

static float
easysimd_test_f32_precision_to_slop(int precision) {
  return HEDLEY_UNLIKELY(precision == INT_MAX) ? EASYSIMD_FLOAT32_C(0.0) : easysimd_math_powf(EASYSIMD_FLOAT32_C(10.0), -HEDLEY_STATIC_CAST(float, precision));
}

static double
easysimd_test_f64_precision_to_slop(int precision) {
  return HEDLEY_UNLIKELY(precision == INT_MAX) ? EASYSIMD_FLOAT64_C(0.0) : easysimd_math_pow(EASYSIMD_FLOAT64_C(10.0), -HEDLEY_STATIC_CAST(double, precision));
}

static int
easysimd_assert_equal_vf16_(
    size_t vec_len, easysimd_float16 const a[HEDLEY_ARRAY_PARAM(vec_len)], easysimd_float16 const b[HEDLEY_ARRAY_PARAM(vec_len)], easysimd_float16 slop,
    const char* filename, int line, const char* astr, const char* bstr) {
  easysimd_float32 slop_ = easysimd_float16_to_float32(slop);
  for (size_t i = 0 ; i < vec_len ; i++) {
    easysimd_float32 a_ = easysimd_float16_to_float32(a[i]);
    easysimd_float32 b_ = easysimd_float16_to_float32(b[i]);
    
    if (HEDLEY_UNLIKELY(!easysimd_test_equal_f32(a_, b_, slop_))) {
      easysimd_test_debug_printf_("%s:%d: assertion failed: %s[%zu] ~= %s[%zu] (%f ~= %f)\n",
              filename, line, astr, i, bstr, i, HEDLEY_STATIC_CAST(double, a_), 
              HEDLEY_STATIC_CAST(double, b_));
      EASYSIMD_TEST_ASSERT_RETURN(1);
    }
  }
  return 0;
}
#define easysimd_assert_equal_vf16(vec_len, a, b, precision) easysimd_assert_equal_vf16_(vec_len, a, b, easysimd_test_f16_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)

static int
easysimd_assert_equal_f16_(easysimd_float16 a, easysimd_float16 b, easysimd_float16 slop,
    const char* filename, int line, const char* astr, const char* bstr) {
  easysimd_float32 a_ = easysimd_float16_to_float32(a);
  easysimd_float32 b_ = easysimd_float16_to_float32(b);
  easysimd_float32 slop_ = easysimd_float16_to_float32(slop);
  if (HEDLEY_UNLIKELY(!easysimd_test_equal_f32(a_, b_, slop_))) {
    easysimd_test_debug_printf_("%s:%d: assertion failed: %s ~= %s (%f ~= %f)\n",
        filename, line, astr, bstr, HEDLEY_STATIC_CAST(double, a_),
        HEDLEY_STATIC_CAST(double, b_));
    EASYSIMD_TEST_ASSERT_RETURN(1);
  }
  return 0;
}
#define easysimd_assert_equal_f16(a, b, precision) easysimd_assert_equal_f16_(a, b, easysimd_test_f16_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)

static int
easysimd_assert_equal_vf32_(
    size_t vec_len, easysimd_float32 const a[HEDLEY_ARRAY_PARAM(vec_len)], easysimd_float32 const b[HEDLEY_ARRAY_PARAM(vec_len)], easysimd_float32 slop,
    const char* filename, int line, const char* astr, const char* bstr) {
  for (size_t i = 0 ; i < vec_len ; i++) {
    if (HEDLEY_UNLIKELY(!easysimd_test_equal_f32(a[i], b[i], slop))) {
      easysimd_test_debug_printf_("%s:%d: assertion failed: %s[%zu] ~= %s[%zu] (%f ~= %f)\n",
              filename, line, astr, i, bstr, i, HEDLEY_STATIC_CAST(double, a[i]), HEDLEY_STATIC_CAST(double, b[i]));
      EASYSIMD_TEST_ASSERT_RETURN(1);
    }
  }
  return 0;
}
#define easysimd_assert_equal_vf32(vec_len, a, b, precision) easysimd_assert_equal_vf32_(vec_len, a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)

static int
easysimd_assert_equal_f32_(easysimd_float32 a, easysimd_float32 b, easysimd_float32 slop,
    const char* filename, int line, const char* astr, const char* bstr) {
  if (HEDLEY_UNLIKELY(!easysimd_test_equal_f32(a, b, slop))) {
    easysimd_test_debug_printf_("%s:%d: assertion failed: %s ~= %s (%f ~= %f)\n",
        filename, line, astr, bstr, HEDLEY_STATIC_CAST(double, a), HEDLEY_STATIC_CAST(double, b));
    EASYSIMD_TEST_ASSERT_RETURN(1);
  }
  return 0;
}
#define easysimd_assert_equal_f32(a, b, precision) easysimd_assert_equal_f32_(a, b, easysimd_test_f32_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)

static int
easysimd_assert_equal_vf64_(
    size_t vec_len, easysimd_float64 const a[HEDLEY_ARRAY_PARAM(vec_len)], easysimd_float64 const b[HEDLEY_ARRAY_PARAM(vec_len)], easysimd_float64 slop,
    const char* filename, int line, const char* astr, const char* bstr) {
  for (size_t i = 0 ; i < vec_len ; i++) {
    if (HEDLEY_UNLIKELY(!easysimd_test_equal_f64(a[i], b[i], slop))) {
      easysimd_test_debug_printf_("%s:%d: assertion failed: %s[%zu] ~= %s[%zu] (%f ~= %f)\n",
              filename, line, astr, i, bstr, i, HEDLEY_STATIC_CAST(double, a[i]), HEDLEY_STATIC_CAST(double, b[i]));
      EASYSIMD_TEST_ASSERT_RETURN(1);
    }
  }
  return 0;
}
#define easysimd_assert_equal_vf64(vec_len, a, b, precision) easysimd_assert_equal_vf64_(vec_len, a, b, easysimd_test_f64_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)

static int
easysimd_assert_equal_f64_(easysimd_float64 a, easysimd_float64 b, easysimd_float64 slop,
    const char* filename, int line, const char* astr, const char* bstr) {
  if (HEDLEY_UNLIKELY(!easysimd_test_equal_f64(a, b, slop))) {
    easysimd_test_debug_printf_("%s:%d: assertion failed: %s ~= %s (%f ~= %f)\n",
        filename, line, astr, bstr, a, b);
    EASYSIMD_TEST_ASSERT_RETURN(1);
  }
  return 0;
}
#define easysimd_assert_equal_f64(a, b, precision) easysimd_assert_equal_f64_(a, b, easysimd_test_f64_precision_to_slop(precision), __FILE__, __LINE__, #a, #b)

#define EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(T, symbol_identifier, fmt) \
  static int \
  easysimd_assert_equal_v##symbol_identifier##_( \
      size_t vec_len, const T a[HEDLEY_ARRAY_PARAM(vec_len)], const T b[HEDLEY_ARRAY_PARAM(vec_len)], \
      const char* filename, int line, const char* astr, const char* bstr) { \
    for (size_t i = 0 ; i < vec_len ; i++) { \
      if (HEDLEY_UNLIKELY(a[i] != b[i])) { \
        easysimd_test_debug_printf_("%s:%d: assertion failed: %s[%zu] == %s[%zu] (%" fmt " == %" fmt ")\n", \
              filename, line, astr, i, bstr, i, a[i], b[i]); \
        EASYSIMD_TEST_ASSERT_RETURN(1); \
      } \
    } \
    return 0; \
  } \
  \
  static int \
  easysimd_assert_equal_##symbol_identifier##_(T a, T b, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    if (HEDLEY_UNLIKELY(a != b)) { \
      easysimd_test_debug_printf_("%s:%d: assertion failed: %s == %s (%" fmt " == %" fmt ")\n", \
            filename, line, astr, bstr, a, b); \
      EASYSIMD_TEST_ASSERT_RETURN(1); \
    } \
    return 0; \
  } \
  \
  static int \
  easysimd_assert_close_v##symbol_identifier##_( \
      size_t vec_len, const T a[HEDLEY_ARRAY_PARAM(vec_len)], const T b[HEDLEY_ARRAY_PARAM(vec_len)], const T slop, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    for (size_t i = 0 ; i < vec_len ; i++) { \
      if (((a[i] + slop) < b[i]) || ((a[i] - slop) > b[i])) { \
        easysimd_test_debug_printf_("%s:%d: assertion failed: %s[%zu] == %s[%zu] (%" fmt " == %" fmt ")\n", \
              filename, line, astr, i, bstr, i, a[i], b[i]); \
        EASYSIMD_TEST_ASSERT_RETURN(1); \
      } \
    } \
    return 0; \
  } \
  \
  static int \
  easysimd_assert_close_##symbol_identifier##_(T a, T b, T slop, \
      const char* filename, int line, const char* astr, const char* bstr) { \
    if (((a + slop) < b) || ((a - slop) > b)) { \
      easysimd_test_debug_printf_("%s:%d: assertion failed: %s == %s +/- %" fmt " (%" fmt " == %" fmt ")\n", \
            filename, line, astr, bstr, slop, a, b); \
      EASYSIMD_TEST_ASSERT_RETURN(1); \
    } \
    return 0; \
  }

static int
easysimd_assert_equal_i_(int a, int b, const char* filename, int line, const char* astr, const char* bstr) {
  if (HEDLEY_UNLIKELY(a != b)) {
    easysimd_test_debug_printf_("%s:%d: assertion failed: %s == %s (%d == %d)\n",
          filename, line, astr, bstr, a, b);
    EASYSIMD_TEST_ASSERT_RETURN(1);
  }
  return 0;
}

EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(int8_t,    i8,  PRId8)
EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(int16_t,  i16, PRId16)
EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(int32_t,  i32, PRId32)
EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(int64_t,  i64, PRId64)
EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(uint8_t,   u8,  PRIu8)
EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(uint16_t, u16, PRIu16)
EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(uint32_t, u32, PRIu32)
EASYSIMD_TEST_GENERATE_ASSERT_EQUAL_FUNC_(uint64_t, u64, PRIu64)

#define easysimd_assert_equal_vi8(vec_len, a, b) do { if (easysimd_assert_equal_vi8_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_vi16(vec_len, a, b) do { if (easysimd_assert_equal_vi16_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_vi32(vec_len, a, b) do { if (easysimd_assert_equal_vi32_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_vi64(vec_len, a, b) do { if (easysimd_assert_equal_vi64_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_vu8(vec_len, a, b) do { if (easysimd_assert_equal_vu8_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_vu16(vec_len, a, b) do { if (easysimd_assert_equal_vu16_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_vu32(vec_len, a, b) do { if (easysimd_assert_equal_vu32_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_vu64(vec_len, a, b) do { if (easysimd_assert_equal_vu64_(vec_len, a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#define easysimd_assert_equal_i8(a, b) do { if (easysimd_assert_equal_i8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_i16(a, b) do { if (easysimd_assert_equal_i16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_i32(a, b) do { if (easysimd_assert_equal_i32_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_i64(a, b) do { if (easysimd_assert_equal_i64_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_u8(a, b) do { if (easysimd_assert_equal_u8_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_u16(a, b) do { if (easysimd_assert_equal_u16_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_u32(a, b) do { if (easysimd_assert_equal_u32_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_u64(a, b) do { if (easysimd_assert_equal_u64_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_equal_i(a, b) do { if (easysimd_assert_equal_i_(a, b, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#define easysimd_assert_close_vi8(vec_len, a, b, slop) do { if (easysimd_assert_close_vi8_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_vi16(vec_len, a, b, slop) do { if (easysimd_assert_close_vi16_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_vi32(vec_len, a, b, slop) do { if (easysimd_assert_close_vi32_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_vi64(vec_len, a, b, slop) do { if (easysimd_assert_close_vi64_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_vu8(vec_len, a, b, slop) do { if (easysimd_assert_close_vu8_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_vu16(vec_len, a, b, slop) do { if (easysimd_assert_close_vu16_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_vu32(vec_len, a, b, slop) do { if (easysimd_assert_close_vu32_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_vu64(vec_len, a, b, slop) do { if (easysimd_assert_close_vu64_(vec_len, a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

#define easysimd_assert_close_i8(a, b, slop) do { if (easysimd_assert_close_i8_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_i16(a, b, slop) do { if (easysimd_assert_close_i16_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_i32(a, b, slop) do { if (easysimd_assert_close_i32_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_i64(a, b, slop) do { if (easysimd_assert_close_i64_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_u8(a, b, slop) do { if (easysimd_assert_close_u8_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_u16(a, b, slop) do { if (easysimd_assert_close_u16_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_u32(a, b, slop) do { if (easysimd_assert_close_u32_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_u64(a, b, slop) do { if (easysimd_assert_close_u64_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)
#define easysimd_assert_close_i(a, b, slop) do { if (easysimd_assert_close_i_(a, b, slop, __FILE__, __LINE__, #a, #b)) { return 1; } } while (0)

/* Since each test is compiled in 4 different versions (C/C++ and
 * native/emul), we need to be able to generate different symbols
 * depending on preprocessor macros. */
#if defined(EASYSIMD_NO_NATIVE)
  #if defined(__cplusplus)
    #define EASYSIMD_TEST_GENERATE_VARIANT_SYMBOL_CURRENT(name) HEDLEY_CONCAT(name,_emul_cpp)
    #define EASYSIMD_TEST_GENERATE_VARIANT_NAME_CURRENT(name) #name "/emul/cpp"
  #else
    #define EASYSIMD_TEST_GENERATE_VARIANT_SYMBOL_CURRENT(name) HEDLEY_CONCAT(name,_emul_c)
    #define EASYSIMD_TEST_GENERATE_VARIANT_NAME_CURRENT(name) #name "/emul/c"
  #endif
#else
  #if defined(__cplusplus)
    #define EASYSIMD_TEST_GENERATE_VARIANT_SYMBOL_CURRENT(name) HEDLEY_CONCAT(name,_native_cpp)
    #define EASYSIMD_TEST_GENERATE_VARIANT_NAME_CURRENT(name) #name "/native/cpp"
  #else
    #define EASYSIMD_TEST_GENERATE_VARIANT_SYMBOL_CURRENT(name) HEDLEY_CONCAT(name,_native_c)
    #define EASYSIMD_TEST_GENERATE_VARIANT_NAME_CURRENT(name) #name "/native/c"
  #endif
#endif

/* The bare version basically assumes you just want to run a single
 * test suite.  It doesn't use munit, or any other dependencies so
 * it's easy to use with creduce. */
#if defined(EASYSIMD_TEST_BARE)
  #define EASYSIMD_TEST_FUNC_LIST_BEGIN static const struct { int (* func)(void); const char* name; } test_suite_tests[] = {
  #define EASYSIMD_TEST_FUNC_LIST_ENTRY(name) { test_easysimd_##name, #name },
  #define EASYSIMD_TEST_FUNC_LIST_END };
  #define EASYSIMD_MUNIT_TEST_ARGS void
#else
  HEDLEY_DIAGNOSTIC_PUSH
  EASYSIMD_DIAGNOSTIC_DISABLE_CPP98_COMPAT_PEDANTIC_
  EASYSIMD_DIAGNOSTIC_DISABLE_OLD_STYLE_CAST_
  EASYSIMD_DIAGNOSTIC_DISABLE_VARIADIC_MACROS_
  EASYSIMD_DIAGNOSTIC_DISABLE_RESERVED_ID_MACRO_
  #include "munit/munit.h"
  HEDLEY_DIAGNOSTIC_POP

  #if \
      HEDLEY_HAS_ATTRIBUTE(unused) || \
      HEDLEY_GCC_VERSION_CHECK(3,1,0)
    #define EASYSIMD_MUNIT_TEST_ARGS __attribute__((__unused__)) const MunitParameter params[], __attribute__((__unused__)) void* data
  #else
    /* Compilers other than emscripten are fine with casting away
     * arguments. */
    #define EASYSIMD_MUNIT_TEST_ARGS void
  #endif

  #define EASYSIMD_TEST_FUNC_LIST_BEGIN static MunitTest test_suite_tests[] = {
  #if defined(__cplusplus)
    #define EASYSIMD_TEST_FUNC_LIST_ENTRY(name) { \
        const_cast<char*>("/" EASYSIMD_TEST_GENERATE_VARIANT_NAME_CURRENT(name)), \
        HEDLEY_REINTERPRET_CAST(MunitTestFunc, test_easysimd_##name), \
        NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
  #else
    #define EASYSIMD_TEST_FUNC_LIST_ENTRY(name) { \
        (char*) "/" EASYSIMD_TEST_GENERATE_VARIANT_NAME_CURRENT(name), \
        HEDLEY_REINTERPRET_CAST(MunitTestFunc, test_easysimd_##name), \
        NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
  #endif
  #define EASYSIMD_TEST_FUNC_LIST_END { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL } };

  #define EASYSIMD_TEST_SUITE_DECLARE_GETTERS(name) \
    HEDLEY_C_DECL MunitSuite* HEDLEY_CONCAT(name, _native_c)(void); \
    HEDLEY_C_DECL MunitSuite* HEDLEY_CONCAT(name, _emul_c)(void); \
    HEDLEY_C_DECL MunitSuite* HEDLEY_CONCAT(name, _native_cpp)(void); \
    HEDLEY_C_DECL MunitSuite* HEDLEY_CONCAT(name, _emul_cpp)(void);
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
static inline uint64_t easysimd_test_timer_real(void)
{
	struct timeval tm;
	gettimeofday( &tm, NULL );
	return tm.tv_sec * UINT64_C(1000000) + tm.tv_usec;
}

static inline void easysimd_do_not_optimeze(int value) {
  __asm__ __volatile__ ("" : : "r,m"(value) : "memory");
}

#define EASYSIMD_TEST_PERF_END(name) \
  easysimd_do_not_optimeze(i_);} \
	uint64_t minunit_end_real_timer = easysimd_test_timer_real(); \
	printf("Test %s performance, Finished in %lu us\n", name, minunit_end_real_timer - minunit_real_timer); }

#define EASYSIMD_TEST_PERF_WITH_LOOP_START(n_) { \
  uint64_t minunit_real_timer = easysimd_test_timer_real(); \
  for (int i_ = 0; i_ < n_; i_++) { \
    easysimd_do_not_optimeze(i_); \

#define EASYSIMD_TEST_PERF_TIMES 1000000
#else
  #define EASYSIMD_TEST_PERF_END(name)
  #define EASYSIMD_TEST_PERF_WITH_LOOP_START(n)
  #define EASYSIMD_TEST_PERF_DEFAULT_TIMES
#endif

#endif /* !defined(EASYSIMD_TESTS_H) */
