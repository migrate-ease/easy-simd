/* Check (assertions)
 * Portable Snippets - https://gitub.com/nemequ/portable-snippets
 * Created by Evan Nemerson <evan@nemerson.com>
 *
 *   To the extent possible under law, the authors have waived all
 *   copyright and related or neighboring rights to this code.  For
 *   details, see the Creative Commons Zero 1.0 Universal license at
 *   https://creativecommons.org/publicdomain/zero/1.0/
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#if !defined(EASYSIMD_CHECK_H)
#define EASYSIMD_CHECK_H

#if !defined(EASYSIMD_NDEBUG) && !defined(EASYSIMD_DEBUG)
#  define EASYSIMD_NDEBUG 1
#endif

#include "hedley.h"
#include "easysimd-diagnostic.h"
#include <stdint.h>

#if defined(_MSC_VER) &&  (_MSC_VER >= 1500)
#  define EASYSIMD_PUSH_DISABLE_MSVC_C4127_ __pragma(warning(push)) __pragma(warning(disable:4127))
#  define EASYSIMD_POP_DISABLE_MSVC_C4127_ __pragma(warning(pop))
#else
#  define EASYSIMD_PUSH_DISABLE_MSVC_C4127_
#  define EASYSIMD_POP_DISABLE_MSVC_C4127_
#endif

#if !defined(easysimd_errorf)
#  if defined(__has_include)
#    if __has_include(<stdio.h>)
#      include <stdio.h>
#    endif
#  elif defined(EASYSIMD_STDC_HOSTED)
#    if EASYSIMD_STDC_HOSTED == 1
#      include <stdio.h>
#    endif
#  elif defined(__STDC_HOSTED__)
#    if __STDC_HOSTETD__ == 1
#      include <stdio.h>
#    endif
#  endif

#  include "debug-trap.h"

   HEDLEY_DIAGNOSTIC_PUSH
   EASYSIMD_DIAGNOSTIC_DISABLE_VARIADIC_MACROS_
#  if defined(EOF)
#    define easysimd_errorf(format, ...) (fprintf(stderr, format, __VA_ARGS__), abort())
#  else
#    define easysimd_errorf(format, ...) (easysimd_trap())
#  endif
   HEDLEY_DIAGNOSTIC_POP
#endif

#define easysimd_error(msg) easysimd_errorf("%s", msg)

#if defined(EASYSIMD_NDEBUG) || \
    (defined(__cplusplus) && (__cplusplus < 201103L)) || \
    (defined(__STDC__) && (__STDC__ < 199901L))
#  if defined(EASYSIMD_CHECK_FAIL_DEFINED)
#    define easysimd_assert(expr)
#  else
#    if defined(HEDLEY_ASSUME)
#      define easysimd_assert(expr) HEDLEY_ASSUME(expr)
#    elif HEDLEY_GCC_VERSION_CHECK(4,5,0)
#      define easysimd_assert(expr) ((void) (!!(expr) ? 1 : (__builtin_unreachable(), 1)))
#    elif HEDLEY_MSVC_VERSION_CHECK(13,10,0)
#      define easysimd_assert(expr) __assume(expr)
#    else
#      define easysimd_assert(expr)
#    endif
#  endif
#  define easysimd_assert_true(expr) easysimd_assert(expr)
#  define easysimd_assert_false(expr) easysimd_assert(!(expr))
#  define easysimd_assert_type_full(prefix, suffix, T, fmt, a, op, b) easysimd_assert(((a) op (b)))
#  define easysimd_assert_double_equal(a, b, precision)
#  define easysimd_assert_string_equal(a, b)
#  define easysimd_assert_string_not_equal(a, b)
#  define easysimd_assert_memory_equal(size, a, b)
#  define easysimd_assert_memory_not_equal(size, a, b)
#else
#  define easysimd_assert(expr) \
    do { \
      if (!HEDLEY_LIKELY(expr)) { \
        easysimd_error("assertion failed: " #expr "\n"); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  define easysimd_assert_true(expr) \
    do { \
      if (!HEDLEY_LIKELY(expr)) { \
        easysimd_error("assertion failed: " #expr " is not true\n"); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  define easysimd_assert_false(expr) \
    do { \
      if (!HEDLEY_LIKELY(!(expr))) { \
        easysimd_error("assertion failed: " #expr " is not false\n"); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  define easysimd_assert_type_full(prefix, suffix, T, fmt, a, op, b)   \
    do { \
      T easysimd_tmp_a_ = (a); \
      T easysimd_tmp_b_ = (b); \
      if (!(easysimd_tmp_a_ op easysimd_tmp_b_)) { \
        easysimd_errorf("assertion failed: %s %s %s (" prefix "%" fmt suffix " %s " prefix "%" fmt suffix ")\n", \
                     #a, #op, #b, easysimd_tmp_a_, #op, easysimd_tmp_b_); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  define easysimd_assert_double_equal(a, b, precision) \
    do { \
      const double easysimd_tmp_a_ = (a); \
      const double easysimd_tmp_b_ = (b); \
      const double easysimd_tmp_diff_ = ((easysimd_tmp_a_ - easysimd_tmp_b_) < 0) ? \
        -(easysimd_tmp_a_ - easysimd_tmp_b_) : \
        (easysimd_tmp_a_ - easysimd_tmp_b_); \
      if (HEDLEY_UNLIKELY(easysimd_tmp_diff_ > 1e-##precision)) { \
        easysimd_errorf("assertion failed: %s == %s (%0." #precision "g == %0." #precision "g)\n", \
                     #a, #b, easysimd_tmp_a_, easysimd_tmp_b_); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  include <string.h>
#  define easysimd_assert_string_equal(a, b) \
    do { \
      const char* easysimd_tmp_a_ = a; \
      const char* easysimd_tmp_b_ = b; \
      if (HEDLEY_UNLIKELY(strcmp(easysimd_tmp_a_, easysimd_tmp_b_) != 0)) { \
        easysimd_errorf("assertion failed: string %s == %s (\"%s\" == \"%s\")\n", \
                     #a, #b, easysimd_tmp_a_, easysimd_tmp_b_); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  define easysimd_assert_string_not_equal(a, b) \
    do { \
      const char* easysimd_tmp_a_ = a; \
      const char* easysimd_tmp_b_ = b; \
      if (HEDLEY_UNLIKELY(strcmp(easysimd_tmp_a_, easysimd_tmp_b_) == 0)) { \
        easysimd_errorf("assertion failed: string %s != %s (\"%s\" == \"%s\")\n", \
                     #a, #b, easysimd_tmp_a_, easysimd_tmp_b_); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  define easysimd_assert_memory_equal(size, a, b) \
    do { \
      const unsigned char* easysimd_tmp_a_ = (const unsigned char*) (a); \
      const unsigned char* easysimd_tmp_b_ = (const unsigned char*) (b); \
      const size_t easysimd_tmp_size_ = (size); \
      if (HEDLEY_UNLIKELY(memcmp(easysimd_tmp_a_, easysimd_tmp_b_, easysimd_tmp_size_)) != 0) { \
        size_t easysimd_tmp_pos_; \
        for (easysimd_tmp_pos_ = 0 ; easysimd_tmp_pos_ < easysimd_tmp_size_ ; easysimd_tmp_pos_++) { \
          if (easysimd_tmp_a_[easysimd_tmp_pos_] != easysimd_tmp_b_[easysimd_tmp_pos_]) { \
            easysimd_errorf("assertion failed: memory %s == %s, at offset %" EASYSIMD_SIZE_MODIFIER "u\n", \
                         #a, #b, easysimd_tmp_pos_); \
            break; \
          } \
        } \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_

#  define easysimd_assert_memory_not_equal(size, a, b) \
    do { \
      const unsigned char* easysimd_tmp_a_ = (const unsigned char*) (a); \
      const unsigned char* easysimd_tmp_b_ = (const unsigned char*) (b); \
      const size_t easysimd_tmp_size_ = (size); \
      if (HEDLEY_UNLIKELY(memcmp(easysimd_tmp_a_, easysimd_tmp_b_, easysimd_tmp_size_)) == 0) { \
        easysimd_errorf("assertion failed: memory %s != %s (%" EASYSIMD_SIZE_MODIFIER "u bytes)\n", \
                     #a, #b, easysimd_tmp_size_); \
      } \
      EASYSIMD_PUSH_DISABLE_MSVC_C4127_ \
    } while (0) \
    EASYSIMD_POP_DISABLE_MSVC_C4127_
#endif

#define easysimd_assert_type(T, fmt, a, op, b) \
  easysimd_assert_type_full("", "", T, fmt, a, op, b)

#define easysimd_assert_char(a, op, b) \
  easysimd_assert_type_full("'\\x", "'", char, "02" EASYSIMD_CHAR_MODIFIER "x", a, op, b)
#define easysimd_assert_uchar(a, op, b) \
  easysimd_assert_type_full("'\\x", "'", unsigned char, "02" EASYSIMD_CHAR_MODIFIER "x", a, op, b)
#define easysimd_assert_short(a, op, b) \
  easysimd_assert_type(short, EASYSIMD_SHORT_MODIFIER "d", a, op, b)
#define easysimd_assert_ushort(a, op, b) \
  easysimd_assert_type(unsigned short, EASYSIMD_SHORT_MODIFIER "u", a, op, b)
#define easysimd_assert_int(a, op, b) \
  easysimd_assert_type(int, "d", a, op, b)
#define easysimd_assert_uint(a, op, b) \
  easysimd_assert_type(unsigned int, "u", a, op, b)
#define easysimd_assert_long(a, op, b) \
  easysimd_assert_type(long int, "ld", a, op, b)
#define easysimd_assert_ulong(a, op, b) \
  easysimd_assert_type(unsigned long int, "lu", a, op, b)
#define easysimd_assert_llong(a, op, b) \
  easysimd_assert_type(long long int, "lld", a, op, b)
#define easysimd_assert_ullong(a, op, b) \
  easysimd_assert_type(unsigned long long int, "llu", a, op, b)

#define easysimd_assert_size(a, op, b) \
  easysimd_assert_type(size_t, EASYSIMD_SIZE_MODIFIER "u", a, op, b)

#define easysimd_assert_float(a, op, b) \
  easysimd_assert_type(float, "f", a, op, b)
#define easysimd_assert_double(a, op, b) \
  easysimd_assert_type(double, "g", a, op, b)
#define easysimd_assert_ptr(a, op, b) \
  easysimd_assert_type(const void*, "p", a, op, b)

#define easysimd_assert_int8(a, op, b) \
  easysimd_assert_type(int8_t, PRIi8, a, op, b)
#define easysimd_assert_uint8(a, op, b) \
  easysimd_assert_type(uint8_t, PRIu8, a, op, b)
#define easysimd_assert_int16(a, op, b) \
  easysimd_assert_type(int16_t, PRIi16, a, op, b)
#define easysimd_assert_uint16(a, op, b) \
  easysimd_assert_type(uint16_t, PRIu16, a, op, b)
#define easysimd_assert_int32(a, op, b) \
  easysimd_assert_type(int32_t, PRIi32, a, op, b)
#define easysimd_assert_uint32(a, op, b) \
  easysimd_assert_type(uint32_t, PRIu32, a, op, b)
#define easysimd_assert_int64(a, op, b) \
  easysimd_assert_type(int64_t, PRIi64, a, op, b)
#define easysimd_assert_uint64(a, op, b) \
  easysimd_assert_type(uint64_t, PRIu64, a, op, b)

#define easysimd_assert_ptr_equal(a, b) \
  easysimd_assert_ptr(a, ==, b)
#define easysimd_assert_ptr_not_equal(a, b) \
  easysimd_assert_ptr(a, !=, b)
#define easysimd_assert_null(ptr) \
  easysimd_assert_ptr(ptr, ==, NULL)
#define easysimd_assert_not_null(ptr) \
  easysimd_assert_ptr(ptr, !=, NULL)
#define easysimd_assert_ptr_null(ptr) \
  easysimd_assert_ptr(ptr, ==, NULL)
#define easysimd_assert_ptr_not_null(ptr) \
  easysimd_assert_ptr(ptr, !=, NULL)

#endif /* !defined(EASYSIMD_CHECK_H) */
