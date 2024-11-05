/* Debugging assertions and traps
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

#if !defined(EASYSIMD_DEBUG_TRAP_H)
#define EASYSIMD_DEBUG_TRAP_H

#if !defined(EASYSIMD_NDEBUG) && defined(NDEBUG) && !defined(EASYSIMD_DEBUG)
#  define EASYSIMD_NDEBUG 1
#endif

#if defined(__has_builtin) && !defined(__ibmxl__)
#  if __has_builtin(__builtin_debugtrap)
#    define easysimd_trap() __builtin_debugtrap()
#  elif __has_builtin(__debugbreak)
#    define easysimd_trap() __debugbreak()
#  endif
#endif
#if !defined(easysimd_trap)
#  if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#    define easysimd_trap() __debugbreak()
#  elif defined(__ARMCC_VERSION)
#    define easysimd_trap() __breakpoint(42)
#  elif defined(__ibmxl__) || defined(__xlC__)
#    include <builtins.h>
#    define easysimd_trap() __trap(42)
#  elif defined(__DMC__) && defined(_M_IX86)
     static inline void easysimd_trap(void) { __asm int 3h; }
#  elif defined(__i386__) || defined(__x86_64__)
     static inline void easysimd_trap(void) { __asm__ __volatile__("int $03"); }
#  elif defined(__thumb__)
     static inline void easysimd_trap(void) { __asm__ __volatile__(".inst 0xde01"); }
#  elif defined(__aarch64__)
     static inline void easysimd_trap(void) { __asm__ __volatile__(".inst 0xd4200000"); }
#  elif defined(__arm__)
     static inline void easysimd_trap(void) { __asm__ __volatile__(".inst 0xe7f001f0"); }
#  elif defined (__alpha__) && !defined(__osf__)
     static inline void easysimd_trap(void) { __asm__ __volatile__("bpt"); }
#  elif defined(_54_)
     static inline void easysimd_trap(void) { __asm__ __volatile__("ESTOP"); }
#  elif defined(_55_)
     static inline void easysimd_trap(void) { __asm__ __volatile__(";\n .if (.MNEMONIC)\n ESTOP_1\n .else\n ESTOP_1()\n .endif\n NOP"); }
#  elif defined(_64P_)
     static inline void easysimd_trap(void) { __asm__ __volatile__("SWBP 0"); }
#  elif defined(_6x_)
     static inline void easysimd_trap(void) { __asm__ __volatile__("NOP\n .word 0x10000000"); }
#  elif defined(__STDC_HOSTED__) && (__STDC_HOSTED__ == 0) && defined(__GNUC__)
#    define easysimd_trap() __builtin_trap()
#  else
#    include <signal.h>
#    if defined(SIGTRAP)
#      define easysimd_trap() raise(SIGTRAP)
#    else
#      define easysimd_trap() raise(SIGABRT)
#    endif
#  endif
#endif

#if defined(HEDLEY_LIKELY)
#  define EASYSIMD_DBG_LIKELY(expr) HEDLEY_LIKELY(expr)
#elif defined(__GNUC__) && (__GNUC__ >= 3)
#  define EASYSIMD_DBG_LIKELY(expr) __builtin_expect(!!(expr), 1)
#else
#  define EASYSIMD_DBG_LIKELY(expr) (!!(expr))
#endif

#if !defined(EASYSIMD_NDEBUG) || (EASYSIMD_NDEBUG == 0)
#  define easysimd_dbg_assert(expr) do { \
    if (!EASYSIMD_DBG_LIKELY(expr)) { \
      easysimd_trap(); \
    } \
  } while (0)
#else
#  define easysimd_dbg_assert(expr)
#endif

#endif /* !defined(EASYSIMD_DEBUG_TRAP_H) */
