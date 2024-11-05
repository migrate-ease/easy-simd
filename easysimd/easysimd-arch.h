/* Architecture detection
 * Created by Evan Nemerson <evan@nemerson.com>
 *
 *   To the extent possible under law, the authors have waived all
 *   copyright and related or neighboring rights to this code.  For
 *   details, see the Creative Commons Zero 1.0 Universal license at
 *   <https://creativecommons.org/publicdomain/zero/1.0/>
 *
 * SPDX-License-Identifier: CC0-1.0
 *
 * Different compilers define different preprocessor macros for the
 * same architecture.  This is an attempt to provide a single
 * interface which is usable on any compiler.
 *
 * In general, a macro named EASYSIMD_ARCH_* is defined for each
 * architecture the CPU supports.  When there are multiple possible
 * versions, we try to define the macro to the target version.  For
 * example, if you want to check for i586+, you could do something
 * like:
 *
 *   #if defined(EASYSIMD_ARCH_X86) && (EASYSIMD_ARCH_X86 >= 5)
 *   ...
 *   #endif
 *
 * You could also just check that EASYSIMD_ARCH_X86 >= 5 without checking
 * if it's defined first, but some compilers may emit a warning about
 * an undefined macro being used (e.g., GCC with -Wundef).
 *
 * This was originally created for SIMDe
 * <https://github.com/simd-everywhere/simde> (hence the prefix), but this
 * header has no dependencies and may be used anywhere.  It is
 * originally based on information from
 * <https://sourceforge.net/p/predef/wiki/Architectures/>, though it
 * has been enhanced with additional information.
 *
 * If you improve this file, or find a bug, please file the issue at
 * <https://github.com/simd-everywhere/simde/issues>.  If you copy this into
 * your project, even if you change the prefix, please keep the links
 * to SIMDe intact so others know where to report issues, submit
 * enhancements, and find the latest version. */

#if !defined(EASYSIMD_ARCH_H)
#define EASYSIMD_ARCH_H

/* AMD64 / x86_64
   <https://en.wikipedia.org/wiki/X86-64> */
#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
#  if !defined(_M_ARM64EC)
#     define EASYSIMD_ARCH_AMD64 1000
#  endif
#endif

/* ARM
   <https://en.wikipedia.org/wiki/ARM_architecture> */
#if defined(__ARM_ARCH)
#  if __ARM_ARCH > 100
#    define EASYSIMD_ARCH_ARM (__ARM_ARCH)
#  else
#    define EASYSIMD_ARCH_ARM (__ARM_ARCH * 100)
#  endif
#elif defined(_M_ARM)
#  if _M_ARM > 100
#    define EASYSIMD_ARCH_ARM (_M_ARM)
#  else
#    define EASYSIMD_ARCH_ARM (_M_ARM * 100)
#  endif
#elif defined(_M_ARM64) || defined(_M_ARM64EC)
#  define EASYSIMD_ARCH_ARM 800
#elif defined(__arm__) || defined(__thumb__) || defined(__TARGET_ARCH_ARM) || defined(_ARM) || defined(_M_ARM) || defined(_M_ARM)
#  define EASYSIMD_ARCH_ARM 1
#endif
#if defined(EASYSIMD_ARCH_ARM)
#  define EASYSIMD_ARCH_ARM_CHECK(major, minor) (((major * 100) + (minor)) <= EASYSIMD_ARCH_ARM)
#else
#  define EASYSIMD_ARCH_ARM_CHECK(major, minor) (0)
#endif

/* AArch64
   <https://en.wikipedia.org/wiki/ARM_architecture> */
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#  define EASYSIMD_ARCH_AARCH64 1000
#endif
#if defined(EASYSIMD_ARCH_AARCH64)
#  define EASYSIMD_ARCH_AARCH64_CHECK(version) ((version) <= EASYSIMD_ARCH_AARCH64)
#else
#  define EASYSIMD_ARCH_AARCH64_CHECK(version) (0)
#endif

/* ARM SIMD ISA extensions */
#if defined(__ARM_NEON) || defined(EASYSIMD_ARCH_AARCH64)
#  if defined(EASYSIMD_ARCH_AARCH64)
#    define EASYSIMD_ARCH_ARM_NEON EASYSIMD_ARCH_AARCH64
#  elif defined(EASYSIMD_ARCH_ARM)
#    define EASYSIMD_ARCH_ARM_NEON EASYSIMD_ARCH_ARM
#  endif
#endif
#if defined(__ARM_FEATURE_SVE)
#  define EASYSIMD_ARCH_ARM_SVE
#endif

/* x86
   <https://en.wikipedia.org/wiki/X86> */
#if defined(_M_IX86)
#  define EASYSIMD_ARCH_X86 (_M_IX86 / 100)
#elif defined(__I86__)
#  define EASYSIMD_ARCH_X86 __I86__
#elif defined(i686) || defined(__i686) || defined(__i686__)
#  define EASYSIMD_ARCH_X86 6
#elif defined(i586) || defined(__i586) || defined(__i586__)
#  define EASYSIMD_ARCH_X86 5
#elif defined(i486) || defined(__i486) || defined(__i486__)
#  define EASYSIMD_ARCH_X86 4
#elif defined(i386) || defined(__i386) || defined(__i386__)
#  define EASYSIMD_ARCH_X86 3
#elif defined(_X86_) || defined(__X86__) || defined(__THW_INTEL__)
#  define EASYSIMD_ARCH_X86 3
#endif
#if defined(EASYSIMD_ARCH_X86)
#  define EASYSIMD_ARCH_X86_CHECK(version) ((version) <= EASYSIMD_ARCH_X86)
#else
#  define EASYSIMD_ARCH_X86_CHECK(version) (0)
#endif

/* SIMD ISA extensions for x86/x86_64 and Elbrus */
#if defined(EASYSIMD_ARCH_X86) || defined(EASYSIMD_ARCH_AMD64) || defined(EASYSIMD_ARCH_E2K)
#  if defined(_M_IX86_FP)
#    define EASYSIMD_ARCH_X86_MMX
#    if (_M_IX86_FP >= 1)
#      define EASYSIMD_ARCH_X86_SSE 1
#    endif
#    if (_M_IX86_FP >= 2)
#      define EASYSIMD_ARCH_X86_SSE2 1
#    endif
#  elif defined(_M_X64)
#    define EASYSIMD_ARCH_X86_SSE 1
#    define EASYSIMD_ARCH_X86_SSE2 1
#  else
#    if defined(__MMX__)
#      define EASYSIMD_ARCH_X86_MMX 1
#    endif
#    if defined(__SSE__)
#      define EASYSIMD_ARCH_X86_SSE 1
#    endif
#    if defined(__SSE2__)
#      define EASYSIMD_ARCH_X86_SSE2 1
#    endif
#  endif
#  if defined(__SSE3__)
#    define EASYSIMD_ARCH_X86_SSE3 1
#  endif
#  if defined(__SSSE3__)
#    define EASYSIMD_ARCH_X86_SSSE3 1
#  endif
#  if defined(__SSE4_1__)
#    define EASYSIMD_ARCH_X86_SSE4_1 1
#  endif
#  if defined(__SSE4_2__)
#    define EASYSIMD_ARCH_X86_SSE4_2 1
#  endif
#  if defined(__XOP__)
#    define EASYSIMD_ARCH_X86_XOP 1
#  endif
#  if defined(__AVX__)
#    define EASYSIMD_ARCH_X86_AVX 1
#    if !defined(EASYSIMD_ARCH_X86_SSE3)
#      define EASYSIMD_ARCH_X86_SSE3 1
#    endif
#    if !defined(EASYSIMD_ARCH_X86_SSE4_1)
#      define EASYSIMD_ARCH_X86_SSE4_1 1
#    endif
#    if !defined(EASYSIMD_ARCH_X86_SSE4_1)
#      define EASYSIMD_ARCH_X86_SSE4_2 1
#    endif
#  endif
#  if defined(__AVX2__)
#    define EASYSIMD_ARCH_X86_AVX2 1
#  endif
#  if defined(__FMA__)
#    define EASYSIMD_ARCH_X86_FMA 1
#    if !defined(EASYSIMD_ARCH_X86_AVX)
#      define EASYSIMD_ARCH_X86_AVX 1
#    endif
#  endif
#  if defined(__AVX512VP2INTERSECT__)
#    define EASYSIMD_ARCH_X86_AVX512VP2INTERSECT 1
#  endif
#  if defined(__AVX512BITALG__)
#    define EASYSIMD_ARCH_X86_AVX512BITALG 1
#  endif
#  if defined(__AVX512VPOPCNTDQ__)
#    define EASYSIMD_ARCH_X86_AVX512VPOPCNTDQ 1
#  endif
#  if defined(__AVX512VBMI__)
#    define EASYSIMD_ARCH_X86_AVX512VBMI 1
#  endif
#  if defined(__AVX512VBMI2__)
#    define EASYSIMD_ARCH_X86_AVX512VBMI2 1
#  endif
#  if defined(__AVX512VNNI__)
#    define EASYSIMD_ARCH_X86_AVX512VNNI 1
#  endif
#  if defined(__AVX5124VNNIW__)
#    define EASYSIMD_ARCH_X86_AVX5124VNNIW 1
#  endif
#  if defined(__AVX512BW__)
#    define EASYSIMD_ARCH_X86_AVX512BW 1
#  endif
#  if defined(__AVX512BF16__)
#    define EASYSIMD_ARCH_X86_AVX512BF16 1
#  endif
#  if defined(__AVX512CD__)
#    define EASYSIMD_ARCH_X86_AVX512CD 1
#  endif
#  if defined(__AVX512DQ__)
#    define EASYSIMD_ARCH_X86_AVX512DQ 1
#  endif
#  if defined(__AVX512F__)
#    define EASYSIMD_ARCH_X86_AVX512F 1
#  endif
#  if defined(__AVX512VL__)
#    define EASYSIMD_ARCH_X86_AVX512VL 1
#  endif
#  if defined(__GFNI__)
#    define EASYSIMD_ARCH_X86_GFNI 1
#  endif
#  if defined(__PCLMUL__)
#    define EASYSIMD_ARCH_X86_PCLMUL 1
#  endif
#  if defined(__VPCLMULQDQ__)
#    define EASYSIMD_ARCH_X86_VPCLMULQDQ 1
#  endif
#  if defined(__F16C__)
#    define EASYSIMD_ARCH_X86_F16C 1
#  endif
#endif

/* Availability of 16-bit floating-point arithmetic intrinsics */
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#  define EASYSIMD_ARCH_ARM_NEON_FP16
#endif

#endif /* !defined(EASYSIMD_ARCH_H) */
