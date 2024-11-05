/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *  2020      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_X86_AVX512_TYPES_H)
#define EASYSIMD_X86_AVX512_TYPES_H

#include "../avx.h"
#if defined(EASYSIMD_ARM_SVE_NATIVE) 
#include "../../arm/sve.h"
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

/* The problem is that Microsoft doesn't support 64-byte aligned parameters, except for
 * __m512/__m512i/__m512d.  Since our private union has an __m512 member it will be 64-byte
 * aligned even if we reduce the alignment requirements of other members.
 *
 * Even if we're on x86 and use the native AVX-512 types for arguments/return values, the
 * to/from private functions will break, and I'm not willing to change their APIs to use
 * pointers (which would also require more verbose code on the caller side) just to make
 * MSVC happy.
 *
 * If you want to use AVX-512 in SIMDe, you'll need to either upgrade to MSVC 2017 or later,
 * or upgrade to a different compiler (clang-cl, perhaps?).  If you have an idea of how to
 * fix this without requiring API changes (except transparently through macros), patches
 * are welcome.
 */

#  if defined(HEDLEY_MSVC_VERSION) && !HEDLEY_MSVC_VERSION_CHECK(19,10,0)
#    if defined(EASYSIMD_X86_AVX512F_NATIVE)
#      undef EASYSIMD_X86_AVX512F_NATIVE
#      pragma message("Native AVX-512 support requires MSVC 2017 or later.  See comment above (in code) for details.")
#    endif
#    define EASYSIMD_AVX512_ALIGN EASYSIMD_ALIGN_TO_32
#  else
#    define EASYSIMD_AVX512_ALIGN EASYSIMD_ALIGN_TO_64
#  endif


typedef easysimd__m128_private  easysimd__m128bh_private;
typedef easysimd__m256_private  easysimd__m256bh_private;

typedef union {
 #if defined(EASYSIMD_VECTOR_SUBSCRIPT)
    EASYSIMD_AVX512_ALIGN int8_t          i8 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN int16_t        i16 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN int32_t        i32 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN int64_t        i64 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN uint8_t         u8 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN uint16_t       u16 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN uint32_t       u32 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN uint64_t       u64 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    #if defined(EASYSIMD_HAVE_INT128_)
    EASYSIMD_AVX512_ALIGN easysimd_int128  i128 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN easysimd_uint128 u128 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    #endif
    EASYSIMD_AVX512_ALIGN easysimd_float32  f32 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN easysimd_float64  f64 EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN int_fast32_t  i32f EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    EASYSIMD_AVX512_ALIGN uint_fast32_t u32f EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
  #else
    EASYSIMD_AVX512_ALIGN int8_t          i8[64];
    EASYSIMD_AVX512_ALIGN int16_t        i16[32];
    EASYSIMD_AVX512_ALIGN int32_t        i32[16];
    EASYSIMD_AVX512_ALIGN int64_t        i64[8];
    EASYSIMD_AVX512_ALIGN uint8_t         u8[64];
    EASYSIMD_AVX512_ALIGN uint16_t       u16[32];
    EASYSIMD_AVX512_ALIGN uint32_t       u32[16];
    EASYSIMD_AVX512_ALIGN uint64_t       u64[8];
    #if defined(EASYSIMD_HAVE_INT128_)
    EASYSIMD_AVX512_ALIGN easysimd_int128  i128[4];
    EASYSIMD_AVX512_ALIGN easysimd_uint128 u128[4];
    #endif
    EASYSIMD_AVX512_ALIGN easysimd_float32  f32[16];
    EASYSIMD_AVX512_ALIGN easysimd_float64  f64[8];
    EASYSIMD_AVX512_ALIGN int_fast32_t  i32f[64 / sizeof(int_fast32_t)];
    EASYSIMD_AVX512_ALIGN uint_fast32_t u32f[64 / sizeof(uint_fast32_t)];
  #endif

    EASYSIMD_AVX512_ALIGN easysimd__m128d_private m128d_private[4];
    EASYSIMD_AVX512_ALIGN easysimd__m128d         m128d[4];
    EASYSIMD_AVX512_ALIGN easysimd__m256d_private m256d_private[2];
    EASYSIMD_AVX512_ALIGN easysimd__m256d         m256d[2];
    EASYSIMD_AVX512_ALIGN easysimd__m128i_private m128i_private[4];
    EASYSIMD_AVX512_ALIGN easysimd__m128i         m128i[4];
    EASYSIMD_AVX512_ALIGN easysimd__m256i_private m256i_private[2];
    EASYSIMD_AVX512_ALIGN easysimd__m256i         m256i[2];
    EASYSIMD_AVX512_ALIGN easysimd__m128_private m128_private[4];
    EASYSIMD_AVX512_ALIGN easysimd__m128         m128[4];
    EASYSIMD_AVX512_ALIGN easysimd__m256_private m256_private[2];
    EASYSIMD_AVX512_ALIGN easysimd__m256         m256[2];

   #if defined(EASYSIMD_ARM_SVE_NATIVE)
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint8_t    sve_i8[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint16_t   sve_i16[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint32_t   sve_i32[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint64_t   sve_i64[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint8_t   sve_u8[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint16_t  sve_u16[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint32_t  sve_u32[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint64_t  sve_u64[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat8_t  sve_f8[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat16_t sve_f16[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat32_t sve_f32[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat64_t sve_f64[EASYSIMD_512_BITS_SV_ARRAY_SIZE];
  #endif

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_ALIGN_TO_32 int8x16x4_t  neon_i8x4;
    EASYSIMD_ALIGN_TO_32 uint8x16x4_t neon_u8x4;
    EASYSIMD_ALIGN_TO_32 int32x4x4_t  neon_i32x4;
    EASYSIMD_ALIGN_TO_32 int64x2x4_t  neon_i64x4;
    EASYSIMD_ALIGN_TO_32 uint64x2x4_t neon_u64x4;
  #endif
   

  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE)
    EASYSIMD_AVX512_ALIGN __m512bh         nbh;
  #endif
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    EASYSIMD_AVX512_ALIGN __m512d        nd;
    EASYSIMD_AVX512_ALIGN __m512i        ni;
    EASYSIMD_AVX512_ALIGN __m512         n;
  #endif
} easysimd__m512_private;
typedef easysimd__m512_private  easysimd__m512i_private;
typedef easysimd__m512_private  easysimd__m512d_private;
typedef easysimd__m512_private  easysimd__m512bh_private;


/* Intel uses the same header (immintrin.h) for everything AVX and
 * later.  If native aliases are enabled, and the machine has native
 * support for AVX imintrin.h will already have been included, which
 * means easysimd__m512* will already have been defined.  So, even
 * if the machine doesn't support AVX512F we need to use the native
 * type; it has already been defined.
 *
 * However, we also can't just assume that including immintrin.h does
 * actually define these.  It could be a compiler which supports AVX
 * but not AVX512F, such as GCC < 4.9 or VS < 2017.  That's why we
 * check to see if _MM_CMPINT_GE is defined; it's part of AVX512F,
 * so we assume that if it's present AVX-512F has already been
 * declared.
 *
 * Note that the choice of _MM_CMPINT_GE is deliberate; while GCC
 * uses the preprocessor to define all the _MM_CMPINT_* members,
 * in most compilers they are simply normal enum members.  However,
 * all compilers I've looked at use an object-like macro for
 * _MM_CMPINT_GE, which is defined to _MM_CMPINT_NLT.  _MM_CMPINT_NLT
 * is included in case a compiler does the reverse, though I haven't
 * run into one which does.
 *
 * As for the ICC check, unlike other compilers, merely using the
 * AVX-512 types causes ICC to generate AVX-512 instructions. */
#if (defined(_MM_CMPINT_GE) || defined(_MM_CMPINT_NLT)) && (defined(EASYSIMD_X86_AVX512F_NATIVE) || !defined(HEDLEY_INTEL_VERSION))
  typedef __m512 easysimd__m512;
  typedef __m512i easysimd__m512i;
  typedef __m512d easysimd__m512d;

  typedef __mmask8 easysimd__mmask8;
  typedef __mmask16 easysimd__mmask16;
#else
 #if defined(EASYSIMD_VECTOR_SUBSCRIPT)
  #if defined(EASYSIMD_CONVERT_TO_PRIVATE)
   typedef easysimd_float32 easysimd__m512  EASYSIMD_AVX512_ALIGN EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
   typedef int_fast32_t  easysimd__m512i EASYSIMD_AVX512_ALIGN EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
   typedef easysimd_float64 easysimd__m512d EASYSIMD_AVX512_ALIGN EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
  #else
    typedef easysimd__m512_private  easysimd__m512;
    typedef easysimd__m512i_private easysimd__m512i;
    typedef easysimd__m512d_private easysimd__m512d;
  #endif

  #else
    typedef easysimd__m512_private  easysimd__m512;
    typedef easysimd__m512i_private easysimd__m512i;
    typedef easysimd__m512d_private easysimd__m512d;
  #endif

  typedef uint8_t easysimd__mmask8;
  typedef uint16_t easysimd__mmask16;
#endif

#if (defined(_AVX512BF16INTRIN_H_INCLUDED) || defined(__AVX512BF16INTRIN_H)) && (defined(EASYSIMD_X86_AVX512BF16_NATIVE) || !defined(HEDLEY_INTEL_VERSION))
  typedef __m128bh easysimd__m128bh;
  typedef __m256bh easysimd__m256bh;
  typedef __m512bh easysimd__m512bh;
#else
 #if defined(EASYSIMD_VECTOR_SUBSCRIPT)
    #if defined(EASYSIMD_CONVERT_TO_PRIVATE)
      typedef easysimd_float32 easysimd__m128bh  EASYSIMD_ALIGN_TO_16  EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
      typedef easysimd_float32 easysimd__m256bh  EASYSIMD_ALIGN_TO_32  EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
      typedef easysimd_float32 easysimd__m512bh  EASYSIMD_AVX512_ALIGN EASYSIMD_VECTOR(64) EASYSIMD_MAY_ALIAS;
    #else
      typedef easysimd__m128bh_private easysimd__m128bh;
      typedef easysimd__m256bh_private easysimd__m256bh;
      typedef easysimd__m512bh_private easysimd__m512bh;
    #endif
  #else
    typedef easysimd__m128bh_private easysimd__m128bh;
    typedef easysimd__m256bh_private easysimd__m256bh;
    typedef easysimd__m512bh_private easysimd__m512bh;
  #endif
#endif

/* These are really part of AVX-512VL / AVX-512BW (in GCC __mmask32 is
 * in avx512vlintrin.h and __mmask64 is in avx512bwintrin.h, in clang
 * both are in avx512bwintrin.h), not AVX-512F.  However, we don't have
 * a good (not-compiler-specific) way to detect if these headers have
 * been included.  In compilers which support AVX-512F but not
 * AVX-512BW/VL (e.g., GCC 4.9) we need typedefs since __mmask{32,64)
 * won't exist.
 *
 * AFAICT __mmask{32,64} are always just typedefs to uint{32,64}_t
 * in all compilers, so it's safe to use these instead of typedefs to
 * __mmask{16,32}. If you run into a problem with this please file an
 * issue and we'll try to figure out a work-around. */
typedef uint32_t easysimd__mmask32;
typedef uint64_t easysimd__mmask64;

#if !defined(__mmask16) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
  #if !defined(HEDLEY_INTEL_VERSION)
    typedef uint16_t __mmask16;
  #else
    #define __mmask16 uint16_t;
  #endif
#endif
#if !defined(__mmask32) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
  #if !defined(HEDLEY_INTEL_VERSION)
    typedef uint32_t __mmask32;
  #else
    #define __mmask32 uint32_t;
  #endif
#endif
#if !defined(__mmask64) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
  #if !defined(HEDLEY_INTEL_VERSION)
    #if defined(HEDLEY_GCC_VERSION)
      typedef unsigned long long __mmask64;
    #else
      typedef uint64_t __mmask64;
    #endif
  #else
    #define __mmask64 uint64_t;
  #endif
#endif

#if !defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
  #if !defined(HEDLEY_INTEL_VERSION)
    typedef easysimd__m512 __m512;
    typedef easysimd__m512i __m512i;
    typedef easysimd__m512d __m512d;
  #else
    #define __m512 easysimd__m512
    #define __m512i easysimd__m512i
    #define __m512d easysimd__m512d
  #endif
#endif

#if !defined(EASYSIMD_X86_AVX512BF16_NATIVE) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
  #if !defined(HEDLEY_INTEL_VERSION)
    typedef easysimd__m128bh __m128bh;
    typedef easysimd__m256bh __m256bh;
    typedef easysimd__m512bh __m512bh;
  #else
    #define __m128bh easysimd__m128bh
    #define __m256bh easysimd__m256bh
    #define __m512bh easysimd__m512bh
  #endif
#endif

HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128bh), "easysimd__m128bh size incorrect");
HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128bh_private), "easysimd__m128bh_private size incorrect");
HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256bh), "easysimd__m256bh size incorrect");
HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256bh_private), "easysimd__m256bh_private size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512bh), "easysimd__m512bh size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512bh_private), "easysimd__m512bh_private size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512), "easysimd__m512 size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512_private), "easysimd__m512_private size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512i), "easysimd__m512i size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512i_private), "easysimd__m512i_private size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512d), "easysimd__m512d size incorrect");
HEDLEY_STATIC_ASSERT(64 == sizeof(easysimd__m512d_private), "easysimd__m512d_private size incorrect");
#if defined(EASYSIMD_CHECK_ALIGNMENT) && defined(EASYSIMD_ALIGN_OF)
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128bh) == 16, "easysimd__m128bh is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128bh_private) == 16, "easysimd__m128bh_private is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256bh) == 32, "easysimd__m256bh is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256bh_private) == 32, "easysimd__m256bh_private is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512bh) == 32, "easysimd__m512bh is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512bh_private) == 32, "easysimd__m512bh_private is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512) == 32, "easysimd__m512 is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512_private) == 32, "easysimd__m512_private is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512i) == 32, "easysimd__m512i is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512i_private) == 32, "easysimd__m512i_private is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512d) == 32, "easysimd__m512d is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m512d_private) == 32, "easysimd__m512d_private is not 32-byte aligned");
#endif

#if defined(EASYSIMD_CONVERT_TO_PRIVATE) || defined(_AVX512BF16INTRIN_H_INCLUDED)
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128bh
easysimd__m128bh_from_private(easysimd__m128bh_private v) {
  easysimd__m128bh r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128bh_private
easysimd__m128bh_to_private(easysimd__m128bh v) {
  easysimd__m128bh_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256bh
easysimd__m256bh_from_private(easysimd__m256bh_private v) {
  easysimd__m256bh r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256bh_private
easysimd__m256bh_to_private(easysimd__m256bh v) {
  easysimd__m256bh_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512bh
easysimd__m512bh_from_private(easysimd__m512bh_private v) {
  easysimd__m512bh r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512bh_private
easysimd__m512bh_to_private(easysimd__m512bh v) {
  easysimd__m512bh_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd__m512_from_private(easysimd__m512_private v) {
  easysimd__m512 r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512_private
easysimd__m512_to_private(easysimd__m512 v) {
  easysimd__m512_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd__m512i_from_private(easysimd__m512i_private v) {
  easysimd__m512i r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i_private
easysimd__m512i_to_private(easysimd__m512i v) {
  easysimd__m512i_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd__m512d_from_private(easysimd__m512d_private v) {
  easysimd__m512d r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d_private
easysimd__m512d_to_private(easysimd__m512d v) {
  easysimd__m512d_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}
#else 
#define easysimd__m128bh_from_private(v) v
#define easysimd__m128bh_to_private(v) v
#define easysimd__m256bh_from_private(v) v
#define easysimd__m256bh_to_private(v) v
#define easysimd__m512bh_from_private(v) v
#define easysimd__m512bh_to_private(v) v
#define easysimd__m512_from_private(v) v
#define easysimd__m512_to_private(v) v
#define easysimd__m512i_from_private(v) v
#define easysimd__m512i_to_private(v) v
#define easysimd__m512d_from_private(v) v
#define easysimd__m512d_to_private(v) v
#endif

#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
/* extract highest bit from every 32bit */ 
#define EXTRACT_HB_32x16(res, sign)                                                                             \
    {                                                                                                        \
        res.m128i[0].neon_u32 = vshrq_n_u32(res.m128i[0].neon_u32, 31);                                                  \
        res.m128i[1].neon_u32 = vshrq_n_u32(res.m128i[1].neon_u32, 31);                                                  \
        res.m128i[2].neon_u32 = vshrq_n_u32(res.m128i[2].neon_u32, 31);                                                  \
        res.m128i[3].neon_u32 = vshrq_n_u32(res.m128i[3].neon_u32, 31);                                                  \
        res.m128i[0].neon_u64 = vsraq_n_u64(res.m128i[0].neon_u64, res.m128i[0].neon_u64, 31);                                 \
        res.m128i[1].neon_u64 = vsraq_n_u64(res.m128i[1].neon_u64, res.m128i[1].neon_u64, 31);                                 \
        res.m128i[2].neon_u64 = vsraq_n_u64(res.m128i[2].neon_u64, res.m128i[2].neon_u64, 31);                                 \
        res.m128i[3].neon_u64 = vsraq_n_u64(res.m128i[3].neon_u64, res.m128i[3].neon_u64, 31);                                 \
        *sign = (vgetq_lane_u8(res.m128i[0].neon_u8, 0) | (vgetq_lane_u8(res.m128i[0].neon_u8, 8) << 2) |                \
                (vgetq_lane_u8(res.m128i[1].neon_u8, 0) << 4) | (vgetq_lane_u8(res.m128i[1].neon_u8, 8) << 6) |          \
                (vgetq_lane_u8(res.m128i[2].neon_u8, 0) << 8) | (vgetq_lane_u8(res.m128i[2].neon_u8, 8) << 10) |         \
                (vgetq_lane_u8(res.m128i[3].neon_u8, 0) << 12) | (vgetq_lane_u8(res.m128i[3].neon_u8, 8) << 14));        \
    };
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_TYPES_H) */
