# easy-simd
## Introduction
easy-simd is a header-only intrinsic portable library launched by T-HEAD, which allows intrinsic code to run on CPU architectures that do not support this instruction set, such as allowing calls on the arm64 platform
x86 SSE instructions. easy-simd reduces the workload of application software migration, and at the same time, it can also accelerate the performance of applications. easy-simd has been applied in many businesses, such as:
- Native Engine in Big Data
- Roaring BitMap in redis 
- GATK, Gromacs in HPC 
- ffmeg, The popular video codec library

## License
easy-simd is distributed under an MIT-style license; see license for details.
## Requirements
- Linux
- GCC 10 or better

## Build && Test
You can compile and test easy-simd with the following two commands:
Compile:
./build.sh all
Test:
./build.sh test

## How to Use
- Copy the easysimd directory to the /usr/local/include/
- Modify the x86 intrinsic header to <easysimd/easysimd.h>
- Add the following compilation options
    -DEASYSIMD_ENABLE_NATIVE_ALIASES
    -march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3  -msve-vector-bits=128
    -I/usr/local/include/easysimd
- It is recommended that the application use compilation optimization options above O2

## Example
```
#include <stdlib.h>
#include <string.h>
#include <easysimd/easysimd.h>

int main()
{
  struct {
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec = {
      {  INT32_C(  1587156417),  INT32_C(  1768270179), -INT32_C(  1942404587),  INT32_C(   346970517) },
      {  INT32_C(  2141391970),  INT32_C(  1584534422),  INT32_C(  1144809083), -INT32_C(   446909148) },
      { -INT32_C(   566418909), -INT32_C(   942162695), -INT32_C(   797595504), -INT32_C(    99938631) }
  };

    __m128i a = _mm_loadu_epi32(test_vec.a);
    __m128i b = _mm_loadu_epi32(test_vec.b);
    __m128i r = _mm_loadu_epi32(test_vec.r);
    __m128i sum = _mm_add_epi32(a, b);

    if (memcmp(&sum, &r, sizeof(__m128i)) != 0) {
        fprintf(stderr, "Example test failed.\n");
    } else {
        fprintf(stderr, "Example test OK.\n");
    }

    return 0;
}
Compile command:
gcc -O2 -fPIC -Wall add.c -o add -I/usr/local/include/easysimd -DEASYSIMD_ENABLE_NATIVE_ALIASES -msve-vector-bits=128 -march=armv8.5-a+crc+sve2

## Performace
When easy-simd translates x86 intrinsic instructions, it makes full use of the characteristics of arm SVE and NEON intrinsic, and implements the x86 intrinsic interface in the best performance way, making its performance better than AvxtoNeon.


